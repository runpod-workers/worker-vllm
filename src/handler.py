#!/usr/bin/env python
from typing import Generator
from vllm.utils import random_uuid
import runpod
from utils import validate_sampling_params
from engine import vLLMEngine

vllm_engine = vLLMEngine()

async def handler(job: dict) -> Generator[dict, None, None]:
    job_input = job["input"]
    llm_input = job_input.get("messages", job_input.get("prompt"))
    if job_input.get("apply_chat_template", False) or isinstance(llm_input, list):
       llm_input = vllm_engine.tokenizer.apply_chat_template(llm_input)
    
    stream = job_input.get("stream", False)
    batch_size = job_input.get("batch_size", vllm_engine.serverless_config.batch_size)

    validated_params = validate_sampling_params(job_input.get("sampling_params", {}))
    results_generator = vllm_engine.llm.generate(
        llm_input, validated_params, random_uuid()
    )
    
    n_responses, n_input_tokens, is_first_output =  validated_params.n, 0, True
    last_output_texts, token_counters= ["" for _ in range(n_responses)], {"batch": 0, "total": 0}
 
    batch = {
        "choices": [{"tokens": []} for _ in range(n_responses)],
    }
    
    async for request_output in results_generator:
        if is_first_output: # Count input tokens only once
            n_input_tokens = len(request_output.prompt_token_ids)
            is_first_output = False
            
        for output in request_output.outputs:
            output_index = output.index
            token_counters["total"] += 1
            if stream:
                new_output = output.text[len(last_output_texts[output_index]):]
                batch["choices"][output_index]["tokens"].append(new_output)
                token_counters["batch"] += 1
                
                if token_counters["batch"] >= batch_size:
                    batch["usage"] = {
                        "input": n_input_tokens,
                        "output": token_counters["total"],
                    }
                    yield batch
                    batch = {
                        "choices": [{"tokens": []} for _ in range(n_responses)],
                    }
                    token_counters["batch"] = 0
                    
            last_output_texts[output_index] = output.text
    
    if not stream:
        for output_index, output in enumerate(last_output_texts):
            batch["choices"][output_index]["tokens"] = [output]
        token_counters["batch"] += 1
    
    if token_counters["batch"] > 0:
        batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.serverless_config.max_concurrency,
        "return_aggregate_stream": True,
    }
)
