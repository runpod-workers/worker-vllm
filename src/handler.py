#!/usr/bin/env python
from typing import Generator
import runpod
from utils import validate_sampling_params, random_uuid
from engine import VLLMEngine

vllm_engine = VLLMEngine()
async def handler(job: dict) -> Generator[dict, None, None]:
    job_input = job["input"]
    llm_input = job_input.get("messages", job_input.get("prompt"))
    apply_chat_template = job_input.get("apply_chat_template", False)

    if apply_chat_template or isinstance(llm_input, list):
        llm_input = vllm_engine.tokenizer.apply_chat_template(llm_input)

    stream = job_input.get("stream", False)
    batch_size = job_input.get("batch_size", vllm_engine.serverless_config.default_batch_size)
    sampling_params = job_input.get("sampling_params", {})

    validated_params = validate_sampling_params(sampling_params)
    request_id = random_uuid()
    results_generator = vllm_engine.llm.generate(
        llm_input, validated_params, request_id
    )

    batch = {"tokens": []}
    last_output_text = ""
    n_input_tokens, is_first_output = 0, True
    
    async for request_output in results_generator:
        if is_first_output: # Count input tokens only once
            n_input_tokens = len(request_output.prompt_token_ids) 
            is_first_output = False
            
        for output in request_output.outputs:
            if stream:
                
                batch["tokens"].append(
                    output.text[len(last_output_text):]
                )
                
                if len(batch["tokens"]) >= batch_size or request_output.finished:
                    batch["usage"] = {
                        "input": n_input_tokens,
                        "output": len(output.token_ids),
                    }
                    yield batch
                    last_usage = batch["usage"]
                    batch = {"tokens": []}
                    
            last_output_text = output.text
    
    if not stream:
        yield {"tokens": [last_output_text], 
                "usage": last_usage}

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.serverless_config.max_concurrency,
        "return_aggregate_stream": True,
    }
)
