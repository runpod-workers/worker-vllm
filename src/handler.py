#!/usr/bin/env python
from typing import Generator
import runpod
from utils import validate_and_convert_sampling_params, initialize_llm_engine, ServerlessConfig
from vllm.utils import random_uuid

serverless_config = ServerlessConfig()
llm, tokenizer = initialize_llm_engine()

def concurrency_modifier(current_concurrency) -> int:
    return max(0, serverless_config.max_concurrency - current_concurrency)

async def handler(job: dict) -> Generator[dict, None, None]:
    job_input = job["input"]
    prompt = job_input.get("prompt")
    apply_chat_template = job_input.get("apply_chat_template", False)
    messages = job_input.get("messages")
    
    if messages:
        prompt = tokenizer.apply_chat_template(messages)
    elif prompt and apply_chat_template:
        prompt = tokenizer.apply_chat_template(prompt)
    elif not prompt:
        raise ValueError("Must specify prompt or messages")
        
    stream = job_input.get("stream", False)
    batch_size = job_input.get("batch_size", serverless_config.default_batch_size)
    sampling_params = job_input.get("sampling_params", {})

    validated_params = validate_and_convert_sampling_params(sampling_params)
    request_id = random_uuid()
    results_generator = llm.generate(prompt, validated_params, request_id)

    batch, last_output_text = [], ""
    async for request_output in results_generator:
        for output in request_output.outputs:
            usage = {"input": len(request_output.prompt_token_ids), "output": len(output.token_ids)}
            
            if stream:
                batch.append({"text": output.text[len(last_output_text):], "usage": usage})
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            last_output_text = output.text

    if not stream:
        yield [{"text": last_output_text, "usage": usage}]

    if batch:
        yield batch
        
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": concurrency_modifier,
    "return_aggregate_stream": True
})
