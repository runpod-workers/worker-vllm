#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless worker. '''

import os
from typing import Generator

import runpod
from utils import EngineConfig, validate_and_convert_sampling_params
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, utils



# Load the configuration
config = EngineConfig()

# Prepare the engine's arguments
engine_args = AsyncEngineArgs(
    model=config.model_name,
    download_dir=config.model_base_path,
    tokenizer=config.tokenizer,
    tensor_parallel_size=config.num_gpu_shard,
    dtype=config.dtype,
    disable_log_stats=True,
    quantization=config.quantization,
    gpu_memory_utilization=0.97,
)

# Create the vLLM asynchronous engine
llm = AsyncLLMEngine.from_engine_args(engine_args)


# Handler function that will be called by the serverless worker
async def handler(job: dict) -> Generator[str, None, None]:
    job_input = job['input']
    prompt = job_input['prompt']
    streaming = job_input.get("streaming", False)
    sampling_params = job_input.get('sampling_params', {})
    validated_params = validate_and_convert_sampling_params(sampling_params)
    sampling_params_obj = SamplingParams(**validated_params)

    request_id = utils.random_uuid()
    results_generator = llm.generate(prompt, sampling_params_obj, request_id)
    last_output_text = ""
    
    async for request_output in results_generator:
        for output in request_output.outputs:
            if output.text:
                if streaming:
                    yield output.text[len(last_output_text):]
                last_output_text = output.text
    if not streaming:
        yield last_output_text

runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: int(os.environ.get('CONCURRENCY_MODIFIER', 100)),
    "return_aggregate_stream": True
})