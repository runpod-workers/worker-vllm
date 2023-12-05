#!/usr/bin/env python

import os
from typing import Generator
import runpod
from utils import EngineConfig, validate_and_convert_sampling_params
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, utils

# Default batch size, configurable via environment variable set in the Endpoint Template
DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', 10))

# Load the configuration for the vLLM engine
config = EngineConfig()
engine_args = AsyncEngineArgs(
    model=config.model_name,
    download_dir=config.model_base_path,
    tokenizer=config.tokenizer,
    tensor_parallel_size=config.num_gpu_shard,
    dtype=config.dtype,
    disable_log_stats=config.disable_log_stats,
    quantization=config.quantization,
    gpu_memory_utilization=0.98,
)

# Create the asynchronous vLLM engine
llm = AsyncLLMEngine.from_engine_args(engine_args)

async def handler(job: dict) -> Generator[str, None, None]:
    """
    Asynchronous Generator Handler for the vLLM worker.

    Args:
        job (dict): A dictionary containing job details, including the prompt and other parameters.

    Yields:
        Generator[str, None, None]: A generator that yields generated text outputs. Format: List[str]
    """
    # Extract the job inputs
    job_input = job["input"]
    prompt = job_input["prompt"]
    streaming = job_input.get("streaming", False)
    batch_size = job_input.get("batch_size", DEFAULT_BATCH_SIZE)
    sampling_params = job_input.get("sampling_params", {})
    
    # Validate and convert sampling parameters
    validated_params = validate_and_convert_sampling_params(sampling_params)
    sampling_params_obj = SamplingParams(**validated_params)

    # Generate a unique request ID
    request_id = utils.random_uuid()

    # Initialize the vLLM generator
    results_generator = llm.generate(prompt, sampling_params_obj, request_id)
    last_output_text = ""
    batch = []

    # Process and yield the generated text
    async for request_output in results_generator:
        for output in request_output.outputs:
            if streaming:
                batch.append(output.text[len(last_output_text):])
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            last_output_text = output.text

    if not streaming:
        yield [last_output_text]

    if batch and streaming:
        yield batch

# Start the serverless worker
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: int(os.environ.get('CONCURRENCY_MODIFIER', 100)),
    "return_aggregate_stream": True
})