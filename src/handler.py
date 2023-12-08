#!/usr/bin/env python

import os
from typing import Generator
import runpod
from utils import validate_and_convert_sampling_params, intialize_llm_engine
from vllm.utils import random_uuid

# Default batch size, configurable via environment variable set in the Endpoint Template
DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', 10))

# Initialize the vLLM engine
llm = intialize_llm_engine()

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
    return_token_counts = job_input.get("count_tokens", False)
    
    # Validate and convert sampling parameters
    validated_params = validate_and_convert_sampling_params(sampling_params)

    # Generate a unique request ID
    request_id = random_uuid()

    # Initialize the vLLM generator
    results_generator = llm.generate(prompt, validated_params, request_id)
    last_output_text = ""
    batch = []

    # Process and yield the generated text
    async for request_output in results_generator:
        for output in request_output.outputs:
            if streaming:
                batch.append({"text": output.text[len(last_output_text):]})
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            last_output_text = output.text

    if not streaming:
        yield [{"text":last_output_text}]

    if batch and streaming:
        yield batch
    
    if return_token_counts and request_output is not None:
        token_counts = {"token_counts":{"input": len(request_output.prompt_token_ids),
               "output": len(output.outputs[-1].token_ids)}}
        yield token_counts

# Start the serverless worker
runpod.serverless.start({
    "handler": handler,
    "concurrency_modifier": lambda _: int(os.environ.get('CONCURRENCY_MODIFIER', 100)),
    "return_aggregate_stream": True
})