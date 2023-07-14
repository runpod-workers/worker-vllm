#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless. '''
from typing import Dict
import runpod

# Start the VLLM serving layer on our RunPod worker.
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.utils import random_uuid

# Prepare the model and tokenizer
MODEL = 'lmsys/vicuna-13b-v1.3'
TOKENIZER = 'hf-internal-testing/llama-tokenizer'

# Prepare the engine's arguments
engine_args = AsyncEngineArgs(
    model=MODEL,
    tokenizer=TOKENIZER,
    tokenizer_mode= "auto",
    tensor_parallel_size= 1,
    dtype = "auto",
    seed = 0,
    worker_use_ray=False,
)
llm = AsyncLLMEngine.from_engine_args(engine_args)

# Validation
def validate_sampling_params(sampling_params):
    def validate_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def validate_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def validate_bool(value, default):
        if isinstance(value, bool):
            return value
        return default

    n = validate_int(sampling_params.get('n'), 1)
    best_of = validate_int(sampling_params.get('best_of'), None)
    presence_penalty = validate_float(sampling_params.get('presence_penalty'), 0.0)
    frequency_penalty = validate_float(sampling_params.get('frequency_penalty'), 0.0)
    temperature = validate_float(sampling_params.get('temperature'), 1.0)
    top_p = validate_float(sampling_params.get('top_p'), 1.0)
    top_k = validate_int(sampling_params.get('top_k'), -1)
    use_beam_search = validate_bool(sampling_params.get('use_beam_search'), False)
    stop = sampling_params.get('stop')
    ignore_eos = validate_bool(sampling_params.get('ignore_eos'), False)
    max_tokens = validate_int(sampling_params.get('max_tokens'), 16)
    logprobs = validate_float(sampling_params.get('logprobs'), None)

    return {
        'n': n,
        'best_of': best_of,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'use_beam_search': use_beam_search,
        'stop': stop,
        'ignore_eos': ignore_eos,
        'max_tokens': max_tokens,
        'logprobs': logprobs,
    }

async def handler(job):
    '''
    This is the handler function that will be called by the serverless worker.
    '''
    # Prompts
    prompts = job['prompts']

    # Validate the inputs
    sampling_params = job['sampling_params']
    sampling_params = validate_sampling_params(sampling_params)

    # Sampling parameters
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L7
    sampling_params = SamplingParams(**sampling_params)

    # Send request to VLLM
    request_id = random_uuid()
    results_generator = llm.generate(prompts, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return ret

runpod.serverless.start({"handler": handler})
