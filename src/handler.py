#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless worker. '''

import os
from typing import Generator

import runpod
from metrics import vllm_log_system_stats
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs, utils


NUM_GPU_SHARD = int(os.environ.get('NUM_GPU_SHARD', 1))  # Number of GPUs to shard the model across

# Prepare the model and tokenizer
MODEL_NAME = os.environ["MODEL_NAME"]
TOKENIZER = os.environ.get('TOKENIZER', MODEL_NAME)


MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', "/runpod-volume/")
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

USE_FULL_METRICS = os.environ.get('USE_FULL_METRICS', True)  # From the SDK, need to review later.

# Set up quantization-related parameters
QUANTIZATION = os.environ.get('QUANTIZATION', None)
DTYPE = "auto" if str(QUANTIZATION).lower() not in ['squeezellm', 'awq'] else "half"


# Prepare the engine's arguments
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    download_dir=MODEL_BASE_PATH,
    tokenizer=TOKENIZER,
    tokenizer_mode="auto",
    tensor_parallel_size=NUM_GPU_SHARD,
    dtype=DTYPE,
    disable_log_stats=False,
    quantization=QUANTIZATION,
)

# Create the vLLM asynchronous engine
llm = AsyncLLMEngine.from_engine_args(engine_args)

# Incorporate metrics tracking
llm.engine._log_system_stats = lambda x, y: vllm_log_system_stats(
    llm.engine, x, y)


def prepare_metrics() -> dict:
    # The vLLM metrics are updated every 5 seconds, see metrics.py for the _LOGGING_INTERVAL_SEC field.
    if hasattr(llm.engine, 'metrics'):
        return llm.engine.metrics
    else:
        return {}


# Validation
def validate_sampling_params(sampling_params):
    def validate_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def validate_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def validate_bool(value):
        if isinstance(value, bool):
            return value
        return None

    n = validate_int(sampling_params.get('n'))
    best_of = validate_int(sampling_params.get('best_of'))
    presence_penalty = validate_float(
        sampling_params.get('presence_penalty'))
    frequency_penalty = validate_float(
        sampling_params.get('frequency_penalty'))
    temperature = validate_float(sampling_params.get('temperature'))
    top_p = validate_float(sampling_params.get('top_p'))
    top_k = validate_int(sampling_params.get('top_k'))
    use_beam_search = validate_bool(
        sampling_params.get('use_beam_search'))
    stop = sampling_params.get('stop')
    ignore_eos = validate_bool(sampling_params.get('ignore_eos'))
    max_tokens = validate_int(sampling_params.get('max_tokens'))
    logprobs = validate_float(sampling_params.get('logprobs'))

    params = {
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
    return {k: v for k, v in params.items() if v is not None}


def validate_and_set_sampling_params(sampling_params):
    """
    Validates the given sampling parameters and creates a SamplingParams object.
    If no sampling parameters are provided, defaults are used.
    """
    if sampling_params:
        validated_params = validate_sampling_params(sampling_params)
        # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L7
        return SamplingParams(**validated_params)
    return SamplingParams()


async def handler_streaming(job: dict) -> Generator[dict[str, list], None, None]:
    '''
    This is the handler function that will be called by the serverless worker.
    '''
    print(f"Job received by handler: {job}")

    job_input = job['input']
    prompt = job_input['prompt']
    sampling_params = validate_and_set_sampling_params(job_input.get('sampling_params', None))

    request_id = utils.random_uuid()
    results_generator = llm.generate(prompt, sampling_params, request_id)
    aggregate_text = ""
    last_output_text = ""
    async for request_output in results_generator:
        for output in request_output.outputs:
            if output.text:
                yield {"text": output.text[len(last_output_text):]}
                last_output_text = output.text
                aggregate_text += output.text
    yield {"aggregate_text": aggregate_text}


runpod.serverless.start({
    "handler": handler_streaming,
    "concurrency_modifier": lambda _: int(os.environ.get('CONCURRENCY_MODIFIER', 100)),
    "return_aggregate_stream": True
})
