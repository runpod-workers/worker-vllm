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
if not os.path.exists(MODEL_BASE_PATH):
    os.makedirs(MODEL_BASE_PATH)

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
    presence_penalty = validate_float(
        sampling_params.get('presence_penalty'), 0.0)
    frequency_penalty = validate_float(
        sampling_params.get('frequency_penalty'), 0.0)
    temperature = validate_float(sampling_params.get('temperature'), 1.0)
    top_p = validate_float(sampling_params.get('top_p'), 1.0)
    top_k = validate_int(sampling_params.get('top_k'), -1)
    use_beam_search = validate_bool(
        sampling_params.get('use_beam_search'), False)
    stop = sampling_params.get('stop', None)
    ignore_eos = validate_bool(sampling_params.get('ignore_eos'), False)
    max_tokens = validate_int(sampling_params.get('max_tokens'), 256)
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
    sampling_params = job_input.get('sampling_params', None)

    # Might be able to remove this later
    sampling_params = validate_and_set_sampling_params(sampling_params)

    # Send request to VLLM
    request_id = utils.random_uuid()
    results_generator = llm.generate(prompt, sampling_params, request_id)

    stream_index = 0
    chunk_positions = None
    aggregate_text = []
    aggregate_metrics = {'input_tokens': 0, 'output_tokens': 0}

    async for request_output in results_generator:
        if chunk_positions is None:
            chunk_positions = [0] * len(request_output.outputs)

        text_outputs, output_tokens = [], []
        for idx, output in enumerate(request_output.outputs):
            chunk_pos = chunk_positions[idx]
            text_outputs.append(output.text[chunk_pos:])
            chunk_positions[idx] = len(output.text)
            output_tokens.append(len(output.token_ids) - chunk_pos)

        runpod_metrics = prepare_metrics() if USE_FULL_METRICS else {}
        if stream_index == 0:
            input_tokens_count = len(request_output.prompt_token_ids)
        else:
            input_tokens_count = 0

        runpod_metrics.update({
            "input_tokens": input_tokens_count,
            "output_tokens": sum(output_tokens),
            "scenario": 'stream',
            "stream_index": stream_index
        })

        stream_index += 1

        yield {
            "text": text_outputs,
            "input_tokens": runpod_metrics['input_tokens'],
            "output_tokens": runpod_metrics['output_tokens']
        }

        # Aggregate text and metrics
        if not aggregate_text:
            aggregate_text = [""] * len(text_outputs)
        for idx, text in enumerate(text_outputs):
            aggregate_text[idx] += text
        aggregate_metrics['input_tokens'] += runpod_metrics['input_tokens']
        aggregate_metrics['output_tokens'] += runpod_metrics['output_tokens']

    yield {
        "text": aggregate_text,
        "input_tokens": aggregate_metrics['input_tokens'],
        "output_tokens": aggregate_metrics['output_tokens']
    }


def concurrency_modifier() -> int:
    return os.environ.get('CONCURRENCY_MODIFIER', 100)


# Start the serverless worker with appropriate settings
runpod.serverless.start({
    "handler": handler_streaming,
    "concurrency_modifier": concurrency_modifier,
    "return_aggregate_stream": True
})
