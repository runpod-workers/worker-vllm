#!/usr/bin/env python
''' Contains the handler function that will be called by the serverless worker. '''

# Start the vLLM serving layer on our RunPod worker.
from typing import Generator
from metrics import vllm_log_system_stats
from templates import DEFAULT_TEMPLATE, LLAMA2_TEMPLATE
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.utils import random_uuid
import runpod
import os

# Prepare the model and tokenizer
MODEL_NAME = os.environ.get('MODEL_NAME')
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/')
STREAMING = os.environ.get('STREAMING', False) == 'True'
TOKENIZER = os.environ.get('TOKENIZER', None)
USE_FULL_METRICS = os.environ.get('USE_FULL_METRICS', True)

if not MODEL_NAME:
    print("Error: The model has not been provided.")

# Tensor parallelism
try:
    NUM_GPU_SHARD = int(os.environ.get('NUM_GPU_SHARD', 1))
except ValueError:
    print("Error: NUM_GPU_SHARD should be an integer. Using default value of 1.")
    NUM_GPU_SHARD = 1

# Prepare the engine's arguments
engine_args = AsyncEngineArgs(
    model=f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}",
    tokenizer=TOKENIZER,
    tokenizer_mode="auto",
    tensor_parallel_size=NUM_GPU_SHARD,
    dtype="auto",
    seed=0,
    max_num_batched_tokens=8192,
    disable_log_stats=False,
    # max_num_seqs=256,
)

# Create the vLLM asynchronous engine
llm = AsyncLLMEngine.from_engine_args(engine_args)

# Incorporate metrics tracking
llm.engine._log_system_stats = lambda x, y: vllm_log_system_stats(
    llm.engine, x, y)

def concurrency_controller() -> bool:
    # Calculate pending sequences
    total_pending_sequences = len(llm.engine.scheduler.waiting) + len(llm.engine.scheduler.swapped)
    print("Total pending sequences in vLLM queue: {}".format(total_pending_sequences))

    # Enable auto-scaling if pending sequences exist
    return total_pending_sequences > 30

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
    print("Job received by handler: {}".format(job))

    # Retrieve the job input.
    job_input = job['input']

    # Utilize the built-in llama2 template if a llama2 base model is being employed.
    llama_models = ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf", "elinas/chronos-13b-v2"]
    if any(model_name.lower() in MODEL_NAME.lower() for model_name in llama_models):
        template = LLAMA2_TEMPLATE
    else:
        template = DEFAULT_TEMPLATE

    # Create the prompt using the template.
    prompt = template(job_input['prompt'])

    # Validate and set sampling parameters
    sampling_params = validate_and_set_sampling_params(job_input.get('sampling_params', None))

    # Print job input and sampling parameters
    print("Job Input:", job_input)
    print("Sampling Parameters:", sampling_params)

    # Send request to VLLM
    request_id = random_uuid()
    results_generator = llm.generate(prompt, sampling_params, request_id)

    # Keep track of the stream's information to perform the appropriate chunking.
    class Tracker():
        def __init__(self):
            self.positions = None
            self.stream_index = 0

        def inc_stream_idx(self):
            self.stream_index +=1

    tracker = Tracker()

    def extract_next_chunk(request_output):
        """
        Extracts and processes generated chunks and token counts from the request output.

        Args:
            request_output (CompletionOutput): The output of a language model request.

        Returns:
            tuple: A tuple containing two lists - chunk_outputs (extracted chunks) and num_output_tokens (generated token counts).
        """
        chunk_outputs = []  # List to store extracted chunks
        num_output_tokens = []  # List to store generated token counts

        # Iterate over each completion in the request output
        for idx, completion in enumerate(request_output.outputs):
            # Extract the current chunk position from the tracker
            chunk_pos = tracker.positions[idx]['chunk_pos']

            # Append the chunk to the output
            chunk_outputs.append(completion.text[chunk_pos:])

            # Update the chunk position in the tracker
            tracker.positions[idx]['chunk_pos'] = len(completion.text)

            # Calculate the number of generated tokens in the current completion
            num_generated_tokens = len(completion.token_ids) - tracker.positions[idx]['token_pos']

            # Append the token count to the output
            num_output_tokens.append(num_generated_tokens)

            # Update the token position in the tracker
            tracker.positions[idx]['token_pos'] = len(completion.token_ids)

        return chunk_outputs, num_output_tokens

    async for request_output in results_generator:
        # Initialize chunk positions if not already done
        if tracker.positions is None:
            tracker.positions = [{'chunk_pos': 0, 'token_pos': 0}] * len(request_output.outputs)

        # Metrics for the vLLM serverless worker
        runpod_metrics = prepare_metrics() if USE_FULL_METRICS else {}

        # Number of generated sequences
        num_seqs = sampling_params.n

        # Extract the next chunk from the output
        text_outputs, output_tokens = extract_next_chunk(request_output)

        # Record job input and token counts
        # runpod_metrics['job_input'] = job_input

        # Only include the input_tokens count for the very first stream response. This is to avoid duplicate counting.
        if tracker.stream_index == 0:
            input_tokens_count = len(request_output.prompt_token_ids)
            runpod_metrics['input_tokens'] = sum([input_tokens_count] * num_seqs)
        else:
            runpod_metrics['input_tokens'] = sum([0] * num_seqs)

        # Include the output tokens count [#, #, #, ...]
        runpod_metrics['output_tokens'] = sum(output_tokens)

        # Store the scenario type and stream index
        runpod_metrics['scenario'] = 'stream'
        runpod_metrics['stream_index'] = tracker.stream_index

        # Increment the index within the stream
        tracker.inc_stream_idx()

        ret = {
            "text": text_outputs,
            "input_tokens": runpod_metrics['input_tokens'],
            "output_tokens": runpod_metrics['output_tokens']
        }

        # Include metrics for the job.
        runpod.serverless.modules.rp_metrics.metrics_collector.push_metrics_internal(
            job_id=job['id'], 
            metrics=runpod_metrics
        )        

        # Keep track of the final output
        final_output = request_output

        # Include metrics in the highest level for the job output for aggregrate.
        def aggregate_function(streamed_outputs):
            aggregate_output = [""] * len(streamed_outputs[0]['text'])
            for stream in streamed_outputs:
                for id, seq in enumerate(stream['text']):
                    aggregate_output[id] += seq

            # Number of generated sequences
            num_seqs = sampling_params.n

            # Aggregate metrics to expose to the user
            input_tokens = len(final_output.prompt_token_ids) * num_seqs
            output_tokens = sum([len(output.token_ids) for output in final_output.outputs])

            return {
                "text": aggregate_output,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
    
        # Update the aggregate transformation function
        runpod.serverless.modules.rp_metrics.metrics_collector.update_stream_aggregate(
            job_id=job['id'], 
            aggregate_function=aggregate_function
        )

        # Yield the output
        yield ret


async def handler(job: dict) -> dict[str, list]:
    '''
    This is the handler function that will be called by the serverless worker.
    '''
    print("Job received by handler: {}".format(job))

    # Retrieve the job input.
    job_input = job['input']

    # Utilize the built-in llama2 template if a llama2 base model is being employed.
    llama_models = ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf", "elinas/chronos-13b-v2"]
    if any(model_name.lower() in MODEL_NAME.lower() for model_name in llama_models):
        template = LLAMA2_TEMPLATE
    else:
        template = DEFAULT_TEMPLATE

    # Create the prompt using the template.
    prompt = template(job_input['prompt'])

    # Validate and set sampling parameters
    sampling_params = validate_and_set_sampling_params(job_input.get('sampling_params', None))

    # Print job input and sampling parameters
    print("Job Input:", job_input)
    print("Sampling Parameters:", sampling_params)

    # Send request to VLLM
    request_id = random_uuid()
    results_generator = llm.generate(prompt, sampling_params, request_id)

    # Get the final generated output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Extract prompt and text outputs
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]

    # Number of generated sequences
    num_seqs = sampling_params.n

    # Prepare metrics if full metrics are enabled
    runpod_metrics = prepare_metrics() if USE_FULL_METRICS else {}

    # Record job input and token counts
    # runpod_metrics['job_input'] = job_input

    runpod_metrics['input_tokens'] = len(final_output.prompt_token_ids) * num_seqs
    runpod_metrics['output_tokens'] = sum([len(output.token_ids) for output in final_output.outputs])

    # Store the scenario type
    runpod_metrics['scenario'] = 'batch'

    # Include metrics for the job.
    runpod.serverless.modules.rp_metrics.metrics_collector.push_metrics_internal(
        job_id=job['id'], 
        metrics=runpod_metrics
    )

    ret = {
        "text": text_outputs,
        "input_tokens": runpod_metrics['input_tokens'],
        "output_tokens": runpod_metrics['output_tokens']
    }
    return ret

# Start the serverless worker with appropriate settings
if STREAMING:
    print("Starting the vLLM serverless worker with streaming enabled.")
    runpod.serverless.start({
        "handler": handler_streaming, 
        "concurrency_controller": concurrency_controller, 
        "return_aggregate_stream": True
    })
else:
    print("Starting the vLLM serverless worker with streaming disabled.")
    runpod.serverless.start({
        "handler": handler, 
        "concurrency_controller": 
        concurrency_controller
    })
