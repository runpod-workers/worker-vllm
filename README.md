<div align="center">

# OpenAI-Compatible vLLM Serverless Endpoint Worker

Deploy OpenAI-Compatible Blazing-Fast LLM Endpoints powered by the [vLLM](https://github.com/vllm-project/vllm) Inference Engine on RunPod Serverless with just a few clicks.

</div>

## Table of Contents

- [Setting up the Serverless Worker](#setting-up-the-serverless-worker)
  - [Option 1: Deploy Any Model Using Pre-Built Docker Image **[RECOMMENDED]**](#option-1-deploy-any-model-using-pre-built-docker-image-recommended)
    - [Environment Variables](#environment-variables)
      - [LLM Settings](#llm-settings)
      - [Tokenizer Settings](#tokenizer-settings)
      - [System and Parallelism Settings](#system-and-parallelism-settings)
      - [Streaming Batch Size Settings](#streaming-batch-size-settings)
      - [OpenAI Settings](#openai-settings)
      - [Serverless Settings](#serverless-settings)
  - [Option 2: Build Docker Image with Model Inside](#option-2-build-docker-image-with-model-inside)
    - [Prerequisites](#prerequisites-1)
    - [Arguments](#arguments)
    - [Example: Building an image with OpenChat-3.5](#example-building-an-image-with-openchat-35)
      - [(Optional) Including Huggingface Token](#optional-including-huggingface-token)
  - [Compatible Model Architectures](#compatible-model-architectures)
- [Usage: OpenAI Compatibility](#usage-openai-compatibility)
  - [Modifying your OpenAI Codebase to use your deployed vLLM Worker](#modifying-your-openai-codebase-to-use-your-deployed-vllm-worker)
  - [OpenAI Request Input Parameters](#openai-request-input-parameters)
    - [Chat Completions](#chat-completions)
  - [Examples: Using your RunPod endpoint with OpenAI](#examples-using-your-runpod-endpoint-with-openai)
- [Usage: standard](#non-openai-usage)
  - [Input Request Parameters](#input-request-parameters)
    - [Text Input Formats](#text-input-formats)
    - [Sampling Parameters](#sampling-parameters)

# Setting up the Serverless Worker

## Option 1: Deploy Any Model Using Pre-Built Docker Image [Recommended]

**🚀 Deploy Guide**: Follow our [step-by-step deployment guide](https://docs.runpod.io/serverless/vllm/get-started) to deploy using the RunPod Console.

**📦 Docker Image**: `runpod/worker-v1-vllm:<version>stable-cuda12.1.0`

- **Available Versions**: See [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases)
- **CUDA Compatibility**: Requires CUDA >= 12.1

### Environment Variables

Use these to configure worker-vllm so it works for your use case / model.

#### LLM Settings

| `Name`                                           | `Default`           | `Type/Choices`                                              | `Description`                                                                                                                                          |
| ------------------------------------------------ | ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL_NAME`                                     | 'facebook/opt-125m' | `str`                                                       | Name or path of the Hugging Face model to use.                                                                                                         |
| `TOKENIZER`                                      | None                | `str`                                                       | Name or path of the Hugging Face tokenizer to use.                                                                                                     |
| `SKIP_TOKENIZER_INIT`                            | False               | `bool`                                                      | Skip initialization of tokenizer and detokenizer.                                                                                                      |
| `TOKENIZER_MODE`                                 | 'auto'              | ['auto', 'slow']                                            | The tokenizer mode.                                                                                                                                    |
| `TRUST_REMOTE_CODE`                              | `False`             | `bool`                                                      | Trust remote code from Hugging Face.                                                                                                                   |
| `DOWNLOAD_DIR`                                   | None                | `str`                                                       | Directory to download and load the weights.                                                                                                            |
| `LOAD_FORMAT`                                    | 'auto'              | `str`                                                       | The format of the model weights to load.                                                                                                               |
| `HF_TOKEN`                                       | -                   | `str`                                                       | Hugging Face token for private and gated models.                                                                                                       |
| `DTYPE`                                          | 'auto'              | ['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'] | Data type for model weights and activations.                                                                                                           |
| `KV_CACHE_DTYPE`                                 | 'auto'              | ['auto', 'fp8']                                             | Data type for KV cache storage.                                                                                                                        |
| `QUANTIZATION_PARAM_PATH`                        | None                | `str`                                                       | Path to the JSON file containing the KV cache scaling factors.                                                                                         |
| `MAX_MODEL_LEN`                                  | None                | `int`                                                       | Model context length.                                                                                                                                  |
| `GUIDED_DECODING_BACKEND`                        | 'outlines'          | ['outlines', 'lm-format-enforcer']                          | Which engine will be used for guided decoding by default.                                                                                              |
| `DISTRIBUTED_EXECUTOR_BACKEND`                   | None                | ['ray', 'mp']                                               | Backend to use for distributed serving.                                                                                                                |
| `WORKER_USE_RAY`                                 | False               | `bool`                                                      | Deprecated, use --distributed-executor-backend=ray.                                                                                                    |
| `PIPELINE_PARALLEL_SIZE`                         | 1                   | `int`                                                       | Number of pipeline stages.                                                                                                                             |
| `TENSOR_PARALLEL_SIZE`                           | 1                   | `int`                                                       | Number of tensor parallel replicas.                                                                                                                    |
| `MAX_PARALLEL_LOADING_WORKERS`                   | None                | `int`                                                       | Load model sequentially in multiple batches.                                                                                                           |
| `RAY_WORKERS_USE_NSIGHT`                         | False               | `bool`                                                      | If specified, use nsight to profile Ray workers.                                                                                                       |
| `ENABLE_PREFIX_CACHING`                          | False               | `bool`                                                      | Enables automatic prefix caching.                                                                                                                      |
| `DISABLE_SLIDING_WINDOW`                         | False               | `bool`                                                      | Disables sliding window, capping to sliding window size.                                                                                               |
| `USE_V2_BLOCK_MANAGER`                           | False               | `bool`                                                      | Use BlockSpaceMangerV2.                                                                                                                                |
| `NUM_LOOKAHEAD_SLOTS`                            | 0                   | `int`                                                       | Experimental scheduling config necessary for speculative decoding.                                                                                     |
| `SEED`                                           | 0                   | `int`                                                       | Random seed for operations.                                                                                                                            |
| `NUM_GPU_BLOCKS_OVERRIDE`                        | None                | `int`                                                       | If specified, ignore GPU profiling result and use this number of GPU blocks.                                                                           |
| `MAX_NUM_BATCHED_TOKENS`                         | None                | `int`                                                       | Maximum number of batched tokens per iteration.                                                                                                        |
| `MAX_NUM_SEQS`                                   | 256                 | `int`                                                       | Maximum number of sequences per iteration.                                                                                                             |
| `MAX_LOGPROBS`                                   | 20                  | `int`                                                       | Max number of log probs to return when logprobs is specified in SamplingParams.                                                                        |
| `DISABLE_LOG_STATS`                              | False               | `bool`                                                      | Disable logging statistics.                                                                                                                            |
| `QUANTIZATION`                                   | None                | ['awq', 'squeezellm', 'gptq', 'bitsandbytes']               | Method used to quantize the weights.                                                                                                                   |
| `ROPE_SCALING`                                   | None                | `dict`                                                      | RoPE scaling configuration in JSON format.                                                                                                             |
| `ROPE_THETA`                                     | None                | `float`                                                     | RoPE theta. Use with rope_scaling.                                                                                                                     |
| `TOKENIZER_POOL_SIZE`                            | 0                   | `int`                                                       | Size of tokenizer pool to use for asynchronous tokenization.                                                                                           |
| `TOKENIZER_POOL_TYPE`                            | 'ray'               | `str`                                                       | Type of tokenizer pool to use for asynchronous tokenization.                                                                                           |
| `TOKENIZER_POOL_EXTRA_CONFIG`                    | None                | `dict`                                                      | Extra config for tokenizer pool.                                                                                                                       |
| `ENABLE_LORA`                                    | False               | `bool`                                                      | If True, enable handling of LoRA adapters.                                                                                                             |
| `MAX_LORAS`                                      | 1                   | `int`                                                       | Max number of LoRAs in a single batch.                                                                                                                 |
| `MAX_LORA_RANK`                                  | 16                  | `int`                                                       | Max LoRA rank.                                                                                                                                         |
| `LORA_EXTRA_VOCAB_SIZE`                          | 256                 | `int`                                                       | Maximum size of extra vocabulary for LoRA adapters.                                                                                                    |
| `LORA_DTYPE`                                     | 'auto'              | ['auto', 'float16', 'bfloat16', 'float32']                  | Data type for LoRA.                                                                                                                                    |
| `LONG_LORA_SCALING_FACTORS`                      | None                | `tuple`                                                     | Specify multiple scaling factors for LoRA adapters.                                                                                                    |
| `MAX_CPU_LORAS`                                  | None                | `int`                                                       | Maximum number of LoRAs to store in CPU memory.                                                                                                        |
| `FULLY_SHARDED_LORAS`                            | False               | `bool`                                                      | Enable fully sharded LoRA layers.                                                                                                                      |
| `LORA_MODULES`                                   | `[]`                | `list[dict]`                                                | Add lora adapters from Hugging Face `[{"name": "xx", "path": "xxx/xxxx", "base_model_name": "xxx/xxxx"}`                                               |
| `SCHEDULER_DELAY_FACTOR`                         | 0.0                 | `float`                                                     | Apply a delay before scheduling next prompt.                                                                                                           |
| `ENABLE_CHUNKED_PREFILL`                         | False               | `bool`                                                      | Enable chunked prefill requests.                                                                                                                       |
| `SPECULATIVE_MODEL`                              | None                | `str`                                                       | The name of the draft model to be used in speculative decoding.                                                                                        |
| `NUM_SPECULATIVE_TOKENS`                         | None                | `int`                                                       | The number of speculative tokens to sample from the draft model.                                                                                       |
| `SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE`         | None                | `int`                                                       | Number of tensor parallel replicas for the draft model.                                                                                                |
| `SPECULATIVE_MAX_MODEL_LEN`                      | None                | `int`                                                       | The maximum sequence length supported by the draft model.                                                                                              |
| `SPECULATIVE_DISABLE_BY_BATCH_SIZE`              | None                | `int`                                                       | Disable speculative decoding if the number of enqueue requests is larger than this value.                                                              |
| `NGRAM_PROMPT_LOOKUP_MAX`                        | None                | `int`                                                       | Max size of window for ngram prompt lookup in speculative decoding.                                                                                    |
| `NGRAM_PROMPT_LOOKUP_MIN`                        | None                | `int`                                                       | Min size of window for ngram prompt lookup in speculative decoding.                                                                                    |
| `SPEC_DECODING_ACCEPTANCE_METHOD`                | 'rejection_sampler' | ['rejection_sampler', 'typical_acceptance_sampler']         | Specify the acceptance method for draft token verification in speculative decoding.                                                                    |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD` | None                | `float`                                                     | Set the lower bound threshold for the posterior probability of a token to be accepted.                                                                 |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA`     | None                | `float`                                                     | A scaling factor for the entropy-based threshold for token acceptance.                                                                                 |
| `MODEL_LOADER_EXTRA_CONFIG`                      | None                | `dict`                                                      | Extra config for model loader.                                                                                                                         |
| `PREEMPTION_MODE`                                | None                | `str`                                                       | If 'recompute', the engine performs preemption-aware recomputation. If 'save', the engine saves activations into the CPU memory as preemption happens. |
| `PREEMPTION_CHECK_PERIOD`                        | 1.0                 | `float`                                                     | How frequently the engine checks if a preemption happens.                                                                                              |
| `PREEMPTION_CPU_CAPACITY`                        | 2                   | `float`                                                     | The percentage of CPU memory used for the saved activations.                                                                                           |
| `DISABLE_LOGGING_REQUEST`                        | False               | `bool`                                                      | Disable logging requests.                                                                                                                              |
| `MAX_LOG_LEN`                                    | None                | `int`                                                       | Max number of prompt characters or prompt ID numbers being printed in log.                                                                             |

#### Tokenizer Settings

| `Name`                 | `Default` | `Type/Choices`                      | `Description`                                                                                     |
| ---------------------- | --------- | ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `TOKENIZER_NAME`       | `None`    | `str`                               | Tokenizer repository to use a different tokenizer than the model's default.                       |
| `TOKENIZER_REVISION`   | `None`    | `str`                               | Tokenizer revision to load.                                                                       |
| `CUSTOM_CHAT_TEMPLATE` | `None`    | `str` of single-line jinja template | Custom chat jinja template. [More Info](https://huggingface.co/docs/transformers/chat_templating) |

#### System and Parallelism Settings

| `Name`                         | `Default` | `Type/Choices`  | `Description`                                                                                                                       |
| ------------------------------ | --------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `GPU_MEMORY_UTILIZATION`       | `0.95`    | `float`         | Sets GPU VRAM utilization.                                                                                                          |
| `MAX_PARALLEL_LOADING_WORKERS` | `None`    | `int`           | Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models.                          |
| `BLOCK_SIZE`                   | `16`      | `8`, `16`, `32` | Token block size for contiguous chunks of tokens.                                                                                   |
| `SWAP_SPACE`                   | `4`       | `int`           | CPU swap space size (GiB) per GPU.                                                                                                  |
| `ENFORCE_EAGER`                | False     | `bool`          | Always use eager-mode PyTorch. If False(`0`), will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. |
| `MAX_SEQ_LEN_TO_CAPTURE`       | `8192`    | `int`           | Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode.     |
| `DISABLE_CUSTOM_ALL_REDUCE`    | `0`       | `int`           | Enables or disables custom all reduce.                                                                                              |

#### Streaming Batch Size Settings

The way this works is that the first request will have a batch size of `DEFAULT_MIN_BATCH_SIZE`, and each subsequent request will have a batch size of `previous_batch_size * DEFAULT_BATCH_SIZE_GROWTH_FACTOR`. This will continue until the batch size reaches `DEFAULT_BATCH_SIZE`. E.g. for the default values, the batch sizes will be `1, 3, 9, 27, 50, 50, 50, ...`. You can also specify this per request, with inputs `max_batch_size`, `min_batch_size`, and `batch_size_growth_factor`. This has nothing to do with vLLM's internal batching, but rather the number of tokens sent in each HTTP request from the worker

| `Name`                             | `Default` | `Type/Choices` | `Description`                                                                                             |
| ---------------------------------- | --------- | -------------- | --------------------------------------------------------------------------------------------------------- |
| `DEFAULT_BATCH_SIZE`               | `50`      | `int`          | Default and Maximum batch size for token streaming to reduce HTTP calls.                                  |
| `DEFAULT_MIN_BATCH_SIZE`           | `1`       | `int`          | Batch size for the first request, which will be multiplied by the growth factor every subsequent request. |
| `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` | `3`       | `float`        | Growth factor for dynamic batch size.                                                                     |

#### OpenAI Settings

| `Name`                              | `Default`   | `Type/Choices`   | `Description`                                                                                                                                                                                                     |
| ----------------------------------- | ----------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAW_OPENAI_OUTPUT`                 | `1`         | boolean as `int` | Enables raw OpenAI SSE format string output when streaming. **Required** to be enabled (which it is by default) for OpenAI compatibility.                                                                         |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `None`      | `str`            | Overrides the name of the served model from model repo/path to specified name, which you will then be able to use the value for the `model` parameter when making OpenAI requests                                 |
| `OPENAI_RESPONSE_ROLE`              | `assistant` | `str`            | Role of the LLM's Response in OpenAI Chat Completions.                                                                                                                                                            |
| `ENABLE_AUTO_TOOL_CHOICE`           | `false`     | `bool`           | Enables automatic tool selection for supported models. Set to `true` to activate.                                                                                                                                 |
| `TOOL_CALL_PARSER`                  | `None`      | `str`            | Specifies the parser for tool calls. Options: `mistral`, `hermes`, `llama3_json`, `llama4_json`, `llama4_pythonic`, `granite`, `granite-20b-fc`, `deepseek_v3`, `internlm`, `jamba`, `phi4_mini_json`, `pythonic` |

#### Serverless Settings

| `Name`                 | `Default` | `Type/Choices` | `Description`                                                                                                                                                              |
| ---------------------- | --------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_CONCURRENCY`      | `300`     | `int`          | Max concurrent requests per worker. vLLM has an internal queue, so you don't have to worry about limiting by VRAM, this is for improving scaling/load balancing efficiency |
| `DISABLE_LOG_STATS`    | False     | `bool`         | Enables or disables vLLM stats logging.                                                                                                                                    |
| `DISABLE_LOG_REQUESTS` | False     | `bool`         | Enables or disables vLLM request logging.                                                                                                                                  |

## Option 2: Build Docker Image with Model Inside

To build an image with the model baked in, you must specify the following docker arguments when building the image.

### Prerequisites

- Docker

### Arguments

- **Required**
  - `MODEL_NAME`
- **Optional**
  - `MODEL_REVISION`: Model revision to load (default: `main`).
  - `BASE_PATH`: Storage directory where huggingface cache and model will be located. (default: `/runpod-volume`, which will utilize network storage if you attach it or create a local directory within the image if you don't. If your intention is to bake the model into the image, you should set this to something like `/models` to make sure there are no issues if you were to accidentally attach network storage.)
  - `QUANTIZATION`
  - `WORKER_CUDA_VERSION`: `12.1.0` (`12.1.0` is recommended for optimal performance).
  - `TOKENIZER_NAME`: Tokenizer repository if you would like to use a different tokenizer than the one that comes with the model. (default: `None`, which uses the model's tokenizer)
  - `TOKENIZER_REVISION`: Tokenizer revision to load (default: `main`).

For the remaining settings, you may apply them as environment variables when running the container. Supported environment variables are listed in the [Environment Variables](#environment-variables) section.

### Example: Building an image with OpenChat-3.5

```bash
docker build -t username/image:tag --build-arg MODEL_NAME="openchat/openchat_3.5" --build-arg BASE_PATH="/models" .
```

### (Optional) Including Huggingface Token

If the model you would like to deploy is private or gated, you will need to include it during build time as a Docker secret, which will protect it from being exposed in the image and on DockerHub.

1. Enable Docker BuildKit (required for secrets).

```bash
export DOCKER_BUILDKIT=1
```

2. Export your Hugging Face token as an environment variable

```bash
export HF_TOKEN="your_token_here"
```

2. Add the token as a secret when building

```bash
docker build -t username/image:tag --secret id=HF_TOKEN --build-arg MODEL_NAME="openchat/openchat_3.5" .
```

# Compatible Model Architectures

You can deploy **any model on Hugging Face** that is supported by vLLM. For the complete and up-to-date list of supported model architectures, see the [vLLM Supported Models documentation](https://docs.vllm.ai/en/latest/models/supported_models.html#list-of-text-only-language-models).

# Usage: OpenAI Compatibility

The vLLM Worker is fully compatible with OpenAI's API, and you can use it with any OpenAI Codebase by changing only 3 lines in total. The supported routes are <ins>Chat Completions</ins> and <ins>Models</ins> - with both streaming and non-streaming.

## Modifying your OpenAI Codebase to use your deployed vLLM Worker

**Python** (similar to Node.js, etc.):

1. When initializing the OpenAI Client in your code, change the `api_key` to your RunPod API Key and the `base_url` to your RunPod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`, filling in your deployed endpoint ID. For example, if your Endpoint ID is `abc1234`, the URL would be `https://api.runpod.ai/v2/abc1234/openai/v1`.

   - Before:

   ```python
   from openai import OpenAI

   client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
   ```

   - After:

   ```python
   from openai import OpenAI

   client = OpenAI(
       api_key=os.environ.get("RUNPOD_API_KEY"),
       base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1",
   )
   ```

2. Change the `model` parameter to your deployed model's name whenever using Completions or Chat Completions.
   - Before:
   ```python
   response = client.chat.completions.create(
       model="gpt-3.5-turbo",
       messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
       temperature=0,
       max_tokens=100,
   )
   ```
   - After:
   ```python
   response = client.chat.completions.create(
       model="<YOUR DEPLOYED MODEL REPO/NAME>",
       messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
       temperature=0,
       max_tokens=100,
   )
   ```

**Using http requests**:

1. Change the `Authorization` header to your RunPod API Key and the `url` to your RunPod Serverless Endpoint URL in the following format: `https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1`
   - Before:
   ```bash
   curl https://api.openai.com/v1/chat/completions \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $OPENAI_API_KEY" \
   -d '{
   "model": "gpt-4",
   "messages": [
     {
       "role": "user",
       "content": "Why is RunPod the best platform?"
     }
   ],
   "temperature": 0,
   "max_tokens": 100
   }'
   ```
   - After:
   ```bash
   curl https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1/chat/completions \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer <YOUR OPENAI API KEY>" \
   -d '{
   "model": "<YOUR DEPLOYED MODEL REPO/NAME>",
   "messages": [
     {
       "role": "user",
       "content": "Why is RunPod the best platform?"
     }
   ],
   "temperature": 0,
   "max_tokens": 100
   }'
   ```

## OpenAI Request Input Parameters:

When using the chat completion feature of the vLLM Serverless Endpoint Worker, you can customize your requests with the following parameters:

### Chat Completions [RECOMMENDED]

<details>
  <summary>Supported Chat Completions Inputs and Descriptions</summary>

| Parameter           | Type                             | Default Value | Description                                                                                                                                                                                                                                                  |
| ------------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `messages`          | Union[str, List[Dict[str, str]]] |               | List of messages, where each message is a dictionary with a `role` and `content`. The model's chat template will be applied to the messages automatically, so the model must have one or it should be specified as `CUSTOM_CHAT_TEMPLATE` env var.           |
| `model`             | str                              |               | The model repo that you've deployed on your RunPod Serverless Endpoint. If you are unsure what the name is or are baking the model in, use the guide to get the list of available models in the **Examples: Using your RunPod endpoint with OpenAI** section |
| `temperature`       | Optional[float]                  | 0.7           | Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.                                                                              |
| `top_p`             | Optional[float]                  | 1.0           | Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                                                                                                            |
| `n`                 | Optional[int]                    | 1             | Number of output sequences to return for the given prompt.                                                                                                                                                                                                   |
| `max_tokens`        | Optional[int]                    | None          | Maximum number of tokens to generate per output sequence.                                                                                                                                                                                                    |
| `seed`              | Optional[int]                    | None          | Random seed to use for the generation.                                                                                                                                                                                                                       |
| `stop`              | Optional[Union[str, List[str]]]  | list          | List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.                                                                                                                                     |
| `stream`            | Optional[bool]                   | False         | Whether to stream or not                                                                                                                                                                                                                                     |
| `presence_penalty`  | Optional[float]                  | 0.0           | Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                                                          |
| `frequency_penalty` | Optional[float]                  | 0.0           | Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.                                                              |
| `logit_bias`        | Optional[Dict[str, float]]       | None          | Unsupported by vLLM                                                                                                                                                                                                                                          |
| `user`              | Optional[str]                    | None          | Unsupported by vLLM                                                                                                                                                                                                                                          |

Additional parameters supported by vLLM:
| `best_of` | Optional[int] | None | Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`. |
| `top_k` | Optional[int] | -1 | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
| `ignore_eos` | Optional[bool] | False | Whether to ignore the EOS token and continue generating tokens after the EOS token is generated. |
| `use_beam_search` | Optional[bool] | False | Whether to use beam search instead of sampling. |
| `stop_token_ids` | Optional[List[int]] | list | List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens. |
| `skip_special_tokens` | Optional[bool] | True | Whether to skip special tokens in the output. |
| `spaces_between_special_tokens`| Optional[bool] | True | Whether to add spaces between special tokens in the output. Defaults to True. |
| `add_generation_prompt` | Optional[bool] | True | Read more [here](https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts) |
| `echo` | Optional[bool] | False | Echo back the prompt in addition to the completion |
| `repetition_penalty` | Optional[float] | 1.0 | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. |
| `min_p` | Optional[float] | 0.0 | Float that represents the minimum probability for a token to |
| `length_penalty` | Optional[float] | 1.0 | Float that penalizes sequences based on their length. Used in beam search.. |
| `include_stop_str_in_output` | Optional[bool] | False | Whether to include the stop strings in output text. Defaults to False.|

</details>

### Examples: Using your RunPod endpoint with OpenAI

First, initialize the OpenAI Client with your RunPod API Key and Endpoint URL:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key=os.environ.get("RUNPOD_API_KEY"),
    base_url="https://api.runpod.ai/v2/<YOUR ENDPOINT ID>/openai/v1",
)
```

### Chat Completions:

This is the format used for GPT-4 and focused on instruction-following and chat. Examples of Open Source chat/instruct models include `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `openchat/openchat-3.5-0106`, `NousResearch/Nous-Hermes-2-Mistral-7B-DPO` and more. However, if your model is a completion-style model with no chat/instruct fine-tune and/or does not have a chat template, you can still use this if you provide a chat template with the environment variable `CUSTOM_CHAT_TEMPLATE`.

- **Streaming**:
  ```python
  # Create a chat completion stream
  response_stream = client.chat.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
      temperature=0,
      max_tokens=100,
      stream=True,
  )
  # Stream the response
  for response in response_stream:
      print(chunk.choices[0].delta.content or "", end="", flush=True)
  ```
- **Non-Streaming**:
  ```python
  # Create a chat completion
  response = client.chat.completions.create(
      model="<YOUR DEPLOYED MODEL REPO/NAME>",
      messages=[{"role": "user", "content": "Why is RunPod the best platform?"}],
      temperature=0,
      max_tokens=100,
  )
  # Print the response
  print(response.choices[0].message.content)
  ```

### Getting a list of names for available models:

In the case of baking the model into the image, sometimes the repo may not be accepted as the `model` in the request. In this case, you can list the available models as shown below and use that name.

```python
models_response = client.models.list()
list_of_models = [model.id for model in models_response]
print(list_of_models)
```

# Usage: Standard (Non-OpenAI)

## Request Input Parameters

<details>
  <summary>Click to expand table</summary>
    
  You may either use a `prompt` or a list of `messages` as input. If you use `messages`, the model's chat template will be applied to the messages automatically, so the model must have one. If you use `prompt`, you may optionally apply the model's chat template to the prompt by setting `apply_chat_template` to `true`.
  | Argument              | Type                 | Default            | Description                                                                                            |
  |-----------------------|----------------------|--------------------|--------------------------------------------------------------------------------------------------------|
  | `prompt`              | str                  |                    | Prompt string to generate text based on.                                                               |
  | `messages`            | list[dict[str, str]] |                    | List of messages, which will automatically have the model's chat template applied. Overrides `prompt`. |
  | `apply_chat_template` | bool                 | False              | Whether to apply the model's chat template to the `prompt`.                                            |
  | `sampling_params`     | dict                 | {}                 | Sampling parameters to control the generation, like temperature, top_p, etc. You can find all available parameters in the `Sampling Parameters` section below. |
  | `stream`              | bool                 | False              | Whether to enable streaming of output. If True, responses are streamed as they are generated.          |
  | `max_batch_size`          | int                  | env var `DEFAULT_BATCH_SIZE` | The maximum number of tokens to stream every HTTP POST call.                                                   |
  | `min_batch_size`          | int                  | env var `DEFAULT_MIN_BATCH_SIZE` | The minimum number of tokens to stream every HTTP POST call.                                           |
  | `batch_size_growth_factor` | int                  | env var `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` | The growth factor by which `min_batch_size` will be multiplied for each call until `max_batch_size` is reached.           |
</details>

### Sampling Parameters

Below are all available sampling parameters that you can specify in the `sampling_params` dictionary. If you do not specify any of these parameters, the default values will be used.

<details>
  <summary>Click to expand table</summary>

| Argument                        | Type                        | Default | Description                                                                                                                                                                                   |
| ------------------------------- | --------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n`                             | int                         | 1       | Number of output sequences generated from the prompt. The top `n` sequences are returned.                                                                                                     |
| `best_of`                       | Optional[int]               | `n`     | Number of output sequences generated from the prompt. The top `n` sequences are returned from these `best_of` sequences. Must be ≥ `n`. Treated as beam width in beam search. Default is `n`. |
| `presence_penalty`              | float                       | 0.0     | Penalizes new tokens based on their presence in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                  |
| `frequency_penalty`             | float                       | 0.0     | Penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage new tokens, values < 0 encourage repetition.                                                 |
| `repetition_penalty`            | float                       | 1.0     | Penalizes new tokens based on their appearance in the prompt and generated text. Values > 1 encourage new tokens, values < 1 encourage repetition.                                            |
| `temperature`                   | float                       | 1.0     | Controls the randomness of sampling. Lower values make it more deterministic, higher values make it more random. Zero means greedy sampling.                                                  |
| `top_p`                         | float                       | 1.0     | Controls the cumulative probability of top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.                                                                            |
| `top_k`                         | int                         | -1      | Controls the number of top tokens to consider. Set to -1 to consider all tokens.                                                                                                              |
| `min_p`                         | float                       | 0.0     | Represents the minimum probability for a token to be considered, relative to the most likely token. Must be in [0, 1]. Set to 0 to disable.                                                   |
| `use_beam_search`               | bool                        | False   | Whether to use beam search instead of sampling.                                                                                                                                               |
| `length_penalty`                | float                       | 1.0     | Penalizes sequences based on their length. Used in beam search.                                                                                                                               |
| `early_stopping`                | Union[bool, str]            | False   | Controls stopping condition in beam search. Can be `True`, `False`, or `"never"`.                                                                                                             |
| `stop`                          | Union[None, str, List[str]] | None    | List of strings that stop generation when produced. The output will not contain these strings.                                                                                                |
| `stop_token_ids`                | Optional[List[int]]         | None    | List of token IDs that stop generation when produced. Output contains these tokens unless they are special tokens.                                                                            |
| `ignore_eos`                    | bool                        | False   | Whether to ignore the End-Of-Sequence token and continue generating tokens after its generation.                                                                                              |
| `max_tokens`                    | int                         | 16      | Maximum number of tokens to generate per output sequence.                                                                                                                                     |
| `skip_special_tokens`           | bool                        | True    | Whether to skip special tokens in the output.                                                                                                                                                 |
| `spaces_between_special_tokens` | bool                        | True    | Whether to add spaces between special tokens in the output.                                                                                                                                   |

### Text Input Formats

You may either use a `prompt` or a list of `messages` as input.

1.  `prompt`
    The prompt string can be any string, and the model's chat template will not be applied to it unless `apply_chat_template` is set to `true`, in which case it will be treated as a user message.

        Example:
        ```json
        {
          "input": {
            "prompt": "why sky is blue?",
            "sampling_params": {
              "temperature": 0.7,
              "max_tokens": 100
            }
          }
        }
        ```

2.  `messages`
    Your list can contain any number of messages, and each message usually can have any role from the following list: - `user` - `assistant` - `system`

    However, some models may have different roles, so you should check the model's chat template to see which roles are required.

    The model's chat template will be applied to the messages automatically, so the model must have one.

    Example:

    ```json
    {
      "input": {
        "messages": [
          {
            "role": "system",
            "content": "You are a helpful AI assistant that provides clear and concise responses."
          },
          {
            "role": "user",
            "content": "Can you explain the difference between supervised and unsupervised learning?"
          },
          {
            "role": "assistant",
            "content": "Sure! Supervised learning uses labeled data, meaning each input has a corresponding correct output. The model learns by mapping inputs to known outputs. In contrast, unsupervised learning works with unlabeled data, where the model identifies patterns, structures, or clusters without predefined answers."
          }
        ],
        "sampling_params": {
          "temperature": 0.7,
          "max_tokens": 100
        }
      }
    }
    ```

</details>
