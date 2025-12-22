# Configuration Reference

Complete guide to all environment variables and configuration options for worker-vllm.

## LLM Settings

| Variable                       | Default             | Type/Choices                                                | Description                                                                     |
| ------------------------------ | ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `MODEL_NAME`                   | 'facebook/opt-125m' | `str`                                                       | Hugging Face repo ID or local filesystem path for the model weights. Change this to deploy a different model. |
| `MODEL_REVISION`               | 'main'              | `str`                                                       | Model revision to load (default: main).                                         |
| `TOKENIZER`                    | None                | `str`                                                       | Name or path of the Hugging Face tokenizer to use.                              |
| `SKIP_TOKENIZER_INIT`          | False               | `bool`                                                      | Skip initialization of tokenizer and detokenizer.                               |
| `TOKENIZER_MODE`               | 'auto'              | ['auto', 'slow']                                            | The tokenizer mode.                                                             |
| `TRUST_REMOTE_CODE`            | `False`             | `bool`                                                      | Trust remote code from Hugging Face.                                            |
| `DOWNLOAD_DIR`                 | None                | `str`                                                       | Directory to download and load the weights.                                     |
| `LOAD_FORMAT`                  | 'auto'              | `str`                                                       | The format of the model weights to load.                                        |
| `HF_TOKEN`                     | -                   | `str`                                                       | Hugging Face token required to download gated/private models. Not needed for public models. Provide it via secrets. |
| `DTYPE`                        | 'auto'              | ['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'] | Data type for model weights and activations.                                    |
| `KV_CACHE_DTYPE`               | 'auto'              | ['auto', 'fp8']                                             | Data type for KV cache storage.                                                 |
| `QUANTIZATION_PARAM_PATH`      | None                | `str`                                                       | Path to the JSON file containing the KV cache scaling factors.                  |
| `MAX_MODEL_LEN`                | None                | `int`                                                       | Maximum context length (tokens) the engine will allocate KV cache for. Lower it to reduce VRAM usage; raise it for long-context models if supported and you have VRAM. |
| `GUIDED_DECODING_BACKEND`      | 'outlines'          | ['outlines', 'lm-format-enforcer']                          | Which engine will be used for guided decoding by default.                       |
| `DISTRIBUTED_EXECUTOR_BACKEND` | None                | ['ray', 'mp']                                               | Backend to use for distributed serving.                                         |
| `WORKER_USE_RAY`               | False               | `bool`                                                      | Deprecated, use --distributed-executor-backend=ray.                             |
| `PIPELINE_PARALLEL_SIZE`       | 1                   | `int`                                                       | Number of pipeline stages.                                                      |
| `TENSOR_PARALLEL_SIZE`         | 1                   | `int`                                                       | Tensor parallel degree (number of GPUs to shard across). On multi-GPU machines, this worker auto-sets it to the number of visible GPUs. |
| `MAX_PARALLEL_LOADING_WORKERS` | None                | `int`                                                       | Load model sequentially in multiple batches.                                    |
| `RAY_WORKERS_USE_NSIGHT`       | False               | `bool`                                                      | If specified, use nsight to profile Ray workers.                                |
| `ENABLE_PREFIX_CACHING`        | False               | `bool`                                                      | Enables automatic prefix caching.                                               |
| `DISABLE_SLIDING_WINDOW`       | False               | `bool`                                                      | Disables sliding window, capping to sliding window size.                        |
| `USE_V2_BLOCK_MANAGER`         | False               | `bool`                                                      | Use BlockSpaceMangerV2.                                                         |
| `NUM_LOOKAHEAD_SLOTS`          | 0                   | `int`                                                       | Experimental scheduling config necessary for speculative decoding.              |
| `SEED`                         | 0                   | `int`                                                       | Random seed for operations.                                                     |
| `NUM_GPU_BLOCKS_OVERRIDE`      | None                | `int`                                                       | If specified, ignore GPU profiling result and use this number of GPU blocks.    |
| `MAX_NUM_BATCHED_TOKENS`       | None                | `int`                                                       | Maximum number of batched tokens per iteration.                                 |
| `MAX_NUM_SEQS`                 | 256                 | `int`                                                       | Upper bound on sequences batched per iteration (affects throughput, VRAM, and tail latency). Higher can improve throughput for many concurrent short requests; lower reduces VRAM usage. |
| `MAX_LOGPROBS`                 | 20                  | `int`                                                       | Max number of log probs to return when logprobs is specified in SamplingParams. |
| `DISABLE_LOG_STATS`            | False               | `bool`                                                      | Disable logging statistics.                                                     |
| `QUANTIZATION`                 | None                | ['awq', 'squeezellm', 'gptq', 'bitsandbytes']               | Quantization backend for loading quantized checkpoints (AWQ/GPTQ/...) or BitsAndBytes. Must match the checkpoint format. |
| `ROPE_SCALING`                 | None                | `dict`                                                      | RoPE scaling configuration in JSON format.                                      |
| `ROPE_THETA`                   | None                | `float`                                                     | RoPE theta. Use with rope_scaling.                                              |
| `TOKENIZER_POOL_SIZE`          | 0                   | `int`                                                       | Size of tokenizer pool to use for asynchronous tokenization.                    |
| `TOKENIZER_POOL_TYPE`          | 'ray'               | `str`                                                       | Type of tokenizer pool to use for asynchronous tokenization.                    |
| `TOKENIZER_POOL_EXTRA_CONFIG`  | None                | `dict`                                                      | Extra config for tokenizer pool.                                                |

## LoRA (Low-Rank Adaptation) Settings

| Variable                    | Default | Type                                       | Description                                                                                               |
| --------------------------- | ------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `ENABLE_LORA`               | False   | `bool`                                     | If True, enable handling of LoRA adapters.                                                                |
| `MAX_LORAS`                 | 1       | `int`                                      | Max number of LoRAs in a single batch.                                                                    |
| `MAX_LORA_RANK`             | 16      | `int`                                      | Max LoRA rank.                                                                                            |
| `LORA_EXTRA_VOCAB_SIZE`     | 256     | `int`                                      | Maximum size of extra vocabulary for LoRA adapters.                                                       |
| `LORA_DTYPE`                | 'auto'  | ['auto', 'float16', 'bfloat16', 'float32'] | Data type for LoRA.                                                                                       |
| `LONG_LORA_SCALING_FACTORS` | None    | `tuple`                                    | Specify multiple scaling factors for LoRA adapters.                                                       |
| `MAX_CPU_LORAS`             | None    | `int`                                      | Maximum number of LoRAs to store in CPU memory.                                                           |
| `FULLY_SHARDED_LORAS`       | False   | `bool`                                     | Enable fully sharded LoRA layers.                                                                         |
| `LORA_MODULES`              | `[]`    | `list[dict]`                               | Add lora adapters from Hugging Face `[{"name": "xx", "path": "xxx/xxxx", "base_model_name": "xxx/xxxx"}]` |

## Speculative Decoding Settings

| Variable                                         | Default             | Type/Choices                                        | Description                                                                               |
| ------------------------------------------------ | ------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `SCHEDULER_DELAY_FACTOR`                         | 0.0                 | `float`                                             | Apply a delay before scheduling next prompt.                                              |
| `ENABLE_CHUNKED_PREFILL`                         | False               | `bool`                                              | Enable chunked prefill requests.                                                          |
| `SPECULATIVE_MODEL`                              | None                | `str`                                               | The name of the draft model to be used in speculative decoding.                           |
| `NUM_SPECULATIVE_TOKENS`                         | None                | `int`                                               | The number of speculative tokens to sample from the draft model.                          |
| `SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE`         | None                | `int`                                               | Number of tensor parallel replicas for the draft model.                                   |
| `SPECULATIVE_MAX_MODEL_LEN`                      | None                | `int`                                               | The maximum sequence length supported by the draft model.                                 |
| `SPECULATIVE_DISABLE_BY_BATCH_SIZE`              | None                | `int`                                               | Disable speculative decoding if the number of enqueue requests is larger than this value. |
| `NGRAM_PROMPT_LOOKUP_MAX`                        | None                | `int`                                               | Max size of window for ngram prompt lookup in speculative decoding.                       |
| `NGRAM_PROMPT_LOOKUP_MIN`                        | None                | `int`                                               | Min size of window for ngram prompt lookup in speculative decoding.                       |
| `SPEC_DECODING_ACCEPTANCE_METHOD`                | 'rejection_sampler' | ['rejection_sampler', 'typical_acceptance_sampler'] | Specify the acceptance method for draft token verification in speculative decoding.       |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD` | None                | `float`                                             | Set the lower bound threshold for the posterior probability of a token to be accepted.    |
| `TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA`     | None                | `float`                                             | A scaling factor for the entropy-based threshold for token acceptance.                    |

## System Performance Settings

| Variable                       | Default | Type/Choices    | Description                                                                                                                         |
| ------------------------------ | ------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `GPU_MEMORY_UTILIZATION`       | `0.95`  | `float`         | Fraction of GPU VRAM vLLM is allowed to use for KV cache + runtime allocations. Lower if you hit CUDA OOM; raise if you have VRAM headroom. |
| `MAX_PARALLEL_LOADING_WORKERS` | `None`  | `int`           | Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and large models.                          |
| `BLOCK_SIZE`                   | `16`    | `8`, `16`, `32` | Token block size for contiguous chunks of tokens.                                                                                   |
| `SWAP_SPACE`                   | `4`     | `int`           | CPU swap space size (GiB) per GPU.                                                                                                  |
| `ENFORCE_EAGER`                | False   | `bool`          | Always use eager-mode PyTorch. If False(`0`), will use eager mode and CUDA graph in hybrid for maximal performance and flexibility. |
| `MAX_SEQ_LEN_TO_CAPTURE`       | `8192`  | `int`           | Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode.     |
| `DISABLE_CUSTOM_ALL_REDUCE`    | `0`     | `int`           | Enables or disables custom all reduce.                                                                                              |
| `ENABLE_EXPERT_PARALLEL`       | `False` | `bool`           |  Enable Expert Parallel for MoE models  |

## Tokenizer Settings

| Variable               | Default | Type/Choices                        | Description                                                                                       |
| ---------------------- | ------- | ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `TOKENIZER_NAME`       | `None`  | `str`                               | Tokenizer repository to use a different tokenizer than the model's default.                       |
| `TOKENIZER_REVISION`   | `None`  | `str`                               | Tokenizer revision to load.                                                                       |
| `CUSTOM_CHAT_TEMPLATE` | `None`  | `str` of single-line jinja template | Override the model chat template (single-line Jinja2). Useful when sending `messages` to a base model without a built-in chat template. [More Info](https://huggingface.co/docs/transformers/chat_templating) |

## Streaming & Batch Settings

The way this works is that the first request will have a batch size of `DEFAULT_MIN_BATCH_SIZE`, and each subsequent request will have a batch size of `previous_batch_size * DEFAULT_BATCH_SIZE_GROWTH_FACTOR`. This will continue until the batch size reaches `DEFAULT_BATCH_SIZE`. E.g. for the default values, the batch sizes will be `1, 3, 9, 27, 50, 50, 50, ...`. You can also specify this per request, with inputs `max_batch_size`, `min_batch_size`, and `batch_size_growth_factor`. This has nothing to do with vLLM's internal batching, but rather the number of tokens sent in each HTTP request from the worker.

| Variable                           | Default | Type/Choices | Description                                                                                               |
| ---------------------------------- | ------- | ------------ | --------------------------------------------------------------------------------------------------------- |
| `DEFAULT_BATCH_SIZE`               | `50`    | `int`        | Default and Maximum batch size for token streaming to reduce HTTP calls.                                  |
| `DEFAULT_MIN_BATCH_SIZE`           | `1`     | `int`        | Batch size for the first request, which will be multiplied by the growth factor every subsequent request. |
| `DEFAULT_BATCH_SIZE_GROWTH_FACTOR` | `3`     | `float`      | Growth factor for dynamic batch size.                                                                     |

## OpenAI Compatibility Settings

| Variable                            | Default     | Type/Choices     | Description                                                                                                                                                                                                       |
| ----------------------------------- | ----------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAW_OPENAI_OUTPUT`                 | `1`         | boolean as `int` | Enables raw OpenAI SSE format string output when streaming. **Required** to be enabled (which it is by default) for OpenAI compatibility.                                                                         |
| `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `None`      | `str`            | Exposes a custom model id via `/v1/models` and accepts it as the `model` field in OpenAI requests (alias for the served model).                                                                     |
| `OPENAI_RESPONSE_ROLE`              | `assistant` | `str`            | Role of the LLM's Response in OpenAI Chat Completions.                                                                                                                                                            |
| `ENABLE_AUTO_TOOL_CHOICE`           | `false`     | `bool`           | Enables vLLM automatic tool selection for OpenAI Chat Completions. Only enable for tool-capable models.                                                                                               |
| `TOOL_CALL_PARSER`                  | `None`      | `str`            | Tool-call parser that matches your model’s tool-call format (required for most tool-calling models). Supported values: `mistral`, `hermes`, `llama3_json`, `llama4_json`, `llama4_pythonic`, `granite`, `granite-20b-fc`, `deepseek_v3`, `internlm`, `jamba`, `phi4_mini_json`, `pythonic`. |
| `REASONING_PARSER`                  | `None`      | `str`            | Parser for reasoning-capable models (enables reasoning mode). Examples: `deepseek_r1`, `qwen3`, `granite`, `hunyuan_a13b`. Leave unset to disable.                                                                |

Notes:
- `TOOL_CALL_PARSER` tells vLLM how to interpret a model’s tool-call output. If the parser doesn’t match the model’s format, tool calls may not be detected (or may error during parsing).

## Serverless & Concurrency Settings

| Variable               | Default | Type/Choices | Description                                                                                                                                                                |
| ---------------------- | ------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_CONCURRENCY`      | `30`    | `int`        | Max concurrent requests per worker instance (Runpod-side). Not a vLLM engine arg; it controls Runpod worker concurrency and affects how requests are fed into vLLM (queueing/throughput/latency). |
| `DISABLE_LOG_STATS`    | False   | `bool`       | Enables or disables vLLM stats logging.                                                                                                                                    |
| `DISABLE_LOG_REQUESTS` | False   | `bool`       | Enables or disables vLLM request logging.                                                                                                                                  |

## Advanced Settings

| Variable                    | Default | Type    | Description                                                                                                                                            |
| --------------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MODEL_LOADER_EXTRA_CONFIG` | None    | `dict`  | Extra config for model loader.                                                                                                                         |
| `PREEMPTION_MODE`           | None    | `str`   | If 'recompute', the engine performs preemption-aware recomputation. If 'save', the engine saves activations into the CPU memory as preemption happens. |
| `PREEMPTION_CHECK_PERIOD`   | 1.0     | `float` | How frequently the engine checks if a preemption happens.                                                                                              |
| `PREEMPTION_CPU_CAPACITY`   | 2       | `float` | The percentage of CPU memory used for the saved activations.                                                                                           |
| `DISABLE_LOGGING_REQUEST`   | False   | `bool`  | Disable logging requests.                                                                                                                              |
| `MAX_LOG_LEN`               | None    | `int`   | Max number of prompt characters or prompt ID numbers being printed in log.                                                                             |

## Docker Build Arguments

These variables are used when building custom Docker images with models baked in:

| Variable              | Default          | Type  | Description                                       |
| --------------------- | ---------------- | ----- | ------------------------------------------------- |
| `BASE_PATH`           | `/runpod-volume` | `str` | Storage directory for huggingface cache and model |
| `WORKER_CUDA_VERSION` | `12.1.0`         | `str` | CUDA version for the worker image                 |

## Deprecated Variables

⚠️ **The following variables are deprecated and will be removed in future versions:**

| Old Variable                 | New Variable             | Note                  |
| ---------------------------- | ------------------------ | --------------------- |
| `MAX_CONTEXT_LEN_TO_CAPTURE` | `MAX_SEQ_LEN_TO_CAPTURE` | Use new variable name |
| `kv_cache_dtype=fp8_e5m2`    | `kv_cache_dtype=fp8`     | Simplified fp8 format |
