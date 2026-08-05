# Configuration Reference

How to configure the RunPod serverless vLLM worker. The worker is a thin wrapper around
the official `vllm/vllm-openai` server image: at startup it builds a `vllm serve` command
from your environment variables, waits for the server to report healthy, and then proxies
RunPod jobs to it over localhost HTTP. All vLLM behavior is configured through env vars.

## How env vars become engine args

`src/args_builder.py` translates environment variables into `vllm serve` CLI flags:

1. **Generic mapping** — any env var whose name is the UPPERCASED form of a `vllm serve`
   flag is converted automatically:

   | Environment Variable     | CLI Flag                   | Example Value                              |
   | ------------------------ | -------------------------- | ------------------------------------------ |
   | `MAX_MODEL_LEN`          | `--max-model-len`          | `4096`                                     |
   | `TENSOR_PARALLEL_SIZE`   | `--tensor-parallel-size`   | `2`                                        |
   | `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | `0.9`                                      |
   | `DTYPE`                  | `--dtype`                  | `auto`, `half`, `bfloat16`, ...            |
   | `QUANTIZATION`           | `--quantization`           | `awq`, `gptq`, `fp8`, ...                  |
   | `KV_CACHE_DTYPE`         | `--kv-cache-dtype`         | `auto`, `fp8`                              |
   | `ENFORCE_EAGER`          | `--enforce-eager`          | `true` / `false`                           |
   | `ENABLE_PREFIX_CACHING`  | `--enable-prefix-caching`  | `true` / `false`                           |
   | `ENABLE_LORA`            | `--enable-lora`            | `true` / `false`                           |
   | `MAX_LORAS`              | `--max-loras`              | `2`                                        |
   | `SPECULATIVE_CONFIG`     | `--speculative-config`     | JSON string                                |
   | `TOOL_CALL_PARSER`       | `--tool-call-parser`       | `hermes`, `llama3_json`, ...               |
   | `ENABLE_AUTO_TOOL_CHOICE`| `--enable-auto-tool-choice`| `true` / `false`                          |
   | `REASONING_PARSER`       | `--reasoning-parser`       | `deepseek_r1`, ...                         |
   | `MAX_NUM_SEQS`           | `--max-num-seqs`           | `256`                                      |
   | `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `8192`                                     |
   | `TRUST_REMOTE_CODE`      | `--trust-remote-code`      | `true` / `false`                           |

   Values are passed verbatim to vLLM — strings, numbers, and JSON blobs like
   `SPECULATIVE_CONFIG='{"model":"org/draft","num_speculative_tokens":3}'` all work.
   Boolean variables keep `true`/`false` env syntax, but the worker emits vLLM's
   pair forms (`TRUST_REMOTE_CODE=false` becomes `--no-trust-remote-code`) —
   current vLLM rejects `--flag false` as an unrecognized argument.

2. **Aliases** — historical worker env names that don't match a CLI flag:

   | Env Variable                        | Maps To              |
   | ----------------------------------- | -------------------- |
   | `MODEL_NAME`                        | `--model`            |
   | `MODEL_REVISION`                    | `--revision`         |
   | `TOKENIZER_NAME`                    | `--tokenizer`        |
   | `OPENAI_SERVED_MODEL_NAME_OVERRIDE` | `--served-model-name`|
   | `CUSTOM_CHAT_TEMPLATE`              | `--chat-template`    |
   | `VLLM_CONFIG_FILE`                  | `--config`           |

3. **`VLLM_EXTRA_ARGS`** — raw passthrough appended last to the command line (and
   therefore able to override anything above). Use it for flags newer than the
   worker's allowlist:

   ```bash
   VLLM_EXTRA_ARGS="--override-generation-config '{\"max_new_tokens\": 512}' --uvicorn-log-level warning"
   ```

The full allowlist lives in `src/args_builder.py` (`VALUE_FLAGS`, `PAIRED_BOOL_FLAGS`,
`BOOL_FLAGS`). `REMOVED_FLAGS` there lists env vars for flags vLLM has deleted upstream —
setting one logs a startup warning and emits nothing.

## Wrapper-level environment variables

These are consumed by the wrapper itself, not passed to vLLM:

| Variable              | Default | Description                                                            |
| --------------------- | ------- | ---------------------------------------------------------------------- |
| `MODEL_NAME`          | —       | Required. HF repo id or local path of the model.                        |
| `HF_TOKEN`            | —       | Hugging Face token for gated/private models.                            |
| `BASE_PATH`           | `/runpod-volume` | Root for the HF cache (persists on a network volume).          |
| `MAX_CONCURRENCY`     | `30`    | Max concurrent jobs per worker (RunPod concurrency modifier). vLLM queues internally beyond this. |
| `VLLM_PORT`           | `8000`  | Loopback port the internal `vllm serve` binds to.                       |
| `VLLM_STARTUP_TIMEOUT`| `1200`  | Seconds to wait for vLLM `/health` before failing the worker.           |
| `REQUEST_TIMEOUT`     | `3600`  | Per-request timeout to the vLLM server, in seconds.                     |
| `VLLM_EXTRA_ARGS`     | —       | Raw extra CLI args (see above).                                         |
| `VLLM_CONFIG_FILE`    | —       | Path to a `vllm serve` YAML config file (`--config`).                   |

vLLM-native env vars (e.g. `VLLM_USE_DEEP_GEMM`, `PYTORCH_ALLOC_CONF`) are **not**
converted to flags — they reach the `vllm serve` subprocess directly through the
environment, exactly as upstream supports them.

## Speculative decoding

Pass a full JSON config, which vLLM parses natively:

```bash
SPECULATIVE_CONFIG='{"model":"RedHatAI/Qwen3-8B-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
```

(The previous per-field env vars `SPECULATIVE_METHOD`, `SPECULATIVE_MODEL`,
`NUM_SPECULATIVE_TOKENS`, `NGRAM_PROMPT_LOOKUP_MIN/MAX` are no longer assembled by the
worker — use the JSON form.)

## LoRA

```bash
ENABLE_LORA=true
MAX_LORAS=2
LORA_MODULES='[{"name":"my-adapter","path":"org/adapter-repo"}]'
```

`LORA_MODULES` is passed to `--lora-modules` verbatim; vLLM loads the adapters.

## Docker build arguments

| ARG                 | Default          | Description                                                     |
| ------------------- | ---------------- | --------------------------------------------------------------- |
| `VLLM_VERSION`      | `v0.23.0`        | Tag of the official `vllm/vllm-openai` base image.              |
| `MODEL_NAME`        | —                | If set, the model is baked into the image at build time.        |
| `MODEL_REVISION`    | `main`           | Model revision for the baked model.                             |
| `TOKENIZER_NAME`    | same as model    | Optional separate tokenizer repo.                               |
| `TOKENIZER_REVISION`| model revision   | Tokenizer revision.                                             |
| `QUANTIZATION`      | —                | Recorded for baked models.                                      |
| `BASE_PATH`         | `/runpod-volume` | HF cache location baked into the image env.                     |

The build secret `HF_TOKEN` (`--secret id=HF_TOKEN`) is used during the bake step for
gated models and never lands in the image.

## Removed/deprecated knobs

The worker no longer intercepts these; either vLLM handles them natively or the feature
is gone:

- `DEFAULT_BATCH_SIZE`, `DEFAULT_MIN_BATCH_SIZE`, `DEFAULT_BATCH_SIZE_GROWTH_FACTOR`,
  per-request `max_batch_size`/`min_batch_size`/`batch_size_growth_factor`, and
  `RAW_OPENAI_OUTPUT` — client-side output batching is gone; streaming now proxies
  vLLM's SSE chunks directly (RunPod aggregate streaming is still used).
- `SPECULATIVE_METHOD` et al. — use `SPECULATIVE_CONFIG` JSON.
- `VLLM_API_KEY` — no longer wired to `--api-key`; RunPod authenticates callers at
  the platform and the internal vLLM server binds to loopback only, so the env var
  has no effect.
- `TRANSFORMERS_VERSION` runtime reinstall — the `transformers` version is pinned by
  the base image; build with a different `VLLM_VERSION` instead.
- `MAX_CONTEXT_LEN_TO_CAPTURE` and `VLLM_ATTENTION_BACKEND` — gone upstream; use
  `ATTENTION_BACKEND`. `DISABLE_LOG_REQUESTS` was inverted upstream — use
  `ENABLE_LOG_REQUESTS=false` (emits `--no-enable-log-requests`). `CUDA_GRAPH_SIZES`
  moved into `COMPILATION_CONFIG` as `cudagraph_capture_sizes`. Removals like
  `SWAP_SPACE`, `ROPE_SCALING`/`ROPE_THETA`, `NUM_LOOKAHEAD_SLOTS`, and
  `GUIDED_DECODING_*` are in `REMOVED_FLAGS`: setting them logs a warning and they
  are ignored instead of crashing the worker at startup.
