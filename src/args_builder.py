"""Translate environment variables into `vllm serve` CLI arguments.

Rules (evaluated in order, later wins):

1. Generic scan: every env var whose name matches a known vLLM flag in
   UPPER_SNAKE_CASE form is emitted as a CLI flag. ``MAX_MODEL_LEN=4096`` turns
   into ``--max-model-len 4096``. Anything the allowlist does not cover can be
   passed verbatim through ``VLLM_EXTRA_ARGS``.

   Flag categories (regenerated against the current vLLM CLI — stable docs and
   ``main`` sources — in 2026-07; earlier versions of this list came from
   ``vllm serve --help`` of vLLM 0.11):

   - ``VALUE_FLAGS`` take a value, passed through verbatim and parsed by vLLM.
     Exception: a ``0`` for a flag in ``ZERO_MEANS_UNSET`` (currently just
     ``--max-num-batched-tokens``) drops the flag so vLLM picks its own default.
   - ``PAIRED_BOOL_FLAGS`` are argparse ``BooleanOptionalAction`` flags
     (``--flag`` / ``--no-flag``) that take NO value. A truthy env value emits
     the bare ``--flag``; a falsy value emits the ``--no-flag`` form. Older
     vLLM releases also accepted ``--flag false``; current vLLM rejects that
     with ``unrecognized arguments``, so never emit a value after these.
   - ``BOOL_FLAGS`` are plain store-true flags: truthy emits the bare flag,
     falsy omits it entirely.
   - ``REMOVED_FLAGS`` are gone from current vLLM. Setting their env var logs
     a warning naming the replacement and emits nothing, so a stale template
     default cannot crash startup.

2. Aliases: a handful of historical env names that do not match a flag
   (``MODEL_NAME`` -> ``--model``, ...).

3. ``VLLM_EXTRA_ARGS``: raw argument string appended last, so it overrides
   anything above when argparse sees the same flag twice.
   Example: ``VLLM_EXTRA_ARGS="--override-generation-config '{"max_new_tokens": 512}'"``.
"""

import logging
import os
import shlex
from typing import Mapping, Optional

TRUE_VALUES = {"true", "1", "yes", "on"}
FALSE_VALUES = {"false", "0", "no", "off"}

# Historical env names that do not match the UPPER_SNAKE of their flag.
ENV_ALIASES = {
    "MODEL_NAME": "--model",
    "MODEL_REVISION": "--revision",
    "TOKENIZER_NAME": "--tokenizer",
    "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "--served-model-name",
    "CUSTOM_CHAT_TEMPLATE": "--chat-template",
    "VLLM_CONFIG_FILE": "--config",
}

# Flags owned by the wrapper itself (or deliberately not exposed generically):
# --host/--port are pinned by main.py, --config comes from the VLLM_CONFIG_FILE
# alias, --model has the MODEL_NAME alias. --api-key stays reserved because the
# worker intentionally does not support it: the internal server binds to
# loopback only and RunPod authenticates callers at the platform.
RESERVED_ENV_VARS = frozenset({"HOST", "PORT", "API_KEY", "MODEL", "CONFIG", "HELP"})

# Flags that take a value. vLLM parses int/float/JSON values itself.
VALUE_FLAGS = frozenset({
    "--additional-config",
    "--allowed-headers",
    "--allowed-local-media-path",
    "--allowed-media-domains",
    "--allowed-methods",
    "--allowed-origins",
    "--api-server-count",
    "--attention-backend",
    "--block-size",
    "--chat-template",
    "--chat-template-content-format",
    "--code-revision",
    "--collect-detailed-traces",
    "--compilation-config",
    "--config-format",
    "--convert",
    "--cpu-offload-gb",
    "--data-parallel-address",
    "--data-parallel-backend",
    "--data-parallel-rank",
    "--data-parallel-rpc-port",
    "--data-parallel-size",
    "--data-parallel-size-local",
    "--data-parallel-start-rank",
    "--dbo-decode-token-threshold",
    "--dbo-prefill-token-threshold",
    "--decode-context-parallel-size",
    "--default-mm-loras",
    "--distributed-executor-backend",
    "--download-dir",
    "--dtype",
    "--eplb-config",
    "--expert-placement-strategy",
    "--generation-config",
    "--gpu-memory-utilization",
    "--h11-max-header-count",
    "--h11-max-incomplete-event-size",
    "--hf-config-path",
    "--hf-overrides",
    "--hf-token",
    "--ignore-patterns",
    "--io-processor-plugin",
    "--kv-cache-dtype",
    "--kv-cache-memory-bytes",
    "--kv-events-config",
    "--kv-offloading-backend",
    "--kv-offloading-size",
    "--kv-transfer-config",
    "--limit-mm-per-prompt",
    "--load-format",
    "--log-config-file",
    "--logits-processors",
    "--logprobs-mode",
    "--long-prefill-token-threshold",
    "--lora-dtype",
    "--lora-modules",
    "--mamba-cache-dtype",
    "--mamba-ssm-cache-dtype",
    "--max-cpu-loras",
    "--max-log-len",
    "--max-logprobs",
    "--max-lora-rank",
    "--max-loras",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--max-parallel-loading-workers",
    "--media-io-kwargs",
    "--middleware",
    "--mm-encoder-tp-mode",
    "--mm-processor-cache-gb",
    "--mm-processor-cache-type",
    "--mm-processor-kwargs",
    "--mm-shm-cache-max-object-size-mb",
    "--model",
    "--model-impl",
    "--model-loader-extra-config",
    "--num-gpu-blocks-override",
    "--otlp-traces-endpoint",
    "--override-generation-config",
    "--pipeline-parallel-size",
    "--pooler-config",
    "--prefix-caching-hash-algo",
    "--pt-load-map-location",
    "--quantization",
    "--reasoning-parser",
    "--response-role",
    "--revision",
    "--root-path",
    "--runner",
    "--safetensors-load-strategy",
    "--scheduler-cls",
    "--scheduling-policy",
    "--seed",
    "--served-model-name",
    "--show-hidden-metrics-for-version",
    "--speculative-config",
    "--ssl-ca-certs",
    "--ssl-cert-reqs",
    "--ssl-certfile",
    "--ssl-keyfile",
    "--structured-outputs-config",
    "--tensor-parallel-size",
    "--tokenizer",
    "--tokenizer-mode",
    "--tokenizer-revision",
    "--tool-call-parser",
    "--tool-parser-plugin",
    "--tool-server",
    "--uds",
    "--uvicorn-log-level",
    "--video-pruning-rate",
    "--worker-cls",
    "--worker-extension-cls",
})

# argparse BooleanOptionalAction flags: `--flag` / `--no-flag`, no value.
# Truthy env -> bare flag, falsy env -> `--no-flag`.
PAIRED_BOOL_FLAGS = frozenset({
    "--allow-credentials",
    "--async-scheduling",
    "--data-parallel-hybrid-lb",
    "--disable-cascade-attn",
    "--disable-chunked-mm-input",
    "--disable-custom-all-reduce",
    "--disable-fastapi-docs",
    "--disable-hybrid-kv-cache-manager",
    "--disable-sliding-window",
    "--disable-uvicorn-access-log",
    "--enable-auto-tool-choice",
    "--enable-chunked-prefill",
    "--enable-dbo",
    "--enable-eplb",
    "--enable-expert-parallel",
    "--enable-force-include-usage",
    "--enable-log-outputs",
    "--enable-log-requests",
    "--enable-lora",
    "--enable-prefix-caching",
    "--enable-prompt-embeds",
    "--enable-prompt-tokens-details",
    "--enable-request-id-headers",
    "--enable-server-load-tracking",
    "--enable-sleep-mode",
    "--enable-ssl-refresh",
    "--enable-tokenizer-info-endpoint",
    "--enforce-eager",
    "--exclude-tools-when-tool-choice-none",
    "--fully-sharded-loras",
    "--interleave-mm-strings",
    "--kv-sharing-fast-prefill",
    "--log-error-stack",
    "--ray-workers-use-nsight",
    "--return-tokens-as-token-ids",
    "--skip-mm-profiling",
    "--skip-tokenizer-init",
    "--trust-remote-code",
    "--trust-request-chat-template",
    "--use-tqdm-on-load",
})

# argparse store-true flags: truthy env value -> bare flag, falsy -> omitted.
BOOL_FLAGS = frozenset({
    "--disable-log-stats",
})

# Value flags where "0" means "leave it to vLLM", because 0 is rejected by
# vLLM's config validation (SchedulerConfig requires max_num_batched_tokens >= 1)
# while endpoint templates commonly carry a stray MAX_NUM_BATCHED_TOKENS=0.
# Omitting the flag lets vLLM size the batch for the actual GPU. Note this is NOT
# generalized: plenty of flags take a legitimate 0 (--seed 0, --cpu-offload-gb 0).
ZERO_MEANS_UNSET = frozenset({
    "--max-num-batched-tokens",
})

# Flags current vLLM no longer accepts. Setting one logs a warning and emits
# nothing instead of dying with `unrecognized arguments` at container start.
# The note names the replacement (if any) and appears in the warning.
REMOVED_FLAGS = {
    "--calculate-kv-scales": "removed in v0.28.0 (#49389); no replacement",
    "--cuda-graph-sizes": "moved into COMPILATION_CONFIG as cudagraph_capture_sizes",
    "--disable-frontend-multiprocessing": "no replacement",
    "--disable-log-requests": "inverted upstream; use ENABLE_LOG_REQUESTS=false",
    "--disable-mm-preprocessor-cache": "no replacement; tune via MM_PROCESSOR_CACHE_GB",
    "--enable-lora-bias": "no replacement",
    "--enable-multimodal-encoder-data-parallel": "no replacement",
    "--eplb-log-balancedness": "now a field of EPLB_CONFIG (--eplb-config)",
    "--eplb-step-interval": "now a field of EPLB_CONFIG (--eplb-config)",
    "--eplb-window-size": "now a field of EPLB_CONFIG (--eplb-config)",
    "--guided-decoding-backend": "use STRUCTURED_OUTPUTS_CONFIG (--structured-outputs-config)",
    "--guided-decoding-disable-additional-properties": "use STRUCTURED_OUTPUTS_CONFIG",
    "--guided-decoding-disable-any-whitespace": "use STRUCTURED_OUTPUTS_CONFIG",
    "--guided-decoding-disable-fallback": "use STRUCTURED_OUTPUTS_CONFIG",
    "--logits-processor-pattern": "use LOGITS_PROCESSORS (--logits-processors)",
    "--lora-extra-vocab-size": "no replacement",
    "--max-long-partial-prefills": "removed in v0.27.0; no replacement",
    "--max-num-partial-prefills": "removed in v0.27.0; no replacement",
    "--num-lookahead-slots": "speculative decoding is configured via SPECULATIVE_CONFIG",
    "--num-redundant-experts": "now a field of EPLB_CONFIG (--eplb-config)",
    "--override-attention-dtype": "removed in v0.28.0 (#48684); no replacement",
    "--override-pooler-config": "use POOLER_CONFIG (--pooler-config)",
    "--rope-scaling": "pass rope_scaling via HF_OVERRIDES (--hf-overrides)",
    "--rope-theta": "pass rope_theta via HF_OVERRIDES (--hf-overrides)",
    "--swap-space": "V1 has no swap space; no replacement",
    "--task": "use RUNNER (--runner) / CONVERT (--convert)",
}

ALL_FLAGS = VALUE_FLAGS | PAIRED_BOOL_FLAGS | BOOL_FLAGS


def flag_to_env_name(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_").upper()


def _boolean_value(flag: str, raw_value: str) -> Optional[bool]:
    value = str(raw_value).strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"invalid boolean value {raw_value!r} for {flag}")


def _apply(flag: str, raw_value, flags: dict) -> None:
    value = str(raw_value).strip()
    if value == "":
        return
    if flag in PAIRED_BOOL_FLAGS or flag in BOOL_FLAGS:
        is_true = _boolean_value(flag, raw_value)
        if flag in PAIRED_BOOL_FLAGS:
            # `--flag` / `--no-flag`, never with a value after it.
            flags[flag if is_true else f"--no-{flag.lstrip('-')}"] = None
        elif is_true:
            flags[flag] = None          # store-true: bare flag
        else:
            flags.pop(flag, None)       # omitted entirely
    elif flag in ZERO_MEANS_UNSET and value == "0":
        logging.warning("%s=0 is invalid for vLLM; treating it as unset (vLLM default)", flag)
    else:
        flags[flag] = value


def build_vllm_args(env: Optional[Mapping[str, str]] = None) -> list[str]:
    """Build the `vllm serve` argument list from environment variables."""
    env = os.environ if env is None else env
    flags: dict[str, Optional[str]] = {}

    # 0. Warn about env vars for flags current vLLM no longer accepts.
    for flag, note in sorted(REMOVED_FLAGS.items()):
        env_name = flag_to_env_name(flag)
        if env.get(env_name, "").strip():
            logging.warning(
                "%s is set but vLLM no longer supports %s; ignoring it (%s).",
                env_name, flag, note,
            )

    # 1. Generic scan (env name == UPPER_SNAKE of the flag).
    for flag in sorted(ALL_FLAGS):
        env_name = flag_to_env_name(flag)
        if env_name in RESERVED_ENV_VARS:
            continue
        value = env.get(env_name)
        if value is not None:
            _apply(flag, value, flags)

    # 2. Historical aliases (win over the generic scan).
    for env_name, flag in ENV_ALIASES.items():
        value = env.get(env_name)
        if value is not None:
            _apply(flag, value, flags)

    argv: list[str] = []
    for flag, value in flags.items():
        argv.append(flag)
        if value is not None:
            argv.append(value)

    # 3. Raw passthrough, appended last so argparse "last occurrence wins".
    extra = env.get("VLLM_EXTRA_ARGS", "")
    argv.extend(shlex.split(extra))
    return argv
