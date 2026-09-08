"""Turn a failed `vllm serve` start into a sentence the user can act on.

vLLM reports its problems as Python tracebacks a hundred lines long, and the
worker only sees them as stdout. When a known one goes by we would rather answer
the next job with the cause and the fix than let the container exit and be
restarted into the same wall: a crash-loop pays for the download and the model
load on every attempt and the console shows nothing but silent restarts.

Only failures that a restart cannot fix belong here. Anything unrecognised is
left alone (``classify`` returns None) so the platform retries it, which is the
right move for a flaky download or a host that lost its GPU.
"""

import re
from typing import Optional

# --- GPU memory -------------------------------------------------------------
# torch prints both the request and the card's capacity; the capacity is the
# useful half, since the request is whatever happened to be next.
_OOM = re.compile(r"torch\.OutOfMemoryError|CUDA out of memory", re.I)
_OOM_CAPACITY = re.compile(r"total capacity of ([\d.]+) GiB", re.I)
# vllm/v1/core/kv_cache_utils.py: the weights fit but nothing is left for the
# KV cache, or not enough is left to hold a single request of max_model_len.
_NO_KV_MEMORY = re.compile(r"No available memory for the cache blocks", re.I)
_KV_TOO_SMALL = re.compile(
    r"larger than the available KV cache memory"
    r"|larger than the maximum number of tokens that can be stored in KV cache",
    re.I,
)
_KV_ESTIMATED_LEN = re.compile(r"estimated maximum model length is (\d+)", re.I)

# --- Configuration ----------------------------------------------------------
# vllm/config/model.py: MAX_MODEL_LEN above what the model's config allows.
_MAX_LEN_EXCEEDS_MODEL = re.compile(
    r"User-specified max_model_len \((\d+)\) is greater than the derived max_model_len", re.I
)
# argparse: exit code 2 before the engine even starts.
_BAD_ARGS = re.compile(r"(?:vllm serve|vllm): error: (.+)")

# --- Model access -----------------------------------------------------------
_GATED = re.compile(r"GatedRepoError|401 Client Error", re.I)
_NOT_FOUND = re.compile(r"RepositoryNotFoundError|404 Client Error", re.I)
_UNSUPPORTED_ARCH = re.compile(r"Model architectures \[.*?\] (?:are not supported|failed to be inspected)", re.I)

# --- Disk -------------------------------------------------------------------
_NO_SPACE = re.compile(r"No space left on device|ENOSPC|errno 28", re.I)


def classify(output: str, model: Optional[str] = None) -> Optional[str]:
    """One actionable message for a known fatal failure, or None to let it retry."""
    named = model or "The model"

    # Memory problems first: an OOM traceback is often surrounded by secondary
    # errors (engine core died, connection reset) that would match nothing.
    if _OOM.search(output) or _NO_KV_MEMORY.search(output) or _KV_TOO_SMALL.search(output):
        capacity = _OOM_CAPACITY.search(output)
        card = f" This GPU has {capacity.group(1)} GiB." if capacity else ""
        estimated = _KV_ESTIMATED_LEN.search(output)
        hint = (
            f" vLLM estimates this GPU can serve a context of about {estimated.group(1)} tokens."
            if estimated
            else ""
        )
        return (
            f"{named} ran out of GPU memory during startup.{card}{hint} "
            f"Lower MAX_MODEL_LEN or MAX_NUM_SEQS, set ENFORCE_EAGER=true to skip "
            f"CUDA graph capture, use a quantized checkpoint, or redeploy on a "
            f"larger GPU (or more GPUs with TENSOR_PARALLEL_SIZE). If the model "
            f"loaded but the KV cache did not fit, raising GPU_MEMORY_UTILIZATION "
            f"a little (default 0.9) can also help."
        )

    match = _MAX_LEN_EXCEEDS_MODEL.search(output)
    if match:
        return (
            f"MAX_MODEL_LEN={match.group(1)} is larger than the context length "
            f"{named} declares in its config.json. Lower MAX_MODEL_LEN (or unset it "
            f"to use the model's own limit). To override the limit anyway, set "
            f"VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and expect degraded output past it."
        )

    match = _BAD_ARGS.search(output)
    if match:
        return (
            f"vLLM rejected its command line: {match.group(1).strip()} "
            f"Check the environment variables that map to vLLM flags and "
            f"VLLM_EXTRA_ARGS; run `vllm serve --help` in the image for the "
            f"accepted flags and values."
        )

    if _GATED.search(output):
        return (
            f"{named} is gated or private on Hugging Face and the worker was not "
            f"allowed to download it. Set HF_TOKEN to a token whose account has "
            f"accepted the model's license, or pick a public model."
        )

    if _NOT_FOUND.search(output):
        return (
            f"{named} was not found on Hugging Face. Check MODEL_NAME for typos "
            f"(it must be the full `org/repo` id) and MODEL_REVISION if set. Private "
            f"repositories also return not-found until HF_TOKEN grants access."
        )

    if _UNSUPPORTED_ARCH.search(output):
        return (
            f"{named} uses an architecture this vLLM version cannot serve. Check "
            f"the vLLM supported-models list, or try a newer worker release."
        )

    if _NO_SPACE.search(output):
        return (
            f"{named} ran out of disk while downloading. Increase the endpoint's "
            f"container disk to comfortably exceed the size of the repository, or "
            f"attach a network volume so the weights are cached there instead."
        )

    return None
