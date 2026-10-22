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
# huggingface_hub raises RevisionNotFoundError when a commit/branch/tag does not
# exist. Since v0.28 vLLM resolves every model ref to an exact commit hash at
# load time (upstream's fix for "Artifact Pin Decay"), so a repo whose history
# was force-pushed 404s on a hash that no longer exists. Matched before
# _NOT_FOUND because the transport error underneath is also a 404.
_REVISION_NOT_FOUND = re.compile(r"RevisionNotFoundError|Revision Not Found", re.I)
_GATED = re.compile(r"GatedRepoError|401 Client Error", re.I)
_NOT_FOUND = re.compile(r"RepositoryNotFoundError|404 Client Error", re.I)
# transformers/dynamic_module_utils.py: config.json has an auto_map pointing at
# the repo's own modeling code, and vLLM won't run it without trust_remote_code.
# Wrapped in a pydantic ValidationError by vLLM's ModelConfig, so match the text.
_NEEDS_REMOTE_CODE = re.compile(
    r"contains custom code which must be executed|pass the argument `trust_remote_code=True`", re.I
)
_UNSUPPORTED_ARCH = re.compile(r"Model architectures \[.*?\] (?:are not supported|failed to be inspected)", re.I)

# --- Disk -------------------------------------------------------------------
_NO_SPACE = re.compile(r"No space left on device|ENOSPC|errno 28|Disk quota exceeded|EDQUOT|errno 122", re.I)

# vLLM's log prefixes: "(EngineCore pid=1014) ERROR 10-21 20:41:12 [core.py:1543] "
# on logged lines, and a bare "(APIServer pid=496) " on raw output (vLLM 0.30).
_LOG_PREFIX = re.compile(r"^\([\w.-]+ pid=\d+\) (?:[A-Z]+ [\d-]+ [\d:]+ \[[^\]]*\] ?)?")
# A logged record's prefix has a level and a timestamp; only lines that share one
# belong to the same record. Raw output has no such boundary.
_RECORD = re.compile(r"\) [A-Z]+ [\d-]+ [\d:]+ \[")
_FRAME = re.compile(r'^\s+File "([^"]+)", line \d+')
_EXCEPTION = re.compile(r"^(?:AssertionError|RuntimeError|NotImplementedError|ValueError)(?:: .*)?$")
_ANY_EXCEPTION = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit)\b(?::|$)")
# Wrappers vLLM raises after an engine process fails; they don't replace the cause.
_WRAPPER = re.compile(r"Engine core initialization failed|WorkerProc initialization failed")
_TRACEBACK_BREAK = re.compile(
    r"^(?:Traceback \(most recent call last\):|The above exception|During handling of the above)"
)
# Model implementations: vllm/model_executor/models/<name>.py, and the per-model
# packages under vllm/models/<name>/ (since v0.22; v0.32.0 ships deepseek_v4,
# kimi_k3, minimax_m3 and others there).
_MODEL_CODE = re.compile(r"/vllm/(?:model_executor/)?models/")
# The model's code states what it needs: "Flint requires FlashInfer attention",
# "... requires the FlashInfer TRTLLM BF16 MoE backend", "only supported on ...".
# Where an error is raised says nothing about permanence (a driver fault inside
# load_weights also ends in model code); a stated requirement does. Autograd's
# "requires grad" errors are not configuration requirements.
_REQUIREMENT = re.compile(
    r"\brequires?\b(?![ _]grad)|\bonly supports?\b|\bonly supported\b|\bnot supported (?:on|with|for|by)\b",
    re.I,
)
# Host faults anywhere in the failed launch, including a CUDA error wrapped by a
# model-code exception: another host can succeed. Fault signatures only, so
# torch's NVML/pynvml warnings on a healthy host don't count.
_HOST_FAULT = re.compile(
    r"CUDA (?:driver |unknown )?error|cudaError|CUDA-capable device|NVIDIA driver on your system is too old|"
    r"NCCL error|nccl(?:System|Unhandled|Internal)\w*Error|Watchdog caught collective operation timeout|"
    r"NVML_SUCCESS == r|NVMLError|nvml\w*Error|CUBLAS_STATUS|CUDNN_STATUS|illegal memory access|"
    r"uncorrectable ECC|\bXid\b",
    re.I,
)
# Transport trouble inside the quoted error itself (a retried download that
# succeeded elsewhere in the log must not count, so this only checks the message).
_TRANSPORT_FAULT = re.compile(
    r"Connection (?:reset|refused|aborted)|timed out|Max retries exceeded|Temporary failure in name resolution|"
    r"Too Many Requests|Service Unavailable|Bad Gateway|Gateway Time-?out|Internal Server Error",
    re.I,
)

# An engine process killed by a signal ("Failed core proc(s): {'EngineCore': -9}")
# died of something else, whatever model code logged earlier.
_ENGINE_KILLED = re.compile(r"Failed core proc\(s\): \{[^}]*: -\d+")
# A traceback logged below ERROR was caught and handled; it didn't end the launch.
_NOT_AN_ERROR_RECORD = re.compile(r"\) (?:DEBUG|INFO|WARNING) ")


def _without_nvlink_probe(output: str) -> str:
    """Drop vLLM's "NVLink detection failed. This is normal ..." record (an NVMLError traceback
    logged on PCIe hosts with tensor parallelism), so it isn't read as a host fault."""
    kept, skip_prefix = [], None
    for raw in output.splitlines():
        prefix, line = _split_prefix(raw)
        if "NVLink detection failed" in line and _RECORD.search(prefix):
            skip_prefix = prefix
            continue
        if skip_prefix is not None and prefix == skip_prefix:
            continue
        skip_prefix = None
        kept.append(raw)
    return "\n".join(kept)


def _split_prefix(raw: str):
    line = raw.rstrip()
    match = _LOG_PREFIX.match(line)
    return (match.group(0), line[match.end():]) if match else ("", line)


def model_code_error(output: str) -> Optional[str]:
    """The stated requirement a model's own vLLM code failed on, if that is all that went wrong.

    Matches an exception whose innermost frame is model code and whose message
    states a requirement the setup does not meet: flint asserts the FlashInfer
    attention backend, and its BF16 MoE needs a Blackwell-only kernel. Those
    fail identically on every start. Returns None when the failed launch shows
    a host fault, when the matched exception's chain shows a transport fault,
    or when a different exception (other than vLLM's wrappers) comes after the
    match, since then the match isn't what ended the launch.
    """
    if _HOST_FAULT.search(_without_nvlink_probe(output)) or _ENGINE_KILLED.search(output):
        return None
    found = None
    last_file = None
    message, message_prefix = None, None
    chain, after_separator = [], False

    def finish(lines):
        text = " ".join(lines)
        if _REQUIREMENT.search(text) and not _TRANSPORT_FAULT.search(" ".join(chain + [text])):
            return text[:300]
        return None

    for raw in output.splitlines():
        prefix, line = _split_prefix(raw)
        if message is not None:
            # A multi-line message continues only inside one logged record (same
            # timestamped prefix). The next record, multiprocessing's "Process
            # EngineCore:", or a warning printed after a raw traceback are not part of it.
            if (_RECORD.search(message_prefix) and prefix == message_prefix and line
                    and not line.startswith(" ") and not _FRAME.match(line) and not _TRACEBACK_BREAK.match(line)):
                message.append(line)
                continue
            found = finish(message) or found
            chain.append(" ".join(message))
            message = None
        if _TRACEBACK_BREAK.match(line):
            if line.startswith("Traceback") and not after_separator:
                chain = []  # a new, unchained traceback
            after_separator = not line.startswith("Traceback")
            last_file = None
            continue
        frame = _FRAME.match(line)
        if frame:
            last_file = frame.group(1)
            continue
        if line.startswith(" ") or not line:
            continue  # source and caret lines under a frame
        if (_EXCEPTION.match(line) and last_file and _MODEL_CODE.search(last_file)
                and not _NOT_AN_ERROR_RECORD.search(prefix)):
            message, message_prefix = [line], prefix
        elif _ANY_EXCEPTION.match(line):
            chain.append(line)
            # A different exception after the match, other than vLLM's wrapper or a
            # repeat of the same cause (logged copy, then raw copy), ended the launch.
            if found and not _WRAPPER.search(line) and not found.startswith(line[:80]):
                found = None
        last_file = None  # any other unindented line ends the frame list
    if message is not None:
        found = finish(message) or found
    return found


# The model-access messages are shared with model_preflight.py, which detects
# the same failures before the cold start; whether the user hits the fast path
# or the post-crash path, the wording must be identical.
def gated_message(named: str) -> str:
    return (
        f"{named} is gated or private on Hugging Face and the worker was not "
        f"allowed to download it. Set HF_TOKEN to a token whose account has "
        f"accepted the model's license, or pick a public model."
    )


def not_found_message(named: str) -> str:
    return (
        f"{named} was not found on Hugging Face. Check MODEL_NAME for typos "
        f"(it must be the full `org/repo` id) and MODEL_REVISION if set. Private "
        f"repositories also return not-found until HF_TOKEN grants access."
    )


def revision_not_found_message(named: str, revision: str) -> str:
    return (
        f"MODEL_REVISION={revision} does not exist for {named} on Hugging Face. "
        f"Check the model page for the branch, tag, or commit hash, or unset "
        f"MODEL_REVISION to use the default branch."
    )


def memory_shortfall(output: str) -> bool:
    """True when startup died for lack of GPU memory in any of its three forms.

    Split out from classify() because main.py treats this like the vanished
    revision: one relaunch with a smaller memory footprint can genuinely
    succeed (CUDA graph memory is reserved up front since v0.29 and the
    batched-tokens default doubled in v0.28, so a boot can OOM where a more
    conservative setup fits), so the worker retries once before answering
    jobs with the error.
    """
    return bool(_OOM.search(output) or _NO_KV_MEMORY.search(output) or _KV_TOO_SMALL.search(output))


def out_of_disk(output: str) -> bool:
    """True when startup died for lack of disk space (or a full disk quota).

    Split out from classify() because main.py relaunches once with vLLM's
    compile cache moved off the network volume: the pre-launch check in
    compile_cache.py runs before the weights download onto the same volume, so
    the volume can still fill during the start.
    """
    return bool(_NO_SPACE.search(output))


def revision_not_found(output: str) -> bool:
    """True when the failure is a Hugging Face revision that no longer exists.

    Split out from classify() because main.py treats this one specially: it is
    the only startup failure where a fresh attempt against the repo's current
    state can genuinely succeed (vLLM re-resolves the revision on every launch),
    so the worker retries once before answering jobs with the error.
    """
    return bool(_REVISION_NOT_FOUND.search(output))


def classify(output: str, model: Optional[str] = None, oom_retried: bool = False) -> Optional[str]:
    """One actionable message for a known fatal failure, or None to let it retry.

    ``oom_retried`` mirrors the revision case: set it when main.py already ran
    its one reduced-footprint relaunch, so the wording reflects what was tried.
    """
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
        if oom_retried:
            # By the time classify sees this output, main.py's reduced-footprint
            # relaunch has already run, so the advice skips knobs that were
            # already applied and names the fixes that actually remain.
            return (
                f"{named} ran out of GPU memory during startup.{card}{hint} "
                f"The worker already retried once with a smaller footprint "
                f"(ENFORCE_EAGER=true and MAX_NUM_BATCHED_TOKENS reduced to 8192 "
                f"where it was higher or unset) and it still does not fit. What "
                f"remains: a smaller or quantized checkpoint, a larger GPU (or "
                f"more GPUs with TENSOR_PARALLEL_SIZE), a lower MAX_MODEL_LEN, or, "
                f"if KV_CACHE_DTYPE=fp8 is set on an Ampere/Ada GPU, "
                f"KV_CACHE_DTYPE=auto (fp8 KV forces the FlashInfer backend there, "
                f"which needs extra workspace memory)."
            )
        return (
            f"{named} ran out of GPU memory during startup.{card}{hint} "
            f"Lower MAX_MODEL_LEN or MAX_NUM_SEQS, set ENFORCE_EAGER=true to skip "
            f"CUDA graph capture, use a quantized checkpoint, or redeploy on a "
            f"larger GPU (or more GPUs with TENSOR_PARALLEL_SIZE). If the model "
            f"loaded but the KV cache did not fit, raising GPU_MEMORY_UTILIZATION "
            f"a little (default 0.9) can also help. On this vLLM version two more "
            f"knobs recover memory: MAX_NUM_BATCHED_TOKENS=8192 (the default "
            f"doubled to 16384 in v0.28, which doubles peak activation memory), "
            f"and, if KV_CACHE_DTYPE=fp8 is set on an Ampere/Ada GPU, "
            f"KV_CACHE_DTYPE=auto (fp8 KV forces the FlashInfer backend there, "
            f"which needs extra workspace memory)."
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

    if _REVISION_NOT_FOUND.search(output):
        # Distinct from revision_not_found_message: by the time classify sees
        # this, main.py has already relaunched once against the repo's current
        # state, so the message must say that and point at the deeper causes.
        return (
            f"The Hugging Face revision requested for {named} does not exist "
            f"(the worker already retried once against the repo's current main "
            f"branch). If MODEL_REVISION or TOKENIZER_REVISION is set, check it "
            f"against the revisions listed on the model page, or unset it to use "
            f"the default branch. This also happens when the repository's history "
            f"was rewritten (force-pushed) between deploys."
        )

    if _GATED.search(output):
        return gated_message(named)

    if _NOT_FOUND.search(output):
        return not_found_message(named)

    if _NEEDS_REMOTE_CODE.search(output):
        # Checked before the architecture match: with remote code allowed, vLLM
        # can run an architecture it has no native class for through its
        # Transformers backend, so "unsupported" would be the wrong advice.
        return (
            f"{named} ships its own modeling code on Hugging Face (an auto_map in "
            f"its config.json), and vLLM only runs that code with "
            f"TRUST_REMOTE_CODE=true. Review the repository's code, then set "
            f"TRUST_REMOTE_CODE=true if you trust it. This also lets vLLM serve "
            f"architectures it has no native implementation for yet, through its "
            f"Transformers backend."
        )

    if _UNSUPPORTED_ARCH.search(output):
        return (
            f"{named} uses an architecture this vLLM version cannot serve. Check "
            f"the vLLM supported-models list, or try a newer worker release."
        )

    if _NO_SPACE.search(output):
        return (
            f"{named} ran out of disk during startup. With a network volume "
            f"attached, the model download (huggingface-cache) and vLLM's compile "
            f"cache (vllm-cache) are stored on it: free space there (vllm-cache is "
            f"safe to delete while no worker is starting) or use a larger volume. "
            f"Without one, increase the endpoint's container disk to comfortably "
            f"exceed the size of the repository, or attach a network volume so the "
            f"weights are cached there instead."
        )

    raised = model_code_error(output)
    if raised:
        return (
            f"{named} failed inside vLLM's implementation of the model during "
            f"startup: {raised.rstrip('.')}. The model's code requires something this "
            f"worker's GPU or settings don't provide, so restarting this worker will not "
            f"change it. Check the model card and vLLM's recipe for the attention backend "
            f"(ATTENTION_BACKEND), GPU generation or dtype it needs. If the endpoint lists "
            f"several GPU types, also remove the ones the model can't use."
        )

    return None
