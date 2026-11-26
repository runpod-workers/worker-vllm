"""Container entrypoint: start RunPod serverless with vLLM startup as a prestart hook.

The worker never imports vLLM. The prestart hook builds the CLI from environment
variables (see args_builder.py), launches `vllm serve` on the loopback interface,
and polls /health until the server (and model) is ready. Jobs stay queued until
startup settles.

Before launching vLLM, a metadata-only pre-flight (model_preflight.py) checks
that the configured model is actually fetchable from the Hugging Face Hub, so
a typo'd MODEL_NAME or a gated model without HF_TOKEN fails in seconds instead
of at the end of the download.

If vLLM dies during startup for a reason a restart cannot fix (CUDA OOM, a
MAX_MODEL_LEN the GPU cannot hold, a bad flag, a gated model), the worker stays
up and answers every job with the cause instead of crash-looping; see
startup_errors.py. Three startup failures get relaunches first, because
retrying them can genuinely succeed: a Hugging Face revision that no longer
exists (vLLM pins refs to commit hashes since v0.28, so a force-pushed repo
invalidates the pin) is retried once against the repo's current state, running
out of disk with vLLM's compile cache on the network volume is retried once
with the cache in the container (see compile_cache.py), and a startup OOM is
retried in up to two stages — first a smaller batch with CUDA graphs kept
(MAX_NUM_BATCHED_TOKENS halved toward 8192 and MAX_NUM_SEQS bounded to
MAX_CONCURRENCY, the two settings whose defaults grew at init time in recent
vLLM releases), then ENFORCE_EAGER=true if the OOM comes back. Unrecognised
failures are reported by the SDK before the worker exits non-zero, so the
platform retries them.
"""

import collections
import contextvars
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import compile_cache
import model_preflight
import startup_errors
from args_builder import TRUE_VALUES, build_vllm_args, redact_argv
from download_model import LOCAL_MODEL_ARGS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1800"))  # seconds
HEALTH_POLL_INTERVAL = 2  # seconds
# How long to wait for vLLM to exit after SIGTERM before SIGKILL.
SHUTDOWN_GRACE = 30  # seconds

vllm_process: subprocess.Popen | None = None

backend_settled = threading.Event()

# Enough of vLLM's output to recognise why it died. The traceback that matters
# is always the last thing it prints.
recent_output: collections.deque[str] = collections.deque(maxlen=400)
output_pump: threading.Thread | None = None


def apply_local_model_args() -> None:
    """Load args baked into the image by download_model.py (Option 2 builds).

    The baked model path wins over MODEL_NAME env vars, and HF hub access is
    forced offline since the weights are already on disk.
    """
    if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
        return
    try:
        with open(LOCAL_MODEL_ARGS_PATH) as f:
            local_args = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.error("Failed to read %s: %s", LOCAL_MODEL_ARGS_PATH, e)
        return

    logging.info("Using baked-in model args: %s", local_args)
    for key, value in local_args.items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def pump_output(proc: subprocess.Popen) -> None:
    """Echo vLLM's output to our stdout while keeping the tail for diagnosis."""
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
        recent_output.append(line)


def start_vllm() -> subprocess.Popen:
    global output_pump

    argv = ["vllm", "serve", "--host", VLLM_HOST, "--port", VLLM_PORT]
    argv += build_vllm_args()

    logging.info("Starting vLLM: %s", " ".join(redact_argv(argv)))
    # vLLM's stdout+stderr flow through a pipe so we can both forward them to the
    # worker logs and keep the tail to classify a startup failure. Unbuffered so
    # the child's log lines arrive as they are written, not when its buffer fills.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    # Carry the hook's log capture into the output thread.
    output_pump = threading.Thread(
        target=contextvars.copy_context().run, args=(pump_output, proc), daemon=True
    )
    output_pump.start()
    return proc


def stop_vllm(proc: subprocess.Popen) -> None:
    """Make sure a failed vLLM is gone (and its GPU memory released)."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Let the pump drain whatever the pipe still holds so the traceback tail
    # is in recent_output before we look at it.
    if output_pump is not None:
        output_pump.join(timeout=10)


def wait_for_vllm(proc: subprocess.Popen) -> None:
    """Poll GET /health until vLLM is ready; fail fast if it crashes or times out."""
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/health"

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM serve exited during startup with code {proc.returncode}")
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("vLLM is healthy")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    raise RuntimeError(f"vLLM did not become healthy within {STARTUP_TIMEOUT}s")


def _forward_signal(signum, _frame):
    logging.info("Received signal %s, shutting down vLLM", signum)
    if vllm_process and vllm_process.poll() is None:
        vllm_process.send_signal(signum)
    sys.exit(128 + signum)


def drop_pinned_revisions() -> None:
    """Clear revision pins so the next launch resolves the repo's current state.

    Since v0.28 vLLM resolves every Hugging Face ref to an exact commit hash at
    launch, so a repo whose history was rewritten 404s on a hash that no longer
    exists. Relaunching is enough for implicit pins (a fresh launch resolves a
    fresh hash); explicit *_REVISION values are dropped too, loudly, because
    serving the current branch beats crash-looping on a commit that is gone.
    """
    for name in ("MODEL_REVISION", "TOKENIZER_REVISION", "CODE_REVISION"):
        value = os.environ.pop(name, None)
        if value:
            logging.warning(
                "Dropping %s=%s — that revision no longer exists on Hugging Face; "
                "retrying against the repository's current default branch.",
                name,
                value,
            )


# vLLM doubled the max-num-batched-tokens default in v0.28, doubling peak
# activation memory at boot; the recovery halves it back toward this.
OOM_TOKEN_BUDGET = "8192"
# runpod's concurrency default when MAX_CONCURRENCY is unset (mirrors main()).
DEFAULT_MAX_CONCURRENCY = 30
# vLLM's OpenAI-server sequence default is GPU-dependent, but never below 256
# on supported versions: 256 on smaller GPUs, 1024 on larger ones. Without an
# explicit value, a target above this floor could increase rather than shrink
# the launch's allocation.
MIN_VLLM_DEFAULT_MAX_NUM_SEQS = 256


def _positive_int(raw: str) -> Optional[int]:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _extra_args_split() -> list[str]:
    try:
        return shlex.split(os.getenv("VLLM_EXTRA_ARGS", ""))
    except ValueError:
        # Unbalanced quotes: `vllm serve` fails on the same string either way.
        return []


def _extra_args_pin(flag: str) -> Optional[str]:
    """The value VLLM_EXTRA_ARGS pins for a value flag, or None when unpinned.

    VLLM_EXTRA_ARGS is appended after every env-derived flag and argparse's
    last-occurrence-wins makes it the highest-precedence source, so a pin here
    beats any env change the recovery could make: the knob must be left alone
    (and reported) instead of silently "adjusted". Returns the last occurrence,
    or "" for a trailing bare flag.
    """
    pinned = None
    tokens = _extra_args_split()
    for i, token in enumerate(tokens):
        if token == flag:
            pinned = tokens[i + 1] if i + 1 < len(tokens) else ""
        elif token.startswith(flag + "="):
            pinned = token[len(flag) + 1:]
    return pinned


def _extra_args_enforce_eager() -> Optional[bool]:
    """The eager polarity VLLM_EXTRA_ARGS pins with --enforce-eager/--no-enforce-eager."""
    pinned = None
    for token in _extra_args_split():
        if token == "--enforce-eager":
            pinned = True
        elif token == "--no-enforce-eager":
            pinned = False
    return pinned


def _enforce_eager_active() -> bool:
    """Whether the next launch will run without CUDA graphs."""
    pinned = _extra_args_enforce_eager()
    if pinned is not None:
        return pinned
    return os.getenv("ENFORCE_EAGER", "").strip().lower() in TRUE_VALUES


def _max_concurrency() -> int:
    raw = os.getenv("MAX_CONCURRENCY")
    if raw is None:
        return DEFAULT_MAX_CONCURRENCY
    value = _positive_int(raw)
    if value is None:
        logging.warning(
            "MAX_CONCURRENCY=%r is not a positive integer; bounding MAX_NUM_SEQS "
            "to the default %d.",
            raw,
            DEFAULT_MAX_CONCURRENCY,
        )
        return DEFAULT_MAX_CONCURRENCY
    return value


class StagedOomRecovery:
    """Bounded relaunch plan for a startup OOM: shrink the batch, then drop the graphs.

    Stage 1 frees memory while keeping CUDA graphs (measured in #366 to cost
    small models no throughput at the worker's concurrency): the peak-activation
    budget is shrunk toward MAX_NUM_BATCHED_TOKENS=8192 and MAX_NUM_SEQS is
    bounded to the worker's MAX_CONCURRENCY, the number of sequences CUDA graphs
    are captured for and the KV cache is budgeted against. Only a repeat OOM
    reaches stage 2, ENFORCE_EAGER=true, which gives up the graphs whose memory
    has been reserved up front since v0.29. Every relaunch must change
    something: a stage with nothing left to change (already-eager included) is
    skipped, so the plan fires at most twice and never relaunches an identical
    footprint. ``adjustments`` records what the relaunches actually did and
    ``suppressed`` the knobs found already in force or pinned by
    VLLM_EXTRA_ARGS, so the terminal error neither claims untried changes nor
    offers changes that cannot take effect.
    """

    def __init__(self) -> None:
        self.adjustments: list[str] = []
        self.suppressed: list[str] = []
        self._stages = [self._shrink_allocations, self._enforce_eager]

    def next_relaunch(self) -> list[str]:
        """Apply the next stage that changes the launch; empty when none remains."""
        while self._stages:
            applied = self._stages.pop(0)()
            if applied:
                self.adjustments.extend(applied)
                return applied
        return []

    def _shrink_allocations(self) -> list[str]:
        applied, token_budget = self._shrink_token_budget()
        applied += self._lower_max_num_seqs(token_budget)
        if applied:
            logging.warning(
                "Shrinking the batch for the OOM retry without changing the "
                "CUDA graph mode: %s.",
                ", ".join(applied),
            )
        return applied

    def _shrink_token_budget(self) -> tuple[list[str], Optional[int]]:
        """Shrink MAX_NUM_BATCHED_TOKENS toward 8192; tighter explicit values win.

        Returns the adjustment (if any) and the budget the next launch will
        actually run with, for clamping MAX_NUM_SEQS (vLLM requires
        max_num_batched_tokens >= max_num_seqs).
        """
        pinned = _extra_args_pin("--max-num-batched-tokens")
        if pinned is not None:
            logging.warning(
                "VLLM_EXTRA_ARGS pins --max-num-batched-tokens, which wins over "
                "MAX_NUM_BATCHED_TOKENS; leaving the peak-activation budget untouched."
            )
            self.suppressed.append("MAX_NUM_BATCHED_TOKENS (pinned by VLLM_EXTRA_ARGS)")
            return [], _positive_int(pinned)
        raw = os.getenv("MAX_NUM_BATCHED_TOKENS", "").strip()
        if raw and raw != "0":
            current = _positive_int(raw)
            if current is None:
                # The operator's value; vLLM rejects it with or without the retry.
                self.suppressed.append(
                    f"MAX_NUM_BATCHED_TOKENS={raw} (kept; not a positive integer)"
                )
                return [], None
            if current <= int(OOM_TOKEN_BUDGET):
                self.suppressed.append(
                    f"MAX_NUM_BATCHED_TOKENS={current} (kept; already within {OOM_TOKEN_BUDGET})"
                )
                return [], current
        os.environ["MAX_NUM_BATCHED_TOKENS"] = OOM_TOKEN_BUDGET
        return [f"MAX_NUM_BATCHED_TOKENS={OOM_TOKEN_BUDGET}"], int(OOM_TOKEN_BUDGET)

    def _lower_max_num_seqs(self, token_budget: Optional[int]) -> list[str]:
        """Bound MAX_NUM_SEQS to MAX_CONCURRENCY, never above an explicit tighter cap."""
        target = _max_concurrency()
        if token_budget is not None:
            target = min(target, token_budget)
        pinned = _extra_args_pin("--max-num-seqs")
        if pinned is not None:
            logging.warning(
                "VLLM_EXTRA_ARGS pins --max-num-seqs, which wins over MAX_NUM_SEQS; "
                "leaving the captured batch size untouched."
            )
            self.suppressed.append("MAX_NUM_SEQS (pinned by VLLM_EXTRA_ARGS)")
            return []
        raw = os.getenv("MAX_NUM_SEQS", "").strip()
        if raw:
            current = _positive_int(raw)
            if current is None:
                self.suppressed.append(f"MAX_NUM_SEQS={raw} (kept; not a positive integer)")
                return []
            if current <= target:
                # Raising an explicit tighter cap to the target would loosen
                # memory, not tighten it.
                self.suppressed.append(
                    f"MAX_NUM_SEQS={current} (kept; already at or below {target})"
                )
                return []
        elif target >= MIN_VLLM_DEFAULT_MAX_NUM_SEQS:
            # The effective default varies by GPU. It might be 1024, but setting
            # this target on a 256-default GPU would grow the footprint; leave
            # it unset unless we know the target is below every valid default.
            return []
        os.environ["MAX_NUM_SEQS"] = str(target)
        return [f"MAX_NUM_SEQS={target}"]

    def _enforce_eager(self) -> list[str]:
        """Final stage: drop CUDA graphs, whose memory is reserved up front."""
        pinned = _extra_args_enforce_eager()
        if pinned is True:
            self.suppressed.append("ENFORCE_EAGER=true (already set by VLLM_EXTRA_ARGS)")
            return []
        if pinned is False:
            logging.warning(
                "VLLM_EXTRA_ARGS pins --no-enforce-eager, which wins over "
                "ENFORCE_EAGER; keeping CUDA graphs rather than claiming a "
                "change the launch would not see."
            )
            self.suppressed.append("ENFORCE_EAGER (blocked by VLLM_EXTRA_ARGS --no-enforce-eager)")
            return []
        raw = os.getenv("ENFORCE_EAGER", "").strip().lower()
        if raw in TRUE_VALUES:
            # Already eager: the OOM'd launches ran without graphs, so this
            # stage would replay the same footprint.
            self.suppressed.append("ENFORCE_EAGER=true (already set)")
            return []
        os.environ["ENFORCE_EAGER"] = "true"
        if raw:
            logging.warning(
                "Overriding ENFORCE_EAGER=%s: graphs are how the boot OOM'd, so "
                "dropping them for the final retry.",
                raw,
            )
        else:
            logging.warning(
                "Disabling CUDA graphs for the final OOM retry (ENFORCE_EAGER=true): "
                "their memory has been reserved up front since v0.29, and the "
                "previous configuration still did not fit."
            )
        return ["ENFORCE_EAGER=true"]


def start_backend() -> None:
    global vllm_process

    import handler as proxy_handler

    # Ask the HF Hub whether the model is fetchable before paying for the
    # download: a typo'd MODEL_NAME, a gated model without HF_TOKEN, or a bad
    # MODEL_REVISION fails here in seconds instead of at the end of the cold
    # start. Only definitive answers fail the boot; see model_preflight.py.
    startup_error = model_preflight.check_model_access()
    if startup_error:
        logging.error(
            "Model pre-flight failed; answering jobs with the cause instead of starting vLLM: %s",
            startup_error,
        )
    # At most five launches: each of the three recoverable startup failures gets
    # relaunches — a vanished Hugging Face revision gets one (vLLM re-resolves
    # the pin on every launch; the pre-flight catches a *configured* revision
    # that never existed, this catches one gone by load time or resolved by
    # vLLM itself), running out of disk with the compile cache on the network
    # volume gets one (the volume can fill after the pre-launch check, during
    # the weight download; a relaunch compiles into the container instead), and
    # a startup OOM gets up to two staged relaunches (a smaller batch with CUDA
    # graphs kept first, graphs off only if the OOM comes back).
    can_retry_revision = True
    can_retry_disk = True
    recovery = StagedOomRecovery()
    while startup_error is None:
        vllm_process = start_vllm()
        try:
            wait_for_vllm(vllm_process)
            if recovery.adjustments:
                logging.info(
                    "vLLM settled after OOM recovery (%s); CUDA graphs %s.",
                    ", ".join(recovery.adjustments),
                    "disabled (ENFORCE_EAGER=true)" if _enforce_eager_active() else "enabled",
                )
            break
        except RuntimeError as e:
            logging.error("%s", e)
            stop_vllm(vllm_process)
            output = "".join(recent_output)
            if can_retry_revision and startup_errors.revision_not_found(output):
                can_retry_revision = False
                logging.warning(
                    "vLLM could not fetch the pinned Hugging Face revision "
                    "(vLLM pins refs to commit hashes; a force-pushed repo "
                    "invalidates them). Relaunching once against the current "
                    "revision."
                )
                drop_pinned_revisions()
                recent_output.clear()
                continue
            if (
                can_retry_disk
                and startup_errors.out_of_disk(output)
                and compile_cache.fall_back("out of space during startup", out_of_space=True)
            ):
                can_retry_disk = False
                logging.warning(
                    "vLLM ran out of disk during startup with its compile cache on "
                    "the network volume. Relaunching once with it in the container."
                )
                recent_output.clear()
                continue
            if startup_errors.memory_shortfall(output):
                applied = recovery.next_relaunch()
                if applied:
                    logging.warning(
                        "vLLM ran out of GPU memory during startup. Relaunching "
                        "with a smaller memory footprint (%s).",
                        ", ".join(applied),
                    )
                    recent_output.clear()
                    continue
            startup_error = startup_errors.classify(
                output,
                model=os.getenv("MODEL_NAME"),
                oom_adjustments=recovery.adjustments,
                oom_suppressed=recovery.suppressed,
            )
            if startup_error is None:
                # Nothing recognisable, so let the platform restart us: a failed
                # download or a bad host is worth another attempt.
                raise
            # A restart cannot fix this one. Exiting would crash-loop the worker,
            # paying for the download and the load on every attempt and showing the
            # user a traceback instead of a cause, so stay up and answer jobs with it.
            logging.error("vLLM cannot start on this configuration; answering jobs with the cause: %s", startup_error)

    proxy_handler.vllm_process = vllm_process
    proxy_handler.startup_error = startup_error
    backend_settled.set()


def main() -> None:
    apply_local_model_args()
    # The compile cache lives on the network volume (Dockerfile); a full or
    # read-only volume would fail the start, so compile in the container instead.
    compile_cache.ensure_usable()

    if not (os.getenv("MODEL_NAME") or os.getenv("VLLM_CONFIG_FILE") or os.getenv("MODEL")):
        logging.warning("MODEL_NAME is not set; `vllm serve` will fail without a --model argument")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    # SDK startup checks read the model environment at import time.
    import handler as proxy_handler
    import runpod

    if int(os.getenv("RUNPOD_REALTIME_PORT", "0")):
        # The SDK rejects prestart hooks in realtime mode.
        start_backend()
    else:
        runpod.serverless.register_prestart_hook(start_backend)

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            # Don't start job execution timeouts during model loading.
            "concurrency_modifier": lambda _current: max_concurrency if backend_settled.is_set() else 0,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
