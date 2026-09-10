"""Container entrypoint: spawn `vllm serve`, wait for it, start RunPod serverless.

The worker never imports vLLM. We build the CLI from environment variables
(see args_builder.py), launch `vllm serve` on the loopback interface, poll
/health until the server (and model) is ready, and only then start the RunPod
serverless job loop so no job is pulled before the backend can serve it.

If vLLM dies during startup for a reason a restart cannot fix (CUDA OOM, a
MAX_MODEL_LEN the GPU cannot hold, a bad flag, a gated model), the worker stays
up and answers every job with the cause instead of crash-looping; see
startup_errors.py. Unrecognised failures still exit non-zero so the platform
retries them.
"""

import collections
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import startup_errors
from args_builder import build_vllm_args
from download_model import LOCAL_MODEL_ARGS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
STARTUP_TIMEOUT = int(os.getenv("VLLM_STARTUP_TIMEOUT", "1200"))  # seconds
HEALTH_POLL_INTERVAL = 2  # seconds
# How long to wait for vLLM to exit after SIGTERM before SIGKILL.
SHUTDOWN_GRACE = 30  # seconds

vllm_process: subprocess.Popen | None = None

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

    logging.info("Starting vLLM: %s", " ".join(argv))
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
    output_pump = threading.Thread(target=pump_output, args=(proc,), daemon=True)
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


def main() -> None:
    global vllm_process

    apply_local_model_args()

    if not (os.getenv("MODEL_NAME") or os.getenv("VLLM_CONFIG_FILE") or os.getenv("MODEL")):
        logging.warning("MODEL_NAME is not set; `vllm serve` will fail without a --model argument")

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    vllm_process = start_vllm()
    startup_error = None
    try:
        wait_for_vllm(vllm_process)
    except RuntimeError as e:
        logging.error("%s", e)
        stop_vllm(vllm_process)
        startup_error = startup_errors.classify("".join(recent_output), model=os.getenv("MODEL_NAME"))
        if startup_error is None:
            # Nothing recognisable, so let the platform restart us: a failed
            # download or a bad host is worth another attempt.
            sys.exit(1)
        # A restart cannot fix this one. Exiting would crash-loop the worker,
        # paying for the download and the load on every attempt and showing the
        # user a traceback instead of a cause, so stay up and answer jobs with it.
        logging.error("vLLM cannot start on this configuration; answering jobs with the cause: %s", startup_error)

    # Import here (not at module import time) so the RunPod SDK and handler
    # start only after the backend is confirmed healthy (or confirmed dead).
    import handler as proxy_handler
    import runpod

    proxy_handler.vllm_process = vllm_process
    proxy_handler.startup_error = startup_error

    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "30"))
    runpod.serverless.start(
        {
            "handler": proxy_handler.handler,
            "concurrency_modifier": lambda _current: max_concurrency,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
