"""RunPod Serverless handler that proxies jobs to the local vLLM OpenAI server.

`main.py` starts `vllm serve` on 127.0.0.1:VLLM_PORT and only then starts the
RunPod serverless loop, so by the time a job arrives the vLLM HTTP server is up
and fully backwards/forwards compatible — we never import vLLM.

Accepted job input shapes (all under job["input"]):

1. RunPod OpenAI passthrough (what the platform sends on /openai/v1/...):
       {"openai_route": "/v1/chat/completions", "openai_input": {...}}
2. Generic proxy to any vLLM route:
       {"route": "/v1/completions", "body": {...}, "method": "POST"}
       ("method" is optional — bodies POST, body-less requests GET)
3. Legacy shorthand:
       {"prompt": "...", "sampling_params": {...}, "stream": false}
       {"messages": [...], "sampling_params": {...}, "stream": true}

When the request body has "stream": true, raw SSE chunks are yielded as they
arrive; otherwise the parsed JSON response is yielded once.
"""

import logging
import os
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_PORT}")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))

DEFAULT_CHAT_ROUTE = "/v1/chat/completions"
DEFAULT_COMPLETION_ROUTE = "/v1/completions"

# Set by main.py once the vLLM subprocess is running, so we can fail fast
# instead of hanging on a dead server.
vllm_process = None

_default_model_cache: Optional[str] = None


def _is_vllm_alive() -> bool:
    return vllm_process is None or vllm_process.poll() is None


async def _default_model(session: aiohttp.ClientSession) -> Optional[str]:
    """Best-effort model id for legacy shorthand requests that omit "model"."""
    global _default_model_cache
    if _default_model_cache is not None:
        return _default_model_cache

    served = os.getenv("SERVED_MODEL_NAME") or os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
    if served:
        _default_model_cache = served.split(",")[0].strip()
        return _default_model_cache

    try:
        async with session.get(f"{VLLM_BASE_URL}/v1/models") as resp:
            data = await resp.json(content_type=None)
            model_id = (data.get("data") or [{}])[0].get("id")
            if model_id:
                _default_model_cache = model_id
                return model_id
    except Exception as e:
        logging.warning("Could not resolve default model from /v1/models: %s", e)

    return os.getenv("MODEL_NAME")


def _normalize_job_input(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    """Return (route, method, body) for any accepted job input shape."""
    if job_input.get("openai_input"):
        return job_input.get("openai_route") or DEFAULT_CHAT_ROUTE, "POST", job_input["openai_input"]

    if job_input.get("openai_route"):
        # Bare route without a body (e.g. /v1/models) is proxied as a GET.
        return job_input["openai_route"], "GET", None

    if job_input.get("route"):
        body = job_input.get("body")
        # The verb defaults to the payload: bodies POST, body-less requests GET
        # so read-only routes (/v1/models, /health) don't need "method" spelled out.
        method = (job_input.get("method") or ("POST" if body else "GET")).upper()
        return job_input["route"], method, body

    # Legacy shorthand: prompt/messages + sampling_params at the top level.
    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if messages is None and prompt is None:
        raise ValueError(
            "Job input must contain one of: openai_input (+openai_route), "
            "route (+body), or prompt/messages."
        )

    sampling_params = dict(job_input.get("sampling_params") or {})
    body = {**sampling_params, "stream": bool(job_input.get("stream", False))}
    if messages is not None:
        body["messages"] = messages
        return DEFAULT_CHAT_ROUTE, "POST", body
    body["prompt"] = prompt
    return DEFAULT_COMPLETION_ROUTE, "POST", body


def _error(message: str) -> dict:
    return {"error": {"message": message, "type": "worker_error", "code": None}}


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    job_input = job.get("input") or {}

    try:
        route, method, body = _normalize_job_input(job_input)
    except ValueError as e:
        yield _error(str(e))
        return

    if not _is_vllm_alive():
        yield _error("vLLM server process is not running; worker is unhealthy")
        return

    headers = {"Content-Type": "application/json"}

    if method != "GET" and body is not None and "model" not in body:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                model = await _default_model(session)
                if model:
                    body = {**body, "model": model}
        except Exception:
            pass

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.request(method, f"{VLLM_BASE_URL}{route}", json=body) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    logging.error("vLLM %s %s returned HTTP %s: %s", method, route, resp.status, detail)
                    yield _error(f"vLLM returned HTTP {resp.status}: {detail}")
                    return

                wants_stream = isinstance(body, dict) and body.get("stream") is True
                if wants_stream:
                    async for chunk in resp.content.iter_any():
                        yield chunk.decode("utf-8", errors="replace")
                else:
                    yield await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        logging.exception("Request to vLLM failed")
        yield _error(f"Request to vLLM failed: {e}")
