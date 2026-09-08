"""Unit tests for job-input normalization (src/handler.py). No HTTP is made."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from handler import _normalize_job_input  # noqa: E402


class TestOpenAIPassthrough:
    def test_openai_input_posts_chat_by_default(self):
        route, method, body = _normalize_job_input({"openai_input": {"messages": []}})
        assert (route, method) == ("/v1/chat/completions", "POST")
        assert body == {"messages": []}

    def test_bare_openai_route_is_get(self):
        assert _normalize_job_input({"openai_route": "/v1/models"}) == ("/v1/models", "GET", None)

    def test_empty_openai_input_falls_back_to_get(self):
        # A RunPod GET on /openai/* must not turn into a POST (405 on /v1/models).
        assert _normalize_job_input({"openai_route": "/v1/models", "openai_input": {}}) == (
            "/v1/models", "GET", None)


class TestGenericRoute:
    def test_route_without_body_defaults_to_get(self):
        assert _normalize_job_input({"route": "/v1/models"}) == ("/v1/models", "GET", None)

    def test_route_with_body_defaults_to_post(self):
        route, method, body = _normalize_job_input({"route": "/v1/completions", "body": {"prompt": "hi"}})
        assert (route, method, body) == ("/v1/completions", "POST", {"prompt": "hi"})

    def test_explicit_method_wins(self):
        route, method, _ = _normalize_job_input({"route": "/v1/models", "method": "post"})
        assert method == "POST"


class TestLegacyShorthand:
    def test_messages_maps_to_chat_completions(self):
        route, method, body = _normalize_job_input({"messages": [{"role": "user", "content": "hi"}]})
        assert (route, method) == ("/v1/chat/completions", "POST")
        assert body["messages"][0]["content"] == "hi"
        assert body["stream"] is False

    def test_prompt_maps_to_completions(self):
        route, method, body = _normalize_job_input({"prompt": "hi", "sampling_params": {"temperature": 0.5}})
        assert (route, method) == ("/v1/completions", "POST")
        assert body["prompt"] == "hi"
        assert body["temperature"] == 0.5

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            _normalize_job_input({})


def _collect(job: dict) -> list:
    """Drain the async-generator handler synchronously."""
    import asyncio

    import handler as handler_module

    async def run():
        return [item async for item in handler_module.handler(job)]

    return asyncio.run(run())


class TestStartupError:
    """main.py sets handler.startup_error when vLLM died for a reason a restart cannot fix."""

    def test_answers_every_job_with_the_cause(self, monkeypatch):
        import handler as handler_module

        monkeypatch.setattr(handler_module, "startup_error", "Ran out of GPU memory.")

        outputs = _collect({"input": {"prompt": "x"}})

        assert outputs == [{"error": {"message": "Ran out of GPU memory.", "type": "startup_error", "code": None}}]

    def test_wins_over_a_dead_process_and_bad_input(self, monkeypatch):
        import handler as handler_module

        monkeypatch.setattr(handler_module, "startup_error", "Ran out of GPU memory.")
        # Either of these would otherwise be reported instead, and both say less.
        monkeypatch.setattr(handler_module, "_is_vllm_alive", lambda: False)

        outputs = _collect({"input": {}})

        assert "GPU memory" in outputs[0]["error"]["message"]

    def test_a_healthy_start_leaves_jobs_alone(self, monkeypatch):
        import handler as handler_module

        monkeypatch.setattr(handler_module, "startup_error", None)
        monkeypatch.setattr(handler_module, "_is_vllm_alive", lambda: False)

        outputs = _collect({"input": {"prompt": "x"}})

        # Falls through to the normal liveness check rather than short-circuiting.
        assert "not running" in outputs[0]["error"]["message"]
