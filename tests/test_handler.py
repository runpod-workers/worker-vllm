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


def _stream(chunks: list) -> list:
    """Run a streaming job against a local server that writes `chunks` as separate TCP writes."""
    import asyncio

    from aiohttp import web

    import handler as handler_module

    async def sse(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for chunk in chunks:
            await resp.write(chunk)
            await asyncio.sleep(0.05)  # let each write arrive as its own chunk
        await resp.write_eof()
        return resp

    async def run():
        app = web.Application()
        app.router.add_post("/v1/completions", sse)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        old_url = handler_module.VLLM_BASE_URL
        handler_module.VLLM_BASE_URL = f"http://127.0.0.1:{port}"
        try:
            job = {"input": {"route": "/v1/completions", "body": {"model": "m", "prompt": "x", "stream": True}}}
            return [item async for item in handler_module.handler(job)]
        finally:
            handler_module.VLLM_BASE_URL = old_url
            await runner.cleanup()

    return asyncio.run(run())


class TestStreamingUtf8:
    """iter_any() chunk boundaries are arbitrary; decoding must not corrupt split characters."""

    def test_two_byte_char_split_across_chunks(self):
        data = "data: héllo\n\n".encode()
        cut = data.index("é".encode()) + 1

        outputs = _stream([data[:cut], data[cut:]])

        assert "�" not in "".join(outputs)
        assert "".join(outputs) == "data: héllo\n\n"

    @pytest.mark.parametrize("offset", [1, 2, 3])
    def test_four_byte_emoji_split_at_each_position(self, offset):
        data = "data: 😀 done\n\n".encode()
        cut = data.index("😀".encode()) + offset

        outputs = _stream([data[:cut], data[cut:]])

        assert "".join(outputs) == "data: 😀 done\n\n"

    def test_ascii_stream_is_unchanged(self):
        outputs = _stream([b"data: one\n\n", b"data: two\n\n"])

        assert outputs == ["data: one\n\n", "data: two\n\n"]

    def test_stream_ending_mid_character_flushes_a_replacement(self):
        data = "data: ok 😀".encode()

        outputs = _stream([data[:-2]])

        assert "".join(outputs) == "data: ok �"

    def test_joined_output_equals_the_original_text(self):
        text = "data: héllo 😀 こんにちは\n\ndata: [DONE]\n\n"
        data = text.encode()
        # One byte per chunk splits every multibyte character at every position.
        outputs = _stream([bytes([b]) for b in data])

        assert "".join(outputs) == text
