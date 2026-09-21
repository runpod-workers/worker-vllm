"""main() relaunches once for a vanished HF revision, and only for that."""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main  # noqa: E402

REVISION_NOT_FOUND = (
    "huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. Revision Not Found "
    "for url https://huggingface.co/org/model/resolve/abc123def/config.json."
)
TORCH_OOM = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB."


@pytest.fixture
def harness(monkeypatch):
    """Run main() without vLLM, the RunPod SDK, or a GPU.

    Returns a driver: call it with the outputs each launch attempt should die
    with (None = the attempt becomes healthy) and it returns the handler module
    stub, whose startup_error records what jobs would be answered with.
    """
    monkeypatch.setenv("MODEL_NAME", "org/model")
    # The merged pre-flight (model_preflight.py) would otherwise ask the real
    # HF Hub about "org/model"; these tests are about what happens after it.
    monkeypatch.setattr(main.model_preflight, "check_model_access", lambda: None)
    monkeypatch.setattr(main, "stop_vllm", lambda proc: None)

    handler_stub = types.ModuleType("handler")
    handler_stub.handler = lambda job: None
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=mock.MagicMock())
    monkeypatch.setitem(sys.modules, "handler", handler_stub)
    monkeypatch.setitem(sys.modules, "runpod", runpod_stub)

    start_vllm = mock.MagicMock(return_value=mock.MagicMock())
    monkeypatch.setattr(main, "start_vllm", start_vllm)

    def drive(*attempt_outputs):
        outputs = iter(attempt_outputs)

        def fake_wait(proc):
            output = next(outputs)
            if output is not None:
                main.recent_output.clear()
                main.recent_output.append(output)
                raise RuntimeError("vLLM serve exited during startup with code 1")

        monkeypatch.setattr(main, "wait_for_vllm", fake_wait)
        main.recent_output.clear()
        main.main()
        handler_stub.launches = start_vllm.call_count
        return handler_stub

    return drive


class TestRevisionRelaunch:
    def test_vanished_revision_gets_one_relaunch_that_can_succeed(self, harness, monkeypatch):
        monkeypatch.setenv("MODEL_REVISION", "abc123def")

        handler = harness(REVISION_NOT_FOUND, None)

        assert handler.launches == 2
        assert handler.startup_error is None
        # The stale pin must not be replayed into the second launch.
        import os

        assert "MODEL_REVISION" not in os.environ

    def test_second_revision_failure_answers_jobs_with_the_cause(self, harness):
        handler = harness(REVISION_NOT_FOUND, REVISION_NOT_FOUND)

        assert handler.launches == 2  # never a third attempt
        assert "revision" in handler.startup_error.lower()

    def test_other_fatal_failures_do_not_relaunch(self, harness):
        handler = harness(TORCH_OOM)

        assert handler.launches == 1
        assert "ran out of GPU memory" in handler.startup_error
