"""main() relaunches a startup OOM once with a smaller footprint, and only once."""

import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main  # noqa: E402

TORCH_OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 340.00 MiB. "
    "GPU 0 has a total capacity of 44.42 GiB of which 85.88 MiB is free."
)
KV_TOO_SMALL = (
    "ValueError: The model's max seq len (65536) is larger than the available KV cache "
    "memory (4.21 GiB). Based on the available memory, the estimated maximum model "
    "length is 34480. Try increasing `gpu_memory_utilization` or decreasing "
    "`max_model_len` when initializing the engine."
)
REVISION_NOT_FOUND = (
    "huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. Revision Not Found "
    "for url https://huggingface.co/org/model/resolve/abc123def/config.json."
)
GATED = "huggingface_hub.errors.GatedRepoError: 401 Client Error: Cannot access gated repo"


@pytest.fixture
def harness(monkeypatch):
    """Run main() without vLLM, the RunPod SDK, or a GPU.

    Returns a driver: call it with the outputs each launch attempt should die
    with (None = the attempt becomes healthy) and it returns the handler module
    stub, whose startup_error records what jobs would be answered with.
    """
    monkeypatch.setenv("MODEL_NAME", "org/model")
    monkeypatch.delenv("ENFORCE_EAGER", raising=False)
    monkeypatch.delenv("MAX_NUM_BATCHED_TOKENS", raising=False)
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


class TestOomRelaunch:
    def test_oom_gets_one_reduced_footprint_relaunch_that_can_succeed(self, harness):
        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert handler.startup_error is None
        assert os.environ["ENFORCE_EAGER"] == "true"
        assert os.environ["MAX_NUM_BATCHED_TOKENS"] == "8192"

    def test_second_oom_answers_jobs_with_what_was_tried(self, harness):
        handler = harness(TORCH_OOM, TORCH_OOM)

        assert handler.launches == 2  # never a third attempt
        assert "ran out of GPU memory" in handler.startup_error
        assert "already retried once" in handler.startup_error
        assert "44.42 GiB" in handler.startup_error

    def test_no_kv_memory_and_kv_too_small_are_the_same_class(self, harness):
        handler = harness(KV_TOO_SMALL, None)

        assert handler.launches == 2
        assert handler.startup_error is None

    def test_fully_tightened_config_means_nothing_left_to_relax(self, harness, monkeypatch):
        # Both knobs already at/over their recovery values: another attempt would
        # run the same footprint and OOM identically, so don't pay for it.
        monkeypatch.setenv("ENFORCE_EAGER", "true")
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "8192")

        handler = harness(TORCH_OOM)

        assert handler.launches == 1
        assert "ran out of GPU memory" in handler.startup_error
        assert "already retried" not in handler.startup_error

    def test_partial_user_tightening_still_earns_one_relaunch(self, harness, monkeypatch):
        # Eager is already on but the token budget is untouched (default 16384):
        # halving it is a real change, so retry once shrinking only that knob.
        monkeypatch.setenv("ENFORCE_EAGER", "true")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert os.environ["ENFORCE_EAGER"] == "true"
        assert os.environ["MAX_NUM_BATCHED_TOKENS"] == "8192"

    def test_user_tightened_token_budget_is_kept_for_the_relaunch(self, harness, monkeypatch):
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "4096")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert os.environ["MAX_NUM_BATCHED_TOKENS"] == "4096"
        assert os.environ["ENFORCE_EAGER"] == "true"

    def test_explicit_eager_false_is_still_retried_but_loudly_overridden(self, harness, monkeypatch):
        monkeypatch.setenv("ENFORCE_EAGER", "false")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert os.environ["ENFORCE_EAGER"] == "true"

    def test_non_memory_failures_do_not_take_the_oom_relaunch(self, harness):
        handler = harness(REVISION_NOT_FOUND, GATED)

        # The revision relaunch fires once; the OOM budget stays untouched, so
        # the second failure is answered, not re-launched.
        assert handler.launches == 2
        assert "gated or private" in handler.startup_error
        assert "MAX_NUM_BATCHED_TOKENS" not in os.environ

    def test_revision_and_oom_each_get_their_single_retry(self, harness):
        handler = harness(REVISION_NOT_FOUND, TORCH_OOM, None)

        assert handler.launches == 3
        assert handler.startup_error is None
