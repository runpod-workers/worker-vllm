from unittest import mock

import pytest

import main

GATED = "huggingface_hub.errors.GatedRepoError: 401 Client Error: Cannot access gated repo"
UNKNOWN = "RuntimeError: something nobody has seen before"


@pytest.fixture
def run_main(monkeypatch, fake_sdk):
    monkeypatch.setenv("MODEL_NAME", "org/model")
    monkeypatch.setenv("MAX_CONCURRENCY", "7")
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    monkeypatch.setattr(main.model_preflight, "check_model_access", lambda: None)
    monkeypatch.setattr(main, "stop_vllm", lambda proc: None)
    monkeypatch.setattr(main, "start_vllm", mock.MagicMock(return_value=mock.MagicMock()))

    seen = {}
    real_start = fake_sdk.start.side_effect

    def start(config):
        seen["before"] = config["concurrency_modifier"](1)
        try:
            real_start(config)
        finally:
            seen["after"] = config["concurrency_modifier"](1)

    fake_sdk.start.side_effect = start

    def drive(output):
        def fake_wait(proc):
            if output is not None:
                main.recent_output.clear()
                main.recent_output.append(output)
                raise RuntimeError("vLLM serve exited during startup with code 1")

        monkeypatch.setattr(main, "wait_for_vllm", fake_wait)
        main.recent_output.clear()
        main.main()
        return seen

    return drive


def test_healthy_start_opens_job_intake(run_main, fake_sdk):
    seen = run_main(None)

    assert fake_sdk.hooks == [main.start_backend]
    assert seen == {"before": 0, "after": 7}
    assert fake_sdk.handler.startup_error is None


def test_known_fatal_failure_opens_intake_to_answer_jobs_with_the_cause(run_main, fake_sdk):
    seen = run_main(GATED)

    assert seen == {"before": 0, "after": 7}
    assert "gated or private" in fake_sdk.handler.startup_error


def test_unknown_failure_raises_from_the_hook_and_never_opens_intake(run_main, fake_sdk):
    with pytest.raises(RuntimeError, match="exited during startup"):
        run_main(UNKNOWN)

    assert fake_sdk.hooks == [main.start_backend]
    assert main.backend_settled.is_set() is False


def test_realtime_starts_vllm_before_the_sdk_without_a_hook(run_main, fake_sdk, monkeypatch):
    monkeypatch.setenv("RUNPOD_REALTIME_PORT", "8888")

    seen = run_main(None)

    assert fake_sdk.hooks == []
    assert seen == {"before": 7, "after": 7}


def test_realtime_unknown_failure_exits_before_the_sdk_starts(run_main, fake_sdk, monkeypatch):
    monkeypatch.setenv("RUNPOD_REALTIME_PORT", "8888")

    with pytest.raises(RuntimeError):
        run_main(UNKNOWN)

    fake_sdk.start.assert_not_called()
