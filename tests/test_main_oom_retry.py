"""main() relaunches a startup OOM in bounded stages: shrink the batch with
CUDA graphs kept, then eager as the last resort — and never an identical launch."""

import logging
import os
import sys
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
# vllm/v1/core/kv_cache_planning.py (kv_cache_utils.py before v0.34) under
# MAX_MODEL_LEN=auto: the weights left no room for the KV cache. Same memory
# class; must be retried and answered,
# not restarted into the same wall.
CANNOT_AUTOFIT = (
    "ValueError: Cannot auto-fit max_model_len: not enough GPU memory available "
    "to serve even a single token. Try reducing `gpu_memory_utilization`."
)
REVISION_NOT_FOUND = (
    "huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. Revision Not Found "
    "for url https://huggingface.co/org/model/resolve/abc123def/config.json."
)
GATED = "huggingface_hub.errors.GatedRepoError: 401 Client Error: Cannot access gated repo"

# The knobs OOM recovery (and its precedence rules) can touch; snapshotting them
# at every launch shows exactly what each retry changed.
SNAPSHOT_ENV = (
    "ENFORCE_EAGER",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_SEQS",
    "MAX_CONCURRENCY",
    "VLLM_EXTRA_ARGS",
)


@pytest.fixture
def harness(monkeypatch, fake_sdk):
    """Run main() without vLLM, the RunPod SDK, or a GPU.

    Returns a driver: call it with the outputs each launch attempt should die
    with (None = the attempt becomes healthy). It returns the handler stub,
    whose startup_error records what jobs would be answered with, launches the
    launch count, and launch_envs a snapshot of the OOM knobs' env at each
    launch.
    """
    monkeypatch.setenv("MODEL_NAME", "org/model")
    # main() probes VLLM_CACHE_ROOT (compile_cache.py); keep a developer's own out of it.
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    for name in SNAPSHOT_ENV:
        monkeypatch.delenv(name, raising=False)
    # The merged pre-flight (model_preflight.py) would otherwise ask the real
    # HF Hub about "org/model"; these tests are about what happens after it.
    monkeypatch.setattr(main.model_preflight, "check_model_access", lambda: None)
    monkeypatch.setattr(main, "stop_vllm", lambda proc: None)

    launches = []

    def fake_start():
        launches.append({name: os.environ.get(name) for name in SNAPSHOT_ENV})
        return mock.MagicMock()

    monkeypatch.setattr(main, "start_vllm", fake_start)

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
        fake_sdk.handler.launches = len(launches)
        fake_sdk.handler.launch_envs = list(launches)
        return fake_sdk.handler

    return drive


class TestFirstStageKeepsCudaGraphs:
    def test_oom_relaunches_with_a_smaller_batch_and_graphs_kept(self, harness):
        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert handler.startup_error is None
        first, second = handler.launch_envs
        assert first["ENFORCE_EAGER"] is None
        assert second["ENFORCE_EAGER"] is None  # graphs kept on the first retry
        assert second["MAX_NUM_BATCHED_TOKENS"] == "8192"
        assert second["MAX_NUM_SEQS"] == "30"  # bounded to MAX_CONCURRENCY's default
        assert second != first  # every retry must change something

    def test_kv_shortfall_is_the_same_memory_class(self, harness):
        handler = harness(KV_TOO_SMALL, None)

        assert handler.launches == 2
        assert handler.startup_error is None

    def test_cannot_autofit_gets_the_same_relaunch(self, harness):
        handler = harness(CANNOT_AUTOFIT, None)

        assert handler.launches == 2
        assert handler.startup_error is None
        assert handler.launch_envs[1]["ENFORCE_EAGER"] is None


class TestSecondStageDropsCudaGraphs:
    def test_a_repeat_oom_reaches_eager_only_on_the_third_launch(self, harness):
        handler = harness(TORCH_OOM, TORCH_OOM, None)

        assert handler.launches == 3
        assert handler.startup_error is None
        first, second, third = handler.launch_envs
        assert second["ENFORCE_EAGER"] is None  # graphs survived the first retry
        assert third["ENFORCE_EAGER"] == "true"
        # The eager retry changes exactly one knob on top of the batch stage.
        assert third == {**second, "ENFORCE_EAGER": "true"}
        assert first != second

    def test_explicit_eager_false_is_overridden_only_at_the_eager_stage(self, harness, monkeypatch):
        monkeypatch.setenv("ENFORCE_EAGER", "false")

        handler = harness(TORCH_OOM, TORCH_OOM, None)

        _, second, third = handler.launch_envs
        assert second["ENFORCE_EAGER"] == "false"  # the user's choice survives the batch stage
        assert third["ENFORCE_EAGER"] == "true"


class TestRetriesAreBounded:
    def test_a_third_oom_answers_jobs_without_a_fourth_identical_launch(self, harness):
        handler = harness(TORCH_OOM, TORCH_OOM, TORCH_OOM)

        assert handler.launches == 3  # initial + two staged retries, never a fourth
        assert "ran out of GPU memory" in handler.startup_error
        assert "already relaunched" in handler.startup_error
        # The terminal message reports every adjustment that was actually tried.
        for tried in ("MAX_NUM_BATCHED_TOKENS", "MAX_NUM_SEQS", "ENFORCE_EAGER"):
            assert tried in handler.startup_error
        assert "44.42 GiB" in handler.startup_error

    def test_cannot_autofit_persists_as_an_answered_memory_failure(self, harness):
        handler = harness(CANNOT_AUTOFIT, CANNOT_AUTOFIT, CANNOT_AUTOFIT)

        assert handler.launches == 3
        assert "ran out of GPU memory" in handler.startup_error
        assert "already relaunched" in handler.startup_error


class TestAlreadyTightenedSettings:
    def test_eager_already_on_skips_the_eager_stage(self, harness, monkeypatch):
        monkeypatch.setenv("ENFORCE_EAGER", "true")

        handler = harness(TORCH_OOM, TORCH_OOM, TORCH_OOM)

        # Only the batch stage changed anything; the eager stage was already
        # enabled, so a third launch would replay the same footprint.
        assert handler.launches == 2
        assert "MAX_NUM_BATCHED_TOKENS" in handler.startup_error
        # In force all along: neither claimed as a retry nor offered as a fix.
        assert "ENFORCE_EAGER" not in handler.startup_error

    def test_fully_tightened_config_has_nothing_left_to_retry(self, harness, monkeypatch):
        monkeypatch.setenv("ENFORCE_EAGER", "true")
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "8192")
        monkeypatch.setenv("MAX_NUM_SEQS", "30")

        handler = harness(TORCH_OOM)

        assert handler.launches == 1  # every knob already at/over the recovery values
        assert "ran out of GPU memory" in handler.startup_error
        assert "already relaunched" not in handler.startup_error

    def test_an_explicit_smaller_max_num_seqs_is_never_raised(self, harness, monkeypatch):
        monkeypatch.setenv("MAX_NUM_SEQS", "16")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        second = handler.launch_envs[1]
        assert second["MAX_NUM_SEQS"] == "16"  # kept, not raised to the target
        assert second["MAX_NUM_BATCHED_TOKENS"] == "8192"
        assert second["ENFORCE_EAGER"] is None

    def test_a_user_tightened_token_budget_is_kept_for_the_retry(self, harness, monkeypatch):
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "4096")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert handler.launch_envs[1]["MAX_NUM_BATCHED_TOKENS"] == "4096"
        assert handler.launch_envs[1]["MAX_NUM_SEQS"] == "30"

    def test_an_unparsable_token_budget_is_kept_without_blocking_the_batch_stage(
        self, harness, monkeypatch
    ):
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "lots")

        handler = harness(TORCH_OOM, None)

        # The operator's value stands; MAX_NUM_SEQS still tightens on its own.
        assert handler.launches == 2
        assert handler.launch_envs[1]["MAX_NUM_BATCHED_TOKENS"] == "lots"
        assert handler.launch_envs[1]["MAX_NUM_SEQS"] == "30"


class TestCustomConstraints:
    def test_max_num_seqs_follows_a_custom_max_concurrency(self, harness, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENCY", "8")

        handler = harness(TORCH_OOM, None)

        assert handler.launch_envs[1]["MAX_NUM_SEQS"] == "8"

    @pytest.mark.parametrize("configured, expected", [(None, None), ("256", "256"), ("1024", "500")])
    def test_high_custom_concurrency_only_shrinks_max_num_seqs(
        self, harness, monkeypatch, configured, expected
    ):
        monkeypatch.setenv("MAX_CONCURRENCY", "500")
        if configured is not None:
            monkeypatch.setenv("MAX_NUM_SEQS", configured)

        handler = harness(TORCH_OOM, None)

        # Unset and explicit 256 might live on a 256-default GPU: setting 500
        # would grow them. An explicit 1024 is a known reduction to 500.
        assert handler.launch_envs[1]["MAX_NUM_SEQS"] == expected
        assert handler.launch_envs[1]["MAX_NUM_BATCHED_TOKENS"] == "8192"

    def test_max_num_seqs_never_overruns_the_effective_token_budget(self, harness, monkeypatch):
        # vLLM requires max_num_batched_tokens >= max_num_seqs.
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "16")
        monkeypatch.setenv("MAX_NUM_SEQS", "256")

        handler = harness(TORCH_OOM, None)

        assert handler.launch_envs[1]["MAX_NUM_SEQS"] == "16"

    @pytest.mark.parametrize("raw", ["lots", "0", "-3", ""])
    def test_an_unusual_max_concurrency_falls_back_to_the_default(self, monkeypatch, raw):
        # main() itself parses MAX_CONCURRENCY with int(), so this can't be
        # driven through a full launch; the recovery's own read must stay
        # conservative instead of crashing earlier than main() would.
        monkeypatch.setenv("MAX_CONCURRENCY", raw)

        assert main._max_concurrency() == main.DEFAULT_MAX_CONCURRENCY == 30


class TestVllmExtraArgsKeepsTheLastWord:
    def test_a_pinned_max_num_seqs_is_left_alone_with_a_warning(self, harness, monkeypatch, caplog):
        monkeypatch.setenv("VLLM_EXTRA_ARGS", "--max-num-seqs 256")

        with caplog.at_level(logging.WARNING):
            handler = harness(TORCH_OOM, None)

        # The env knob is skipped (the pin would win over it); the budget knob
        # still changes, so the retry is not a silent no-op.
        assert handler.launches == 2
        assert "pins --max-num-seqs" in caplog.text
        assert handler.launch_envs[1]["MAX_NUM_SEQS"] is None
        assert handler.launch_envs[1]["MAX_NUM_BATCHED_TOKENS"] == "8192"

    def test_fully_pinned_allocations_send_the_first_retry_straight_to_eager(
        self, harness, monkeypatch
    ):
        monkeypatch.setenv("VLLM_EXTRA_ARGS", "--max-num-batched-tokens 4096 --max-num-seqs 16")
        monkeypatch.setenv("MAX_NUM_SEQS", "16")

        handler = harness(TORCH_OOM, None)

        assert handler.launches == 2
        assert handler.launch_envs[1]["ENFORCE_EAGER"] == "true"

    def test_a_no_enforce_eager_pin_skips_the_eager_stage_with_a_warning(
        self, harness, monkeypatch, caplog
    ):
        monkeypatch.setenv("VLLM_EXTRA_ARGS", "--no-enforce-eager")

        with caplog.at_level(logging.WARNING):
            handler = harness(TORCH_OOM, TORCH_OOM, TORCH_OOM)

        # The batch stage ran once; with eager pinned off, the second OOM had
        # nothing left to change and is answered instead of relaunched.
        assert handler.launches == 2
        assert "pins --no-enforce-eager" in caplog.text
        assert "ENFORCE_EAGER" not in handler.startup_error
        assert "ran out of GPU memory" in handler.startup_error

    def test_an_extra_args_enforce_eager_counts_as_eager_already_on(self, harness, monkeypatch):
        monkeypatch.setenv("VLLM_EXTRA_ARGS", "--enforce-eager")

        handler = harness(TORCH_OOM, TORCH_OOM, TORCH_OOM)

        assert handler.launches == 2  # batch stage only; eager already in force
        assert "ENFORCE_EAGER" not in handler.startup_error


class TestSettleLog:
    def test_settling_after_the_batch_stage_reports_graphs_enabled(self, harness, caplog):
        with caplog.at_level(logging.INFO):
            harness(TORCH_OOM, None)

        assert "settled after OOM recovery" in caplog.text
        assert "CUDA graphs enabled" in caplog.text

    def test_settling_after_the_eager_stage_reports_graphs_disabled(self, harness, caplog):
        with caplog.at_level(logging.INFO):
            harness(TORCH_OOM, TORCH_OOM, None)

        assert "CUDA graphs disabled (ENFORCE_EAGER=true)" in caplog.text


class TestOtherFailures:
    def test_non_memory_failures_do_not_take_the_oom_relaunches(self, harness):
        handler = harness(REVISION_NOT_FOUND, GATED)

        # The revision relaunch fires once; the OOM plan stays untouched, so
        # the second failure is answered, not re-launched.
        assert handler.launches == 2
        assert "gated or private" in handler.startup_error
        assert all(
            env["MAX_NUM_BATCHED_TOKENS"] is None and env["MAX_NUM_SEQS"] is None
            for env in handler.launch_envs
        )

    def test_revision_and_each_oom_stage_get_their_own_retry(self, harness):
        handler = harness(REVISION_NOT_FOUND, TORCH_OOM, TORCH_OOM, None)

        assert handler.launches == 4
        assert handler.startup_error is None
        assert handler.launch_envs[-1]["ENFORCE_EAGER"] == "true"
