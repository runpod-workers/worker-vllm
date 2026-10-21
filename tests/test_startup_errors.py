"""Which vLLM startup failures are worth answering, and which are worth a restart."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import startup_errors  # noqa: E402
from startup_errors import classify  # noqa: E402

TORCH_OOM = """
(EngineCore_DP0 pid=245) ERROR 08-19 04:02:11 [core.py:1346] EngineCore failed to start.
(EngineCore_DP0 pid=245) ERROR 08-19 04:02:11 [core.py:1346] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 108.00 MiB.
GPU 0 has a total capacity of 19.57 GiB of which 105.81 MiB is free.
"""
NO_KV_MEMORY = (
    "ValueError: No available memory for the cache blocks. Try increasing "
    "`gpu_memory_utilization` when initializing the engine."
)
KV_TOO_SMALL = (
    "ValueError: To serve at least one request with the model's max seq len (131072), "
    "(16.00 GiB KV cache is needed, which is larger than the available KV cache memory "
    "(4.21 GiB). Based on the available memory, the estimated maximum model length is 34480. "
    "Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine."
)
KV_TOO_SMALL_OLD = (
    "ValueError: The model's max seq len (131072) is larger than the maximum number of "
    "tokens that can be stored in KV cache (34480)."
)
MAX_LEN_EXCEEDS_MODEL = (
    "ValueError: User-specified max_model_len (65536) is greater than the derived "
    "max_model_len (max_position_embeddings=32768 or model_max_length=None in model's config.json)."
)
BAD_ARGS = "vllm serve: error: unrecognized arguments: --foo-bar 3"
BAD_VALUE = "vllm serve: error: argument --max-model-len: invalid int value: 'lots'"
GATED = (
    "huggingface_hub.errors.GatedRepoError: 401 Client Error. Cannot access gated repo "
    "for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json."
)
NOT_FOUND = (
    "huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. Repository Not Found "
    "for url https://huggingface.co/org/nope/resolve/main/config.json."
)
UNSUPPORTED = "ValueError: Model architectures ['FooForCausalLM'] are not supported for now."
# vLLM v0.30 worker log for IFM/K2-Horizon-7B without TRUST_REMOTE_CODE (the worker crash-looped on it).
NEEDS_REMOTE_CODE = (
    "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig\n"
    "  Value error, The repository IFM/K2-Horizon-7B contains custom code which must be executed to "
    "correctly load the model. You can inspect the repository content at https://hf.co/IFM/K2-Horizon-7B .\n"
    "Please pass the argument `trust_remote_code=True` to allow custom code to be run. [type=value_error]"
)
NO_SPACE = "OSError: [Errno 28] No space left on device"


class TestOutOfMemory:
    def test_torch_oom_names_the_model_and_the_card(self):
        message = classify(TORCH_OOM, model="Qwen/Qwen3-30B-A3B")

        assert message.startswith("Qwen/Qwen3-30B-A3B ran out of GPU memory")
        assert "19.57 GiB" in message

    def test_advice_covers_the_knobs_the_worker_exposes(self):
        message = classify(TORCH_OOM)

        for knob in ("MAX_MODEL_LEN", "ENFORCE_EAGER", "GPU_MEMORY_UTILIZATION", "TENSOR_PARALLEL_SIZE"):
            assert knob in message

    def test_no_kv_cache_memory_is_out_of_memory(self):
        assert "ran out of GPU memory" in classify(NO_KV_MEMORY)

    def test_kv_cache_too_small_passes_on_the_estimated_context(self):
        message = classify(KV_TOO_SMALL)

        assert "ran out of GPU memory" in message
        assert "34480 tokens" in message

    def test_older_kv_cache_wording_is_recognised(self):
        assert classify(KV_TOO_SMALL_OLD) is not None

    def test_oom_wins_over_surrounding_noise(self):
        assert "GPU memory" in classify(f"Connection reset by peer\n{TORCH_OOM}\nEngineCore died")


class TestConfiguration:
    def test_max_model_len_above_the_model_limit_names_the_value(self):
        message = classify(MAX_LEN_EXCEEDS_MODEL, model="org/model")

        assert "MAX_MODEL_LEN=65536" in message
        assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" in message

    @pytest.mark.parametrize("output", [BAD_ARGS, BAD_VALUE])
    def test_argparse_rejections_quote_the_reason(self, output):
        message = classify(output)

        assert message.startswith("vLLM rejected its command line")
        assert output.split("error: ", 1)[1] in message
        assert "VLLM_EXTRA_ARGS" in message


class TestModelAccess:
    def test_gated_repo_points_at_hf_token(self):
        assert "HF_TOKEN" in classify(GATED, model="meta-llama/Llama-3.1-8B-Instruct")

    def test_missing_repo_points_at_model_name(self):
        assert "MODEL_NAME" in classify(NOT_FOUND)

    def test_unsupported_architecture(self):
        assert "architecture" in classify(UNSUPPORTED)

    def test_custom_code_points_at_trust_remote_code(self):
        message = classify(NEEDS_REMOTE_CODE, model="IFM/K2-Horizon-7B")

        assert message.startswith("IFM/K2-Horizon-7B ships its own modeling code")
        assert "TRUST_REMOTE_CODE=true" in message

    def test_custom_code_wins_over_unsupported_architecture(self):
        # With remote code allowed, vLLM's Transformers backend can serve an
        # architecture it has no native class for, so don't say "unsupported".
        assert "TRUST_REMOTE_CODE=true" in classify(UNSUPPORTED + "\n" + NEEDS_REMOTE_CODE)


def test_out_of_disk_points_at_container_disk():
    assert "container disk" in classify(NO_SPACE)


def test_out_of_disk_names_the_volume_caches():
    # With a network volume, the HF cache and vLLM's compile cache live on it.
    assert "vllm-cache" in classify(NO_SPACE)
    assert "network volume" in classify(NO_SPACE)


@pytest.mark.parametrize("output", [NO_SPACE, "OSError: [Errno 122] Disk quota exceeded"])
def test_out_of_disk_predicate_covers_quota(output):
    assert startup_errors.out_of_disk(output)


def test_falls_back_to_a_generic_subject():
    assert classify(NO_SPACE).startswith("The model")


class TestUnknownFailures:
    @pytest.mark.parametrize(
        "output",
        [
            "",
            "Connection reset by peer while downloading",
            "RuntimeError: something nobody has seen before",
            "vLLM serve exited during startup with code 1",
            # Only vLLM's own argparse errors count, not any line with "error:".
            "INFO 08-19 [launcher.py] error: none",
        ],
    )
    def test_are_left_for_the_platform_to_retry(self, output):
        # Answering these forever would turn a flaky download into a dead
        # endpoint; a restart is the right response.
        assert classify(output) is None


REVISION_NOT_FOUND = (
    "huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. Revision Not Found "
    "for url https://huggingface.co/org/model/resolve/abc123def/config.json. "
    "Invalid rev id: abc123def"
)


class TestRevisionNotFound:
    def test_names_the_fix_and_the_force_push_cause(self):
        message = classify(REVISION_NOT_FOUND, model="org/model")

        assert message.startswith("The Hugging Face revision requested for org/model")
        assert "MODEL_REVISION" in message
        assert "force-pushed" in message

    def test_wins_over_the_generic_404_match(self):
        # The transport error underneath a missing revision is also a
        # "404 Client Error", which the repository-not-found pattern matches.
        # A vanished revision must not be reported as a missing repository.
        message = classify(REVISION_NOT_FOUND, model="org/model")

        assert "was not found on Hugging Face" not in message

    def test_predicate_drives_the_relaunch(self):
        from startup_errors import revision_not_found

        assert revision_not_found(REVISION_NOT_FOUND)
        assert not revision_not_found(NOT_FOUND)
        assert not revision_not_found(TORCH_OOM)


class TestOomAdviceForV029:
    def test_mentions_the_batched_tokens_and_fp8_kv_knobs(self):
        message = classify(TORCH_OOM)

        assert "MAX_NUM_BATCHED_TOKENS=8192" in message
        assert "KV_CACHE_DTYPE" in message
        assert "FlashInfer" in message


class TestOomAfterAutoRetry:
    def test_says_the_reduced_footprint_relaunch_already_ran(self):
        message = classify(TORCH_OOM, model="org/model", oom_retried=True)

        assert message.startswith("org/model ran out of GPU memory")
        assert "already retried once" in message
        assert "ENFORCE_EAGER" in message
        # The knobs the retry applied are no longer offered as the fix.
        assert "MAX_NUM_BATCHED_TOKENS=8192 (the default doubled" not in message
        # The fixes that remain are real capacity changes.
        assert "TENSOR_PARALLEL_SIZE" in message
        assert "quantized" in message

    def test_still_carries_the_card_and_the_kV_estimate(self):
        assert "19.57 GiB" in classify(TORCH_OOM, oom_retried=True)
        assert "34480 tokens" in classify(KV_TOO_SMALL, oom_retried=True)


class TestMemoryShortfallPredicate:
    def test_all_three_memory_wording_forms_match(self):
        from startup_errors import memory_shortfall

        assert memory_shortfall(TORCH_OOM)
        assert memory_shortfall(NO_KV_MEMORY)
        assert memory_shortfall(KV_TOO_SMALL)

    def test_non_memory_failures_do_not_match(self):
        from startup_errors import memory_shortfall

        assert not memory_shortfall(REVISION_NOT_FOUND)
        assert not memory_shortfall(GATED)
        assert not memory_shortfall("Connection reset by peer while downloading")


# Tails of real worker logs (RecursiveSI/flint on an H200, vLLM 0.32.0rc1 day-0
# image) in the order vLLM prints them, trimmed to the frames that matter:
# EngineCore logs the root cause with a prefix, multiprocessing prints an
# unprefixed "Process EngineCore:" and the raw traceback, and the API server's
# own traceback ends in a generic RuntimeError.
_EC = "(EngineCore pid=1079) ERROR 10-21 20:16:36 [core.py:1543] "
API_SERVER_TAIL = """(APIServer pid=503) INFO 10-21 20:16:36 [utils.py:628] [shutdown] Process manager: send sigterm to process EngineCore
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/utils.py", line 1468, in wait_for_engine_startup
    raise RuntimeError(
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def _flint_log(frames, exception):
    traceback = ["Traceback (most recent call last):", *frames, exception]
    logged = "".join(_EC + line + "\n" for line in ["EngineCore failed to start.", *traceback])
    raw = "Process EngineCore:\n" + "".join(line + "\n" for line in traceback)
    return logged + raw + API_SERVER_TAIL


FLINT_ATTN = "AssertionError: Flint requires FlashInfer attention"
FLINT_MOE = (
    "RuntimeError: Flint BF16 decode MoE requires the FlashInfer TRTLLM BF16 MoE backend "
    "without padding (got UnquantizedMoeBackend.TRITON)"
)
FLINT_NEEDS_FLASHINFER = _flint_log([
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 1515, in run_engine_core',
    "    engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)",
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/models/flint/model.py", line 479, in __init__',
    "    self.self_attn = FlintAttention(",
    "                     ^^^^^^^^^^^^^^^",
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/models/flint/model.py", line 316, in __init__',
    '    assert isinstance(impl, FlashInferImpl), "Flint requires FlashInfer attention"',
    "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
], FLINT_ATTN)
FLINT_MOE_NEEDS_BLACKWELL = _flint_log([
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py", line 294, in kernel_warmup',
    "    mixed_batch_attention_warmup(worker)",
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/models/flint/ops/moe.py", line 806, in forward',
    "    self._check(routed_experts)",
    '  File "/usr/local/lib/python3.12/dist-packages/vllm/models/flint/ops/moe.py", line 792, in _check',
    "    raise RuntimeError(",
], FLINT_MOE)


def _traceback(innermost: str, exception: str, function: str = "__init__") -> str:
    return (
        "Traceback (most recent call last):\n"
        '  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 157, in __init__\n'
        "    kv_cache_config = self._initialize_kv_caches(vllm_config)\n"
        f'  File "/usr/local/lib/python3.12/dist-packages/{innermost}", line 42, in {function}\n'
        "    raise\n"
        f"{exception}\n"
    )


class TestModelCodeErrors:
    @pytest.mark.parametrize("output,cause", [(FLINT_NEEDS_FLASHINFER, FLINT_ATTN), (FLINT_MOE_NEEDS_BLACKWELL, FLINT_MOE)])
    def test_quote_exactly_the_root_cause(self, output, cause):
        message = classify(output, model="RecursiveSI/flint")

        # Exactly the exception, not the next log line ("Process EngineCore:") or the wrapper.
        assert f"during startup: {cause}. The model's code requires" in message
        assert "Process EngineCore" not in message and "Engine core initialization failed" not in message
        assert "ATTENTION_BACKEND" in message and "several GPU types" in message

    @pytest.mark.parametrize(
        "harmless",
        [
            # torch on a Runpod host with partial NVML support (seen in this image's logs)
            "UserWarning: nvmlDeviceGetComputeRunningProcesses failed; omitting per-process memory info "
            "from CUDA out-of-memory messages. This is expected on platforms with partial NVML support.",
            "FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.",
            # a download retry that then succeeded
            "WARNING huggingface_hub: HTTPSConnectionPool(host='cas-bridge.xethub.hf.co', port=443): "
            "Read timed out. (read timeout=10) Retrying in 1s [Retry 1/5].",
            # c10d's connect retry with tensor parallelism
            "[W1021 20:16:30 socket.cpp:767] [c10d] The client socket has failed to connect to [::1]:29500 "
            "(errno: 111 - Connection refused).",
            "(EngineCore pid=1079) INFO 10-21 20:16:34 [parallel_state.py:1911] world_size=1 rank=0 backend=nccl",
        ],
    )
    def test_harmless_lines_elsewhere_in_the_launch_do_not_block_it(self, harmless):
        message = classify(harmless + "\n" + FLINT_NEEDS_FLASHINFER, model="RecursiveSI/flint")

        assert message and FLINT_ATTN in message

    @pytest.mark.parametrize(
        "exception",
        [
            "NotImplementedError: FooAttention only supports compute capability 9.0 and above",
            "ValueError: Sliding window attention is not supported on this platform",
            "RuntimeError: FooMoE requires the CUTLASS backend",
        ],
    )
    def test_stated_requirements_in_classic_model_code(self, exception):
        assert exception in classify(_traceback("vllm/model_executor/models/foo.py", exception))

    def test_multiline_requirement_within_one_record(self):
        output = _traceback("vllm/models/foo/model.py", "RuntimeError: Foo decode kernel\nrequires sm100 (got sm90)")

        assert "RuntimeError: Foo decode kernel requires sm100 (got sm90)" in classify(output)

    def test_raw_output_with_vllm_030_short_prefix(self):
        lines = _traceback("vllm/models/foo/model.py", "RuntimeError: FooMoE requires the CUTLASS backend").splitlines()
        output = "".join(f"(APIServer pid=496) {line}\n" for line in lines)

        assert "RuntimeError: FooMoE requires the CUTLASS backend" in classify(output)

    @pytest.mark.parametrize(
        "innermost,function,exception",
        [
            # Where an error is raised says nothing about permanence: model code can be
            # the innermost Python frame of a native failure during weight loading.
            ("vllm/model_executor/models/llama.py", "load_weights", "RuntimeError: CUDA driver error: initialization error"),
            ("vllm/model_executor/models/llama.py", "load_weights",
             "RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at \"/pytorch/c10/cuda/CUDACachingAllocator.cpp\":1131"),
            ("vllm/model_executor/models/llama.py", "load_weights", "RuntimeError: Failed to load weights: Connection reset by peer"),
            ("vllm/model_executor/models/llama.py", "load_weights",
             "RuntimeError: Download requires the shard, but the request timed out"),
            ("vllm/models/flint/ops/moe.py", "forward", "RuntimeError: CUDA error: an illegal memory access was encountered"),
            # No stated requirement: could be anything.
            ("vllm/model_executor/models/foo.py", "__init__", "AssertionError"),
            ("vllm/model_executor/models/foo.py", "forward", "ValueError: Error during index put operation"),
            ("safetensors/torch.py", "load_file", "ValueError: truncated file"),
            # Autograd misuse is a bug, not a configuration requirement.
            ("vllm/model_executor/models/foo.py", "forward",
             "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation."),
            # A stated requirement, but not raised by model code.
            ("vllm/v1/engine/core.py", "__init__", "RuntimeError: this step requires a restart"),
        ],
    )
    def test_everything_else_still_restarts(self, innermost, function, exception):
        assert classify(_traceback(innermost, exception, function)) is None

    def test_a_wrapped_cuda_cause_still_restarts(self):
        # Synthetic wrapper text: vLLM's _merge_multimodal_embeddings re-raises a failed
        # index_put as ValueError; "requires" is added so only the CUDA cause can veto it.
        cause = _traceback("vllm/model_executor/models/utils.py", "RuntimeError: CUDA error: too many resources requested for launch")
        wrapper = _traceback("vllm/model_executor/models/utils.py", "ValueError: Error during index put operation requires retry")
        output = cause + "\nThe above exception was the direct cause of the following exception:\n\n" + wrapper

        assert classify(output) is None

    def test_a_cuda_error_on_a_continuation_line_still_restarts(self):
        output = _traceback("vllm/models/foo/model.py", "RuntimeError: Foo requires a warmup pass\nCUDA error: unspecified launch failure")

        assert classify(output) is None

    def test_the_next_log_record_is_not_part_of_the_message(self):
        # A bare model-code assertion, then an INFO record that happens to say "requires".
        logged = "".join(_EC + line + "\n" for line in _traceback("vllm/models/foo/model.py", "AssertionError").splitlines())
        output = logged + "(EngineCore pid=1079) INFO 10-21 20:16:37 [deepep.py:88] DeepEP v2 requires NCCL >= 2.30.4\n"

        assert classify(output) is None

    def test_a_handled_warning_traceback_does_not_count(self):
        warned = _traceback("vllm/models/foo/model.py", "RuntimeError: FooMoE requires the CUTLASS backend")
        output = "".join(f"(EngineCore pid=1079) WARNING 10-21 20:16:36 [moe.py:40] {line}\n" for line in warned.splitlines())

        assert classify(output) is None

    def test_an_engine_killed_by_a_signal_is_not_blamed_on_model_code(self):
        output = FLINT_NEEDS_FLASHINFER.replace("Failed core proc(s): {}", "Failed core proc(s): {'EngineCore': -9}")

        assert classify(output) is None

    def test_a_stale_model_frame_does_not_attach_to_a_later_error(self):
        output = (
            _traceback("vllm/models/foo/model.py", "safetensors_rust.SafetensorError: Error while deserializing header")
            + "RuntimeError: FooMoE requires the CUTLASS backend\n"
        )

        assert classify(output) is None

    def test_oom_raised_in_model_code_keeps_the_memory_advice(self):
        output = _traceback(
            "vllm/model_executor/models/llama.py",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.",
        )

        assert "ran out of GPU memory" in classify(output)
