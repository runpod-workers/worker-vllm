"""Which vLLM startup failures are worth answering, and which are worth a restart."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def test_out_of_disk_points_at_container_disk():
    assert "container disk" in classify(NO_SPACE)


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
