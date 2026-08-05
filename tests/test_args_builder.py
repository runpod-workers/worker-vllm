"""Unit tests for env-var -> `vllm serve` CLI translation (src/args_builder.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from args_builder import (  # noqa: E402
    BOOL_FLAGS,
    ENV_ALIASES,
    PAIRED_BOOL_FLAGS,
    REMOVED_FLAGS,
    VALUE_FLAGS,
    build_vllm_args,
    flag_to_env_name,
)


def argv_to_dict(argv):
    """Fold an argv list into {flag: value|None} (last occurrence wins, like argparse)."""
    out = {}
    for i, token in enumerate(argv):
        if token.startswith("--"):
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if next_token is None or next_token.startswith("--"):
                out[token] = None
            else:
                out[token] = next_token
    return out


class TestGenericMapping:
    def test_value_flag_from_env(self):
        args = build_vllm_args({"MAX_MODEL_LEN": "4096", "GPU_MEMORY_UTILIZATION": "0.9"})
        assert ["--max-model-len", "4096"] == args[args.index("--max-model-len"):][:2]
        assert ["--gpu-memory-utilization", "0.9"] == args[args.index("--gpu-memory-utilization"):][:2]

    def test_paired_bool_flag(self):
        # vLLM booleans are `--flag/--no-flag` pairs that take no value; a
        # falsy env value must emit the `--no-` form, never `--flag false`.
        args = build_vllm_args({"ENFORCE_EAGER": "true", "TRUST_REMOTE_CODE": "false"})
        assert "--enforce-eager" in args
        assert "true" not in args
        assert "--trust-remote-code" not in args
        assert "--no-trust-remote-code" in args

    def test_paired_bool_flag_invalid_value(self):
        with pytest.raises(ValueError, match="invalid boolean value"):
            build_vllm_args({"ENFORCE_EAGER": "maybe"})

    def test_store_true_flag(self):
        flag = next(iter(BOOL_FLAGS))
        env_name = flag_to_env_name(flag)
        assert flag in build_vllm_args({env_name: "true"})
        assert flag not in build_vllm_args({env_name: "false"})

    def test_json_values_pass_verbatim(self):
        blob = '{"model": "org/draft", "num_speculative_tokens": 3}'
        args = argv_to_dict(build_vllm_args({"SPECULATIVE_CONFIG": blob}))
        assert args["--speculative-config"] == blob

    def test_unknown_env_vars_ignored(self):
        args = build_vllm_args({"NOT_A_VLLM_FLAG": "x", "HOME": "/root", "RUNPOD_POD_ID": "abc"})
        assert args == []

    def test_empty_values_skipped(self):
        assert "--max-model-len" not in build_vllm_args({"MAX_MODEL_LEN": "  "})

    def test_reserved_flags_not_scanned(self):
        # HOST/PORT/API_KEY/MODEL are owned by the wrapper (or pinned in main.py).
        args = build_vllm_args({"HOST": "0.0.0.0", "PORT": "9000", "API_KEY": "secret", "MODEL": "org/m"})
        assert "--host" not in args
        assert "--port" not in args
        assert "--api-key" not in args
        assert "--model" not in args


class TestAliases:
    def test_model_name_alias(self):
        assert ["--model", "org/model"] == build_vllm_args({"MODEL_NAME": "org/model"})

    def test_aliases_map_to_real_flags(self):
        for env_name, flag in ENV_ALIASES.items():
            assert env_name not in {flag_to_env_name(flag)}, f"{env_name} should not be needed as alias"
            args = argv_to_dict(build_vllm_args({env_name: "v"}))
            assert flag in args, f"{env_name} should produce {flag}"

    def test_alias_wins_over_generic_scan(self):
        # SERVED_MODEL_NAME scans generically; the override alias must win.
        args = argv_to_dict(
            build_vllm_args({"SERVED_MODEL_NAME": "base", "OPENAI_SERVED_MODEL_NAME_OVERRIDE": "override"})
        )
        assert args["--served-model-name"] == "override"


class TestExtraArgs:
    def test_extra_args_appended_last(self):
        args = build_vllm_args({"MAX_MODEL_LEN": "4096", "VLLM_EXTRA_ARGS": "--max-model-len 2048 --new-flag x"})
        assert args[-4:] == ["--max-model-len", "2048", "--new-flag", "x"]
        # argparse last-occurrence-wins: the extra args entry must be the effective one
        assert argv_to_dict(args)["--max-model-len"] == "2048"

    def test_extra_args_quoting(self):
        args = build_vllm_args({"VLLM_EXTRA_ARGS": "--override-generation-config '{\"max_new_tokens\": 512}'"})
        assert args == ["--override-generation-config", '{"max_new_tokens": 512}']


class TestRemovedFlags:
    def test_removed_flags_warn_and_emit_nothing(self, caplog):
        # Flags deleted upstream must not crash startup; they warn and emit nothing.
        args = build_vllm_args({"SWAP_SPACE": "4", "ROPE_SCALING": '{"factor": 2.0}', "TASK": "generate"})
        assert args == []
        assert "SWAP_SPACE" in caplog.text
        assert "ROPE_SCALING" in caplog.text
        assert "TASK" in caplog.text


class TestZeroMeansUnset:
    def test_zero_max_num_batched_tokens_is_omitted(self):
        # 0 is rejected by vLLM's SchedulerConfig (ge=1) but templates carry it;
        # omitting the flag lets vLLM size the batch for the GPU.
        assert "--max-num-batched-tokens" not in build_vllm_args({"MAX_NUM_BATCHED_TOKENS": "0"})

    def test_nonzero_value_passes_through(self):
        args = argv_to_dict(build_vllm_args({"MAX_NUM_BATCHED_TOKENS": "8192"}))
        assert args["--max-num-batched-tokens"] == "8192"

    def test_zero_not_dropped_for_other_flags(self):
        # SEED=0 et al. are legitimate values; only ZERO_MEANS_UNSET flags drop 0.
        args = argv_to_dict(build_vllm_args({"SEED": "0"}))
        assert args["--seed"] == "0"


class TestAllowlistSanity:
    def test_expected_flags_present(self):
        # Regression: these env vars back the e2e configs and docs.
        expected_value = [
            "--max-model-len", "--tensor-parallel-size", "--gpu-memory-utilization",
            "--dtype", "--quantization", "--kv-cache-dtype", "--speculative-config",
            "--tool-call-parser", "--reasoning-parser",
        ]
        for flag in expected_value:
            assert flag in VALUE_FLAGS, flag
        expected_bool = ["--enforce-eager", "--enable-prefix-caching", "--enable-lora", "--trust-remote-code"]
        for flag in expected_bool:
            assert flag in PAIRED_BOOL_FLAGS, flag

    def test_flag_categories_do_not_overlap(self):
        assert not (VALUE_FLAGS & PAIRED_BOOL_FLAGS)
        assert not (VALUE_FLAGS & BOOL_FLAGS)
        assert not (PAIRED_BOOL_FLAGS & BOOL_FLAGS)
        assert not (set(REMOVED_FLAGS) & (VALUE_FLAGS | PAIRED_BOOL_FLAGS | BOOL_FLAGS))

    def test_every_flag_has_valid_env_name(self):
        all_flags = VALUE_FLAGS | BOOL_FLAGS | PAIRED_BOOL_FLAGS | set(REMOVED_FLAGS)
        for flag in all_flags:
            assert flag.startswith("--")
            name = flag_to_env_name(flag)
            assert name == name.upper() and not name.startswith("_")
