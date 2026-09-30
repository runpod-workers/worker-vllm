"""Credentials never reach the `vllm serve` command line or the launch log."""

import logging
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main  # noqa: E402
from args_builder import REDACTED, build_vllm_args, is_secret_flag, redact_argv  # noqa: E402

# Obviously fake; the tests only check that this exact string never escapes.
FAKE_TOKEN = "hf_FAKEtestTOKENvalue0000000000000000"


class TestSecretFlags:
    def test_credential_flags_are_secret(self):
        for flag in ("--hf-token", "--api-key", "--some-secret", "--db-password"):
            assert is_secret_flag(flag), flag

    def test_look_alike_flags_are_not_secret(self):
        # Substring matches would hide values operators need when debugging.
        for flag in ("--tokenizer", "--max-num-batched-tokens", "--ssl-keyfile",
                     "--tokenizer-mode", "--dbo-decode-token-threshold", "--model"):
            assert not is_secret_flag(flag), flag


class TestRedactArgv:
    def test_space_separated_value_masked(self):
        argv = ["vllm", "serve", "--hf-token", FAKE_TOKEN, "--model", "org/m"]
        assert redact_argv(argv) == ["vllm", "serve", "--hf-token", REDACTED, "--model", "org/m"]

    def test_equals_form_masked(self):
        assert redact_argv(["--api-key=sk-123", "--model=org/m"]) == [f"--api-key={REDACTED}", "--model=org/m"]

    def test_input_not_mutated(self):
        argv = ["--hf-token", FAKE_TOKEN]
        redact_argv(argv)
        assert argv == ["--hf-token", FAKE_TOKEN]

    def test_non_secret_argv_unchanged(self):
        argv = ["--tokenizer", "org/tok", "--max-num-batched-tokens", "8192"]
        assert redact_argv(argv) == argv


class TestHfTokenStaysInEnv:
    def test_hf_token_env_is_not_turned_into_a_flag(self):
        args = build_vllm_args({"HF_TOKEN": FAKE_TOKEN, "MODEL_NAME": "org/m"})
        assert "--hf-token" not in args
        assert FAKE_TOKEN not in args

    def test_start_vllm_keeps_token_out_of_argv_and_log(self, monkeypatch, caplog):
        monkeypatch.setenv("HF_TOKEN", FAKE_TOKEN)
        monkeypatch.setenv("MODEL_NAME", "org/m")
        # Even if someone passes the flag explicitly, the log must mask it.
        monkeypatch.setenv("VLLM_EXTRA_ARGS", f"--api-key=sk-extra --hf-token {FAKE_TOKEN}")
        with mock.patch.object(main.subprocess, "Popen") as popen, \
                mock.patch.object(main.threading, "Thread"), \
                caplog.at_level(logging.INFO):
            main.start_vllm()

        launch_lines = [r.getMessage() for r in caplog.records if "Starting vLLM" in r.getMessage()]
        assert len(launch_lines) == 1
        assert FAKE_TOKEN not in launch_lines[0]
        assert "sk-extra" not in launch_lines[0]
        assert "--model org/m" in launch_lines[0]

        argv = popen.call_args.args[0]
        # HF_TOKEN reaches vLLM through the environment, not the command line.
        assert argv.count("--hf-token") == 1  # only the explicit VLLM_EXTRA_ARGS one
        assert popen.call_args.kwargs["env"]["HF_TOKEN"] == FAKE_TOKEN
