"""Static and CPU-only checks for the optional Telys example; no image or GPU."""

import json
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "examples" / "telys" / "Dockerfile"
ENTRYPOINT = ROOT / "examples" / "telys" / "entrypoint.sh"
README = ROOT / "examples" / "telys" / "README.md"
CONNECTOR_COMMIT = "49662f299b3c17b0f8c51050c8600cc8706362fd"
DEFAULT_CONFIG = (
    '{"kv_connector":"TelysKVConnector",'
    '"kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both"}'
)
STOCK_CONFIG = (
    '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",'
    '"kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec"}}'
)


def write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


@pytest.fixture
def entrypoint_env(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    packages = tmp_path / "packages"
    telys_package = packages / "telys"
    telys_package.mkdir(parents=True)
    (telys_package / "__init__.py").write_text("")
    (telys_package / "paths.py").write_text(
        """import os

def api_url():
    return "https://example.invalid"

def telys_home():
    return os.environ["TELYS_HOME"]
"""
    )
    (telys_package / "login.py").write_text(
        """import os

def register_device(*, api, api_key):
    assert api == "https://example.invalid"
    assert api_key == os.environ["TELYS_API_KEY"]
    if os.environ.get("FAKE_REGISTER_FAIL") == "1":
        raise RuntimeError("provisioning failed")
    with open(os.path.join(os.environ["CAPTURE_DIR"], "api-called"), "w") as marker:
        marker.write("registered")
    return "test-license-token"
"""
    )
    (telys_package / "installer.py").write_text(
        """import os

def install_from_host(*, token, license_path):
    assert token == os.environ["TELYS_API_KEY"]
    with open(license_path) as license_file:
        assert license_file.read() == "test-license-token"
    with open(os.path.join(os.environ["CAPTURE_DIR"], "installed"), "w") as marker:
        marker.write("installed")
"""
    )
    write_executable(
        bin_dir / "python3",
        f"""#!/bin/sh
if [ "${{1:-}}" = "-" ]; then
    exec {shlex.quote(sys.executable)} "$@"
fi
printf '%s\\n' "$@" > "$CAPTURE_DIR/python-args"
env > "$CAPTURE_DIR/python-env"
printf called > "$CAPTURE_DIR/main-called"
""",
    )
    write_executable(
        bin_dir / "telys",
        """#!/bin/sh
if [ "$#" -ne 6 ] || [ "$1" != "login" ] || [ "$2" != "--token" ] || \
    [ "$3" != "$TELYS_OAUTH_TOKEN" ] || [ "$4" != "--no-browser" ] || \
    [ "$5" != "--plan" ] || [ "$6" != "telys_developer" ]; then
    exit 64
fi
printf called > "$CAPTURE_DIR/oauth-called"
""",
    )

    def run(overrides=None, *args):
        env = {
            "CAPTURE_DIR": str(capture),
            "HOME": str(tmp_path / "home"),
            "LC_ALL": "C",
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(packages),
            "TELYS_HOME": str(tmp_path / "telys-home"),
        }
        for key, value in (overrides or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run(
            ["bash", str(ENTRYPOINT), *args],
            capture_output=True,
            env=env,
            text=True,
            timeout=10,
        )

    return run, capture, tmp_path


def captured_env(capture: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (capture / "python-env").read_text().splitlines()
    )


@pytest.mark.parametrize("enabled", [None, "false"])
def test_disabled_preserves_stock_entrypoint_and_environment(entrypoint_env, enabled):
    run, capture, _ = entrypoint_env
    result = run(
        {"TELYS_ENABLED": enabled, "KV_TRANSFER_CONFIG": STOCK_CONFIG, "SENTINEL": "kept"},
        "--worker-arg",
        "two words",
    )

    assert result.returncode == 0, result.stderr
    assert (capture / "python-args").read_text().splitlines() == [
        "/src/main.py",
        "--worker-arg",
        "two words",
    ]
    env = captured_env(capture)
    assert env["KV_TRANSFER_CONFIG"] == STOCK_CONFIG
    assert env["SENTINEL"] == "kept"
    assert "TELYS_VLLM_LOAD_ASYNC" not in env
    assert not (capture / "api-called").exists()
    assert not (capture / "oauth-called").exists()


@pytest.mark.parametrize("credential", ["TELYS_API_KEY", "TELYS_OAUTH_TOKEN"])
def test_disabled_rejects_credentials_without_starting(entrypoint_env, credential):
    run, capture, _ = entrypoint_env
    result = run({credential: "test-secret-value"})

    assert result.returncode == 64
    assert "TELYS_ENABLED=true" in result.stderr
    assert "test-secret-value" not in result.stderr + result.stdout
    assert not (capture / "main-called").exists()


def test_disabled_rejects_telys_connector(entrypoint_env):
    run, capture, _ = entrypoint_env
    result = run({"KV_TRANSFER_CONFIG": DEFAULT_CONFIG})

    assert result.returncode == 64
    assert "requires TELYS_ENABLED=true" in result.stderr
    assert not (capture / "main-called").exists()


def test_opt_in_rejects_ambiguous_boolean(entrypoint_env):
    run, capture, _ = entrypoint_env
    result = run({"TELYS_ENABLED": "1"})

    assert result.returncode == 64
    assert "exactly 'true' or 'false'" in result.stderr
    assert not (capture / "main-called").exists()


def test_enabled_supplies_connector_and_disables_async(entrypoint_env):
    run, capture, _ = entrypoint_env
    result = run({"TELYS_ENABLED": "true"})

    assert result.returncode == 0, result.stderr
    env = captured_env(capture)
    assert json.loads(env["KV_TRANSFER_CONFIG"]) == {
        "kv_connector": "TelysKVConnector",
        "kv_connector_module_path": "telys_vllm.connector",
        "kv_role": "kv_both",
    }
    assert env["TELYS_VLLM_LOAD_ASYNC"] == "0"
    assert not (capture / "api-called").exists()
    assert not (capture / "oauth-called").exists()


def test_enabled_preserves_valid_custom_config(entrypoint_env):
    run, capture, _ = entrypoint_env
    config = json.dumps(
        {
            "kv_role": "kv_both",
            "kv_connector_module_path": "telys_vllm.connector",
            "kv_connector": "TelysKVConnector",
            "kv_connector_extra_config": {
                "telys_namespace": "test",
                "telys_load_async": False,
            },
        }
    )
    result = run({"TELYS_ENABLED": "true", "KV_TRANSFER_CONFIG": config})

    assert result.returncode == 0, result.stderr
    env = captured_env(capture)
    assert env["KV_TRANSFER_CONFIG"] == config
    assert env["TELYS_VLLM_LOAD_ASYNC"] == "0"


@pytest.mark.parametrize(
    "overrides",
    [
        {"KV_TRANSFER_CONFIG": "not-json"},
        {"KV_TRANSFER_CONFIG": STOCK_CONFIG},
        {"KV_TRANSFER_CONFIG": '{"kv_connector":"TelysKVConnector"}'},
        {
            "KV_TRANSFER_CONFIG": json.dumps(
                {
                    "kv_connector": "TelysKVConnector",
                    "kv_connector_module_path": "telys_vllm.connector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"telys_load_async": True},
                }
            )
        },
        {"TELYS_VLLM_LOAD_ASYNC": "true"},
    ],
)
def test_enabled_rejects_invalid_connector_or_async_config(entrypoint_env, overrides):
    run, capture, _ = entrypoint_env
    result = run({"TELYS_ENABLED": "true", **overrides})

    assert result.returncode != 0
    assert not (capture / "main-called").exists()
    assert not (capture / "api-called").exists()
    assert not (capture / "oauth-called").exists()


def test_api_key_provisioning_takes_precedence_and_unsets_credentials(entrypoint_env):
    run, capture, tmp_path = entrypoint_env
    result = run(
        {
            "TELYS_ENABLED": "true",
            "TELYS_API_KEY": "test-api-key",
            "TELYS_OAUTH_TOKEN": "test-oauth-token",
        }
    )

    assert result.returncode == 0, result.stderr
    assert (capture / "api-called").exists()
    assert (capture / "installed").exists()
    assert not (capture / "oauth-called").exists()
    env = captured_env(capture)
    assert "TELYS_API_KEY" not in env
    assert "TELYS_OAUTH_TOKEN" not in env
    license_path = tmp_path / "telys-home" / "login_license.jwt"
    assert license_path.read_text() == "test-license-token"
    assert stat.S_IMODE(license_path.stat().st_mode) == 0o600


def test_oauth_provisioning_and_secret_cleanup(entrypoint_env):
    run, capture, _ = entrypoint_env
    result = run({"TELYS_ENABLED": "true", "TELYS_OAUTH_TOKEN": "test-oauth-token"})

    assert result.returncode == 0, result.stderr
    assert (capture / "oauth-called").exists()
    assert not (capture / "api-called").exists()
    assert "TELYS_OAUTH_TOKEN" not in captured_env(capture)


def test_provisioning_failure_does_not_fall_back_or_start_worker(entrypoint_env):
    run, capture, _ = entrypoint_env
    result = run(
        {
            "TELYS_ENABLED": "true",
            "TELYS_API_KEY": "test-api-key",
            "TELYS_OAUTH_TOKEN": "test-oauth-token",
            "FAKE_REGISTER_FAIL": "1",
        }
    )

    assert result.returncode != 0
    assert "provisioning failed" in result.stderr
    assert "test-api-key" not in result.stderr + result.stdout
    assert "test-oauth-token" not in result.stderr + result.stdout
    assert not (capture / "oauth-called").exists()
    assert not (capture / "main-called").exists()


def test_dockerfile_pins_final_connector_commit():
    dockerfile = DOCKERFILE.read_text()

    assert "FROM runpod/worker-v1-vllm:v2.30.3" in dockerfile
    assert f"ARG TELYS_VLLM_COMMIT={CONNECTOR_COMMIT}" in dockerfile
    assert 'r"[0-9a-f]{40}"' in dockerfile
    assert (
        "telys-vllm @ git+https://github.com/thyn-ai/telys-vllm.git@"
        "${TELYS_VLLM_COMMIT}"
    ) in dockerfile
    assert "apt-get install -y --no-install-recommends git" in dockerfile
    assert "--no-deps" in dockerfile
    assert 'vllm_version == "0.32.0"' in dockerfile
    assert 'connector_version == "0.1.2"' in dockerfile
    assert re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", dockerfile) == [
        CONNECTOR_COMMIT
    ]
    assert "COPY --chmod=0755 examples/telys/entrypoint.sh" in dockerfile


def test_dockerfile_commit_validator_accepts_only_full_lowercase_shas():
    dockerfile = DOCKERFILE.read_text()
    validator = dockerfile.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]

    for invalid in ["not-a-commit", "main", "v0.1.2", "a" * 39, "A" * 40]:
        result = subprocess.run(
            [sys.executable, "-", invalid],
            capture_output=True,
            input=validator,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0, invalid
        assert "full 40-character lowercase SHA" in result.stderr

    valid = subprocess.run(
        [sys.executable, "-", CONNECTOR_COMMIT],
        capture_output=True,
        input=validator,
        text=True,
        timeout=10,
    )
    assert valid.returncode == 0, valid.stderr


def test_readme_documents_pinned_commit_and_remote_acceptance_boundary():
    readme = README.read_text()

    assert CONNECTOR_COMMIT in readme
    assert "placeholder" not in readme.lower()
    assert "intentionally not buildable" not in readme
    assert "TELYS_ENABLED=true" in readme
    assert "not remote acceptance evidence" in readme
    assert "TELYS_VLLM_LOAD_ASYNC=0" in readme
