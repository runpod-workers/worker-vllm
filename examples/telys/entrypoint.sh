#!/usr/bin/env bash
set -euo pipefail
set +x

readonly DEFAULT_KV_TRANSFER_CONFIG='{"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both"}'

fail() {
    printf 'telys-vllm entrypoint: %s\n' "$1" >&2
    exit 64
}

python3 - <<'PY'
import os
import shlex
import sys

extra_args = shlex.split(os.environ.get("VLLM_EXTRA_ARGS", ""))
if any(
    arg == "--kv-transfer-config" or arg.startswith("--kv-transfer-config=")
    for arg in extra_args
):
    print(
        "telys-vllm entrypoint: --kv-transfer-config is not allowed in "
        "VLLM_EXTRA_ARGS; set KV_TRANSFER_CONFIG instead",
        file=sys.stderr,
    )
    raise SystemExit(64)
PY

case "${TELYS_ENABLED:-false}" in
    false)
        if [[ -n "${TELYS_API_KEY:-}" || -n "${TELYS_OAUTH_TOKEN:-}" ]]; then
            fail "Telys credentials require TELYS_ENABLED=true"
        fi
        if [[ "${KV_TRANSFER_CONFIG:-}" == *"TelysKVConnector"* || "${KV_TRANSFER_CONFIG:-}" == *"telys_vllm.connector"* ]]; then
            fail "the Telys connector requires TELYS_ENABLED=true"
        fi
        exec python3 /src/main.py "$@"
        ;;
    true)
        ;;
    *)
        fail "TELYS_ENABLED must be exactly 'true' or 'false'"
        ;;
esac

export KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-$DEFAULT_KV_TRANSFER_CONFIG}"
python3 - <<'PY'
import json
import os

FALSE_VALUES = {"0", "false", "no", "off"}
EXPECTED = {
    "kv_connector": "TelysKVConnector",
    "kv_connector_module_path": "telys_vllm.connector",
    "kv_role": "kv_both",
}


def require_false(value, source):
    if isinstance(value, bool):
        valid = not value
    elif isinstance(value, str):
        valid = value.strip().lower() in FALSE_VALUES
    else:
        valid = False
    if not valid:
        raise SystemExit(
            f"telys-vllm entrypoint: {source} must be false; "
            "asynchronous KV loading is unsupported"
        )


try:
    config = json.loads(os.environ["KV_TRANSFER_CONFIG"])
except json.JSONDecodeError as exc:
    raise SystemExit(
        "telys-vllm entrypoint: KV_TRANSFER_CONFIG must be valid JSON"
    ) from exc

if not isinstance(config, dict):
    raise SystemExit("telys-vllm entrypoint: KV_TRANSFER_CONFIG must be a JSON object")
if any(config.get(key) != value for key, value in EXPECTED.items()):
    raise SystemExit(
        "telys-vllm entrypoint: TELYS_ENABLED=true requires TelysKVConnector "
        "from telys_vllm.connector with kv_role=kv_both"
    )

extra = config.get("kv_connector_extra_config", {})
if not isinstance(extra, dict):
    raise SystemExit(
        "telys-vllm entrypoint: kv_connector_extra_config must be a JSON object"
    )
require_false(extra.get("telys_load_async", False), "telys_load_async")
if "TELYS_VLLM_LOAD_ASYNC" in os.environ:
    require_false(
        os.environ["TELYS_VLLM_LOAD_ASYNC"], "TELYS_VLLM_LOAD_ASYNC"
    )
PY
export TELYS_VLLM_LOAD_ASYNC=0

if [[ -n "${TELYS_API_KEY:-}" ]]; then
    python3 - <<'PY'
import os
import tempfile

import telys.installer
import telys.login
import telys.paths

key = os.environ["TELYS_API_KEY"]
home = telys.paths.telys_home()
os.makedirs(home, mode=0o700, exist_ok=True)
license_token = telys.login.register_device(
    api=telys.paths.api_url(), api_key=key
)
license_path = os.path.join(home, "login_license.jwt")
descriptor, temporary_path = tempfile.mkstemp(dir=home, prefix=".login_license.")
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as license_file:
        license_file.write(license_token)
    os.replace(temporary_path, license_path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
telys.installer.install_from_host(token=key, license_path=license_path)
PY
elif [[ -n "${TELYS_OAUTH_TOKEN:-}" ]]; then
    telys login --token "$TELYS_OAUTH_TOKEN" --no-browser --plan telys_developer
fi

unset TELYS_API_KEY TELYS_OAUTH_TOKEN
exec python3 /src/main.py "$@"
