FROM runpod/worker-v1-vllm:v2.30.3

ARG TELYS_VLLM_COMMIT=3505b767cc9d6f9cfd7751fa3ad9946b92d54bb8
RUN python3 - "$TELYS_VLLM_COMMIT" <<'PY'
import re
import sys

commit = sys.argv[1]
if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
    raise SystemExit(
        "TELYS_VLLM_COMMIT must be a full 40-character lowercase SHA; "
        "branches, tags, short SHAs, and other revision forms are not accepted"
    )
PY

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "telys==0.1.15" \
    && python3 -m pip install --no-cache-dir --no-deps \
        "telys-vllm @ git+https://github.com/thyn-ai/telys-vllm.git@${TELYS_VLLM_COMMIT}" \
    && python3 -c 'from importlib.metadata import version; vllm_version = version("vllm"); assert vllm_version == "0.32.0", f"expected vllm==0.32.0, found {vllm_version}"; telys_version = version("telys"); assert telys_version == "0.1.15", f"expected telys==0.1.15, found {telys_version}"; connector_version = version("telys-vllm"); assert connector_version == "0.1.5", f"expected telys-vllm==0.1.5, found {connector_version}"; from telys_vllm.connector import TelysKVConnector'

COPY --chmod=0755 examples/telys/entrypoint.sh /usr/local/bin/telys-vllm-entrypoint

ENTRYPOINT ["/usr/local/bin/telys-vllm-entrypoint"]
