# Optional Telys KV connector example

This derived image adds the [Telys vLLM connector](https://github.com/thyn-ai/telys-vllm)
to the stock `runpod/worker-v1-vllm:v2.30.3` image (vLLM 0.32.0). The upstream worker's
`/src/main.py` startup path remains unchanged. Telys is disabled unless
`TELYS_ENABLED=true` is set explicitly.

## Build

`TELYS_VLLM_COMMIT` defaults to
`2f0c20ccec56afcb613973e296281f51e23cbe26`, the source commit for connector `v0.1.6`.
The Dockerfile installs from the immutable `git+https` source at that commit. Overrides
must use a full 40-character lowercase SHA; branches, tags, short SHAs, and uppercase
SHAs are rejected.

Build from the worker repository root:

```bash
docker build --platform=linux/amd64 \
  --file examples/telys/Dockerfile \
  --build-arg TELYS_VLLM_COMMIT=2f0c20ccec56afcb613973e296281f51e23cbe26 \
  --tag your-dockerhub-user/telys-worker-vllm:VERSION .
```

The build installs `telys==0.1.15`, installs the connector directly from the pinned
commit with `--no-deps`, and checks that vLLM remains exactly 0.32.0, the connector
distribution is 0.1.6, and `TelysKVConnector` imports. It does not bake in a model,
Telys credentials, or a signed Telys runtime. Publishing the resulting image is a
separate operation.

The stable `telys-vllm==0.1.6` and `telys==0.1.15` packages are public on PyPI. The
connector project's stable GHCR release image is also public and can be pulled without
registry credentials at
`ghcr.io/thyn-ai/telys-worker-vllm@sha256:1b02b3cdac471064e8b9968fe2ab87f99da2d455e04d9cffea6b74b745a4218f`.
The build above uses the immutable connector source commit rather than that prebuilt
image.

## Stock behavior (default)

Leave `TELYS_ENABLED` unset or set it to `false`. The entrypoint immediately executes
`python3 /src/main.py` with the original arguments and environment. Non-Telys vLLM
configuration, including another `KV_TRANSFER_CONFIG` connector such as the stock
filesystem offloader, is passed through unchanged.

To prevent an accidental partial configuration, the disabled path rejects nonempty
`TELYS_API_KEY` or `TELYS_OAUTH_TOKEN`, and rejects a `KV_TRANSFER_CONFIG` that names
`TelysKVConnector` or `telys_vllm.connector`. It never provisions Telys in this mode.
`TELYS_ENABLED` accepts exactly `true` or `false`.

In both modes, the entrypoint rejects `--kv-transfer-config` in `VLLM_EXTRA_ARGS`,
including both `--kv-transfer-config value` and `--kv-transfer-config=value` forms,
instead of letting it bypass the connector checks. Set `KV_TRANSFER_CONFIG` instead.
Other extra arguments pass through unchanged.

## Enable Telys

Attach a network volume at `/runpod-volume`. A minimal endpoint configuration is:

```bash
TELYS_ENABLED=true
TELYS_HOME=/runpod-volume/telys-home
TELYS_PATH=/runpod-volume/telys-data
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
ENABLE_PREFIX_CACHING=false
VLLM_ENABLE_V1_MULTIPROCESSING=0
# HF_TOKEN=<only for a gated/private model; configure it as a secret>
```

When enabled, the entrypoint supplies this connector configuration unless one is
already present:

```bash
KV_TRANSFER_CONFIG={"kv_connector":"TelysKVConnector","kv_connector_module_path":"telys_vllm.connector","kv_role":"kv_both"}
```

A custom `KV_TRANSFER_CONFIG` must be valid JSON with those same connector, module,
and role values. `kv_connector_extra_config` may contain supported Telys settings. Do
not configure another KV connector on the same endpoint. Disabling native prefix
caching and V1 multiprocessing matches the connector-only validation setup; choose
those settings deliberately for other workloads.

The entrypoint sets `TELYS_VLLM_LOAD_ASYNC=0`. Asynchronous KV loading is unsupported
in this connector revision. Startup fails if the environment or
`kv_connector_extra_config.telys_load_async` attempts to enable it.

`TELYS_HOME` holds the SDK runtime and login state. Embedded engine data uses
`TELYS_PATH`; BlobStore payloads use `TELYS_PATH` with `_blobs` appended, and the
fallback slab defaults to `TELYS_PATH` with `_kv.slab` appended. Set
`TELYS_VLLM_SLAB_PATH` explicitly when using a remote engine with the slab fallback.
Keep runtime state, metadata, and payloads on persistent storage, use separate cache
namespaces for incompatible model/KV configurations, and restrict access to the volume.

## Provision the Telys runtime

For a runtime that is not installed yet, configure exactly one credential as a Runpod
secret:

- `TELYS_API_KEY`: direct API-key provisioning. The entrypoint registers the device,
  atomically writes `login_license.jwt` under `TELYS_HOME` with mode `0600`, and calls
  the SDK installer. An API key is not an OAuth token.
- `TELYS_OAUTH_TOKEN`: OAuth-only provisioning through
  `telys login --token ... --no-browser --plan telys_developer`.

If both are present, API-key provisioning takes precedence; there is no fallback from
a failed API-key attempt to OAuth. Any provisioning failure stops startup. After
success, both credential variables are removed from the worker environment before the
upstream entrypoint runs. Do not put credentials in the Dockerfile, `ENV`,
`KV_TRANSFER_CONFIG`, or `VLLM_EXTRA_ARGS`.

After the signed runtime and login state are installed on the volume, remove both
credential variables. The entrypoint skips login when both are empty and uses the
existing runtime. Perform initial provisioning with one worker before allowing
concurrent workers to initialize the same `TELYS_HOME`.

## Checks and required acceptance

The CPU-only checks exercise the entrypoint with fake SDK/CLI boundaries, inspect the
Dockerfile, and execute its commit validator:

```bash
python3 -m pytest tests/test_telys_example.py -v
```

They do not pull the base image, install the connector, download a model, access a GPU,
or contact Telys or Runpod. They are not remote acceptance evidence.

Before production use, complete all of the following with the selected final commit:

- Build the `linux/amd64` image and confirm its vLLM, SDK, and connector import checks.
- Verify the default path with the normal worker smoke test and a non-Telys connector.
- Provision the signed runtime and start a real endpoint with the target model, GPU,
  network volume, and model/KV settings.
- Check cold misses, warm hits from a fresh worker, output equivalence against an
  uncached reference, persistence drain, disk use, and restart behavior.
- Validate multi-writer or multi-group topologies separately rather than assuming them.
