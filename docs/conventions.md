# Worker vLLM - Development Conventions & Architecture Guide

## Project Overview

**worker-vllm** is a RunPod serverless worker that provides OpenAI-compatible endpoints
for Large Language Model (LLM) inference, powered by the vLLM engine. It is a thin
wrapper around the official [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai)
server image: instead of importing vLLM in-process (which broke on every vLLM release),
the worker launches `vllm serve` as a subprocess and proxies RunPod jobs to it over
loopback HTTP.

### Core Purpose

- **Primary Function**: Deploy any Hugging Face LLM as an OpenAI-compatible API endpoint
- **Platform**: RunPod Serverless infrastructure
- **Engine**: vLLM, packaged by upstream in the `vllm/vllm-openai` image
- **Compatibility**: Drop-in replacement for the OpenAI API (Chat Completions, Models,
  Responses, Messages, Completions), plus any other route the vLLM server exposes

## High-Level Architecture

### 1. **Process Layout & Request Flow**

```
Container start:
  ENTRYPOINT python3 /src/main.py
    ├─ (optional) reads /local_model_args.json for baked-in models
    ├─ args_builder.build_vllm_args()  → env vars → CLI flags
    ├─ spawns `vllm serve --host 127.0.0.1 --port $VLLM_PORT ...`
    ├─ polls GET /health until ready (fail fast if vLLM exits or times out)
    └─ starts runpod.serverless loop with handler.handler

Per job:
  RunPod job → handler.py normalize input → aiohttp → 127.0.0.1:$VLLM_PORT/<route>
    → stream SSE chunks back as they arrive, or yield parsed JSON once
```

**Key rule**: worker code must never `import vllm`. Everything goes through the HTTP
boundary or the CLI; that is what makes vLLM upgrades a one-line `VLLM_VERSION` change.

### 2. **Components**

- `src/main.py`: container entrypoint — builds the `vllm serve` command, manages the
  subprocess (health polling, signal forwarding), then starts the RunPod loop.
- `src/args_builder.py`: pure-Python env var → CLI flag translation (allowlist of
  `vllm serve` flags + legacy aliases + `VLLM_EXTRA_ARGS` passthrough). No third-party
  imports, so it is fully unit-testable on CPU runners.
- `src/handler.py`: the RunPod serverless handler; an aiohttp proxy that accepts three
  input shapes:
  1. `{"openai_route": ..., "openai_input": ...}` — RunPod's `/openai/*` passthrough
  2. `{"route": ..., "body": ..., "method": ...}` — generic proxy to any vLLM route
     (`method` is optional: POST when a body is present, GET otherwise)
  3. `{"prompt"|"messages": ..., "sampling_params": ...}` — legacy shorthand, mapped to
     `/v1/completions` / `/v1/chat/completions` (model id resolved from
     `SERVED_MODEL_NAME`/`MODEL_NAME`/the server's `/v1/models` when omitted)
- `src/download_model.py`: build-time model+tokenizer snapshot download; writes
  `/local_model_args.json` for baked-model images.

### 3. **Configuration Sources (precedence)**

```
VLLM_EXTRA_ARGS  >  env aliases (MODEL_NAME, ...)  >  env flag scan
                                                       >  vLLM_CONFIG_FILE yaml
                                                       >  vllm serve defaults
```

- `VLLM_EXTRA_ARGS` is appended last so it wins any argparse conflict.
- A `vllm serve --config` yaml can provide shared defaults; env-derived flags override it.

## Deployment Models

### Option 1: Pre-built Images (Recommended)

- **Image**: `runpod/worker-v1-vllm:<version>` (see [GitHub Releases](https://github.com/runpod-workers/worker-vllm/releases))
- **Configuration**: Entirely via environment variables
- **Model Loading**: Downloaded at container start by `vllm serve` into the HF cache on
  the network volume (`BASE_PATH`, default `/runpod-volume`)

### Option 2: Baked Model Images

- **Build Process**: `docker build --build-arg MODEL_NAME=... [--secret id=HF_TOKEN]`
  downloads weights during the image build via `src/download_model.py`
- **Storage**: Model embedded in the image; resolved snapshot paths land in
  `/local_model_args.json`
- **Runtime**: `main.py` applies the file (overrides `MODEL_NAME`) and sets
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`

## Development Patterns & Best Practices

### 1. **Code Organization**

```
src/
├── main.py            # Entrypoint: vLLM subprocess + RunPod loop lifecycle
├── args_builder.py    # env vars → `vllm serve` CLI flags (pure Python)
├── handler.py         # RunPod handler: aiohttp proxy to the vLLM server
└── download_model.py  # Build-time model download (Option 2)
```

### 2. **Environment Variable Conventions**

- **vLLM flags**: UPPER_SNAKE of the `vllm serve` flag name (`MAX_MODEL_LEN`,
  `ENABLE_PREFIX_CACHING`). New flags work immediately if they are in the
  `args_builder` allowlist; anything else goes through `VLLM_EXTRA_ARGS`.
- **Booleans**: string `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`). The
  worker emits vLLM's pair forms — truthy becomes the bare `--flag`, falsy becomes
  `--no-flag`; current vLLM rejects `--flag false` as an unrecognized argument.
- **Complex values**: JSON strings passed verbatim (`SPECULATIVE_CONFIG`,
  `LORA_MODULES`, `HF_OVERRIDES`, ...).
- **Wrapper envs**: `MAX_CONCURRENCY`, `VLLM_PORT`, `VLLM_STARTUP_TIMEOUT`,
  `REQUEST_TIMEOUT`.

When vLLM changes, don't hand-audit the allowlist in `args_builder.py` — verify it
mechanically with `scripts/check_vllm_flags.py`, which introspects the real
`vllm serve` parser of the target release (run inside the pinned base image, which
is also how macOS dev machines without vllm can do it):

```bash
VLLM_VERSION=$(sed -n 's/^ARG VLLM_VERSION=//p' Dockerfile | head -1)
docker run --rm -v "$PWD:/repo" --entrypoint python3 \
  "vllm/vllm-openai:${VLLM_VERSION}" \
  /repo/scripts/check_vllm_flags.py dump -o /repo/.cache/vllm_serve_flags.json
python3 scripts/check_vllm_flags.py check --snapshot .cache/vllm_serve_flags.json
```

`dump` classifies every long option into `bool_pair` (`--flag/--no-flag`),
`store_true`, or `value`; `check` fails on any worker flag that vLLM removed or
re-categorized. The `vllm-flag-drift` job in `.github/workflows/test.yml` runs
exactly this on every PR/push against the image pinned by the Dockerfile ARG, so
a `VLLM_VERSION` bump that breaks the allowlist fails CI instead of crashing
workers at container start.

### 3. **Error Handling**

- Startup failures (bad flag, missing model, OOM during load) → vLLM exits before
  `/health` → `main.py` fails the worker fast instead of accepting jobs.
- Request failures → the proxy yields a RunPod job error containing the vLLM HTTP
  status and body (vLLM's own OpenAI-compatible error payloads flow through unchanged).
- If the vLLM process dies while serving, the next job gets an immediate
  "worker is unhealthy" error instead of hanging on a dead socket.

### 4. **Docker & Deployment**

- Single `FROM vllm/vllm-openai:${VLLM_VERSION}` stage; pip-installs only
  `runpod` + `aiohttp` + `huggingface-hub` on top of the base image.
- **Build args**: `VLLM_VERSION`, `MODEL_NAME`, `MODEL_REVISION`, `TOKENIZER_NAME`,
  `TOKENIZER_REVISION`, `QUANTIZATION`, `BASE_PATH`.
- **Docker Bake**: `docker-bake.hcl` with `DOCKERHUB_REPO`, `DOCKERHUB_IMG`,
  `RELEASE_VERSION`, `HUGGINGFACE_ACCESS_TOKEN`; platform `linux/amd64`.
- **CI/CD**: non-main branches → `runpod/worker-v1-vllm:dev-<branch-name>`;
  version tags → `runpod/worker-v1-vllm:<version>`. vLLM version bumps are
  `VLLM_VERSION` changes; CI syncs the README version line from the Dockerfile.

## Release & Versioning Strategy

### 1. **Version Tagging**

- **Worker releases**: numeric tags without "v" prefix (`2.7.0`, `2.8.0`) — unchanged.
- **vLLM version**: the Dockerfile `ARG VLLM_VERSION=vX.Y.Z` picks the base image tag.

### 2. **Release Workflow**

1. **Feature Development**: Work on feature branches → triggers dev builds
2. **Main Branch Staging**: Merge features to main → stable codebase (no builds)
3. **Version Release**: Create git tag from main branch (e.g., `2.8.0`) → triggers versioned release + GitHub release
4. **Docker Hub**: Versioned image pushed with tag

### 3. **Branch Strategy**

- **Feature Branches**: `feature/*`, `fix/*`, `feat/*` etc. → Dev builds
- **Main Branch**: Stable codebase ready for release (no automatic builds)
- **Git Tags**: Must be created from main branch: `git checkout main && git tag 2.8.0 && git push origin 2.8.0`

## Testing & Development

### 1. **Unit Tests**

- `tests/test_args_builder.py` covers the env var → CLI translation (mapping, aliases,
  bool handling, JSON passthrough, `VLLM_EXTRA_ARGS` precedence). Pure Python, no GPU,
  no vllm import — runs on any CPU runner with just `pytest`.

### 2. **Local Smoke Testing**

- `src/handler.py` can be exercised against any OpenAI-compatible server by setting
  `VLLM_BASE_URL`; the tests use a local aiohttp stub.
- `src/main.py`'s health/startup logic is thin by design so that the unit surface stays
  in `args_builder.py`.

### 3. **End-to-End**

- `scripts/serverless_e2e_test.py` builds the image, deploys a real RunPod serverless
  endpoint, and runs `.runpod/tests.json` against it. Model configs live in
  `configs/**/*.yaml` using `vllm serve` key names (`model`, `max-model-len`,
  `gpu-memory-utilization`), which the script converts to env vars
  (`model` → `MODEL_NAME`, everything else → UPPER_SNAKE).

## Security & Best Practices

- **Build secrets**: `HF_TOKEN` via Docker secrets; never baked into image layers.
- **Authentication**: handled by RunPod at the platform level (endpoint API key).
  The worker intentionally does not wire `VLLM_API_KEY`/`--api-key`; setting it has
  no effect. The internal vLLM server binds to loopback only and is never
  reachable from outside the worker container.
- **Remote code**: `TRUST_REMOTE_CODE` stays opt-in and is forwarded to vLLM.
