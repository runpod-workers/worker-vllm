#!/usr/bin/env python3
"""End-to-end test: build/push the worker image, deploy it as a real RunPod
Serverless endpoint on H100, run the .runpod/tests.json test cases against it,
then tear the endpoint down.

Usage:
    RUNPOD_API_KEY=... python scripts/serverless_e2e_test.py \\
        --config configs/qwen/qwen3_8b.yaml --build

    RUNPOD_API_KEY=... python scripts/serverless_e2e_test.py \\
        --config configs/qwen/qwen3_8b.yaml --image runpod/worker-v1-vllm:dev-my-branch
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

REST_API_BASE = "https://rest.runpod.io/v1"
JOB_API_BASE = "https://api.runpod.ai/v2"

DEFAULT_GPU_TYPE_IDS = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 NVL",
    "NVIDIA H100 PCIe",
]

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}


def log(msg: str) -> None:
    print(f"[serverless_e2e_test] {msg}", flush=True)


def write_github_output(key: str, value: str) -> None:
    """Expose a value to later CI steps (e.g. an always() cleanup safety net
    for when this process gets killed before its own `finally` can run)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"{key}={value}\n")


def stringify_env_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def load_hub_defaults(hub_json_path: Path) -> dict:
    hub = json.loads(hub_json_path.read_text())
    config = hub.get("config", {})

    base_env = {}
    for entry in config.get("env", []):
        default = entry.get("input", {}).get("default")
        if default is None:
            continue
        base_env[entry["key"]] = stringify_env_value(default)

    return {
        "containerDiskInGb": config.get("containerDiskInGb", 50),
        "gpuCount": config.get("gpuCount", 1),
        "allowedCudaVersions": config.get("allowedCudaVersions"),
        "minCudaVersion": config.get("minCudaVersion"),
        "env": base_env,
    }



# config keys whose naive `KEY.replace("-", "_").upper()` transform doesn't match
# the env var the worker actually reads (see src/args_builder.py ENV_ALIASES).
CONFIG_KEY_ALIASES = {
    "MODEL": "MODEL_NAME",
}


def load_model_env(config_yaml_path: Path) -> dict:
    raw = yaml.safe_load(config_yaml_path.read_text()) or {}
    env = {}
    for key, value in raw.items():
        env_key = key.replace("-", "_").upper()
        env_key = CONFIG_KEY_ALIASES.get(env_key, env_key)
        env[env_key] = stringify_env_value(value)
    return env


def build_and_push_image(dockerhub_repo: str, dockerhub_img: str, release_version: str) -> str:
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN", "")
    dockerhub_user = os.environ.get("DOCKERHUB_USERNAME")
    dockerhub_pass = os.environ.get("DOCKERHUB_TOKEN")

    if dockerhub_user and dockerhub_pass:
        log(f"Logging in to Docker Hub as {dockerhub_user}")
        subprocess.run(
            ["docker", "login", "-u", dockerhub_user, "--password-stdin"],
            input=dockerhub_pass,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    else:
        log("DOCKERHUB_USERNAME/DOCKERHUB_TOKEN not set; assuming docker is already logged in")

    tag = f"{dockerhub_repo}/{dockerhub_img}:{release_version}"
    log(f"Building and pushing {tag} via docker buildx bake")
    # docker-bake.hcl's DOCKERHUB_REPO/DOCKERHUB_IMG/RELEASE_VERSION are bake-level
    # `variable` blocks that compute `tags` - they read from env vars of the same
    # name, NOT from `--set target.args.*` (that sets Dockerfile build ARGs, a
    # separate namespace the Dockerfile doesn't even declare these under).
    bake_env = {
        **os.environ,
        "DOCKERHUB_REPO": dockerhub_repo,
        "DOCKERHUB_IMG": dockerhub_img,
        "RELEASE_VERSION": release_version,
        "HUGGINGFACE_ACCESS_TOKEN": hf_token,
    }
    subprocess.run(
        ["docker", "buildx", "bake", "--push"],
        check=True,
        cwd=REPO_ROOT,
        env=bake_env,
    )
    return tag


def api_request(method: str, url: str, api_key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from e


def create_template(api_key: str, name: str, image: str, env: dict, container_disk_in_gb: int) -> str:
    log(f"Creating template {name!r} for image {image}")
    resp = api_request("POST", f"{REST_API_BASE}/templates", api_key, {
        "name": name,
        "imageName": image,
        "isServerless": True,
        "env": env,
        "containerDiskInGb": container_disk_in_gb,
    })
    return resp["id"]


def create_endpoint(
    api_key: str,
    name: str,
    template_id: str,
    gpu_type_ids: list[str],
    gpu_count: int,
    allowed_cuda_versions: list[str] | None,
    min_cuda_version: str | None,
    idle_timeout: int,
) -> str:
    log(f"Creating endpoint {name!r} (gpuTypeIds={gpu_type_ids})")
    body = {
        "name": name,
        "templateId": template_id,
        "gpuTypeIds": gpu_type_ids,
        "gpuCount": gpu_count,
        "workersMin": 0,
        "workersMax": 1,
        "idleTimeout": idle_timeout,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
    }
    if allowed_cuda_versions:
        body["allowedCudaVersions"] = allowed_cuda_versions
    if min_cuda_version:
        body["minCudaVersion"] = min_cuda_version
    resp = api_request("POST", f"{REST_API_BASE}/endpoints", api_key, body)
    return resp["id"]


def delete_endpoint(api_key: str, endpoint_id: str) -> None:
    log(f"Deleting endpoint {endpoint_id}")
    try:
        api_request("DELETE", f"{REST_API_BASE}/endpoints/{endpoint_id}", api_key)
    except RuntimeError as e:
        log(f"WARNING: failed to delete endpoint {endpoint_id}: {e}")


def delete_template(api_key: str, template_id: str) -> None:
    log(f"Deleting template {template_id}")
    try:
        api_request("DELETE", f"{REST_API_BASE}/templates/{template_id}", api_key)
    except RuntimeError as e:
        log(f"WARNING: failed to delete template {template_id}: {e}")


def response_has_error(output) -> bool:
    if isinstance(output, dict):
        return "error" in output
    if isinstance(output, list):
        return any(isinstance(item, dict) and "error" in item for item in output)
    return False


def run_test_case(api_key: str, endpoint_id: str, test: dict, cold_start_buffer_seconds: int) -> bool:
    name = test.get("name", "unnamed_test")
    deadline = time.monotonic() + test.get("timeout", 300000) / 1000 + cold_start_buffer_seconds

    log(f"Submitting job for test {name!r}")
    submit = api_request("POST", f"{JOB_API_BASE}/{endpoint_id}/run", api_key, {"input": test["input"]})
    job_id = submit["id"]

    while True:
        if time.monotonic() > deadline:
            log(f"FAIL {name}: timed out waiting for job {job_id}")
            return False

        status_resp = api_request("GET", f"{JOB_API_BASE}/{endpoint_id}/status/{job_id}", api_key)
        status = status_resp.get("status")

        if status in TERMINAL_STATUSES:
            if status != "COMPLETED":
                log(f"FAIL {name}: job {job_id} ended with status {status}: {status_resp}")
                return False
            if response_has_error(status_resp.get("output")):
                log(f"FAIL {name}: job {job_id} completed but output contained an error: {status_resp.get('output')}")
                return False
            log(f"PASS {name}")
            return True

        time.sleep(5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a configs/**/*.yaml vLLM config")
    parser.add_argument("--image", help="Existing image tag to test; skips building unless --build is also given")
    parser.add_argument("--build", action="store_true", help="Build and push the image before testing")
    parser.add_argument("--keep", action="store_true", help="Don't tear down the endpoint/template afterward")
    parser.add_argument("--hub-json", type=Path, default=REPO_ROOT / ".runpod" / "hub.json")
    parser.add_argument("--tests-file", type=Path, default=REPO_ROOT / ".runpod" / "tests.json")
    parser.add_argument("--gpu-type-ids", default=",".join(DEFAULT_GPU_TYPE_IDS))
    parser.add_argument("--min-cuda-version", help="Overrides the config/hub.json minCudaVersion")
    parser.add_argument("--dockerhub-repo", default=os.environ.get("DOCKERHUB_REPO", "runpod"))
    parser.add_argument("--dockerhub-img", default=os.environ.get("DOCKERHUB_IMG", "worker-v1-vllm"))
    parser.add_argument("--idle-timeout", type=int, default=60)
    parser.add_argument("--cold-start-buffer-seconds", type=int, default=600)
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        log("ERROR: RUNPOD_API_KEY is not set")
        return 1

    model_slug = args.config.stem
    run_id = uuid.uuid4().hex[:8]

    if args.build or not args.image:
        release_version = f"test-{model_slug}-{run_id}"
        image = build_and_push_image(args.dockerhub_repo, args.dockerhub_img, release_version)
    else:
        image = args.image

    hub_defaults = load_hub_defaults(args.hub_json)
    model_env = load_model_env(args.config)
    env = {**hub_defaults["env"], **model_env}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    tests_data = json.loads(args.tests_file.read_text())
    gpu_type_ids = [g.strip() for g in args.gpu_type_ids.split(",") if g.strip()]

    resource_name = f"worker-vllm-e2e-{model_slug}-{run_id}"
    template_id = None
    endpoint_id = None
    try:
        template_id = create_template(
            api_key, resource_name, image, env, hub_defaults["containerDiskInGb"]
        )
        write_github_output("template_id", template_id)
        endpoint_id = create_endpoint(
            api_key,
            resource_name,
            template_id,
            gpu_type_ids,
            hub_defaults["gpuCount"],
            hub_defaults["allowedCudaVersions"],
            args.min_cuda_version or hub_defaults["minCudaVersion"],
            args.idle_timeout,
        )
        write_github_output("endpoint_id", endpoint_id)

        results = [
            run_test_case(api_key, endpoint_id, test, args.cold_start_buffer_seconds)
            for test in tests_data["tests"]
        ]

        if all(results):
            log(f"All {len(results)} test(s) passed for {model_slug}")
            return 0
        log(f"{results.count(False)}/{len(results)} test(s) failed for {model_slug}")
        return 1
    finally:
        if not args.keep:
            if endpoint_id:
                delete_endpoint(api_key, endpoint_id)
            if template_id:
                delete_template(api_key, template_id)
        else:
            log(f"--keep passed; leaving endpoint={endpoint_id} template={template_id} running")


if __name__ == "__main__":
    sys.exit(main())
