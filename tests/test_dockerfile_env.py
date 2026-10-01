"""Caches that must live under BASE_PATH so they persist on a network volume.

The RunPod network volume mounts at BASE_PATH (default /runpod-volume). Any
cache left at its default under ~/.cache is lost on every cold start, so the
image env points both the HF cache and vLLM's torch.compile cache there.
"""

import re
import shlex
from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def image_env() -> dict[str, str]:
    """KEY=value pairs from the Dockerfile's ENV instructions (values unexpanded)."""
    # Docker drops comment lines before joining continuations, so do the same.
    lines = [
        line
        for line in DOCKERFILE.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    joined = re.sub(r"\\\n", " ", "\n".join(lines))
    env: dict[str, str] = {}
    for line in joined.splitlines():
        if line.startswith("ENV "):
            for pair in shlex.split(line[len("ENV "):]):
                key, _, value = pair.partition("=")
                env[key] = value
    return env


def test_hf_and_vllm_caches_live_under_base_path():
    env = image_env()
    assert env["HF_HOME"].startswith("${BASE_PATH}/")
    assert env["VLLM_CACHE_ROOT"] == "${BASE_PATH}/vllm-cache"
