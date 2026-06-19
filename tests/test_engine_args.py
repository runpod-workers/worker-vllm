"""Tests for HF cache path resolution and served-model-name decoupling.

Regression coverage for issue #310: when MODEL_NAME is served from a lowercased
HF cache dir, the cache resolver rewrites engine_args.model to a snapshot path.
The served model name must stay the original repo id, not the path.
"""

import os

import pytest

from src import engine_args
from src.engine_args import _resolve_cached_model_path, get_engine_args


MODEL = "Qwen/Qwen3.6-27B-FP8"
SNAPSHOT_HASH = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"


def _make_cache(root, folder_name, snapshot=SNAPSHOT_HASH):
    """Create a HF-style ``models--…/snapshots/<hash>/`` dir and return its path."""
    snap_dir = os.path.join(root, folder_name, "snapshots", snapshot)
    os.makedirs(snap_dir)
    return snap_dir


def _is_case_sensitive_fs(path):
    """The lowercase-cache resolution only matters on case-sensitive filesystems.

    On macOS (APFS, case-insensitive by default) ``models--Qwen--…`` and
    ``models--qwen--…`` collide, so the resolver always sees the exact-case dir
    as present. Production runs on Linux (case-sensitive), which is what these
    tests exercise.
    """
    probe = os.path.join(path, "CaseProbe")
    open(probe, "w").close()
    try:
        return not os.path.exists(os.path.join(path, "caseprobe"))
    finally:
        os.remove(probe)


requires_case_sensitive_fs = pytest.mark.skipif(
    not _is_case_sensitive_fs(os.environ.get("TMPDIR", "/tmp")),
    reason="lowercase HF cache resolution only applies on case-sensitive filesystems",
)


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    # Make sure HF_HOME does not shadow the explicit cache dir during the test.
    monkeypatch.delenv("HF_HOME", raising=False)
    return cache


class TestResolveCachedModelPath:
    def test_exact_case_dir_returns_repo_id(self, hf_cache):
        _make_cache(str(hf_cache), "models--Qwen--Qwen3.6-27B-FP8")
        assert _resolve_cached_model_path(MODEL) == MODEL

    def test_no_cache_returns_repo_id(self, hf_cache):
        assert _resolve_cached_model_path(MODEL) == MODEL

    def test_absolute_path_passthrough(self, hf_cache):
        path = "/runpod-volume/some/local/model"
        assert _resolve_cached_model_path(path) == path

    @requires_case_sensitive_fs
    def test_lowercase_dir_returns_snapshot_path(self, hf_cache):
        snap = _make_cache(str(hf_cache), "models--qwen--qwen3.6-27b-fp8")
        assert _resolve_cached_model_path(MODEL) == snap

    def test_lowercase_dir_without_snapshots_returns_repo_id(self, hf_cache):
        # Dir exists but has no snapshots subdir -> nothing to resolve to.
        os.makedirs(os.path.join(str(hf_cache), "models--qwen--qwen3.6-27b-fp8"))
        assert _resolve_cached_model_path(MODEL) == MODEL

    @requires_case_sensitive_fs
    def test_lowercase_dir_picks_latest_snapshot(self, hf_cache):
        folder = "models--qwen--qwen3.6-27b-fp8"
        _make_cache(str(hf_cache), folder, snapshot="aaaa")
        latest = _make_cache(str(hf_cache), folder, snapshot="zzzz")
        assert _resolve_cached_model_path(MODEL) == latest


class TestGetEngineArgsServedName:
    """Issue #310: served name must be decoupled from the resolved on-disk path."""

    @pytest.fixture(autouse=True)
    def base_env(self, monkeypatch):
        # Avoid the network branch in _resolve_max_model_len.
        monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "2048")
        monkeypatch.delenv("SERVED_MODEL_NAME", raising=False)
        # Don't pick up a stray vLLM config file from the environment.
        monkeypatch.setenv("VLLM_CONFIG_FILE", "/nonexistent-vllm-config.yaml")

    @requires_case_sensitive_fs
    def test_served_name_is_repo_id_when_path_rewritten(self, hf_cache, monkeypatch):
        snap = _make_cache(str(hf_cache), "models--qwen--qwen3.6-27b-fp8")
        monkeypatch.setenv("MODEL_NAME", MODEL)

        result = get_engine_args()

        assert result.model == snap  # weights load from the lowercase cache
        assert result.served_model_name == MODEL  # API still serves the repo id

    def test_served_name_untouched_when_no_rewrite(self, hf_cache, monkeypatch):
        _make_cache(str(hf_cache), "models--Qwen--Qwen3.6-27B-FP8")
        monkeypatch.setenv("MODEL_NAME", MODEL)

        result = get_engine_args()

        assert result.model == MODEL
        assert result.served_model_name is None

    def test_explicit_served_name_not_overridden(self, hf_cache, monkeypatch):
        _make_cache(str(hf_cache), "models--qwen--qwen3.6-27b-fp8")
        monkeypatch.setenv("MODEL_NAME", MODEL)
        monkeypatch.setenv("SERVED_MODEL_NAME", "custom-name")

        result = get_engine_args()

        assert result.served_model_name == "custom-name"
