"""The compile cache stays on the volume only when the volume can take the writes.

A full or read-only volume must not fail the start: the worker falls back to
vLLM's default cache root (in the container), which is what every start used
before the cache moved to the volume — before launch when the check catches
it, or with one relaunch when the volume fills during the start.
"""

import errno
import shutil
import sys
import types
from collections import namedtuple
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import compile_cache  # noqa: E402
import main  # noqa: E402

DiskUsage = namedtuple("DiskUsage", "total used free")
NO_SPACE = "OSError: [Errno 28] No space left on device"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A volume and a container cache dir; the volume reports as its own disk."""
    volume = tmp_path / "volume"
    container = tmp_path / "container-cache"
    real_device = compile_cache._device
    monkeypatch.setattr(
        compile_cache,
        "_device",
        lambda p: -1 if str(p).startswith(str(volume)) else real_device(p),
    )
    return types.SimpleNamespace(
        root=str(volume / "vllm-cache"),
        xdg=str(container),
        fallback=str(container / "vllm"),
        volume=volume,
    )


def plenty_free(monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DiskUsage(100 << 30, 0, 100 << 30))


def test_writable_volume_with_room_keeps_the_cache_there(paths, monkeypatch):
    plenty_free(monkeypatch)
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.ensure_usable(env) is None
    assert env["VLLM_CACHE_ROOT"] == paths.root
    assert list(Path(paths.root).iterdir()) == []  # probe file cleaned up


def test_unset_means_vllm_default_and_nothing_to_check():
    env = {}

    assert compile_cache.ensure_usable(env) is None
    assert env == {}


def test_uncreatable_root_falls_back(paths):
    # A path under a regular file can't be created, even as root.
    paths.volume.write_text("")
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.ensure_usable(env).startswith("not writable")
    assert env["VLLM_CACHE_ROOT"] == paths.fallback


def test_existing_read_only_root_falls_back(paths, monkeypatch):
    # makedirs(exist_ok=True) succeeds on an existing dir of a read-only
    # volume; only the write probe catches it.
    Path(paths.root).mkdir(parents=True)

    def read_only(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(compile_cache.tempfile, "NamedTemporaryFile", read_only)
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.ensure_usable(env) == "not writable (Read-only file system)"
    assert env["VLLM_CACHE_ROOT"] == paths.fallback


def test_full_quota_is_caught_before_launch(paths, monkeypatch):
    # statvfs may report the storage pool, not the volume's quota, and an empty
    # file still fits under an exhausted quota; a synced block does not.
    plenty_free(monkeypatch)

    def quota_full(fd):
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(compile_cache.os, "fsync", quota_full)
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.ensure_usable(env) == "not writable (Disk quota exceeded)"
    assert env["VLLM_CACHE_ROOT"] == paths.fallback


def test_device_lookup_failure_still_falls_back(paths, monkeypatch):
    def broken(path):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(compile_cache, "_device", broken)
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.fall_back("out of space during startup", env, out_of_space=True)
    assert env["VLLM_CACHE_ROOT"] == paths.fallback


def test_nearly_full_volume_falls_back(paths, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DiskUsage(20 << 30, 0, 300 << 20))
    env = {"VLLM_CACHE_ROOT": paths.root, "XDG_CACHE_HOME": paths.xdg}

    assert compile_cache.ensure_usable(env) == "nearly full (300 MiB free)"
    assert env["VLLM_CACHE_ROOT"] == paths.fallback


def test_nearly_full_without_a_volume_changes_nothing(tmp_path, monkeypatch):
    # Root and fallback on the same disk: moving the cache can't free space.
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DiskUsage(20 << 30, 0, 300 << 20))
    env = {"VLLM_CACHE_ROOT": str(tmp_path / "vllm-cache"), "XDG_CACHE_HOME": str(tmp_path / "xdg")}

    assert compile_cache.ensure_usable(env) is None
    assert env["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm-cache")


def test_opt_out_is_not_probed(tmp_path):
    env = {"VLLM_CACHE_ROOT": str(tmp_path / "xdg" / "vllm"), "XDG_CACHE_HOME": str(tmp_path / "xdg")}

    assert compile_cache.ensure_usable(env) is None
    assert not (tmp_path / "xdg").exists()


def test_tilde_is_expanded_like_vllm_does(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    plenty_free(monkeypatch)
    env = {"VLLM_CACHE_ROOT": "~/vllm-cache"}

    assert compile_cache.ensure_usable(env) is None
    assert (tmp_path / "home" / "vllm-cache").is_dir()
    assert not (tmp_path / "~").exists()


def test_fallback_root_matches_vllm_default(monkeypatch):
    monkeypatch.setenv("HOME", "/root")

    assert compile_cache.vllm_default_root({}) == "/root/.cache/vllm"
    assert compile_cache.vllm_default_root({"XDG_CACHE_HOME": "/x"}) == "/x/vllm"


@pytest.fixture
def run_main(monkeypatch):
    """Run main() without vLLM or the RunPod SDK; returns (cache roots per launch, handler)."""
    monkeypatch.setenv("MODEL_NAME", "org/model")
    monkeypatch.setattr(main.model_preflight, "check_model_access", lambda: None)
    monkeypatch.setattr(main, "stop_vllm", lambda proc: None)
    handler_stub = types.ModuleType("handler")
    handler_stub.handler = lambda job: None
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=mock.MagicMock())
    monkeypatch.setitem(sys.modules, "handler", handler_stub)
    monkeypatch.setitem(sys.modules, "runpod", runpod_stub)

    def drive(*attempt_outputs):
        roots = []
        outputs = iter(attempt_outputs)

        def fake_start():
            roots.append(main.os.environ["VLLM_CACHE_ROOT"])
            return mock.MagicMock()

        def fake_wait(proc):
            output = next(outputs)
            if output is not None:
                main.recent_output.clear()
                main.recent_output.append(output)
                raise RuntimeError("vLLM serve exited during startup with code 1")

        monkeypatch.setattr(main, "start_vllm", fake_start)
        monkeypatch.setattr(main, "wait_for_vllm", fake_wait)
        main.recent_output.clear()
        main.main()
        return roots, handler_stub

    return drive


def test_main_checks_the_cache_root_before_launching_vllm(paths, run_main, monkeypatch):
    paths.volume.write_text("")
    monkeypatch.setenv("VLLM_CACHE_ROOT", paths.root)
    monkeypatch.setenv("XDG_CACHE_HOME", paths.xdg)

    roots, handler = run_main(None)

    assert roots == [paths.fallback]
    assert handler.startup_error is None


def test_volume_filling_during_start_relaunches_once_in_the_container(paths, run_main, monkeypatch):
    plenty_free(monkeypatch)  # the pre-launch check passes; the download fills it later
    monkeypatch.setenv("VLLM_CACHE_ROOT", paths.root)
    monkeypatch.setenv("XDG_CACHE_HOME", paths.xdg)

    roots, handler = run_main(NO_SPACE, None)

    assert roots == [paths.root, paths.fallback]
    assert handler.startup_error is None


def test_out_of_disk_without_a_volume_answers_jobs_instead_of_relaunching(tmp_path, run_main, monkeypatch):
    plenty_free(monkeypatch)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    roots, handler = run_main(NO_SPACE)

    assert roots == [str(tmp_path / "vllm-cache")]
    assert "vllm-cache" in handler.startup_error


def test_second_out_of_disk_answers_jobs(paths, run_main, monkeypatch):
    plenty_free(monkeypatch)
    monkeypatch.setenv("VLLM_CACHE_ROOT", paths.root)
    monkeypatch.setenv("XDG_CACHE_HOME", paths.xdg)

    roots, handler = run_main(NO_SPACE, NO_SPACE)

    assert roots == [paths.root, paths.fallback]  # never a third attempt
    assert "vllm-cache" in handler.startup_error


def test_out_of_disk_is_relaunched_before_oom(paths, run_main, monkeypatch):
    # The OOM relaunch sets ENFORCE_EAGER, which also turns torch.compile off;
    # moving the cache keeps compile and CUDA graphs, so it must go first.
    plenty_free(monkeypatch)
    monkeypatch.setenv("VLLM_CACHE_ROOT", paths.root)
    monkeypatch.setenv("XDG_CACHE_HOME", paths.xdg)
    # setenv (not delenv) so teardown also undoes a write by the OOM path.
    monkeypatch.setenv("ENFORCE_EAGER", "")
    monkeypatch.setenv("MAX_NUM_BATCHED_TOKENS", "")

    roots, handler = run_main(NO_SPACE + "\ntorch.OutOfMemoryError: CUDA out of memory.", None)

    assert roots == [paths.root, paths.fallback]
    assert main.os.environ["ENFORCE_EAGER"] == ""
    assert handler.startup_error is None
