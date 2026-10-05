"""Keep vLLM's compile cache on the network volume only when the volume can take it.

The image sets VLLM_CACHE_ROOT=$BASE_PATH/vllm-cache so torch.compile output
survives cold starts on a network volume. vLLM does not handle write errors on
most of its compile-cache writes, so a full or read-only volume would fail
startup where compiling into vLLM's default root used to work. Two guards:

- ensure_usable(), before launch: if the directory isn't writable or has less
  than MIN_FREE_BYTES free, use vLLM's default root for this start.
- fall_back(), from main.py when vLLM still runs out of disk mid-start (the
  check runs before the weights download onto the same volume): relaunch once
  with vLLM's default root.

Pure Python, no vLLM import.
"""

import logging
import os
import shutil
import tempfile

# Free space required on the volume before compiling onto it. A heuristic, not
# a measurement: running out later in the start is handled by the relaunch.
MIN_FREE_BYTES = 1 << 30


def vllm_default_root(env=os.environ) -> str:
    """Where vLLM puts its cache when VLLM_CACHE_ROOT is unset (vllm/envs.py)."""
    base = env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "vllm")


def _configured_root(env) -> str | None:
    """VLLM_CACHE_ROOT as vLLM resolves it, or None if unset or already the default."""
    root = env.get("VLLM_CACHE_ROOT")
    if not root:
        return None
    root = os.path.expanduser(root)
    if os.path.realpath(root) == os.path.realpath(vllm_default_root(env)):
        return None
    return root


def _device(path: str) -> int:
    """st_dev of the path, or of its nearest existing ancestor."""
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.stat(path).st_dev


def fall_back(reason: str, env=os.environ, out_of_space: bool = False) -> bool:
    """Point VLLM_CACHE_ROOT at vLLM's default root.

    Returns False (and changes nothing) when that can't help: VLLM_CACHE_ROOT is
    unset or already the default, or, for ``out_of_space``, the default is on
    the same filesystem (no volume attached, so the same disk would run out).
    """
    root = _configured_root(env)
    fallback = vllm_default_root(env)
    if root is None:
        return False
    if out_of_space:
        try:
            if _device(root) == _device(fallback):
                logging.warning(
                    "VLLM_CACHE_ROOT=%s is %s, but the fallback %s is on the same "
                    "disk; not moving the compile cache.",
                    root,
                    reason,
                    fallback,
                )
                return False
        except OSError:
            pass
    logging.warning(
        "VLLM_CACHE_ROOT=%s is %s; compiling into %s for this start instead. "
        "The vllm-cache folder is safe to delete while no worker is starting "
        "(e.g. endpoint scaled to zero); vLLM rebuilds it.",
        root,
        reason,
        fallback,
    )
    env["VLLM_CACHE_ROOT"] = fallback
    return True


def ensure_usable(env=os.environ, min_free_bytes: int | None = None) -> str | None:
    """Fall back to vLLM's default root if VLLM_CACHE_ROOT can't take writes.

    Returns why it fell back, or None when VLLM_CACHE_ROOT is unset, already the
    default, or writable with at least ``min_free_bytes`` (default
    MIN_FREE_BYTES) free.
    """
    if min_free_bytes is None:
        min_free_bytes = MIN_FREE_BYTES
    root = _configured_root(env)
    if root is None:
        return None
    try:
        os.makedirs(root, exist_ok=True)
        # A zero-byte file still fits on a full volume or under an exhausted
        # quota (statvfs may report the storage pool, not the quota); one
        # written and synced block does not.
        with tempfile.NamedTemporaryFile(dir=root, prefix=".write-check-") as probe:
            probe.write(b"\0" * 65536)
            probe.flush()
            os.fsync(probe.fileno())
        free = shutil.disk_usage(root).free
    except OSError as e:
        reason = f"not writable ({e.strerror or e})"
    else:
        if free >= min_free_bytes:
            return None
        reason = f"nearly full ({free >> 20} MiB free)"
        return reason if fall_back(reason, env, out_of_space=True) else None
    return reason if fall_back(reason, env) else None
