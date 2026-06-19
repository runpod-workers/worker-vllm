"""Shared test fixtures.

``src/engine_args.py`` hard-imports ``vllm`` (and a tensorizer submodule) and
``torch.cuda``. Both are only installed inside the GPU Docker image, so when the
tests run on a machine without them we install lightweight stubs. When the real
packages *are* available (e.g. CI inside the worker image) the stubs are skipped
and the real ones are used instead.
"""

import sys
import types
from dataclasses import dataclass
from typing import Optional, Union, List


def _install_torch_stub():
    try:
        import torch  # noqa: F401
        return  # real torch present, nothing to stub
    except Exception:
        pass

    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    # No GPU in the test environment -> 0 devices (skips tensor-parallel setup).
    cuda.device_count = lambda: 0
    torch.cuda = cuda
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = cuda


def _install_vllm_stub():
    try:
        import vllm  # noqa: F401
        return  # real vLLM present, nothing to stub
    except Exception:
        pass

    vllm = types.ModuleType("vllm")

    @dataclass
    class AsyncEngineArgs:
        # Only the fields the worker actually sets/reads need to exist here;
        # get_engine_args() filters args down to AsyncEngineArgs.__dataclass_fields__
        # before construction, so unknown keys are dropped rather than passed.
        model: Optional[str] = None
        served_model_name: Optional[Union[str, List[str]]] = None
        revision: Optional[str] = None
        tokenizer: Optional[str] = None
        trust_remote_code: bool = False
        max_model_len: Optional[int] = None
        max_num_batched_tokens: Optional[int] = None
        disable_log_stats: bool = False
        gpu_memory_utilization: float = 0.9
        tensor_parallel_size: int = 1
        max_parallel_loading_workers: Optional[int] = None
        kv_cache_dtype: Optional[str] = None

    vllm.AsyncEngineArgs = AsyncEngineArgs
    sys.modules["vllm"] = vllm

    # vllm.model_executor.model_loader.tensorizer.TensorizerConfig
    model_executor = types.ModuleType("vllm.model_executor")
    model_loader = types.ModuleType("vllm.model_executor.model_loader")
    tensorizer = types.ModuleType("vllm.model_executor.model_loader.tensorizer")

    class TensorizerConfig:  # pragma: no cover - placeholder
        def __init__(self, *args, **kwargs):
            pass

    tensorizer.TensorizerConfig = TensorizerConfig
    model_loader.tensorizer = tensorizer
    model_executor.model_loader = model_loader
    vllm.model_executor = model_executor
    sys.modules["vllm.model_executor"] = model_executor
    sys.modules["vllm.model_executor.model_loader"] = model_loader
    sys.modules["vllm.model_executor.model_loader.tensorizer"] = tensorizer


_install_torch_stub()
_install_vllm_stub()
