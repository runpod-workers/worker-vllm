"""
vLLM Worker - OpenAI-compatible vLLM inference engine

Usage:
    from vllm_worker import vLLMEngine, OpenAIvLLMEngine, JobInput
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("vllm-worker")
except PackageNotFoundError:
    # Package is not installed, fallback to reading VERSION file
    from pathlib import Path
    _version_file = Path(__file__).parent.parent / "VERSION"
    __version__ = _version_file.read_text().strip()

# Import main classes for easy access
from .engine import vLLMEngine, OpenAIvLLMEngine
from .utils import JobInput, DummyRequest, BatchSize, create_error_response
from .tokenizer import TokenizerWrapper
from .engine_args import get_engine_args

__all__ = [
    "vLLMEngine",
    "OpenAIvLLMEngine",
    "JobInput",
    "DummyRequest",
    "BatchSize",
    "TokenizerWrapper",
    "get_engine_args",
    "create_error_response",
]
