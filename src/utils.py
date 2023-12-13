import os
from typing import Any, Dict, Optional, Union
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from constants import sampling_param_types, DEFAULT_BATCH_SIZE, MAX_CONCURRENCY
import logging

logging.basicConfig(level=logging.INFO)

class ServerlessConfig:
    def __init__(self):
        self._max_concurrency = int(os.environ.get('MAX_CONCURRENCY', DEFAULT_BATCH_SIZE))
        self._default_batch_size = int(os.environ.get('DEFAULT_BATCH_SIZE', MAX_CONCURRENCY))

    @property
    def max_concurrency(self):
        return self._max_concurrency

    @property
    def default_batch_size(self):
        return self._default_batch_size

class EngineConfig:
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'default_model')
        self.tokenizer = os.getenv('TOKENIZER', self.model_name)
        self.model_base_path = os.getenv('MODEL_BASE_PATH', "/runpod-volume/")
        self.num_gpu_shard = int(os.getenv('NUM_GPU_SHARD', 1))
        self.use_full_metrics = os.getenv('USE_FULL_METRICS', 'True') == 'True'
        self.quantization = os.getenv('QUANTIZATION', None)
        self.dtype = "auto" if self.quantization is None else "half"
        self.disable_log_stats = os.getenv('DISABLE_LOG_STATS', 'True') == 'True'
        self.gpu_memory_utilization = float(os.getenv('GPU_MEMORY_UTILIZATION', 0.98))
        os.makedirs(self.model_base_path, exist_ok=True)


def initialize_llm_engine() -> AsyncLLMEngine:
    try:
        config = EngineConfig()
        engine_args = AsyncEngineArgs(
            model=config.model_name,
            download_dir=config.model_base_path,
            tokenizer=config.tokenizer,
            tensor_parallel_size=config.num_gpu_shard,
            dtype=config.dtype,
            disable_log_stats=config.disable_log_stats,
            quantization=config.quantization,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)
    except Exception as e:
        logging.error(f"Error initializing vLLM engine: {e}")
        raise

class JobManager:
    def __init__(self):
        self.total_running_jobs = 0

    def increment_job_count(self):
        self.total_running_jobs += 1

    def decrement_job_count(self):
        self.total_running_jobs -= 1

def validate_and_convert_sampling_params(params: Dict[str, Any]) -> SamplingParams:
    validated_params = {}

    for key, value in params.items():
        expected_type = sampling_param_types.get(key)
        if value is None:
            validated_params[key] = None
            continue
        
        if expected_type is None:
            continue

        if not isinstance(expected_type, tuple):
            expected_type = (expected_type,)

        try:
            validated_params[key] = next(
                casted_value for t in expected_type 
                if (casted_value := t(value)) or True
            )
        except (TypeError, ValueError):
            continue

    return SamplingParams(**validated_params)
