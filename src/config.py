import os
from dotenv import load_dotenv
from typing import Tuple, Optional
from utils import count_physical_cores 
from torch.cuda import device_count


class EngineConfig:
    def __init__(self):
        load_dotenv()
        self.model_name_or_path, self.hf_home, self.model_revision = self._get_local_or_env("/local_model_path.txt", "MODEL_NAME")
        self.tokenizer_name_or_path, _, self.tokenizer_revision = self._get_local_or_env("/local_tokenizer_path.txt", "TOKENIZER_NAME")
        self.tokenizer_name_or_path = self.tokenizer_name_or_path or self.model_name_or_path
        self.quantization = self._get_quantization()
        self.config = self._initialize_config()

    def _get_local_or_env(self, local_path, env_var):
        if os.path.exists(local_path):
            with open(local_path, "r") as file:
                return file.read().strip(), None, None
        return os.getenv(env_var), os.getenv("HF_HOME"), os.getenv(f"{env_var}_REVISION")

    def _get_quantization(self):
        quantization = os.getenv("QUANTIZATION", "").lower()
        return quantization if quantization in ["awq", "squeezellm", "gptq"] else None

    def _initialize_config(self):
        return {
            "model": self.model_name_or_path,
            "revision": self.model_revision,
            "download_dir": self.hf_home,
            "quantization": self.quantization,
            "load_format": os.getenv("LOAD_FORMAT", "auto"),
            "dtype": "half" if self.quantization else "auto",
            "tokenizer": self.tokenizer_name_or_path,
            "tokenizer_revision": self.tokenizer_revision,
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 1))),
            "disable_log_requests": bool(int(os.getenv("DISABLE_LOG_REQUESTS", 1))),
            "trust_remote_code": bool(int(os.getenv("TRUST_REMOTE_CODE", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": self._get_max_parallel_loading_workers(),
            "max_model_len": self._get_max_model_len(),
            "tensor_parallel_size": self._get_num_gpu_shard(),
        }

    def _get_max_parallel_loading_workers(self):
        if int(os.getenv("TENSOR_PARALLEL_SIZE", 1)) > 1:
            return None
        return int(os.getenv("MAX_PARALLEL_LOADING_WORKERS", count_physical_cores()))

    def _get_num_gpu_shard(self):
        num_gpu_shard = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        if num_gpu_shard > 1:
            num_gpu_available = device_count()
            num_gpu_shard = min(num_gpu_shard, num_gpu_available)
        return num_gpu_shard

    def _get_max_model_len(self):
        max_model_len = os.getenv("MAX_MODEL_LENGTH")
        return int(max_model_len) if max_model_len else None
