import os
from dotenv import load_dotenv
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
        args = {
            "model": self.model_name_or_path,
            "revision": self.model_revision,
            "download_dir": self.hf_home,
            "quantization": self.quantization,
            "load_format": os.getenv("LOAD_FORMAT", "auto"),
            "dtype": os.getenv("DTYPE", "half" if self.quantization else "auto"),
            "tokenizer": self.tokenizer_name_or_path,
            "tokenizer_revision": self.tokenizer_revision,
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 1))),
            "disable_log_requests": bool(int(os.getenv("DISABLE_LOG_REQUESTS", 1))),
            "trust_remote_code": bool(int(os.getenv("TRUST_REMOTE_CODE", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": None if device_count() > 1 or not os.getenv("MAX_PARALLEL_LOADING_WORKERS") else int(os.getenv("MAX_PARALLEL_LOADING_WORKERS")),
            "max_model_len": int(os.getenv("MAX_MODEL_LENGTH")) if os.getenv("MAX_MODEL_LENGTH") else None,
            "tensor_parallel_size": device_count(),
            "seed": int(os.getenv("SEED")) if os.getenv("SEED") else None,
            "kv_cache_dtype": os.getenv("KV_CACHE_DTYPE"),
            "block_size": int(os.getenv("BLOCK_SIZE")) if os.getenv("BLOCK_SIZE") else None,
            "swap_space": int(os.getenv("SWAP_SPACE")) if os.getenv("SWAP_SPACE") else None,
            "max_context_len_to_capture": int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE")) if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE") else None,
            "disable_custom_all_reduce": bool(int(os.getenv("DISABLE_CUSTOM_ALL_REDUCE", 0))),
            "enforce_eager": bool(int(os.getenv("ENFORCE_EAGER", 0)))
        }
        
        return {k: v for k, v in args.items() if v is not None}