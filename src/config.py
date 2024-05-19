import os
from dotenv import load_dotenv
from torch.cuda import device_count
import logging

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
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            with open(local_path, "r") as file:
                return file.read().strip(), None, None
        return os.getenv(env_var), os.getenv("HF_HOME"), os.getenv(f"{env_var.split('_')[0]}_REVISION") or None

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
            "max_model_len": int(os.getenv("MAX_MODEL_LEN")) if os.getenv("MAX_MODEL_LEN") else None,
            "tensor_parallel_size": device_count(),
            "seed": int(os.getenv("SEED")) if os.getenv("SEED") else None,
            "kv_cache_dtype": os.getenv("KV_CACHE_DTYPE"),
            "block_size": int(os.getenv("BLOCK_SIZE")) if os.getenv("BLOCK_SIZE") else None,
            "swap_space": int(os.getenv("SWAP_SPACE")) if os.getenv("SWAP_SPACE") else None,
            "max_context_len_to_capture": int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE")) if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE") else None,
            "disable_custom_all_reduce": bool(int(os.getenv("DISABLE_CUSTOM_ALL_REDUCE", 0))),
            "enforce_eager": bool(int(os.getenv("ENFORCE_EAGER", 0))),
            "image_input_type": os.getenv("IMAGE_INPUT_TYPE", "pixel_values"),
            "image_token_id": int(os.getenv("IMAGE_TOKEN_ID")) if os.getenv("IMAGE_TOKEN_ID") else None,
            "image_input_shape": os.getenv("IMAGE_INPUT_SHAPE"),
            "image_feature_size": int(os.getenv("IMAGE_FEATURE_SIZE")) if os.getenv("IMAGE_FEATURE_SIZE") else None
        }
        if args["kv_cache_dtype"] == "fp8_e5m2":
            args["kv_cache_dtype"] = "fp8"
            logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
        return {k: v for k, v in args.items() if v is not None}
