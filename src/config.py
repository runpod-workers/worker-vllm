import os
import json
import logging
from dotenv import load_dotenv
from torch.cuda import device_count
from utils import get_int_bool_env

class EngineConfig:
    def __init__(self):
        load_dotenv()
        self.hf_home = os.getenv("HF_HOME")
        # Check if /local_metadata.json exists
        local_metadata = {}
        if os.path.exists("/local_metadata.json"):
            with open("/local_metadata.json", "r") as f:
                local_metadata = json.load(f)
            if local_metadata.get("model_name") is None:
                raise ValueError("Model name is not found in /local_metadata.json, there was a problem when you baked the model in.")
            logging.info("Using baked-in model")
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
        self.model_name_or_path = local_metadata.get("model_name", os.getenv("MODEL_NAME"))
        self.model_revision = local_metadata.get("revision", os.getenv("MODEL_REVISION")) 
        self.tokenizer_name_or_path = local_metadata.get("tokenizer_name", os.getenv("TOKENIZER_NAME")) or self.model_name_or_path
        self.tokenizer_revision = local_metadata.get("tokenizer_revision", os.getenv("TOKENIZER_REVISION"))  
        self.quantization = local_metadata.get("quantization", os.getenv("QUANTIZATION"))
        self.config = self._initialize_config()
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
            "disable_log_stats": get_int_bool_env("DISABLE_LOG_STATS", True),
            "disable_log_requests": get_int_bool_env("DISABLE_LOG_REQUESTS", True),
            "trust_remote_code": get_int_bool_env("TRUST_REMOTE_CODE", False),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": None if device_count() > 1 or not os.getenv("MAX_PARALLEL_LOADING_WORKERS") else int(os.getenv("MAX_PARALLEL_LOADING_WORKERS")),
            "max_model_len": int(os.getenv("MAX_MODEL_LEN")) if os.getenv("MAX_MODEL_LEN") else None,
            "tensor_parallel_size": device_count(),
            "seed": int(os.getenv("SEED")) if os.getenv("SEED") else None,
            "kv_cache_dtype": os.getenv("KV_CACHE_DTYPE"),
            "block_size": int(os.getenv("BLOCK_SIZE")) if os.getenv("BLOCK_SIZE") else None,
            "swap_space": int(os.getenv("SWAP_SPACE")) if os.getenv("SWAP_SPACE") else None,
            "max_seq_len_to_capture": int(os.getenv("MAX_SEQ_LEN_TO_CAPTURE")) if os.getenv("MAX_SEQ_LEN_TO_CAPTURE") else None,
            "disable_custom_all_reduce": get_int_bool_env("DISABLE_CUSTOM_ALL_REDUCE", False),
            "enforce_eager": get_int_bool_env("ENFORCE_EAGER", False)
        }
        if args["kv_cache_dtype"] == "fp8_e5m2":
            args["kv_cache_dtype"] = "fp8"
            logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
        if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"):
            args["max_seq_len_to_capture"] = int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"))
            logging.warning("Using MAX_CONTEXT_LEN_TO_CAPTURE is deprecated. Please use MAX_SEQ_LEN_TO_CAPTURE instead.")
            
            
        return {k: v for k, v in args.items() if v not in [None, ""]}
