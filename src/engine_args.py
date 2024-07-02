import os
import json
import logging
from torch.cuda import device_count
from vllm import AsyncEngineArgs

env_to_args_map = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
    "TOKENIZER_REVISION": "tokenizer_revision",
    "QUANTIZATION": "quantization"
}
    
def get_local_args():
    if os.path.exists("/local_metadata.json"):
        with open("/local_metadata.json", "r") as f:
            local_metadata = json.load(f)
        if local_metadata.get("model_name") is None:
            raise ValueError("Model name is not found in /local_metadata.json, there was a problem when baking the model in.")
        else:
            local_args = {env_to_args_map[k.upper()]: v for k, v in local_metadata.items() if k in env_to_args_map}
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
    return local_args

def get_engine_args():
    # Start with default args
    args = {
        "disable_log_stats": True,
        "disable_log_requests": True,
        "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.9)),
    }
    
    # Get env args that match keys in AsyncEngineArgs
    env_args = {k.lower(): v for k, v in dict(os.environ).items() if k.lower() in AsyncEngineArgs.__dataclass_fields__}
    args.update(env_args)
    
    # Get local args if model is baked in and overwrite env args
    local_args = get_local_args()
    args.update(local_args)
    
    # Set tensor parallel size and max parallel loading workers if more than 1 GPU is available
    num_gpus = device_count()
    if num_gpus > 1:
        args["tensor_parallel_size"] = num_gpus
        args["max_parallel_loading_workers"] = None
        if os.getenv("MAX_PARALLEL_LOADING_WORKERS"):
            logging.warning("Overriding MAX_PARALLEL_LOADING_WORKERS with None because more than 1 GPU is available.")
    
    # Deprecated env args backwards compatibility
    if args["kv_cache_dtype"] == "fp8_e5m2":
        args["kv_cache_dtype"] = "fp8"
        logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
    if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"):
        args["max_seq_len_to_capture"] = int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"))
        logging.warning("Using MAX_CONTEXT_LEN_TO_CAPTURE is deprecated. Please use MAX_SEQ_LEN_TO_CAPTURE instead.")
    return AsyncEngineArgs(**args)
