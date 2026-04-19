import os
import logging
from vllm import AsyncEngineArgs

def get_engine_args():
    """
    Get hardcoded engine arguments for the specific model.
    """
    # The model path is determined by the download script and stored in /local_model_args.json
    # but we can also use the environment variable set in the Dockerfile.
    model_path = os.getenv("MODEL_NAME", "sakamakismile/Huihui-Qwen3.5-4B-abliterated-NVFP4")
    
    # If /local_model_args.json exists, it means the model was baked in.
    # We should use the path from there as it will be the absolute path to the downloaded files.
    import json
    if os.path.exists("/local_model_args.json"):
        with open("/local_model_args.json", "r") as f:
            local_args = json.load(f)
            model_path = local_args.get("MODEL_NAME", model_path)
            logging.info(f"Using baked-in model at {model_path}")

    args = {
        "model": model_path,
        "kv_cache_dtype": "fp8",
        "max_model_len": 175000,
        "trust_remote_code": False,
        "tokenizer_mode": "auto",
        "load_format": "auto",
        "dtype": "auto",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_num_seqs": 256,
        "max_logprobs": 20,
        "gpu_memory_utilization": 0.95,
        "block_size": 16,
        "swap_space": 4,
        "enforce_eager": False,
        "disable_custom_all_reduce": False,
        "enable_prefix_caching": False,
        "disable_sliding_window": False,
        "seed": 0,
    }

    # Override with environment variables if present (optional, but keeps some flexibility)
    # The user wanted a cut down version, but keeping basic vLLM overrides can be useful.
    # However, to be "cut down", we can just return these.

    # Apply specific overrides mentioned by user
    args["max_num_batched_tokens"] = args["max_model_len"]

    logging.info(f"Final engine args: {args}")
    return AsyncEngineArgs(**args)
