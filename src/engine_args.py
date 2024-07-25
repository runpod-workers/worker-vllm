import os
import json
import logging
from torch.cuda import device_count
from vllm import AsyncEngineArgs
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig

RENAME_ARGS_MAP = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
    "MAX_CONTEXT_LEN_TO_CAPTURE": "max_seq_len_to_capture"
}

DEFAULT_ARGS = {
    "disable_log_stats": True,
    "disable_log_requests": True,
    "gpu_memory_utilization": 0.9,
}

def match_vllm_args(args):
    """Rename args to match vllm by:
    1. Renaming keys to lower case
    2. Renaming keys to match vllm
    3. Filtering args to match vllm's AsyncEngineArgs

    Args:
        args (dict): Dictionary of args

    Returns:
        dict: Dictionary of args with renamed keys
    """
    renamed_args = {RENAME_ARGS_MAP.get(k, k): v for k, v in args.items()}
    matched_args = {k: v for k, v in renamed_args.items() if k in AsyncEngineArgs.__dataclass_fields__}
    return {k: v for k, v in matched_args.items() if v not in [None, ""]}
def get_local_args():
    """
    Retrieve local arguments from a JSON file.

    Returns:
        dict: Local arguments.
    """
    if not os.path.exists("/local_model_args.json"):
        return {}

    with open("/local_model_args.json", "r") as f:
        local_args = json.load(f)

    if local_args.get("MODEL_NAME") is None:
        raise ValueError("Model name not found in /local_model_args.json. There was a problem when baking the model in.")

    logging.info(f"Using baked in model with args: {local_args}")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    return local_args
def get_engine_args():
    # Start with default args
    args = DEFAULT_ARGS
    
    # Get env args that match keys in AsyncEngineArgs
    args.update(os.environ)
    
    # Get local args if model is baked in and overwrite env args
    args.update(get_local_args())
    
    # if args.get("TENSORIZER_URI"): TODO: add back once tensorizer is ready
    #     args["load_format"] = "tensorizer"
    #     args["model_loader_extra_config"] = TensorizerConfig(tensorizer_uri=args["TENSORIZER_URI"], num_readers=None)
    #     logging.info(f"Using tensorized model from {args['TENSORIZER_URI']}")
    
    
    # Rename and match to vllm args
    args = match_vllm_args(args)
    
    # Set tensor parallel size and max parallel loading workers if more than 1 GPU is available
    num_gpus = device_count()
    if num_gpus > 1:
        args["tensor_parallel_size"] = num_gpus
        args["max_parallel_loading_workers"] = None
        if os.getenv("MAX_PARALLEL_LOADING_WORKERS"):
            logging.warning("Overriding MAX_PARALLEL_LOADING_WORKERS with None because more than 1 GPU is available.")
    
    # Deprecated env args backwards compatibility
    if args.get("kv_cache_dtype") == "fp8_e5m2":
        args["kv_cache_dtype"] = "fp8"
        logging.warning("Using fp8_e5m2 is deprecated. Please use fp8 instead.")
    if os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"):
        args["max_seq_len_to_capture"] = int(os.getenv("MAX_CONTEXT_LEN_TO_CAPTURE"))
        logging.warning("Using MAX_CONTEXT_LEN_TO_CAPTURE is deprecated. Please use MAX_SEQ_LEN_TO_CAPTURE instead.")
        
    if "gemma-2" in args.get("model", "").lower():
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
        logging.info("Using FLASHINFER for gemma-2 model.")
        
    return AsyncEngineArgs(**args)
