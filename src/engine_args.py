import os
import json
import logging
from torch.cuda import device_count
from vllm import AsyncEngineArgs
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from src.utils import convert_limit_mm_per_prompt

RENAME_ARGS_MAP = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
    "MAX_CONTEXT_LEN_TO_CAPTURE": "max_seq_len_to_capture"
}

DEFAULT_ARGS = {
    "disable_log_stats": os.getenv('DISABLE_LOG_STATS', 'False').lower() == 'true',
    "disable_log_requests": os.getenv('DISABLE_LOG_REQUESTS', 'False').lower() == 'true',
    "gpu_memory_utilization": float(os.getenv('GPU_MEMORY_UTILIZATION', 0.95)),
    "pipeline_parallel_size": int(os.getenv('PIPELINE_PARALLEL_SIZE', 1)),
    "tensor_parallel_size": int(os.getenv('TENSOR_PARALLEL_SIZE', 1)),
    "served_model_name": os.getenv('SERVED_MODEL_NAME', None),
    "tokenizer": os.getenv('TOKENIZER', None),
    "skip_tokenizer_init": os.getenv('SKIP_TOKENIZER_INIT', 'False').lower() == 'true',
    "tokenizer_mode": os.getenv('TOKENIZER_MODE', 'auto'),
    "trust_remote_code": os.getenv('TRUST_REMOTE_CODE', 'False').lower() == 'true',
    "download_dir": os.getenv('DOWNLOAD_DIR', None),
    "load_format": os.getenv('LOAD_FORMAT', 'auto'),
    "config_format": os.getenv('CONFIG_FORMAT', 'auto'),
    "dtype": os.getenv('DTYPE', 'auto'),
    "kv_cache_dtype": os.getenv('KV_CACHE_DTYPE', 'auto'),
    "quantization_param_path": os.getenv('QUANTIZATION_PARAM_PATH', None),
    "seed": int(os.getenv('SEED', 0)),
    "max_model_len": int(os.getenv('MAX_MODEL_LEN', 0)) or None,
    "worker_use_ray": os.getenv('WORKER_USE_RAY', 'False').lower() == 'true',
    "distributed_executor_backend": os.getenv('DISTRIBUTED_EXECUTOR_BACKEND', None),
    "max_parallel_loading_workers": int(os.getenv('MAX_PARALLEL_LOADING_WORKERS', 0)) or None,
    "block_size": int(os.getenv('BLOCK_SIZE', 16)),
    "enable_prefix_caching": os.getenv('ENABLE_PREFIX_CACHING', 'False').lower() == 'true',
    "disable_sliding_window": os.getenv('DISABLE_SLIDING_WINDOW', 'False').lower() == 'true',
    "use_v2_block_manager": os.getenv('USE_V2_BLOCK_MANAGER', 'False').lower() == 'true',
    "swap_space": int(os.getenv('SWAP_SPACE', 4)),  # GiB
    "cpu_offload_gb": int(os.getenv('CPU_OFFLOAD_GB', 0)),  # GiB
    "max_num_batched_tokens": int(os.getenv('MAX_NUM_BATCHED_TOKENS', 0)) or None,
    "max_num_seqs": int(os.getenv('MAX_NUM_SEQS', 256)),
    "max_logprobs": int(os.getenv('MAX_LOGPROBS', 20)),  # Default value for OpenAI Chat Completions API
    "revision": os.getenv('REVISION', None),
    "code_revision": os.getenv('CODE_REVISION', None),
    "rope_scaling": os.getenv('ROPE_SCALING', None),
    "rope_theta": float(os.getenv('ROPE_THETA', 0)) or None,
    "tokenizer_revision": os.getenv('TOKENIZER_REVISION', None),
    "quantization": os.getenv('QUANTIZATION', None),
    "enforce_eager": os.getenv('ENFORCE_EAGER', 'False').lower() == 'true',
    "max_context_len_to_capture": int(os.getenv('MAX_CONTEXT_LEN_TO_CAPTURE', 0)) or None,
    "max_seq_len_to_capture": int(os.getenv('MAX_SEQ_LEN_TO_CAPTURE', 8192)),
    "disable_custom_all_reduce": os.getenv('DISABLE_CUSTOM_ALL_REDUCE', 'False').lower() == 'true',
    "tokenizer_pool_size": int(os.getenv('TOKENIZER_POOL_SIZE', 0)),
    "tokenizer_pool_type": os.getenv('TOKENIZER_POOL_TYPE', 'ray'),
    "tokenizer_pool_extra_config": os.getenv('TOKENIZER_POOL_EXTRA_CONFIG', None),
    "enable_lora": os.getenv('ENABLE_LORA', 'False').lower() == 'true',
    "max_loras": int(os.getenv('MAX_LORAS', 1)),
    "max_lora_rank": int(os.getenv('MAX_LORA_RANK', 16)),
    "enable_prompt_adapter": os.getenv('ENABLE_PROMPT_ADAPTER', 'False').lower() == 'true',
    "max_prompt_adapters": int(os.getenv('MAX_PROMPT_ADAPTERS', 1)),
    "max_prompt_adapter_token": int(os.getenv('MAX_PROMPT_ADAPTER_TOKEN', 0)),
    "fully_sharded_loras": os.getenv('FULLY_SHARDED_LORAS', 'False').lower() == 'true',
    "lora_extra_vocab_size": int(os.getenv('LORA_EXTRA_VOCAB_SIZE', 256)),
    "long_lora_scaling_factors": tuple(map(float, os.getenv('LONG_LORA_SCALING_FACTORS', '').split(','))) if os.getenv('LONG_LORA_SCALING_FACTORS') else None,
    "lora_dtype": os.getenv('LORA_DTYPE', 'auto'),
    "max_cpu_loras": int(os.getenv('MAX_CPU_LORAS', 0)) or None,
    "device": os.getenv('DEVICE', 'auto'),
    "ray_workers_use_nsight": os.getenv('RAY_WORKERS_USE_NSIGHT', 'False').lower() == 'true',
    "num_gpu_blocks_override": int(os.getenv('NUM_GPU_BLOCKS_OVERRIDE', 0)) or None,
    "num_lookahead_slots": int(os.getenv('NUM_LOOKAHEAD_SLOTS', 0)),
    "model_loader_extra_config": os.getenv('MODEL_LOADER_EXTRA_CONFIG', None),
    "ignore_patterns": os.getenv('IGNORE_PATTERNS', None),
    "preemption_mode": os.getenv('PREEMPTION_MODE', None),
    "scheduler_delay_factor": float(os.getenv('SCHEDULER_DELAY_FACTOR', 0.0)),
    "enable_chunked_prefill": os.getenv('ENABLE_CHUNKED_PREFILL', None),
    "guided_decoding_backend": os.getenv('GUIDED_DECODING_BACKEND', 'outlines'),
    "speculative_model": os.getenv('SPECULATIVE_MODEL', None),
    "speculative_draft_tensor_parallel_size": int(os.getenv('SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE', 0)) or None,
    "enable_expert_parallel": bool(os.getenv('ENABLE_EXPERT_PARALLEL', 'False').lower() == 'true'),
    "num_speculative_tokens": int(os.getenv('NUM_SPECULATIVE_TOKENS', 0)) or None,
    "speculative_max_model_len": int(os.getenv('SPECULATIVE_MAX_MODEL_LEN', 0)) or None,
    "speculative_disable_by_batch_size": int(os.getenv('SPECULATIVE_DISABLE_BY_BATCH_SIZE', 0)) or None,
    "ngram_prompt_lookup_max": int(os.getenv('NGRAM_PROMPT_LOOKUP_MAX', 0)) or None,
    "ngram_prompt_lookup_min": int(os.getenv('NGRAM_PROMPT_LOOKUP_MIN', 0)) or None,
    "spec_decoding_acceptance_method": os.getenv('SPEC_DECODING_ACCEPTANCE_METHOD', 'rejection_sampler'),
    "typical_acceptance_sampler_posterior_threshold": float(os.getenv('TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD', 0)) or None,
    "typical_acceptance_sampler_posterior_alpha": float(os.getenv('TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA', 0)) or None,
    "qlora_adapter_name_or_path": os.getenv('QLORA_ADAPTER_NAME_OR_PATH', None),
    "disable_logprobs_during_spec_decoding": os.getenv('DISABLE_LOGPROBS_DURING_SPEC_DECODING', None),
    "otlp_traces_endpoint": os.getenv('OTLP_TRACES_ENDPOINT', None),
    "use_v2_block_manager": os.getenv('USE_V2_BLOCK_MANAGER', 'true'),
}
limit_mm_env = os.getenv('LIMIT_MM_PER_PROMPT')
if limit_mm_env is not None:
    DEFAULT_ARGS["limit_mm_per_prompt"] = convert_limit_mm_per_prompt(limit_mm_env)

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
    return {k: v for k, v in matched_args.items() if v not in [None, "", "None"]}
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
        logging.warning("Model name not found in /local_model_args.json. There maybe was a problem when baking the model in.")

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

    if args.get("load_format") == "bitsandbytes":
        args["quantization"] = args["load_format"]
    
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
        
    # if "gemma-2" in args.get("model", "").lower():
    #     os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    #     logging.info("Using FLASHINFER for gemma-2 model.")
        
    return AsyncEngineArgs(**args)
