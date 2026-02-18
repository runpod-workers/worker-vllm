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
    # disable_log_requests is deprecated, use enable_log_requests instead
    "enable_log_requests": os.getenv('ENABLE_LOG_REQUESTS', 'False').lower() == 'true',
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
    # attention_backend replaces deprecated VLLM_ATTENTION_BACKEND env var
    "attention_backend": os.getenv('ATTENTION_BACKEND', None),
    # Enabled by default for improved throughput. Set to False to disable if experiencing issues
    "async_scheduling": None if os.getenv('ASYNC_SCHEDULING') is None else os.getenv('ASYNC_SCHEDULING', 'True').lower() == 'true',
    # Controls how often to yield streaming results
    "stream_interval": int(os.getenv('STREAM_INTERVAL', 1)),
    "swap_space": int(os.getenv('SWAP_SPACE', 4)),  # GiB
    "cpu_offload_gb": int(os.getenv('CPU_OFFLOAD_GB', 0)),  # GiB
    # vLLM defaults None to 2048; keep 0 as None to let vLLM auto-calculate
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
}

def get_speculative_config():
    """Build speculative decoding configuration from environment variables.

    Supports two modes:
    1. Full JSON config via SPECULATIVE_CONFIG env var
    2. Individual env vars for common settings
    """
    # Option 1: Full JSON configuration
    spec_config_json = os.getenv('SPECULATIVE_CONFIG')
    if spec_config_json:
        try:
            config = json.loads(spec_config_json)
            logging.info(f"Using speculative config from SPECULATIVE_CONFIG: {config}")
            return config
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse SPECULATIVE_CONFIG JSON: {e}")
            return None

    # Option 2: Build config from individual environment variables
    spec_method = os.getenv('SPECULATIVE_METHOD')
    spec_model = os.getenv('SPECULATIVE_MODEL')
    _num_spec_tokens = os.getenv('NUM_SPECULATIVE_TOKENS')
    _ngram_max = os.getenv('NGRAM_PROMPT_LOOKUP_MAX')
    _ngram_min = os.getenv('NGRAM_PROMPT_LOOKUP_MIN')

    # Convert numeric vars to int so '0' (hub.json default) is treated as unset
    num_spec_tokens = (int(_num_spec_tokens) or None) if _num_spec_tokens else None
    ngram_max = (int(_ngram_max) or None) if _ngram_max else None
    ngram_min = (int(_ngram_min) or None) if _ngram_min else None

    if not any([spec_method, spec_model, ngram_max]):
        return None

    config = {}

    # Determine method
    if spec_method:
        config['method'] = spec_method
    elif ngram_max and not spec_model:
        config['method'] = 'ngram'
    elif spec_model:
        model_lower = spec_model.lower()
        if 'eagle3' in model_lower:
            config['method'] = 'eagle3'
        elif 'eagle' in model_lower:
            config['method'] = 'eagle'
        elif 'medusa' in model_lower:
            config['method'] = 'medusa'
        else:
            config['method'] = 'draft_model'

    if spec_model:
        config['model'] = spec_model
    if num_spec_tokens:
        config['num_speculative_tokens'] = num_spec_tokens
    if ngram_max:
        config['prompt_lookup_max'] = ngram_max
    if ngram_min:
        config['prompt_lookup_min'] = ngram_min

    draft_tp = os.getenv('SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE')
    if draft_tp:
        config['draft_tensor_parallel_size'] = int(draft_tp)

    spec_max_len = os.getenv('SPECULATIVE_MAX_MODEL_LEN')
    if spec_max_len:
        config['max_model_len'] = int(spec_max_len)

    disable_batch = os.getenv('SPECULATIVE_DISABLE_BY_BATCH_SIZE')
    if disable_batch:
        config['disable_by_batch_size'] = int(disable_batch)

    spec_quant = os.getenv('SPECULATIVE_QUANTIZATION')
    if spec_quant:
        config['quantization'] = spec_quant

    spec_revision = os.getenv('SPECULATIVE_MODEL_REVISION')
    if spec_revision:
        config['revision'] = spec_revision

    spec_eager = os.getenv('SPECULATIVE_ENFORCE_EAGER')
    if spec_eager:
        config['enforce_eager'] = spec_eager.lower() == 'true'

    if config:
        logging.info(f"Built speculative config from env vars: {config}")
        return config

    return None

def _resolve_max_model_len(model, trust_remote_code=False, revision=None):
    """Resolve max_model_len from the model's HuggingFace config."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        for attr in ('max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length'):
            val = getattr(config, attr, None)
            if val is not None:
                logging.info(f"Resolved max_model_len={val} from model config ({attr})")
                return val
    except Exception as e:
        logging.warning(f"Could not resolve max_model_len from model config: {e}")
    return None

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
    
    # Set max_num_batched_tokens to max_model_len for unlimited batching.
    # vLLM defaults max_num_batched_tokens to 2048 when None, which is too low.

    if args.get("max_model_len") == 0:
        args["max_model_len"] = None

    if args.get("max_num_batched_tokens") == 0:
        args["max_num_batched_tokens"] = None

    if args.get("max_num_batched_tokens") is None:
        max_model_len = args.get("max_model_len")
        if max_model_len is None:
            max_model_len = _resolve_max_model_len(
                args.get("model"),
                trust_remote_code=args.get("trust_remote_code", False),
                revision=args.get("revision"),
            )
        if max_model_len is not None:
            args["max_num_batched_tokens"] = max_model_len
            logging.info(f"Setting max_num_batched_tokens to {max_model_len}")
    
    # VLLM_ATTENTION_BACKEND is deprecated, migrate to attention_backend
    if os.getenv('VLLM_ATTENTION_BACKEND'):
        logging.warning(
            "VLLM_ATTENTION_BACKEND env var is deprecated. "
            "Use ATTENTION_BACKEND instead (maps to --attention-backend CLI arg)."
        )
        if not args.get('attention_backend'):
            args['attention_backend'] = os.getenv('VLLM_ATTENTION_BACKEND')

    # DISABLE_LOG_REQUESTS is deprecated, use ENABLE_LOG_REQUESTS instead
    if os.getenv('DISABLE_LOG_REQUESTS'):
        logging.warning(
            "DISABLE_LOG_REQUESTS env var is deprecated. "
            "Use ENABLE_LOG_REQUESTS instead (default: False)."
        )
        # Honor old behavior: if DISABLE_LOG_REQUESTS=true, don't enable logging
        if os.getenv('DISABLE_LOG_REQUESTS', 'False').lower() == 'true':
            args['enable_log_requests'] = False

    # Add speculative decoding configuration if present
    speculative_config = get_speculative_config()
    if speculative_config:
        args["speculative_config"] = speculative_config

    return AsyncEngineArgs(**args)
