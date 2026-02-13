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
    "distributed_executor_backend": os.getenv('DISTRIBUTED_EXECUTOR_BACKEND', None),
    "max_parallel_loading_workers": int(os.getenv('MAX_PARALLEL_LOADING_WORKERS', 0)) or None,
    "block_size": int(os.getenv('BLOCK_SIZE', 16)),
    "enable_prefix_caching": os.getenv('ENABLE_PREFIX_CACHING', 'False').lower() == 'true',
    "disable_sliding_window": os.getenv('DISABLE_SLIDING_WINDOW', 'False').lower() == 'true',
    "swap_space": int(os.getenv('SWAP_SPACE', 4)),  # GiB
    "cpu_offload_gb": int(os.getenv('CPU_OFFLOAD_GB', 0)),  # GiB
    # vLLM defaults None to 2048; keep 0 as None to let vLLM auto-calculate
    "max_num_batched_tokens": int(os.getenv('MAX_NUM_BATCHED_TOKENS', 0)) or None,
    "max_num_seqs": int(os.getenv('MAX_NUM_SEQS', 256)),
    "max_logprobs": int(os.getenv('MAX_LOGPROBS', 20)),  # Default value for OpenAI Chat Completions API
    "revision": os.getenv('REVISION', None),
    "code_revision": os.getenv('CODE_REVISION', None),
    "tokenizer_revision": os.getenv('TOKENIZER_REVISION', None),
    "quantization": os.getenv('QUANTIZATION', None),
    "enforce_eager": os.getenv('ENFORCE_EAGER', 'False').lower() == 'true',
    "max_context_len_to_capture": int(os.getenv('MAX_CONTEXT_LEN_TO_CAPTURE', 0)) or None,
    "max_seq_len_to_capture": int(os.getenv('MAX_SEQ_LEN_TO_CAPTURE', 8192)),
    "disable_custom_all_reduce": os.getenv('DISABLE_CUSTOM_ALL_REDUCE', 'False').lower() == 'true',
    "enable_lora": os.getenv('ENABLE_LORA', 'False').lower() == 'true',
    "max_loras": int(os.getenv('MAX_LORAS', 1)),
    "max_lora_rank": int(os.getenv('MAX_LORA_RANK', 16)),
    "enable_prompt_adapter": os.getenv('ENABLE_PROMPT_ADAPTER', 'False').lower() == 'true',
    "max_prompt_adapters": int(os.getenv('MAX_PROMPT_ADAPTERS', 1)),
    "max_prompt_adapter_token": int(os.getenv('MAX_PROMPT_ADAPTER_TOKEN', 0)),
    "fully_sharded_loras": os.getenv('FULLY_SHARDED_LORAS', 'False').lower() == 'true',
    "lora_dtype": os.getenv('LORA_DTYPE', 'auto'),
    "max_cpu_loras": int(os.getenv('MAX_CPU_LORAS', 0)) or None,
    "device": os.getenv('DEVICE', 'auto'),
    "ray_workers_use_nsight": os.getenv('RAY_WORKERS_USE_NSIGHT', 'False').lower() == 'true',
    "num_gpu_blocks_override": int(os.getenv('NUM_GPU_BLOCKS_OVERRIDE', 0)) or None,
    "model_loader_extra_config": os.getenv('MODEL_LOADER_EXTRA_CONFIG', None),
    "ignore_patterns": os.getenv('IGNORE_PATTERNS', None),
    "preemption_mode": os.getenv('PREEMPTION_MODE', None),
    "scheduler_delay_factor": float(os.getenv('SCHEDULER_DELAY_FACTOR', 0.0)),
    "enable_chunked_prefill": os.getenv('ENABLE_CHUNKED_PREFILL', None),
    "guided_decoding_backend": os.getenv('GUIDED_DECODING_BACKEND', 'outlines'),
    "enable_expert_parallel": bool(os.getenv('ENABLE_EXPERT_PARALLEL', 'False').lower() == 'true'),
    "qlora_adapter_name_or_path": os.getenv('QLORA_ADAPTER_NAME_OR_PATH', None),
    "otlp_traces_endpoint": os.getenv('OTLP_TRACES_ENDPOINT', None),
    "attention_backend": os.getenv('ATTENTION_BACKEND', None),
    "async_scheduling": os.getenv('ASYNC_SCHEDULING', 'False').lower() == 'true',
    "stream_interval": float(os.getenv('STREAM_INTERVAL', 0)),
}


def get_speculative_config():
    """
    Build speculative decoding configuration from environment variables.

    Supports two modes:
    1. Full JSON config via SPECULATIVE_CONFIG env var
    2. Individual env vars for common settings

    Speculative Methods:
    - "draft_model": Use a smaller draft model for speculation
    - "ngram": Use n-gram based prompt lookup (no additional model needed)
    - "eagle" / "eagle3": Use EAGLE-based speculation
    - "medusa": Use Medusa heads for speculation
    - "mlp_speculator": Use MLP-based speculator

    Returns:
        dict | None: Speculative config dictionary or None if not configured
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
    spec_method = os.getenv('SPECULATIVE_METHOD')  # ngram, draft_model, eagle, eagle3, medusa, mlp_speculator
    spec_model = os.getenv('SPECULATIVE_MODEL')
    num_spec_tokens = os.getenv('NUM_SPECULATIVE_TOKENS')

    # N-gram specific settings
    ngram_max = os.getenv('NGRAM_PROMPT_LOOKUP_MAX')
    ngram_min = os.getenv('NGRAM_PROMPT_LOOKUP_MIN')

    # Check if any speculative decoding is configured
    if not any([spec_method, spec_model, ngram_max]):
        return None

    config = {}

    # Determine method
    if spec_method:
        config['method'] = spec_method
    elif ngram_max and not spec_model:
        config['method'] = 'ngram'
    elif spec_model:
        # Auto-detect method based on model name if not specified
        model_lower = spec_model.lower()
        if 'eagle3' in model_lower:
            config['method'] = 'eagle3'
        elif 'eagle' in model_lower:
            config['method'] = 'eagle'
        elif 'medusa' in model_lower:
            config['method'] = 'medusa'
        else:
            config['method'] = 'draft_model'

    # Model configuration
    if spec_model:
        config['model'] = spec_model

    # Number of speculative tokens
    if num_spec_tokens:
        config['num_speculative_tokens'] = int(num_spec_tokens)

    # N-gram settings
    if ngram_max:
        config['prompt_lookup_max'] = int(ngram_max)
    if ngram_min:
        config['prompt_lookup_min'] = int(ngram_min)

    # Draft model tensor parallel size
    draft_tp = os.getenv('SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE')
    if draft_tp:
        config['draft_tensor_parallel_size'] = int(draft_tp)

    # Max model length for draft
    spec_max_len = os.getenv('SPECULATIVE_MAX_MODEL_LEN')
    if spec_max_len:
        config['max_model_len'] = int(spec_max_len)

    # Disable by batch size
    disable_batch = os.getenv('SPECULATIVE_DISABLE_BY_BATCH_SIZE')
    if disable_batch:
        config['disable_by_batch_size'] = int(disable_batch)

    # Draft model quantization
    spec_quant = os.getenv('SPECULATIVE_QUANTIZATION')
    if spec_quant:
        config['quantization'] = spec_quant

    # Draft model revision
    spec_revision = os.getenv('SPECULATIVE_MODEL_REVISION')
    if spec_revision:
        config['revision'] = spec_revision

    # Enforce eager mode for draft model
    spec_eager = os.getenv('SPECULATIVE_ENFORCE_EAGER')
    if spec_eager:
        config['enforce_eager'] = spec_eager.lower() == 'true'

    if config:
        logging.info(f"Built speculative config from env vars: {config}")
        return config

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

    # VLLM_ATTENTION_BACKEND env var → attention_backend arg (deprecated shim)
    vllm_attn_backend = os.getenv("VLLM_ATTENTION_BACKEND")
    if vllm_attn_backend and "attention_backend" not in args:
        args["attention_backend"] = vllm_attn_backend
        logging.warning("VLLM_ATTENTION_BACKEND is deprecated. Please use ATTENTION_BACKEND instead.")

    # DISABLE_LOG_REQUESTS → enable_log_requests (inverted, deprecated shim)
    if os.getenv("DISABLE_LOG_REQUESTS") and "enable_log_requests" not in args:
        args["enable_log_requests"] = os.getenv("DISABLE_LOG_REQUESTS", "False").lower() != "true"
        logging.warning("DISABLE_LOG_REQUESTS is deprecated. Please use ENABLE_LOG_REQUESTS instead.")

    # Default max_num_batched_tokens to max_model_len when not explicitly set
    if args.get("max_num_batched_tokens") is None and args.get("max_model_len") is not None:
        args["max_num_batched_tokens"] = args["max_model_len"]

    # Add speculative decoding configuration if present
    speculative_config = get_speculative_config()
    if speculative_config:
        args["speculative_config"] = speculative_config

    return AsyncEngineArgs(**args)
