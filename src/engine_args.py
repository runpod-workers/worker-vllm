import os
import json
import logging
from typing import get_origin, get_args
from torch.cuda import device_count
from vllm import AsyncEngineArgs
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from src.utils import convert_limit_mm_per_prompt

# Backward-compat: env var names users already know → engine arg name
ENV_ALIASES = {
    "MODEL_NAME": "model",
    "MODEL_REVISION": "revision",
    "TOKENIZER_NAME": "tokenizer",
}

# Literal defaults from original worker (used when env/local do not set a value)
DEFAULT_ARGS = {
    "disable_log_stats": False,
    "enable_log_requests": False,
    "gpu_memory_utilization": 0.95,
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": 1,
    "skip_tokenizer_init": False,
    "tokenizer_mode": "auto",
    "trust_remote_code": False,
    "load_format": "auto",
    "dtype": "auto",
    "kv_cache_dtype": "auto",
    "seed": 0,
    "worker_use_ray": False,
    "block_size": 16,
    "enable_prefix_caching": False,
    "disable_sliding_window": False,
    "swap_space": 4,
    "cpu_offload_gb": 0,
    "max_num_seqs": 256,
    "max_logprobs": 20,
    "enforce_eager": False,
    "max_seq_len_to_capture": 8192,
    "disable_custom_all_reduce": False,
    "tokenizer_pool_size": 0,
    "tokenizer_pool_type": "ray",
    "enable_lora": False,
    "max_loras": 1,
    "max_lora_rank": 16,
    "enable_prompt_adapter": False,
    "max_prompt_adapters": 1,
    "max_prompt_adapter_token": 0,
    "fully_sharded_loras": False,
    "lora_extra_vocab_size": 256,
    "lora_dtype": "auto",
    "device": "auto",
    "ray_workers_use_nsight": False,
    "num_lookahead_slots": 0,
    "scheduler_delay_factor": 0.0,
    "guided_decoding_backend": "outlines",
    "spec_decoding_acceptance_method": "rejection_sampler",
    "stream_interval": 1,

}


def _resolve_field_type(field_type: type) -> type:
    """Resolve Optional/Union to the concrete type for conversion."""
    origin = get_origin(field_type)
    args = get_args(field_type) if hasattr(field_type, "__args__") else ()
    if origin is not None:
        # Optional[X] is Union[X, None]; X | None is UnionType
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return non_none[0]
    return field_type


def _convert_env_value_to_field_type(value: str, field_name: str, field_type: type):
    """Convert env var string to the type expected by AsyncEngineArgs for this field."""
    val = value.strip() if isinstance(value, str) else value
    if val in ("", "None", "none"):
        args = get_args(field_type) if hasattr(field_type, "__args__") else ()
        if type(None) in (args or ()):
            return None
        raise ValueError("empty value not allowed for non-optional field")
    effective_type = _resolve_field_type(field_type)
    # bool
    if effective_type is bool:
        return str(val).lower() in ("true", "1", "yes", "on")
    # int
    if effective_type is int:
        return int(val)
    # float
    if effective_type is float:
        return float(val)
    # str
    if effective_type is str:
        return str(val)
    # dict, list, or complex (try JSON)
    origin = get_origin(effective_type)
    if effective_type in (dict, list) or origin in (dict, list):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    # tuple (e.g. long_lora_scaling_factors) — comma-separated or JSON array
    if effective_type is tuple or origin is tuple:
        args = get_args(field_type) if hasattr(field_type, "__args__") else ()
        elem_types = [a for a in args if a is not Ellipsis]
        elem_type = elem_types[0] if elem_types else str
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return tuple(elem_type(x) for x in parsed)
        except (json.JSONDecodeError, TypeError):
            pass
        return tuple(elem_type(x.strip()) for x in str(val).split(",") if x.strip())
    # Fallback: try int, float, then str
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return str(val)


def _get_args_from_env_auto_discover() -> dict:
    """Auto-discover engine args from env vars using UPPERCASED field names.

    For every field in AsyncEngineArgs, check os.getenv(FIELD_NAME).
    E.g. MAX_MODEL_LEN=4096 -> max_model_len=4096.
    Uses same type conversion as before; supports all vLLM engine args without manual listing.
    """
    args = {}
    valid_fields = AsyncEngineArgs.__dataclass_fields__
    for field_name, field in valid_fields.items():
        env_key = field_name.upper()
        value = os.environ.get(env_key)
        if value is None:
            continue
        try:
            args[field_name] = _convert_env_value_to_field_type(
                value, field_name, field.type
            )
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logging.warning(
                "Skip env %s=%r: %s", env_key, value, e
            )
    return args


def _apply_env_aliases(args: dict) -> None:
    """Apply ENV_ALIASES: if MODEL_NAME etc. are set, set the target engine arg."""
    valid_fields = AsyncEngineArgs.__dataclass_fields__
    for alias, target in ENV_ALIASES.items():
        value = os.environ.get(alias)
        if value is None or target not in valid_fields:
            continue
        try:
            args[target] = _convert_env_value_to_field_type(
                value, target, valid_fields[target].type
            )
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logging.warning("Skip env alias %s=%r: %s", alias, value, e)

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


def _local_args_to_engine_args(local: dict) -> dict:
    """Map local args (e.g. from /local_model_args.json) to engine arg names and filter."""
    valid = AsyncEngineArgs.__dataclass_fields__
    out = {}
    for k, v in local.items():
        target = ENV_ALIASES.get(k, k.lower().replace("-", "_"))
        if target not in valid or v in (None, "", "None"):
            continue
        out[target] = v
    return out


def _sanitize_hf_overrides(hf_overrides: dict) -> dict | None:
    """Strip rope_scaling from hf_overrides sub-configs if vLLM rejects them.

    Older vLLM (<0.7) required explicit mrope rope_scaling in hf_overrides for
    models like Qwen2-VL. Newer vLLM auto-detects mrope and raises a ValueError
    in patch_rope_scaling_dict when it finds conflicting rope_type values. Strip
    the offending rope_scaling so the model loads with its native config.
    """
    if not isinstance(hf_overrides, dict):
        return hf_overrides

    try:
        from vllm.transformers_utils.config import patch_rope_scaling_dict
    except ImportError:
        return hf_overrides

    import copy
    cleaned = {}
    changed = False
    for key, value in hf_overrides.items():
        if isinstance(value, dict) and "rope_scaling" in value:
            rope_scaling = value.get("rope_scaling")
            if isinstance(rope_scaling, dict):
                try:
                    patch_rope_scaling_dict(copy.deepcopy(rope_scaling))
                except (ValueError, Exception) as e:
                    logging.warning(
                        "Stripping hf_overrides['%s']['rope_scaling'] because vLLM "
                        "rejected it (%s). Newer vLLM auto-detects rope scaling from "
                        "the model config.", key, e
                    )
                    stripped = {k: v for k, v in value.items() if k != "rope_scaling"}
                    cleaned[key] = stripped if stripped else None
                    changed = True
                    continue
        cleaned[key] = value

    if not changed:
        return hf_overrides

    result = {k: v for k, v in cleaned.items() if v is not None}
    return result or None


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
    # Start with worker custom defaults (only where we differ from vLLM)
    args = dict(DEFAULT_ARGS)

    # Auto-discover: every AsyncEngineArgs field from env UPPERCASED (e.g. MAX_MODEL_LEN)
    args.update(_get_args_from_env_auto_discover())

    # Backward-compat aliases (MODEL_NAME → model, etc.)
    _apply_env_aliases(args)

    # Local baked-in model overrides
    local = get_local_args()
    if local:
        args.update(_local_args_to_engine_args(local))

    # Filter to valid engine args and drop sentinel empty values
    valid_fields = AsyncEngineArgs.__dataclass_fields__
    args = {
        k: v for k, v in args.items()
        if k in valid_fields and v not in (None, "", "None")
    }

    # Special conversion for limit_mm_per_prompt (e.g. "image=1,video=0")
    limit_mm_env = os.getenv("LIMIT_MM_PER_PROMPT")
    if limit_mm_env is not None:
        args["limit_mm_per_prompt"] = convert_limit_mm_per_prompt(limit_mm_env)

    # if args.get("TENSORIZER_URI"): TODO: add back once tensorizer is ready
    #     args["load_format"] = "tensorizer"
    #     args["model_loader_extra_config"] = TensorizerConfig(tensorizer_uri=args["TENSORIZER_URI"], num_readers=None)
    #     logging.info(f"Using tensorized model from {args['TENSORIZER_URI']}")

    if "hf_overrides" in args:
        sanitized = _sanitize_hf_overrides(args["hf_overrides"])
        if sanitized:
            args["hf_overrides"] = sanitized
        else:
            del args["hf_overrides"]

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
