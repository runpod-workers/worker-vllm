DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_CONCURRENCY = 300

sampling_param_types = {
    "n": int,
    "best_of": int,
    "presence_penalty": float,
    "frequency_penalty": float,
    "repetition_penalty": float,
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "use_beam_search": bool,
    "length_penalty": float,
    "early_stopping": (bool, str),
    "stop": (str, list),
    "stop_token_ids": list,
    "ignore_eos": bool,
    "max_tokens": int,
    "logprobs": int,
    "prompt_logprobs": int,
    "skip_special_tokens": bool,
    "spaces_between_special_tokens": bool,
}
