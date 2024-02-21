from typing import Union

DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_CONCURRENCY = 300
DEFAULT_BATCH_SIZE_GROWTH_FACTOR = 3
DEFAULT_MIN_BATCH_SIZE = 1

SAMPLING_PARAM_TYPES = {
    "n": int,
    "best_of": int,
    "presence_penalty": float,
    "frequency_penalty": float,
    "repetition_penalty": float,
    "temperature": Union[float, int],
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "use_beam_search": bool,
    "length_penalty": float,
    "early_stopping": Union[bool, str],
    "stop": Union[str, list],
    "stop_token_ids": list,
    "ignore_eos": bool,
    "max_tokens": int,
    "logprobs": int,
    "prompt_logprobs": int,
    "skip_special_tokens": bool,
    "spaces_between_special_tokens": bool,
    "include_stop_str_in_output": bool
}