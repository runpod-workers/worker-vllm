from typing import Union

DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_CONCURRENCY = 300

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
}
