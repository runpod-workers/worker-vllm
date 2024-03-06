from pydantic import BaseModel
from typing import Union, List, Optional
from vllm import SamplingParams

class InputSamplingParams(BaseModel):
    n: Optional[int]
    best_of: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    repetition_penalty: Optional[float]
    temperature: Optional[Union[float, int]]
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    use_beam_search: Optional[bool]
    length_penalty: Optional[float]
    early_stopping: Optional[Union[bool, str]]
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[List[int]]
    ignore_eos: Optional[bool]
    max_tokens: Optional[int]
    logprobs: Optional[int]
    prompt_logprobs: Optional[int]
    skip_special_tokens: Optional[bool]
    spaces_between_special_tokens: Optional[bool]
    include_stop_str_in_output: Optional[bool]

def validate_sampling_params(params: dict) -> SamplingParams:
    cast_params = InputSamplingParams(**params)
    return SamplingParams(**cast_params.model_dump())

