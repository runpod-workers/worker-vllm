import os
import logging
from typing import Any, Dict
from vllm import SamplingParams
from constants import SAMPLING_PARAM_TYPES, DEFAULT_BATCH_SIZE, DEFAULT_MAX_CONCURRENCY

logging.basicConfig(level=logging.INFO)

class ServerlessConfig:
    def __init__(self):
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        self.batch_size = int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE))

def validate_sampling_params(params: Dict[str, Any]) -> SamplingParams:
    validated_params = {}
    invalid_params = []
    for key, value in params.items():
        expected_type = SAMPLING_PARAM_TYPES.get(key)
        if expected_type and isinstance(value, expected_type):
            validated_params[key] = value
        else:
            invalid_params.append(key)
        
    if len(invalid_params) > 0:
        logging.warning("Ignoring invalid sampling params: %s", invalid_params)
        
    return SamplingParams(**validated_params)