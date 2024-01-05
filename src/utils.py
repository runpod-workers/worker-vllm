import os
import logging
from typing import Any, Dict
from vllm import SamplingParams
from vllm.utils import random_uuid
from constants import sampling_param_types, DEFAULT_BATCH_SIZE, DEFAULT_MAX_CONCURRENCY

logging.basicConfig(level=logging.INFO)


class ServerlessConfig:
    def __init__(self):
        self._max_concurrency = int(
            os.environ.get("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY)
        )
        self._default_batch_size = int(
            os.environ.get("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        )

    @property
    def max_concurrency(self):
        return self._max_concurrency

    @property
    def default_batch_size(self):
        return self._default_batch_size


def validate_sampling_params(params: Dict[str, Any]) -> SamplingParams:
    validated_params = {}

    for key, value in params.items():
        expected_type = sampling_param_types.get(key)
        if value is None:
            validated_params[key] = None
            continue

        if expected_type is None:
            continue

        if isinstance(expected_type, tuple):
            casted_value = next(
                (t(value) for t in expected_type if isinstance(value, t)), None
            )
        else:
            casted_value = value if isinstance(value, expected_type) else None

        if casted_value is not None:
            validated_params[key] = casted_value

    return SamplingParams(**validated_params)
