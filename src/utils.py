import os

class EngineConfig:
    def __init__(self, make_dirs=True):
        self.model_name = os.getenv('MODEL_NAME', 'default_model')
        self.tokenizer = os.getenv('TOKENIZER', self.model_name)
        self.model_base_path = os.getenv('MODEL_BASE_PATH', "/runpod-volume/")
        self.num_gpu_shard = int(os.getenv('NUM_GPU_SHARD', 1))
        self.use_full_metrics = os.getenv('USE_FULL_METRICS', 'True') == 'True'
        self.quantization = os.getenv('QUANTIZATION', None)
        self.dtype = "auto" if str(self.quantization).lower() not in ['squeezellm', 'awq'] else "half"
        self.disable_log_stats = os.getenv('DISABLE_LOG_STATS', 'False') == 'True'
        if make_dirs and not os.path.exists(self.model_base_path):
            os.makedirs(self.model_base_path)

# Map of parameter names to their expected types
sampling_param_types = {
    'n': int,
    'best_of': int,
    'presence_penalty': float,
    'frequency_penalty': float,
    'temperature': float,
    'top_p': float,
    'top_k': int,
    'use_beam_search': bool,
    'stop': str,
    'ignore_eos': bool,
    'max_tokens': int,
    'logprobs': float,
}


# Function to convert sampling parameters to the right types
def cast_sampling_param(value, target_type):
    if value is None:
        return None
    try:
        return target_type(value)
    except (TypeError, ValueError):
        return None


# Function to validate and convert sampling parameters
def validate_and_convert_sampling_params(sampling_params):
    validated_params = {}
    for param_name, param_type in sampling_param_types.items():
        param_value = sampling_params.get(param_name)
        if param_value is not None:
            validated_params[param_name] = cast_sampling_param(param_value, param_type)
    return validated_params