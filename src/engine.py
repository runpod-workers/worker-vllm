import os
import logging
from typing import Union
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer
from utils import ServerlessConfig


class Tokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.has_chat_template = bool(self.tokenizer.chat_template)

    def apply_chat_template(self, input: Union[str, list[dict[str, str]]]) -> str:
        if isinstance(input, list):
            if not self.has_chat_template:
                raise ValueError(
                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                )
        elif isinstance(input, str):
            input = [{"role": "user", "content": input}]
        else:
            raise ValueError("Input must be a string or a list of messages")
        
        return self.tokenizer.apply_chat_template(
            input, tokenize=False, add_generation_prompt=True
        )


class VLLMEngine:
    def __init__(self):
        self.config = self._initialize_config()
        self.serverless_config = ServerlessConfig()
        self.tokenizer = Tokenizer(self.config["model"])
        self.llm = self._initialize_llm()

    def _initialize_config(self):
        return {
            "model": os.getenv("MODEL_NAME", "default_model"),
            "download_dir": os.getenv("MODEL_BASE_PATH", "/runpod-volume/"),
            "quantization": os.getenv("QUANTIZATION"),
            "dtype": "auto" if os.getenv("QUANTIZATION") is None else "half",
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.98)),
            "tensor_parallel_size": self._get_num_gpu_shard(),
        }

    def _initialize_llm(self):
        try:
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.config))
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e
        
    def _get_num_gpu_shard(self):
        final_num_gpu_shard = 1
        if bool(int(os.getenv("USE_TENSOR_PARALLEL", 0))):
            env_num_gpu_shard = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
            num_gpu_available = torch.cuda.device_count()
            final_num_gpu_shard = min(env_num_gpu_shard, num_gpu_available)
            logging.info("Using %s GPU shards", final_num_gpu_shard)
        return final_num_gpu_shard
    
    def _get_n_current_jobs(self):
        total_sequences = len(self.llm.engine.scheduler.waiting) + len(self.llm.engine.scheduler.swapped) + len(self.llm.engine.scheduler.running)
        return total_sequences
    
    def concurrency_modifier(self, current_concurrency):
        n_current_jobs = self._get_n_current_jobs()
        requested_concurrency = max(0, self.serverless_config.max_concurrency - n_current_jobs)
        if not self.config["disable_log_stats"]:
            logging.info("Current Jobs: %s", n_current_jobs)
            logging.info("Concurrency Modifier Requested Jobs: %s", requested_concurrency)
        return requested_concurrency

