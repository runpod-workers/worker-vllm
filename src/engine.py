import os
import logging
from typing import Union, AsyncGenerator
import json
from torch.cuda import device_count
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from transformers import AutoTokenizer
from utils import count_physical_cores
from constants import DEFAULT_MAX_CONCURRENCY
from dotenv import load_dotenv


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


class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        self.config = self._initialize_config()
        logging.info("vLLM config: %s", self.config)
        self.tokenizer = Tokenizer(self.config["model"])
        self.llm = self._initialize_llm() if engine is None else engine
        self.openai_engine = self._initialize_openai()
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))

    async def generate(self, job_input):
        generator_args = job_input.__dict__
        
        if generator_args.pop("use_openai_format"):
            if self.openai_engine is None:
                raise ValueError("OpenAI Chat Completion Format is not enabled for this model")
            generator = self.generate_openai_chat
        else:
            generator = self.generate_vllm        
        
        async for batch in generator(**generator_args):
            yield batch

    async def generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id: str) -> AsyncGenerator[dict, None]:
        
        if apply_chat_template or isinstance(llm_input, list):
            llm_input = self.tokenizer.apply_chat_template(llm_input)
        validated_sampling_params = SamplingParams(**validated_sampling_params)
        results_generator = self.llm.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }

        async for request_output in results_generator:
            if is_first_output:  # Count input tokens only once
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            for output in request_output.outputs:
                output_index = output.index
                token_counters["total"] += 1
                if stream:
                    new_output = output.text[len(last_output_texts[output_index]):]
                    batch["choices"][output_index]["tokens"].append(new_output)
                    token_counters["batch"] += 1

                    if token_counters["batch"] >= batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch
    
    async def generate_openai_chat(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id: str) -> AsyncGenerator[dict, None]:
        
        if not isinstance(llm_input, list):
            raise ValueError("Input must be a list of messages")
        
        if not stream:
            raise ValueError("OpenAI Chat Completion Format only supports streaming")
        
        chat_completion_request = ChatCompletionRequest(
            model=self.config["model"],
            messages=llm_input,
            stream=True,
            **validated_sampling_params, 
        )

        response_generator = await self.openai_engine.create_chat_completion(chat_completion_request, None)  # None for raw_request
        batch_contents = {}
        batch_latest_choices = {}
        batch_token_counter = 0
        last_chunk = {}
        
        async for chunk_str in response_generator:
            try:
                chunk = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) 
            except:
                continue
            
            if "choices" in chunk:
                for choice in chunk["choices"]:
                    choice_index = choice["index"]
                    if "delta" in choice and "content" in choice["delta"]:
                        batch_contents[choice_index] =  batch_contents.get(choice_index, []) + [choice["delta"]["content"]]
                        batch_latest_choices[choice_index] = choice
                        batch_token_counter += 1
                last_chunk = chunk
            
            if batch_token_counter >= batch_size:
                for choice_index in batch_latest_choices:
                    batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
                last_chunk["choices"] = list(batch_latest_choices.values())
                yield last_chunk
                
                batch_contents = {}
                batch_latest_choices = {}
                batch_token_counter = 0

        if batch_contents:
            for choice_index in batch_latest_choices:
                batch_latest_choices[choice_index]["delta"]["content"] = batch_contents[choice_index]
            last_chunk["choices"] = list(batch_latest_choices.values())
            yield last_chunk
    
    def _initialize_config(self):
        quantization = self._get_quantization()
        dtype = "half" if quantization else "auto"
        return {
            "model": os.getenv("MODEL_NAME"),
            "download_dir": os.getenv("MODEL_BASE_PATH", "/runpod-volume/"),
            "quantization": quantization,
            "load_format": os.getenv("LOAD_FORMAT", "auto"),
            "dtype": dtype,
            "disable_log_stats": bool(int(os.getenv("DISABLE_LOG_STATS", 1))),
            "disable_log_requests": bool(int(os.getenv("DISABLE_LOG_REQUESTS", 1))),
            "trust_remote_code": bool(int(os.getenv("TRUST_REMOTE_CODE", 0))),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", 0.95)),
            "max_parallel_loading_workers": int(os.getenv("MAX_PARALLEL_LOADING_WORKERS", count_physical_cores())),
            "max_model_len": self._get_max_model_len(),
            "tensor_parallel_size": self._get_num_gpu_shard(),
        }

    def _initialize_llm(self):
        try:
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.config))
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e
    
    def _initialize_openai(self):
        if bool(int(os.getenv("ALLOW_OPENAI_FORMAT", 1))) and self.tokenizer.has_chat_template:
            return OpenAIServingChat(self.llm, self.config["model"], "assistant")
        else: 
            return None
            
        
    def _get_num_gpu_shard(self):
        final_num_gpu_shard = 1
        if bool(int(os.getenv("USE_TENSOR_PARALLEL", 0))):
            env_num_gpu_shard = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
            num_gpu_available = device_count()
            final_num_gpu_shard = min(env_num_gpu_shard, num_gpu_available)
            logging.info("Using %s GPU shards", final_num_gpu_shard)
        return final_num_gpu_shard
    
    def _get_max_model_len(self):
        max_model_len = os.getenv("MAX_MODEL_LEN")
        return int(max_model_len) if max_model_len is not None else None
    
    def _get_n_current_jobs(self):
        total_sequences = len(self.llm.engine.scheduler.waiting) + len(self.llm.engine.scheduler.swapped) + len(self.llm.engine.scheduler.running)
        return total_sequences

    def _get_quantization(self):
        quantization = os.getenv("QUANTIZATION", "").lower()
        return quantization if quantization in ["awq", "squeezellm", "gptq"] else None