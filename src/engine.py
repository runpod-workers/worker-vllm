import asyncio
import inspect
import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from vllm import AsyncLLMEngine
from vllm.inputs import TextPrompt
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest, AnthropicMessagesResponse, AnthropicError, AnthropicErrorResponse
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest, ResponsesResponse
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.serve.render.serving import OpenAIServingRender

from constants import DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE_GROWTH_FACTOR, DEFAULT_MAX_CONCURRENCY, DEFAULT_MIN_BATCH_SIZE
from engine_args import get_engine_args
from tokenizer import TokenizerWrapper
from utils import BatchSize, DummyRequest, JobInput, create_error_response

class vLLMEngine:
    def __init__(self, engine = None):
        load_dotenv() # For local development
        self.engine_args = get_engine_args()

        if engine is None:
            ea = self.engine_args
            summary = {
                "model": ea.model,
                "dtype": ea.dtype,
                "quantization": ea.quantization,
                "max_model_len": ea.max_model_len,
                "tensor_parallel_size": ea.tensor_parallel_size,
                "gpu_memory_utilization": ea.gpu_memory_utilization,
            }
            if ea.tokenizer and ea.tokenizer != ea.model:
                summary["tokenizer"] = ea.tokenizer
            logging.info("Engine config: %s", summary)
            logging.debug("Full engine args: %s", ea)

            self.llm = self._initialize_llm()

            if self.engine_args.tokenizer_mode != 'mistral':
                self.tokenizer = TokenizerWrapper(self.engine_args.tokenizer or self.engine_args.model,
                                                  self.engine_args.tokenizer_revision,
                                                  self.engine_args.trust_remote_code)
            else:
                self.tokenizer = None
        else:
            self.llm = engine.llm
            self.tokenizer = engine.tokenizer
            
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
        self.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", DEFAULT_BATCH_SIZE))
        self.batch_size_growth_factor = int(os.getenv("BATCH_SIZE_GROWTH_FACTOR", DEFAULT_BATCH_SIZE_GROWTH_FACTOR))
        self.min_batch_size = int(os.getenv("MIN_BATCH_SIZE", DEFAULT_MIN_BATCH_SIZE))

    def _get_tokenizer_for_chat_template(self):
        """Get tokenizer for chat template application"""
        if self.tokenizer is not None:
            return self.tokenizer
        else:
            # For mistral models, get tokenizer from vLLM engine
            # This is a fallback - ideally chat templates should be handled by vLLM directly
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.engine_args.tokenizer or self.engine_args.model,
                    revision=self.engine_args.tokenizer_revision or "main",
                    trust_remote_code=self.engine_args.trust_remote_code
                )
                # Create a minimal wrapper
                class MinimalTokenizerWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
                        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(self.custom_chat_template)
                        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
                            self.tokenizer.chat_template = self.custom_chat_template
                    
                    def apply_chat_template(self, input):
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
                
                return MinimalTokenizerWrapper(tokenizer)
            except Exception as e:
                logging.error(f"Failed to create fallback tokenizer: {e}")
                raise e

    def dynamic_batch_size(self, current_batch_size, batch_size_growth_factor):
        return min(current_batch_size*batch_size_growth_factor, self.default_batch_size)
                           
    async def generate(self, job_input: JobInput):
        try:
            async for batch in self._generate_vllm(
                llm_input=job_input.llm_input,
                validated_sampling_params=job_input.sampling_params,
                batch_size=job_input.max_batch_size,
                stream=job_input.stream,
                apply_chat_template=job_input.apply_chat_template,
                request_id=job_input.request_id,
                batch_size_growth_factor=job_input.batch_size_growth_factor,
                min_batch_size=job_input.min_batch_size
            ):
                yield batch
        except Exception as e:
            yield {"error": create_error_response(str(e)).model_dump()}

    async def _generate_vllm(self, llm_input, validated_sampling_params, batch_size, stream, apply_chat_template, request_id, batch_size_growth_factor, min_batch_size: str) -> AsyncGenerator[dict, None]:
        if apply_chat_template or isinstance(llm_input, list):
            tokenizer_wrapper = self._get_tokenizer_for_chat_template()
            llm_input = tokenizer_wrapper.apply_chat_template(llm_input)
        results_generator = self.llm.generate(TextPrompt(prompt=llm_input), validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}

        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }
        
        max_batch_size = batch_size or self.default_batch_size
        batch_size_growth_factor, min_batch_size = batch_size_growth_factor or self.batch_size_growth_factor, min_batch_size or self.min_batch_size
        batch_size = BatchSize(max_batch_size, min_batch_size, batch_size_growth_factor)
    

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

                    if token_counters["batch"] >= batch_size.current_batch_size:
                        batch["usage"] = {
                            "input": n_input_tokens,
                            "output": token_counters["total"],
                        }
                        yield batch
                        batch = {
                            "choices": [{"tokens": []} for _ in range(n_responses)],
                        }
                        token_counters["batch"] = 0
                        batch_size.update()

                last_output_texts[output_index] = output.text

        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch

    def _initialize_llm(self):
        try:
            start = time.time()
            engine = AsyncLLMEngine.from_engine_args(self.engine_args)
            end = time.time()
            logging.info(f"Initialized vLLM engine in {end - start:.2f}s")
            return engine
        except Exception as e:
            logging.error("Error initializing vLLM engine: %s", e)
            raise e


class OpenAIvLLMEngine(vLLMEngine):
    def __init__(self, vllm_engine):
        super().__init__(vllm_engine)
        self.served_model_name = os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or self.engine_args.served_model_name or self.engine_args.model
        self.response_role = os.getenv("OPENAI_RESPONSE_ROLE") or "assistant"
        self.lora_adapters = self._load_lora_adapters()

        # Always defer OpenAI engine initialization to the first request.
        # asyncio.run() creates a temporary event loop that gets closed, but async
        # components (tokenizer pool, serving engines) bind futures to that loop.
        # When Runpod's serverless handler runs in its own event loop, those futures
        # are "attached to a different loop" causing RuntimeError.
        # This affects all configurations, not just LoRA.
        self._engines_initialized = False
        if self.lora_adapters:
            logging.info(f"LoRA mode: {len(self.lora_adapters)} adapter(s) will load on first request")
            for adapter in self.lora_adapters:
                logging.info(f"  - {adapter.name}: {adapter.path}")
        else:
            logging.info("OpenAI engines will initialize on first request")

        # Handle both integer and boolean string values for RAW_OPENAI_OUTPUT
        raw_output_env = os.getenv("RAW_OPENAI_OUTPUT", "1")
        if raw_output_env.lower() in ('true', 'false'):
            self.raw_openai_output = raw_output_env.lower() == 'true'
        else:
            self.raw_openai_output = bool(int(raw_output_env))

    def _load_lora_adapters(self):
        lora_modules_env = os.getenv("LORA_MODULES", "")
        if not lora_modules_env:
            return []

        try:
            parsed = json.loads(lora_modules_env)
        except json.JSONDecodeError as e:
            logging.error(
                "LORA_MODULES could not be parsed as JSON: %s — no LoRA adapters loaded. Value: %r",
                e, lora_modules_env,
            )
            return []

        # Accept a single adapter dict as well as an array
        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            logging.error(
                "LORA_MODULES must be a JSON array of adapter objects, got %s — no LoRA adapters loaded.",
                type(parsed).__name__,
            )
            return []

        adapters = []
        for i, adapter in enumerate(parsed):
            try:
                adapters.append(LoRAModulePath(**adapter))
                logging.info("Loaded LoRA adapter config [%d]: %s", i, adapter)
            except Exception as e:
                logging.error(
                    "Failed to parse LoRA adapter at index %d: %s. Config: %r",
                    i, e, adapter,
                )

        if parsed and not adapters:
            logging.error(
                "LORA_MODULES specified %d adapter(s) but none could be loaded — "
                "OpenAI model name lookups for LoRA adapters will fail.",
                len(parsed),
            )

        return adapters

    async def _ensure_engines_initialized(self):
        """Initialize engines on first request to avoid event loop mismatch.

        In Runpod Serverless, the startup code runs outside the handler's event
        loop. Deferring initialization to the first request ensures all async
        components (tokenizer pool, serving engines, LoRA state) are created in
        the correct event loop context.
        """
        if not self._engines_initialized:
            logging.info("Initializing OpenAI serving engines...")
            await self._initialize_engines()
            self._engines_initialized = True
            logging.info("OpenAI serving engines initialized successfully")

    async def _initialize_engines(self):
        self.model_config = self.llm.model_config
        self.base_model_paths = [
            BaseModelPath(name=self.served_model_name, model_path=self.engine_args.model)
        ]

        self.serving_models = OpenAIServingModels(
            engine_client=self.llm,
            base_model_paths=self.base_model_paths,
            lora_modules=self.lora_adapters,
        )
        await self.serving_models.init_static_loras()

        # Get chat template from vLLM tokenizer if available
        chat_template = None
        if self.tokenizer and hasattr(self.tokenizer, 'tokenizer'):
            chat_template = self.tokenizer.tokenizer.chat_template

        self.openai_serving_render = OpenAIServingRender(
            model_config=self.llm.model_config,
            renderer=self.llm.renderer,
            model_registry=self.serving_models.registry,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            trust_request_chat_template=os.getenv('TRUST_REQUEST_CHAT_TEMPLATE', 'false').lower() == 'true',
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            exclude_tools_when_tool_choice_none=os.getenv('EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            reasoning_parser=os.getenv('REASONING_PARSER', "") or None,
            log_error_stack=os.getenv('LOG_ERROR_STACK', 'false').lower() == 'true',
        )

        self.chat_engine = OpenAIServingChat(
            engine_client=self.llm,
            models=self.serving_models,
            response_role=self.response_role,
            openai_serving_render=self.openai_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            trust_request_chat_template=os.getenv('TRUST_REQUEST_CHAT_TEMPLATE', 'false').lower() == 'true',
            return_tokens_as_token_ids=os.getenv('RETURN_TOKENS_AS_TOKEN_IDS', 'false').lower() == 'true',
            reasoning_parser=os.getenv('REASONING_PARSER', "") or "",
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            exclude_tools_when_tool_choice_none=os.getenv('EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            enable_prompt_tokens_details=os.getenv('ENABLE_PROMPT_TOKENS_DETAILS', 'false').lower() == 'true',
            enable_force_include_usage=os.getenv('ENABLE_FORCE_INCLUDE_USAGE', 'false').lower() == 'true',
            enable_log_outputs=os.getenv('ENABLE_LOG_OUTPUTS', 'false').lower() == 'true',
        )
        self.completion_engine = OpenAIServingCompletion(
            engine_client=self.llm,
            models=self.serving_models,
            openai_serving_render=self.openai_serving_render,
            request_logger=None,
            return_tokens_as_token_ids=os.getenv('RETURN_TOKENS_AS_TOKEN_IDS', 'false').lower() == 'true',
            enable_prompt_tokens_details=os.getenv('ENABLE_PROMPT_TOKENS_DETAILS', 'false').lower() == 'true',
            enable_force_include_usage=os.getenv('ENABLE_FORCE_INCLUDE_USAGE', 'false').lower() == 'true',
        )
        self.responses_engine = OpenAIServingResponses(
            engine_client=self.llm,
            models=self.serving_models,
            openai_serving_render=self.openai_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv('RETURN_TOKENS_AS_TOKEN_IDS', 'false').lower() == 'true',
            reasoning_parser=os.getenv('REASONING_PARSER', "") or "",
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            tool_server=None,
            enable_prompt_tokens_details=os.getenv('ENABLE_PROMPT_TOKENS_DETAILS', 'false').lower() == 'true',
            enable_force_include_usage=os.getenv('ENABLE_FORCE_INCLUDE_USAGE', 'false').lower() == 'true',
            enable_log_outputs=os.getenv('ENABLE_LOG_OUTPUTS', 'false').lower() == 'true',
        )
        self.messages_engine = AnthropicServingMessages(
            engine_client=self.llm,
            models=self.serving_models,
            response_role=self.response_role,
            openai_serving_render=self.openai_serving_render,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            return_tokens_as_token_ids=os.getenv('RETURN_TOKENS_AS_TOKEN_IDS', 'false').lower() == 'true',
            reasoning_parser=os.getenv('REASONING_PARSER', "") or "",
            enable_auto_tools=os.getenv('ENABLE_AUTO_TOOL_CHOICE', 'false').lower() == 'true',
            tool_parser=os.getenv('TOOL_CALL_PARSER', "") or None,
            enable_prompt_tokens_details=os.getenv('ENABLE_PROMPT_TOKENS_DETAILS', 'false').lower() == 'true',
            enable_force_include_usage=os.getenv('ENABLE_FORCE_INCLUDE_USAGE', 'false').lower() == 'true',
        )

        warmup = getattr(self.chat_engine, 'warmup', None)
        if callable(warmup):
            result = warmup()
            if inspect.isawaitable(result):
                await result

    async def generate(self, openai_request: JobInput):
        # Ensure engines are ready (no-op if already initialized at startup)
        await self._ensure_engines_initialized()

        if openai_request.openai_route == "/v1/models":
            yield await self._handle_model_request()
        elif openai_request.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(openai_request):
                yield response
        elif openai_request.openai_route == "/v1/responses":
            async for response in self._handle_responses_request(openai_request):
                yield response
        elif openai_request.openai_route == "/v1/messages":
            async for response in self._handle_messages_request(openai_request):
                yield response
        else:
            yield create_error_response("Invalid route").model_dump()
    
    async def _handle_model_request(self):
        models = await self.serving_models.show_available_models()
        return models.model_dump()
    
    async def _handle_chat_or_completion_request(self, openai_request: JobInput):
        if openai_request.openai_route == "/v1/chat/completions":
            request_class = ChatCompletionRequest
            generator_function = self.chat_engine.create_chat_completion
        elif openai_request.openai_route == "/v1/completions":
            request_class = CompletionRequest
            generator_function = self.completion_engine.create_completion
        
        try:
            request = request_class(
                **openai_request.openai_input
            )
        except Exception as e:
            yield create_error_response(str(e)).model_dump()
            return
        
        dummy_request = DummyRequest()
        response_generator = await generator_function(request, raw_request=dummy_request)

        if not openai_request.openai_input.get("stream") or isinstance(response_generator, ErrorResponse):
            yield response_generator.model_dump()
        else:
            batch = []
            batch_token_counter = 0
            batch_size = BatchSize(self.default_batch_size, self.min_batch_size, self.batch_size_growth_factor)
        
            async for chunk_str in response_generator:
                if "data" in chunk_str:
                    if self.raw_openai_output:
                        data = chunk_str
                    elif "[DONE]" in chunk_str:
                        continue
                    else:
                        data = json.loads(chunk_str.removeprefix("data: ").rstrip("\n\n")) if not self.raw_openai_output else chunk_str
                    batch.append(data)
                    batch_token_counter += 1
                    if batch_token_counter >= batch_size.current_batch_size:
                        if self.raw_openai_output:
                            batch = "".join(batch)
                        yield batch
                        batch = []
                        batch_token_counter = 0
                        batch_size.update()
            if batch:
                if self.raw_openai_output:
                    batch = "".join(batch)
                yield batch

    async def _handle_responses_request(self, openai_request: JobInput):
        request_id = getattr(openai_request, "request_id", "unknown")

        try:
            request = ResponsesRequest(**openai_request.openai_input)
        except Exception as e:
            logging.error(
                "Invalid ResponsesRequest JSON: %s",
                e,
                extra={"request_id": request_id}
            )
            yield create_error_response(
                "Invalid request format",
                err_type="BadRequestError"
            ).model_dump()
            return

        dummy_request = DummyRequest()
        try:
            response = await self.responses_engine.create_responses(request, raw_request=dummy_request)
        except Exception as e:
            logging.error(
                "Failed to create Responses: %s",
                e,
                extra={"request_id": request_id},
                exc_info=True
            )
            yield create_error_response(
                "Internal server error during response generation",
                err_type="InternalServerError"
            ).model_dump()
            return

        if isinstance(response, (ErrorResponse, ResponsesResponse)):
            yield response.model_dump()
            return

        try:
            async for event in response:
                if not hasattr(event, "type"):
                    continue
                event_type = getattr(event, "type", "unknown")
                yield f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        except Exception as e:
            logging.error(
                "Error processing responses stream: %s",
                e,
                extra={"request_id": request_id},
                exc_info=True
            )
            error_payload = create_error_response(
                "Streaming response failed",
                err_type="InternalServerError"
            ).model_dump_json()
            yield f"event: error\ndata: {error_payload}\n\n"

    async def _handle_messages_request(self, openai_request: JobInput):
        request_id = getattr(openai_request, "request_id", "unknown")

        try:
            request = AnthropicMessagesRequest(**openai_request.openai_input)
        except Exception as e:
            logging.error(
                "Invalid AnthropicMessagesRequest: %s",
                e,
                extra={"request_id": request_id}
            )
            yield AnthropicErrorResponse(
                error=AnthropicError(
                    type="invalid_request_error",
                    message="Invalid request format"
                )
            ).model_dump()
            return

        dummy_request = DummyRequest()

        try:
            response = await self.messages_engine.create_messages(request, raw_request=dummy_request)
        except Exception as e:
            logging.error(
                "Failed to create messages: %s",
                e,
                extra={"request_id": request_id},
                exc_info=True
            )
            yield AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message="Failed to generate messages"
                )
            ).model_dump()
            return

        if isinstance(response, ErrorResponse):
            error_type = getattr(response, "type", "internal_error")
            error_message = getattr(response, "message", "Unknown error")
            yield AnthropicErrorResponse(
                error=AnthropicError(type=error_type, message=error_message)
            ).model_dump()
            return

        if isinstance(response, AnthropicMessagesResponse):
            yield response.model_dump(exclude_none=True)
            return

        try:
            async for chunk in response:
                yield chunk
        except Exception as e:
            logging.error(
                "Error streaming messages: %s",
                e,
                extra={"request_id": request_id},
                exc_info=True
            )
            error_payload = AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message="Error while streaming messages"
                )
            ).model_dump_json()
            yield f"event: error\ndata: {error_payload}\n\n"
