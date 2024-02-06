import os
import runpod
from utils import JobInput, OpenAIRequest
from engine import vLLMEngine, OpenAIvLLMEngine

if not os.getenv("MODEL_NAME"):
    os.environ["MODEL_NAME"] = "facebook/opt-125m"
    os.environ["CUSTOM_CHAT_TEMPLATE"] = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    if "openai" in job:
        openai_request = OpenAIRequest(job)
        results_generator = OpenAIvLLMEngine.generate(openai_request)
    else:
        job_input = JobInput(job["input"])
        results_generator = vllm_engine.generate(job_input)
        
    async for batch in results_generator:
        yield batch
        
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)