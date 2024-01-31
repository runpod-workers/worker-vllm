import os
from utils import JobInput
from engine import vLLMEngine

os.environ["MODEL_NAME"] = "facebook/opt-125m"
os.environ["CUSTOM_CHAT_TEMPLATE"] = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"

vllm_engine = vLLMEngine()

async def handler(job):
    job_input = JobInput(job["input"])
    results_generator = vllm_engine.generate(job_input)
    async for batch in results_generator:
        yield batch
        
test_payload = {
    "input": {
        "messages":  [
            {"role": "user", "content": "Write me a 3000 word long and detailed essay about how the french revolution impacted the rest of europe over the 18th century."},
        ],
        "batch_size": 2, # How many tokens to yield per batch
        "apply_chat_template": True,
        "sampling_params": {
            "max_tokens": 10,
            "temperature": 0,
            "ignore_eos": True,
            "n":1
        },
        "stream": True,
        "use_openai_format": True
    }
}

async def test_handler():
    print("Start of output")
    print("=" *50)
    async for batch in handler(test_payload):
        print(batch, end="")
    print("=" *50)
    print("End of output")
    
import asyncio

asyncio.run(test_handler())