import os
from utils import JobInput
from engine import vLLMEngine

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
        "batch_size": 2, 
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