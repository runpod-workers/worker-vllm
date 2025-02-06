import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    try:
        job_input = JobInput(job["input"])
        engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            # If there's any kind of error in the batch, format it
            if isinstance(batch, dict) and 'error' in batch:
                yield {"error": str(batch)}
            else:
                yield batch
    except Exception as e:
        yield {"error": str(e)}
        return 

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)