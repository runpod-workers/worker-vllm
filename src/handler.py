import runpod
from utils import JobInput
from engine import vLLMEngine

vllm_engine = vLLMEngine()

async def handler(job):
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