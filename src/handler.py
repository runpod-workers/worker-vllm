import os
import runpod
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

# Detect number of visible GPUs
gpu_count = torch.cuda.device_count()

# Fallback to 1 if none detected
if gpu_count < 1:
    gpu_count = 1

# Set the environment variable
os.environ["TENSOR_PARALLEL_SIZE"] = str(gpu_count)

print(f"Detected {gpu_count} GPU(s)")
print(f"Set TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}")


vllm_engine = vLLMEngine()
OpenAIvLLMEngine = OpenAIvLLMEngine(vllm_engine)

async def handler(job):
    job_input = JobInput(job["input"])
    engine = OpenAIvLLMEngine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)