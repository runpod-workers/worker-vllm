import multiprocessing
import os
import sys
import traceback

import runpod
from runpod import RunPodLogger

from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine

log = RunPodLogger()

# Prevent re-initialization in vLLM worker subprocesses
if multiprocessing.current_process().name == "MainProcess":
    vllm_engine = vLLMEngine()
    openai_engine = OpenAIvLLMEngine(vllm_engine)
else:
    vllm_engine = None
    openai_engine = None

async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    try:
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        err_str = str(e)
        is_cuda_error = "CUDA" in err_str or "cuda" in err_str
        is_oom = "out of memory" in err_str.lower()

        if is_cuda_error and not is_oom:
            log.error(f"CUDA error (non-OOM), exiting for worker recycle: {e}")
            traceback.print_exc()
            sys.exit(1)
        elif is_oom:
            log.error(f"CUDA OOM error: {e}")
            traceback.print_exc()
            yield {"error": str(e)}
        else:
            log.error(f"Handler error: {e}")
            traceback.print_exc()
            yield {"error": str(e)}

runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
        "return_aggregate_stream": True,
    }
)
