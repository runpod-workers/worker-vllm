import os
import sys
import multiprocessing
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

# Store startup error to return to user via handler (RunPod error handling pattern)
STARTUP_ERROR = None
vllm_engine = None
openai_engine = None

# RunPod SDK fitness checks (memory, disk, network, CUDA, GPU benchmark)
# run automatically at worker startup - no custom checks needed


# ============================================================================
# Engine Initialization
# ============================================================================

def initialize_engines():
    """Initialize vLLM engines after fitness checks pass."""
    global vllm_engine, openai_engine, STARTUP_ERROR
    
    try:
        from utils import JobInput
        from engine import vLLMEngine, OpenAIvLLMEngine
        
        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        log.info("vLLM engines initialized successfully")
    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        STARTUP_ERROR = f"Failed to initialize vLLM engine: {e}\n{full_error}"
        log.error(STARTUP_ERROR)


# ============================================================================
# Request Handler
# ============================================================================

async def handler(job):
    """Handle inference requests. Startup errors cause worker exit before this runs."""
    try:
        from utils import JobInput
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        import traceback
        
        error_str = str(e)
        full_traceback = traceback.format_exc()
        
        # Log full error with traceback for debugging
        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{full_traceback}")
        
        # CUDA errors = worker is broken, log then exit to flush it out
        if "CUDA" in error_str or "cuda" in error_str or "OutOfMemory" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)
        
        yield {"error": error_str}


# ============================================================================
# Main Entry Point
# ============================================================================

# Only run initialization and start server in main process
# This prevents re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    
    # Initialize vLLM engines
    initialize_engines()
    
    # If engine init failed, crash the worker BEFORE starting serverless loop
    # This way RunPod never sees this worker as available, and jobs go to healthy workers
    if STARTUP_ERROR or vllm_engine is None:
        log.error(f"Worker startup failed, exiting: {STARTUP_ERROR}")
        sys.exit(1)  # Non-zero exit = worker unhealthy, RunPod will spin up another
    
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
            "return_aggregate_stream": True,
        }
    )