import os
import multiprocessing
import runpod
from runpod import RunPodLogger

log = RunPodLogger()

# Store startup error to return to user via handler (RunPod error handling pattern)
STARTUP_ERROR = None
vllm_engine = None
openai_engine = None

# Early CUDA check - fail fast before loading heavy vLLM dependencies
def check_cuda_availability():
    """Check if CUDA is available before loading vLLM to save user money on broken GPU machines."""
    global STARTUP_ERROR
    try:
        import torch
        import traceback
        
        if not torch.cuda.is_available():
            STARTUP_ERROR = "CUDA is not available. torch.cuda.is_available() returned False. Please check your GPU configuration. This worker requires a GPU to run vLLM."
            log.error(STARTUP_ERROR)
            return False
        
        # Try to actually use CUDA to catch driver issues
        device_count = torch.cuda.device_count()
        if device_count == 0:
            STARTUP_ERROR = "No CUDA devices found. torch.cuda.device_count() returned 0. Please ensure GPU is properly attached."
            log.error(STARTUP_ERROR)
            return False
        
        # Quick test to verify CUDA actually works
        try:
            torch.cuda.current_device()
            _ = torch.tensor([1.0], device="cuda")
        except RuntimeError as e:
            full_error = traceback.format_exc()
            STARTUP_ERROR = f"CUDA initialization failed. Full error: {full_error}"
            log.error(STARTUP_ERROR)
            return False
            
        log.info(f"CUDA check passed: {device_count} GPU(s) available")
        for i in range(device_count):
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
            
    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        STARTUP_ERROR = f"Failed to check CUDA availability. Full error: {full_error}"
        log.error(STARTUP_ERROR)
        return False

def initialize_engines():
    """Initialize vLLM engines if CUDA check passed."""
    global vllm_engine, openai_engine, STARTUP_ERROR
    
    if not check_cuda_availability():
        return
    
    try:
        from utils import JobInput
        from engine import vLLMEngine, OpenAIvLLMEngine
        
        vllm_engine = vLLMEngine()
        openai_engine = OpenAIvLLMEngine(vllm_engine)
        log.info("vLLM engines initialized successfully")
    except Exception as e:
        STARTUP_ERROR = f"Failed to initialize vLLM engine: {e}"
        log.error(STARTUP_ERROR)

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
        import sys
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

# Only run initialization and start server in main process
# This prevents re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    import sys
    
    initialize_engines()
    
    # If CUDA/engine init failed, crash the worker BEFORE starting serverless loop
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