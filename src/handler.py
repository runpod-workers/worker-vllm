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
    # Return startup error to user if initialization failed
    if STARTUP_ERROR:
        log.error(f"Returning startup error to user: {STARTUP_ERROR}")
        yield {"error": STARTUP_ERROR}
        return
    
    if vllm_engine is None or openai_engine is None:
        yield {"error": "vLLM engine not initialized. Check worker logs for details."}
        return
    
    try:
        from utils import JobInput
        job_input = JobInput(job["input"])
        engine = openai_engine if job_input.openai_route else vllm_engine
        results_generator = engine.generate(job_input)
        async for batch in results_generator:
            yield batch
    except Exception as e:
        log.error(f"Error during inference: {e}")
        yield {"error": str(e)}

# Only run initialization and start server in main process
# This prevents re-initialization when vLLM spawns worker subprocesses
if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    initialize_engines()
    
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency if vllm_engine else 1,
            "return_aggregate_stream": True,
        }
    )