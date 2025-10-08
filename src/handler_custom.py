import os
import runpod
import logging
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from utils import JobInput
from engine import vLLMEngine, OpenAIvLLMEngine
import json

   
# Bypass model_name since runpod cannot set model_name to local or network volume
if os.getenv("MODEL_NAME_OVERRIDE"):
    os.environ["MODEL_NAME"] = str(os.getenv("MODEL_NAME_OVERRIDE"))


if os.getenv("ENABLE_MODEL_PATCH"):
    
    import sys
    import importlib

    # Get MODEL_NAME path and go back 1 directory
    model_path = str(os.getenv("MODEL_NAME"))
    parent_dir = os.path.dirname(model_path)

    # Get the directory name of MODEL_NAME to use as module name
    model_dir_name = os.path.basename(model_path)

    # Add parent directory to sys.path
    sys.path.append(os.path.abspath(parent_dir))

    # Dynamic import: from {model_dir_name} import modeling_dots_ocr_vllm
    module = importlib.import_module(f"{model_dir_name}.modeling_dots_ocr_vllm")

vllm_engine = vLLMEngine()
openai_vllm_engine = OpenAIvLLMEngine(vllm_engine)

# Use the MODEL environment variable; fallback to a default if not set
mode_to_run = os.getenv("MODE_TO_RUN", "pod")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("------- ENVIRONMENT VARIABLES -------")
print("Mode running: ", mode_to_run)
print("------- -------------------- -------")

# Create FastAPI app
app = FastAPI(title="vLLM OpenAI-Compatible API", version="1.0.0")


async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_vllm_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch

# FastAPI endpoints for OpenAI compatibility
@app.get("/openai/v1/models")
@app.get("/v1/models")
async def get_models():
    """Get available models"""
    try:
        job_input = JobInput({
            "openai_route": "/v1/models",
            "openai_input": {}
        })
        result_generator = openai_vllm_engine.generate(job_input)
        async for result in result_generator:
            return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in get_models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/openai/v1/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions"""
    try:
        # Parse request body
        request_data = await request.json()

        # Create JobInput for OpenAI engine
        job_input = JobInput({
            "openai_route": "/v1/chat/completions",
            "openai_input": request_data
        })

        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            # Return streaming response
            async def stream_generator():
                result_generator = openai_vllm_engine.generate(job_input)
                async for result in result_generator:
                    if isinstance(result, str):
                        # Raw OpenAI output format
                        yield result
                    elif isinstance(result, list):
                        # Batch of responses
                        for item in result:
                            if isinstance(item, str):
                                yield item
                            else:
                                yield f"data: {json.dumps(item)}\n\n"
                    else:
                        # Single response object
                        yield f"data: {json.dumps(result)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return non-streaming response
            result_generator = openai_vllm_engine.generate(job_input)
            async for result in result_generator:
                return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/openai/v1/completions")
@app.post("/v1/completions")
async def completions(request: Request):
    """Handle text completions"""
    try:
        # Parse request body
        request_data = await request.json()

        # Create JobInput for OpenAI engine
        job_input = JobInput({
            "openai_route": "/v1/completions",
            "openai_input": request_data
        })

        # Check if streaming is requested
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            # Return streaming response
            async def stream_generator():
                result_generator = openai_vllm_engine.generate(job_input)
                async for result in result_generator:
                    if isinstance(result, str):
                        # Raw OpenAI output format
                        yield result
                    elif isinstance(result, list):
                        # Batch of responses
                        for item in result:
                            if isinstance(item, str):
                                yield item
                            else:
                                yield f"data: {json.dumps(item)}\n\n"
                    else:
                        # Single response object
                        yield f"data: {json.dumps(result)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Return non-streaming response
            result_generator = openai_vllm_engine.generate(job_input)
            async for result in result_generator:
                return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "healthy"}

if mode_to_run == "pod":
    # Get ports from environment variables
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting vLLM server on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "INFO").lower()
    )
else:
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _: vllm_engine.max_concurrency,
            "return_aggregate_stream": True,
        }
    )