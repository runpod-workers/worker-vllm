FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.19.0" --extra-index-url https://download.pytorch.org/whl/cu129

# Install additional Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade -r /requirements.txt

# Hardcoded Model Configuration
ENV MODEL_NAME="sakamakismile/Huihui-Qwen3.5-4B-abliterated-NVFP4" \
    BASE_PATH="/model" \
    KV_CACHE_DTYPE="fp8" \
    MAX_MODEL_LEN="175000" \
    REASONING_PARSER="qwen3" \
    TOOL_CALL_PARSER="qwen3_coder" \
    ENABLE_AUTO_TOOL_CHOICE="true" \
    HF_HOME="/model/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    PYTHONPATH="/:/vllm-workspace" \
    RAW_OPENAI_OUTPUT=true \
    OPENAI_RESPONSE_ROLE=assistant \
    OPENAI_SERVED_MODEL_NAME_OVERRIDE=q3.6-35a5-uncensored \
    MAX_CONCURRENCY=30

COPY src /src

# Download the model at build time
RUN python3 /src/download_model.py

# Start the handler
CMD ["python3", "/src/handler.py"]
