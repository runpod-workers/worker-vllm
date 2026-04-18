FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM with FlashInfer - use CUDA 12.8 PyTorch wheels
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "vllm[flashinfer]==0.19.0" --extra-index-url https://download.pytorch.org/whl/cu129



# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="sakamakismile/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4"
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/models"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

# Add default args for the config
ARG KV_CACHE_DTYPE="fp8"
ARG MAX_MODEL_LEN=175000
ARG REASONING_PARSER="qwen3"
ARG ENABLE_AUTO_TOOL_CHOICE="true"
ARG TOOL_CALL_PARSER="qwen3_coder"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    # new cfg
    KV_CACHE_DTYPE=$KV_CACHE_DTYPE \
    MAX_MODEL_LEN=$MAX_MODEL_LEN \
    REASONING_PARSER=$REASONING_PARSER \
    ENABLE_AUTO_TOOL_CHOICE=$ENABLE_AUTO_TOOL_CHOICE \
    TOOL_CALL_PARSER=$TOOL_CALL_PARSER

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    pip install git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
