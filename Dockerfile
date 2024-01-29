ARG WORKER_CUDA_VERSION=11.8.0
FROM runpod/worker-vllm:base-0.2.0-cuda${WORKER_CUDA_VERSION} AS vllm-base

RUN apt-get update -y \
    && apt-get install -y python3-pip

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Add source files
COPY src /src

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG MODEL_BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""

ENV MODEL_BASE_PATH=$MODEL_BASE_PATH \
    MODEL_NAME=$MODEL_NAME \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${MODEL_BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${MODEL_BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${MODEL_BASE_PATH}/huggingface-cache/hub" \
    HF_TRANSFER=1 

ENV PYTHONPATH="/:/vllm-installation"
    
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 /src/download_model.py --model $MODEL_NAME; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
