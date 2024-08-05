ARG WORKER_CUDA_VERSION=11.8.0
ARG BASE_IMAGE_VERSION=1.0.0
FROM runpod/worker-vllm:base-${BASE_IMAGE_VERSION}-cuda${WORKER_CUDA_VERSION} AS vllm-base

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME="" \
    TOKENIZER_NAME="" \
    BASE_PATH="/runpod-volume" \
    QUANTIZATION="" \
    MODEL_REVISION="" \
    TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH="/:/vllm-workspace"

COPY src/download_model.py /download_model.py
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 /download_model.py; \
    fi

# Add source files and remove download_model.py
COPY src /src
RUN rm /download_model.py

# Add a health check
HEALTHCHECK CMD python3 -c "import vllm" || exit 1

# Start the handler
CMD ["python3", "/src/handler.py"]