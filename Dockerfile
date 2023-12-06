# Base image - Set default to CUDA 11.8.0
ARG CUDA_VERSION=11.8.0

# Use different base images based on CUDA_VERSION argument
FROM runpod/base:0.4.2-cuda${CUDA_VERSION} as builder

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Install specific packages based on CUDA version
RUN if [ "$CUDA_VERSION" = "12.1.0" ]; then \
        python3.11 -m pip install vllm==0.2.3; \
    else \
        python3.11 -m pip install vllm-0.2.3+cu118-cp311-cp311-manylinux1_x86_64.whl; \
    fi


# Add source files
ADD src .

ARG MODEL_NAME=""
ARG MODEL_BASE_PATH=""

# Conditionally run download_model.py
RUN if [ -n "$MODEL_NAME" ] && [ -n "$MODEL_BASE_PATH" ]; then \
        python3.11 /download_model.py --model $MODEL_NAME --download_dir $MODEL_BASE_PATH; \
        export MODEL_NAME=$MODEL_NAME; \
        export MODEL_BASE_PATH=$MODEL_BASE_PATH; \
    fi

# Start the handler
CMD ["python3.11", "/handler.py"]