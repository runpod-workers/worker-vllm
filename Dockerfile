# Base image - Set default to CUDA 11.8
ARG WORKER_CUDA_VERSION=11.8
FROM runpod/base:0.4.2-cuda${WORKER_CUDA_VERSION}.0 as builder

ARG WORKER_CUDA_VERSION=11.8 # Required duplicate to keep in scope

# Set Environment Variables
ENV WORKER_CUDA_VERSION=${WORKER_CUDA_VERSION} \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub" 


# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt && \
    rm /requirements.txt

# Install torch and vllm based on CUDA version
RUN if [[ "${WORKER_CUDA_VERSION}" == 11.8* ]]; then \
        wget https://github.com/alpayariyak/vllm/releases/download/0.2.4-runpod-11.8/vllm-0.2.4+cu118-cp311-cp311-linux_x86_64.whl && \
        python3.11 -m pip install vllm-0.2.4+cu118-cp311-cp311-linux_x86_64.whl && \
        rm vllm-0.2.4+cu118-cp311-cp311-linux_x86_64.whl; \
        python3.11 -m pip uninstall torch -y; \
        python3.11 -m pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118; \
        python3.11 -m pip uninstall xformers -y; \
        python3.11 -m pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118; \
    else \
        python3.11 -m pip install -e git+https://github.com/alpayariyak/vllm.git#egg=vllm; \
    fi && \
    rm -rf /root/.cache/pip


# Add source files
COPY src .

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG MODEL_BASE_PATH="/runpod-volume/"
ARG HF_TOKEN=""
ARG QUANTIZATION=""
RUN if [ -n "$MODEL_NAME" ]; then \
        python3.11 /download_model.py --model $MODEL_NAME --download_dir $MODEL_BASE_PATH; \
        export MODEL_BASE_PATH=$MODEL_BASE_PATH; \
        export MODEL_NAME=$MODEL_NAME; \
    fi && \
    if [ -n "$QUANTIZATION" ]; then \
        export QUANTIZATION=$QUANTIZATION; \
    fi

# Start the handler
CMD ["python3.11", "/handler.py"]