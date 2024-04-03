FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
  && apt-get install -y python3-pip git

RUN ldconfig /usr/local/cuda-12.1/compat/

# https://github.com/vllm-project/vllm/blob/2ff767b51301e07d1e0ad5887eb26e104e2b3a8a/Dockerfile#L68
FROM dev as flash-attn-builder
# max jobs used for build
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# flash attention version
ARG flash_attn_version=v2.5.6
ENV FLASH_ATTN_VERSION=${flash_attn_version}

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
RUN pip install packaging torch
RUN pip --verbose wheel flash-attn==${FLASH_ATTN_VERSION} \
  --no-build-isolation

FROM runpod/worker-vllm:stable-cuda12.1.0 AS vllm-base

RUN apt-get update -y \
  && apt-get install -y python3-pip

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

COPY --from=flash-attn-builder /usr/src/flash-attention-v2 /usr/src/flash-attention-v2
RUN pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
  MODEL_REVISION=$MODEL_REVISION \
  TOKENIZER_NAME=$TOKENIZER_NAME \
  TOKENIZER_REVISION=$TOKENIZER_REVISION \
  BASE_PATH=$BASE_PATH \
  QUANTIZATION=$QUANTIZATION \
  HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
  HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
  HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
  HF_TRANSFER=1 

ENV PYTHONPATH="/:/vllm-installation"

COPY builder/download_model.py /download_model.py
RUN if [ -f /run/secrets/HF_TOKEN ]; then \
  export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
  fi && \
  if [ -n "$MODEL_NAME" ]; then \
  python3 /download_model.py; \
  fi

# Add source files
COPY src /src


# Start the handler
CMD ["python3", "/src/handler.py"]