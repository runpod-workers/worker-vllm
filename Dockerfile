# FROM alpayariyakrunpod/worker-vllm:base-1.0.0-cuda12.1.0 AS vllm-base
FROM runpod/worker-vllm:base-0.3.2-cuda11.8.0 as vllm-base

RUN apt-get update -y \
  && apt-get install -y python3-pip

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

# Flash atten is installed in base image :-)

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

# apply patch :-) should be changed once upstreamed
# RUN apt install -y wget git
# RUN wget https://github.com/vllm-project/vllm/pull/3804.patch
# RUN cd /vllm-installation && git apply /3804.patch

# Start the handler
CMD ["python3", "/src/handler.py"]