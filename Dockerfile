# Base image
FROM runpod/base:0.4.2-cuda12.1.0

ARG HUGGING_FACE_HUB_TOKEN

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

# Prepare argument for the model and tokenizer
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_REVISION="main"
ENV MODEL_REVISION=$MODEL_REVISION
ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
ARG TOKENIZER=
ENV TOKENIZER=$TOKENIZER
ARG STREAMING=
ENV STREAMING=$STREAMING
ARG QUANTIZATION=
ENV QUANTIZATION=$QUANTIZATION
ARG MAX_CONCURRENCY=
ENV MAX_CONCURRENCY=$MAX_CONCURRENCY

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Download the models
RUN mkdir -p /model

# Set environment variables
ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    MODEL_BASE_PATH=$MODEL_BASE_PATH \
    HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Start the handler
CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER QUANTIZATION=$QUANTIZATION  python3.11 /handler.py
