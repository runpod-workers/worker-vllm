# Base image
FROM runpod/base:0.4.2-cuda11.8.0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN

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

# Run the Python script to download the model
RUN python3.11 -u /download_model.py --model_name $MODEL_NAME --model_revision $MODEL_REVISION --model_base_path $MODEL_BASE_PATH --hugging_face_hub_token $HUGGING_FACE_HUB_TOKEN

# Start the handler
CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER QUANTIZATION=$QUANTIZATION python3.11 /handler.py
