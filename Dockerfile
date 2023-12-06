# Base image
FROM runpod/base:0.4.2-cuda12.1.0

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add source files
ADD src .

ARG MODEL_NAME=""
ARG MODEL_BASE_PATH=""
ARG TOKENIZER=""

# Conditionally run download_model.py
RUN if [ -n "$MODEL_NAME" ] && [ -n "$MODEL_BASE_PATH"]; then \
        python3.11 /download_model.py --model $MODEL_NAME --download_dir $MODEL_BASE_PATH --tokenizer $TOKENIZER; \
    fi

# Start the handler
CMD python3.11 /handler.py
