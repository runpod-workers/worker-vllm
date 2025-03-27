FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip git build-essential curl

# Install uv using pip
RUN pip install uv

RUN ldconfig /usr/local/cuda-12.1/compat/

# Create virtual environment and install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN uv venv /venv
ENV PATH="/venv/bin:$PATH"
RUN uv pip install --upgrade pip
RUN uv pip install --upgrade -r /requirements.txt

# Clone and install vLLM from fork in development mode
ENV VLLM_USE_PRECOMPILED=1
WORKDIR /vllm-fork
RUN git clone https://github.com/TimPietrusky/vllm.git .
RUN uv pip install -e . 
RUN uv pip install --system https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post2/flashinfer_python-0.2.1.post2+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# Reset workdir to root for compatibility with the rest of the image
WORKDIR /

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
    HF_HUB_ENABLE_HF_TRANSFER=0 

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Print all installed packages for debugging purposes and write to a file
RUN /venv/bin/pip list > /installed_packages.txt && cat /installed_packages.txt

# Start the handler with the Python from our virtual environment
CMD ["/venv/bin/python", "/src/handler.py"]