FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y curl git software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

RUN ldconfig /usr/local/cuda-13.0/compat/

# Install vLLM with FlashInfer - use CUDA 130 PyTorch wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.23.0" && \
    uv pip install --system git+https://github.com/deepseek-ai/DeepGEMM.git@714dd1a4a980f7937a74343d19a8eba4fe321480 --no-build-isolation && \
    uv pip install --system --force-reinstall --no-deps nixl-cu13

# Fix CUTLASS DSL cu13 install order: nvidia-cutlass-dsl[cu13] installs
# -libs-base and -libs-cu13 wheels that share paths with different content.
# uv can extract them in either order, leaving base files that break CUDA 13
# CuTe DSL JIT. Force -libs-cu13 last. See vllm-project/vllm#45204.
RUN CUTLASS_DSL_VERSION=$(uv pip show --system nvidia-cutlass-dsl 2>/dev/null | awk '/^Version:/{print $2}') && \
    if [ -n "$CUTLASS_DSL_VERSION" ]; then \
        uv pip install --system --force-reinstall --no-deps \
            "nvidia-cutlass-dsl-libs-cu13==${CUTLASS_DSL_VERSION}"; \
    fi

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    # Disable DeepGEMM MoE kernels by default; override with VLLM_USE_DEEP_GEMM=1 to enable
    VLLM_USE_DEEP_GEMM=0

ENV PYTHONPATH="/:/vllm-workspace" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    uv pip install --system -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    uv pip install --system git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src
RUN chmod +x /src/start.sh
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]
