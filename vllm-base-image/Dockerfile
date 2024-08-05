################### vLLM Base Dockerfile ###################
# This Dockerfile is for building the image that the  
# vLLM worker container will use as its base image. 
# If your changes are outside of the vLLM source code, you
# do not need to build this image.
##########################################################

# Define the CUDA version for the build
ARG WORKER_CUDA_VERSION=11.8.0

FROM nvidia/cuda:${WORKER_CUDA_VERSION}-devel-ubuntu22.04 AS dev

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Update and install dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Set working directory
WORKDIR /vllm-installation

RUN ldconfig /usr/local/cuda-$(echo "$WORKER_CUDA_VERSION" | sed 's/\.0$//')/compat/

# Install build and runtime dependencies
COPY vllm/requirements-common.txt vllm/requirements-cuda${WORKER_CUDA_VERSION}.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache -r requirements-common.txt -r requirements-cuda${WORKER_CUDA_VERSION}.txt

# Install development dependencies
COPY vllm/requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache -r requirements-dev.txt
    
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

FROM dev AS build

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Install build dependencies
COPY vllm/requirements-build.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache -r requirements-build.txt

# Install compiler cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && apt-get install -y --no-install-recommends ccache \
    && rm -rf /var/lib/apt/lists/*

# Copy necessary files
COPY vllm/csrc csrc
COPY vllm/setup.py vllm/cmake vllm/CMakeLists.txt vllm/pyproject.toml ./
COPY vllm/vllm vllm

# Set environment variables for building extensions
ENV WORKER_CUDA_VERSION=${WORKER_CUDA_VERSION} \
    VLLM_INSTALL_PUNICA_KERNELS=0 \
    CCACHE_DIR=/root/.cache/ccache

# Build extensions
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    python3 setup.py bdist_wheel --dist-dir=dist

RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip cache remove vllm_nccl*

FROM dev as flash-attn-builder

ARG flash_attn_version=v2.5.8
ENV FLASH_ATTN_VERSION=${flash_attn_version}

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip wheel flash-attn==${FLASH_ATTN_VERSION} \
    --no-build-isolation --no-deps --no-cache-dir

FROM dev as NCCL-installer

ARG WORKER_CUDA_VERSION

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Install NCCL library
RUN --mount=type=cache,target=/var/cache/apt \
    if [ "$WORKER_CUDA_VERSION" = "11.8.0" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnccl2=2.15.5-1+cuda11.8 libnccl-dev=2.15.5-1+cuda11.8; \
    elif [ "$WORKER_CUDA_VERSION" = "12.1.0" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnccl2=2.17.1-1+cuda12.1 libnccl-dev=2.17.1-1+cuda12.1; \
    else \
        echo "Unsupported CUDA version: $WORKER_CUDA_VERSION"; \
        exit 1; \
    fi \
    && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:${WORKER_CUDA_VERSION}-base-ubuntu22.04 AS vllm-base

ARG WORKER_CUDA_VERSION

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

WORKDIR /vllm-workspace

RUN ldconfig /usr/local/cuda-$(echo "$WORKER_CUDA_VERSION" | sed 's/\.0$//')/compat/

RUN --mount=type=bind,from=build,src=/vllm-installation/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache dist/*.whl

RUN --mount=type=bind,from=flash-attn-builder,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache /usr/src/flash-attention-v2/*.whl

FROM vllm-base AS runtime

# Install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/uv \
    /root/.cargo/bin/uv pip install --system --no-cache accelerate hf_transfer modelscope tensorizer

ENV PYTHONPATH="/" \
    VLLM_NCCL_SO_PATH="/usr/lib/x86_64-linux-gnu/libnccl.so.2"

# Copy NCCL library
COPY --from=NCCL-installer /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2

# Validate the installation
RUN python3 -c "import vllm; print(vllm.__file__)"

# Add a health check
HEALTHCHECK CMD python3 -c "import vllm" || exit 1