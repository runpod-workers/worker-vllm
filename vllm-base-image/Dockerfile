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
RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Set working directory
WORKDIR /vllm-installation

RUN ldconfig /usr/local/cuda-$(echo "$WORKER_CUDA_VERSION" | sed 's/\.0$//')/compat/

# Install build and runtime dependencies
COPY vllm/requirements-common.txt requirements-common.txt
COPY vllm/requirements-cuda${WORKER_CUDA_VERSION}.txt  requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

# Install development dependencies
COPY vllm/requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt
    
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

FROM dev AS build

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Install build dependencies
COPY vllm/requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# Copy necessary files
COPY vllm/csrc csrc
COPY vllm/setup.py setup.py
COPY vllm/cmake cmake
COPY vllm/CMakeLists.txt CMakeLists.txt
COPY vllm/requirements-common.txt requirements-common.txt
COPY vllm/requirements-cuda${WORKER_CUDA_VERSION}.txt requirements-cuda.txt
COPY vllm/pyproject.toml pyproject.toml
COPY vllm/vllm vllm 

# Set environment variables for building extensions
ARG max_jobs=1024
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=512
ENV NVCC_THREADS=${nvcc_threads}
ENV WORKER_CUDA_VERSION=${WORKER_CUDA_VERSION}
ENV VLLM_INSTALL_PUNICA_KERNELS=0
# Build extensions
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist

RUN --mount=type=cache,target=/root/.cache/pip \
    pip cache remove vllm_nccl*

FROM dev as flash-attn-builder
# max jobs used for build
ARG max_jobs=512
ENV MAX_JOBS=${max_jobs}
# flash attention version
ARG flash_attn_version=v2.5.6
ENV FLASH_ATTN_VERSION=${flash_attn_version}

WORKDIR /usr/src/flash-attention-v2

# Download the wheel or build it if a pre-compiled release doesn't exist
RUN pip --verbose wheel flash-attn==${FLASH_ATTN_VERSION} \
    --no-build-isolation --no-deps --no-cache-dir

FROM dev as NCCL-installer

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Update and install necessary libraries
RUN apt-get update -y \
    && apt-get install -y wget

# Install NCCL library
RUN if [ "$WORKER_CUDA_VERSION" = "11.8.0" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update \
        && apt install -y libnccl2=2.15.5-1+cuda11.8 libnccl-dev=2.15.5-1+cuda11.8; \
    elif [ "$WORKER_CUDA_VERSION" = "12.1.0" ]; then \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
        && dpkg -i cuda-keyring_1.0-1_all.deb \
        && apt-get update \
        && apt install -y libnccl2=2.17.1-1+cuda12.1 libnccl-dev=2.17.1-1+cuda12.1; \
    else \
        echo "Unsupported CUDA version: $WORKER_CUDA_VERSION"; \
        exit 1; \
    fi

FROM nvidia/cuda:${WORKER_CUDA_VERSION}-base-ubuntu22.04 AS vllm-base

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Update and install necessary libraries
RUN apt-get update -y \
    && apt-get install -y python3-pip

# Set working directory
WORKDIR /vllm-workspace

RUN ldconfig /usr/local/cuda-$(echo "$WORKER_CUDA_VERSION" | sed 's/\.0$//')/compat/

RUN --mount=type=bind,from=build,src=/vllm-installation/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install dist/*.whl --verbose

RUN --mount=type=bind,from=flash-attn-builder,src=/usr/src/flash-attention-v2,target=/usr/src/flash-attention-v2 \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /usr/src/flash-attention-v2/*.whl --no-cache-dir

FROM vllm-base AS runtime

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer modelscope

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/"

# Copy NCCL library
COPY --from=NCCL-installer /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2
# Set the VLLM_NCCL_SO_PATH environment variable
ENV VLLM_NCCL_SO_PATH="/usr/lib/x86_64-linux-gnu/libnccl.so.2"


# Validate the installation
RUN python3 -c "import vllm; print(vllm.__file__)"