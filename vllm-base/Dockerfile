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

RUN if [ "${WORKER_CUDA_VERSION}" = "12.1.0" ]; then \
        ldconfig /usr/local/cuda-12.1/compat/; \
    fi

# Set working directory
WORKDIR /vllm-installation

# Install build and runtime dependencies
COPY vllm-${WORKER_CUDA_VERSION}/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WORKER_CUDA_VERSION}" = "11.8.0" ]; then \
        pip install -U --force-reinstall torch==2.1.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Install development dependencies
COPY vllm-${WORKER_CUDA_VERSION}/requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

FROM dev AS build

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Install build dependencies
COPY vllm-${WORKER_CUDA_VERSION}/requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# Copy necessary files
COPY vllm-${WORKER_CUDA_VERSION}/csrc csrc
COPY vllm-${WORKER_CUDA_VERSION}/setup.py setup.py
COPY vllm-12.1.0/pyproject.toml pyproject.toml
COPY vllm-${WORKER_CUDA_VERSION}/vllm/__init__.py vllm/__init__.py

# Conditional installation based on CUDA version
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WORKER_CUDA_VERSION}" = "11.8.0" ]; then \
        pip install -U --force-reinstall torch==2.1.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118; \
        rm pyproject.toml; \
    elif [ "${WORKER_CUDA_VERSION}" != "12.1.0" ]; then \
        echo "WORKER_CUDA_VERSION not supported"; \
        exit 1; \
    fi

# Set environment variables for building extensions
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ARG max_jobs=48
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=1024
ENV NVCC_THREADS=${nvcc_threads}

# Build extensions
RUN python3 setup.py build_ext --inplace

FROM nvidia/cuda:${WORKER_CUDA_VERSION}-runtime-ubuntu22.04 AS vllm-base

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Update and install necessary libraries
RUN apt-get update -y \
    && apt-get install -y python3-pip

# Set working directory
WORKDIR /vllm-installation

# Install runtime dependencies
COPY vllm-${WORKER_CUDA_VERSION}/requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WORKER_CUDA_VERSION}" = "11.8.0" ]; then \
        pip install -U --force-reinstall torch==2.1.2 xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118; \
    fi

# Copy built files from the build stage
COPY --from=build /vllm-installation/vllm/*.so /vllm-installation/vllm/
COPY vllm-${WORKER_CUDA_VERSION}/vllm vllm

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/"

# Validate the installation
RUN python3 -c "import sys; print(sys.path); import vllm; print(vllm.__file__)"