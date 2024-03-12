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

# Install build and runtime dependencies
COPY vllm/requirements-${WORKER_CUDA_VERSION}.txt  requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Install development dependencies
COPY vllm/requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

FROM dev AS build

# Re-declare ARG after FROM
ARG WORKER_CUDA_VERSION

# Install build dependencies
COPY vllm/requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# Copy necessary files
COPY vllm/csrc csrc
COPY vllm/setup.py setup.py
COPY vllm/pyproject.toml pyproject.toml
COPY vllm/vllm/__init__.py vllm/__init__.py

# Set environment variables for building extensions
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ARG max_jobs=48
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=1024
ENV NVCC_THREADS=${nvcc_threads}
ENV WORKER_CUDA_VERSION=${WORKER_CUDA_VERSION}
ENV VLLM_INSTALL_PUNICA_KERNELS=0
# Build extensions
RUN ldconfig /usr/local/cuda-$(echo "$WORKER_CUDA_VERSION" | sed 's/\.0$//')/compat/
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
COPY vllm/requirements-${WORKER_CUDA_VERSION}.txt  requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy built files from the build stage
COPY --from=build /vllm-installation/vllm/*.so /vllm-installation/vllm/
COPY vllm/vllm vllm

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/"

# Validate the installation
RUN python3 -c "import sys; print(sys.path); import vllm; print(vllm.__file__)"