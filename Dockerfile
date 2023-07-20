# Base image
# The following docker base image is recommended by VLLM: 
# FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
# FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
ARG DEBIAN_FRONTEND=noninteractive
RUN pip uninstall torch -y
RUN pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh && \
    /setup.sh && \
    rm /setup.sh

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

# Quick temporary updates
RUN pip install git+https://github.com/runpod/runpod-python@multijob2#egg=runpod --compile

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=NONE
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
ENV DOWNLOAD_7B_MODEL=YES
# ENV DOWNLOAD_13B_MODEL=1

# Download the models
RUN mkdir -p /model
RUN DOWNLOAD_7B_MODEL=$DOWNLOAD_7B_MODEL HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN python -u /download_model.py

# Start the handler
CMD python -u /handler.py
