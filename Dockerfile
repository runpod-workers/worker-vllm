# Base image
# The following docker base image is recommended by VLLM: 
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

# Install fast api
RUN pip install fastapi==0.99.1

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add src files (Worker Template)
ADD src .

# Quick temporary updates
RUN pip install git+https://github.com/runpod/runpod-python@main#egg=runpod --compile

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=NONE
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
ARG MODEL=
ARG TOKENIZER=

ENV MODEL=$MODEL
ENV TOKENIZER=$TOKENIZER

# Download the models
RUN mkdir -p /model
RUN MODEL=$MODEL HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN python -u /download_model.py

# Start the handler
CMD MODEL=$MODEL TOKENIZER=$TOKENIZER python -u /handler.py
