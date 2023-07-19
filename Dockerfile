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
RUN pip install git+https://github.com/runpod/runpod-python@multijob2#egg=runpod    --compile

CMD python -u /handler.py
