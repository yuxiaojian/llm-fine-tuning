# Start with the NVIDIA CUDA base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update, install necessary packages, and clean up in a single RUN command
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Python installation
RUN python --version && pip --version

# Set the working directory in the container
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# Command to run when the container starts
CMD ["/bin/bash"]