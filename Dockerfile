# Base image
FROM nvidia/cuda:10.1-devel-ubuntu18.04
MAINTAINER kevin.mets@uantwerpen.be

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    wget \
    zip \
    unzip \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip.
RUN pip3 install --upgrade pip

# Install torch and additional dependencies.
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install jupyter gpustat wandb

# Install PyTorch Geometric and dependencies (graph neural networks).
RUN TORCH_CUDA_ARCH_LIST=7.0 pip3 install install torch-sparse==latest+cu101 torch-scatter==latest+cu101 torch-cluster==latest+cu101 torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
RUN TORCH_CUDA_ARCH_LIST=7.0 pip3 install torch-geometric

# Install ituml package.
COPY . /ituml
WORKDIR /ituml
RUN pip3 install -e .
