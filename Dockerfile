# ✅ Base image with CUDA 12.4
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment vars
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 🔧 Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    gcc g++ make cmake ninja-build \
    libgomp1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 🐍 Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh

# 🧪 Create Conda env and install sparsehash (skip cuda-toolkit since we're using base image CUDA)
RUN conda create -n spatiallm python=3.11 -y && \
    conda run -n spatiallm conda install -c conda-forge sparsehash -y && \
    conda clean -afy

# 📦 Install Poetry
RUN conda run -n spatiallm pip install poetry && \
    conda run -n spatiallm poetry config virtualenvs.create false --local

# 📁 Clone SpatialLM
WORKDIR /workspace
RUN git clone https://github.com/halimo22/SpatialLM
WORKDIR /workspace/SpatialLM

# 📜 Install dependencies via Poetry
RUN conda run -n spatiallm poetry install

# Install basic dependencies first
RUN conda run -n spatiallm pip install --no-cache-dir \
    fastapi uvicorn[standard] peft python-multipart open3d

# 🎯 Set correct CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/opt/conda/envs/spatiallm/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/conda/envs/spatiallm/lib:$LD_LIBRARY_PATH" \
    TORCH_CUDA_ARCH_LIST="6.1;8.6;8.9" \
    MAKEFLAGS="-j1" \
    MAX_JOBS=1 \
    IABN_FORCE_CUDA="1" \
    FORCE_CUDA="1"

# Install torchsparse from source with proper environment
RUN conda run -n spatiallm poe install-torchsparse
# Copy files before heavy compilation
COPY start.sh /workspace/SpatialLM/

COPY app.py /workspace/SpatialLM/

# Make startup script executable
RUN chmod +x /workspace/SpatialLM/start.sh


# Expose port
EXPOSE 8000

# 🧼 Set entrypoint with better error handling
CMD ["/workspace/SpatialLM/start.sh"]