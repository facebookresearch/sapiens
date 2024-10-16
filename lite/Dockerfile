#ARG PYTORCH="2.4.1"
#ARG CUDA="12.1"
#ARG CUDNN="9"
#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

FROM nvcr.io/nvidia/pytorch:23.12-py3

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git wget nano ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Verify nvcc works properly
RUN nvcc --version
# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh && \
    echo "source activate base" > ~/.bashrc

SHELL ["/bin/bash", "-c"]

ENV PATH=/root/miniconda3/bin:$PATH
ARG CONDA_ENV=sapiens_lite

RUN conda --version && \
    conda create -n ${CONDA_ENV} python=3.10 -y && \
    source activate ${CONDA_ENV} && \
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y -q && \
    pip install opencv-python tqdm json-tricks && \
    git clone https://github.com/facebookresearch/sapiens.git && \
    cd sapiens/engine && pip install -e . && \
    cd /workspace/sapiens/cv && pip install -e . && \
    cd /workspace/sapiens/det && pip install -e .

RUN echo "source activate ${CONDA_ENV}" > ~/.bashrc

ENTRYPOINT ["/bin/bash"]
