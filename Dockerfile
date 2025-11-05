# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    CONDA_ALWAYS_YES=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all --yes && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    /opt/conda/bin/conda config --set always_yes yes

# Set working directory
WORKDIR /workspace/QeRL

# Copy the repository files
COPY . .

# Create conda environment and install dependencies
RUN conda create -n qerl python=3.10 -y && \
    echo "source activate qerl" >> ~/.bashrc

# Activate environment and install CUDA toolkit
SHELL ["conda", "run", "-n", "qerl", "/bin/bash", "-c"]

RUN conda install nvidia/label/cuda-12.4.1::cuda -y && \
    conda install -c nvidia/label/cuda-12.4.1 cudatoolkit -y

# Install QeRL dependencies
RUN GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]" && \
    pip install torch==2.7.1 && \
    pip install torchaudio==2.7.1 && \
    pip install flash-attn==2.7.4.post1 --no-build-isolation && \
    pip install trl==0.21.0 && \
    pip install vllm==0.10.1 && \
    pip install peft && \
    pip install accelerate==1.10.1 --no-deps

# Setup vLLM and compressed-tensors replacements
RUN site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    cp -v replacement/vllm_replacement/models.py ${site_pkg_path}/vllm/lora/models.py && \
    cp -v replacement/vllm_replacement/worker_manager.py ${site_pkg_path}/vllm/lora/worker_manager.py && \
    mkdir -p simon_lora_path simon_stub_path

# Install and setup compressed-tensors
RUN git clone --branch 0.11.0 --depth 1 https://github.com/neuralmagic/compressed-tensors.git && \
    cd compressed-tensors && \
    pip install -e . --no-deps && \
    cd .. && \
    cp replacement/compressed-tensors_replacement/compressed_linear.py compressed-tensors/src/compressed_tensors/linear/compressed_linear.py && \
    cp replacement/compressed-tensors_replacement/forward.py compressed-tensors/src/compressed_tensors/quantization/lifecycle/forward.py

# Apply trainer replacement
RUN site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    cp -v replacement/trainer.py ${site_pkg_path}/transformers/trainer.py

# Create a non-root user for better security (rootless Docker compatibility)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} qerluser && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash qerluser && \
    chown -R qerluser:qerluser /workspace/QeRL && \
    chown -R qerluser:qerluser /opt/conda

# Switch to non-root user
USER qerluser

# Set the default shell to bash with conda environment activated
SHELL ["/bin/bash", "-c"]

# Expose common ports (optional, can be modified as needed)
EXPOSE 8000

# Default command
CMD ["bash"]
