# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    CONDA_ALWAYS_YES=true

# Install system dependencies
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ca-certificates \
    # Removed: libmamba (It's not an APT package on this base image)
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda, configure channels, and create the 'qerl' environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p ${CONDA_DIR} && \
    rm miniconda.sh && \
    # Clean up installation to save space
    ${CONDA_DIR}/bin/conda clean -afy && \
    \
    # *** FIX 1: CONDAPATH, TOS, and CHANNEL CONFIGURATION ***
    # Conda is now installed. Ensure its bin directory is in PATH for the remaining commands.
    export PATH=${CONDA_DIR}/bin:$PATH && \
    \
    # Accept TOS non-interactively (no -y flag)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    \
    # Configure channels (conda-forge priority)
    conda config --remove-key channels || true && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    \
    # Create environment (ONLY ONCE)
    conda create -n qerl python=3.10 -y && \
    \
    # Setup activation for non-interactive shell and subsequent commands
    ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate qerl" >> ~/.bashrc

# Set working directory
WORKDIR /workspace/QeRL

# Copy the repository files
COPY . .

# *** FIX 2: Remove redundant environment creation step ***
# The previous step now handles environment creation. This RUN command is deleted.
# OLD: RUN conda create -n qerl python=3.10 -y && \
# OLD:     echo "source activate qerl" >> ~/.bashrc

# Activate environment and install CUDA toolkit
# Use the correct SHELL form for Conda activation in subsequent RUN commands
SHELL ["conda", "run", "-n", "qerl", "/bin/bash", "-c"]

# *** FIX 3: CUDA INSTALLATION (Resolve strict repo priority error) ***
# The fix uses `--override-channels` to allow Conda to pull the CUDA package from the nvidia channel, 
# bypassing the 'strict repo priority' conflict with conda-forge/defaults.
RUN conda install -c nvidia/label/cuda-12.4.1 -c conda-forge cuda=12.4.1 cudatoolkit -y

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

# Switch back to base shell to create llmcompressor environment
SHELL ["/bin/bash", "-c"]

# Create llmcompressor environment for model quantization
RUN source ${CONDA_DIR}/etc/profile.d/conda.sh && \
    conda create -n llmcompressor python=3.12 -y && \
    conda activate llmcompressor && \
    cd llm-compressor && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.11.0 pip install -e . && \
    pip install nvidia-ml-py && \
    cd .. && \
    echo "conda activate llmcompressor" >> ~/.llmcompressor_activate

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