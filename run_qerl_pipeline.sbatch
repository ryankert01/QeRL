#!/bin/bash
#SBATCH --job-name=qerl-pipeline
#SBATCH --output=logs/qerl-pipeline-%j.out
#SBATCH --error=logs/qerl-pipeline-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gputype=a100
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# QeRL Docker Pipeline: Quantization + Training
# ============================================================================
# This script runs the complete QeRL workflow in Docker:
# 1. Build/pull the Docker image
# 2. Quantize a model to NVFP4 format using llmcompressor environment
# 3. Train with QeRL using the qerl environment
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================

# Docker configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-qerl:latest}"  # Can be overridden with registry image
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"  # Optional: set to pull from registry (e.g., "yourusername/qerl:latest")
BUILD_DOCKER=${BUILD_DOCKER:-false}  # Set to true to build locally, false to pull from registry

# Model configuration
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"  # Model to quantize
QUANTIZED_MODEL_NAME="Qwen2.5-3B-Instruct-NVFP4A16-GPTQ"

# Paths (on host)
WORKSPACE_DIR=$(pwd)
MODELS_DIR="${WORKSPACE_DIR}/models"
CKPT_DIR="${WORKSPACE_DIR}/ckpt"
LOGS_DIR="${WORKSPACE_DIR}/logs"

# Create directories if they don't exist
mkdir -p "${MODELS_DIR}"
mkdir -p "${CKPT_DIR}"
mkdir -p "${LOGS_DIR}"

# Environment variables for training
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export HF_TOKEN=${HF_TOKEN:-""}

# ============================================================================
# Step 1: Get Docker Image
# ============================================================================

# Pull from registry if DOCKER_REGISTRY is set
if [ -n "${DOCKER_REGISTRY}" ]; then
    echo "============================================"
    echo "Pulling Docker image from registry..."
    echo "Image: ${DOCKER_REGISTRY}"
    echo "============================================"
    docker pull ${DOCKER_REGISTRY}
    
    # Tag it with local name for convenience
    docker tag ${DOCKER_REGISTRY} ${DOCKER_IMAGE}
    echo "Docker image pulled and tagged successfully!"
    
# Build locally if BUILD_DOCKER is true
elif [ "${BUILD_DOCKER}" = true ]; then
    echo "============================================"
    echo "Building Docker image locally..."
    echo "============================================"
    docker build \
        --build-arg USER_ID=$(id -u) \
        --build-arg GROUP_ID=$(id -g) \
        -t ${DOCKER_IMAGE} \
        ${WORKSPACE_DIR}
    echo "Docker image built successfully!"
else
    echo "============================================"
    echo "Using existing Docker image: ${DOCKER_IMAGE}"
    echo "============================================"
    # Verify image exists
    if ! docker image inspect ${DOCKER_IMAGE} >/dev/null 2>&1; then
        echo "ERROR: Docker image ${DOCKER_IMAGE} not found!"
        echo "Please either:"
        echo "  1. Set DOCKER_REGISTRY to pull from a registry"
        echo "  2. Set BUILD_DOCKER=true to build locally"
        echo "  3. Build the image manually: docker build -t ${DOCKER_IMAGE} ."
        exit 1
    fi
fi

# ============================================================================
# Step 2: Run Model Quantization
# ============================================================================

echo "============================================"
echo "Starting model quantization..."
echo "Model: ${BASE_MODEL}"
echo "============================================"

docker run --rm --gpus all \
    -v ${MODELS_DIR}:/workspace/QeRL/models \
    -e HF_TOKEN=${HF_TOKEN} \
    -e CUDA_VISIBLE_DEVICES=0 \
    ${DOCKER_IMAGE} \
    /bin/bash -c "
        set -e
        source /opt/conda/etc/profile.d/conda.sh
        conda activate llmcompressor
        
        echo 'Running quantization...'
        cd /workspace/QeRL/llm-compressor
        
        # Create output directory
        mkdir -p /workspace/QeRL/models
        
        # Run quantization
        python quantize_nvfp4.py --model ${BASE_MODEL}
        
        # Move quantized model to models directory
        if [ -d './qmodel' ]; then
            mv ./qmodel/* /workspace/QeRL/models/ || true
        fi
        
        echo 'Quantization completed!'
        ls -lh /workspace/QeRL/models/
    "

echo "Quantization completed successfully!"

# ============================================================================
# Step 3: Run Training
# ============================================================================

echo "============================================"
echo "Starting QeRL training..."
echo "============================================"

# Set the quantized model path for the training script
QUANTIZED_MODEL_PATH="/workspace/QeRL/models/${QUANTIZED_MODEL_NAME}"

docker run --rm --gpus all \
    -v ${MODELS_DIR}:/workspace/QeRL/models \
    -v ${CKPT_DIR}:/workspace/QeRL/ckpt \
    -v ${WORKSPACE_DIR}/training:/workspace/QeRL/training \
    -v ${WORKSPACE_DIR}/recipes:/workspace/QeRL/recipes \
    -v ${WORKSPACE_DIR}/qerl.py:/workspace/QeRL/qerl.py \
    -v ${WORKSPACE_DIR}/utils:/workspace/QeRL/utils \
    -v ${WORKSPACE_DIR}/trl_trainer:/workspace/QeRL/trl_trainer \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -e HF_TOKEN=${HF_TOKEN} \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e MODEL_PATH=${QUANTIZED_MODEL_PATH} \
    ${DOCKER_IMAGE} \
    /bin/bash -c "
        set -e
        source /opt/conda/etc/profile.d/conda.sh
        conda activate qerl
        
        echo 'Starting training with quantized model...'
        echo 'Model path: ${QUANTIZED_MODEL_PATH}'
        
        # Export MODEL_PATH for the training script
        export MODEL_PATH=${QUANTIZED_MODEL_PATH}
        
        # Run training script
        cd /workspace/QeRL
        bash training/dapo_qwen2.5-3b_nvfp4_single_gpu.sh
        
        echo 'Training completed!'
    "

echo "============================================"
echo "Pipeline completed successfully!"
echo "============================================"
echo "Quantized model: ${MODELS_DIR}/${QUANTIZED_MODEL_NAME}"
echo "Training checkpoints: ${CKPT_DIR}"
echo "============================================"
