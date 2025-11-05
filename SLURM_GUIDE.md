# SLURM Batch Scripts for QeRL

This directory contains SLURM batch scripts for running QeRL workflows on HPC clusters with Docker support.

## Main Pipeline Script

### `run_qerl_pipeline.sbatch`

A complete pipeline that runs both model quantization and training in Docker containers.

**What it does:**
1. Pulls/builds the Docker image (configurable)
2. Quantizes a model to NVFP4 format using the `llmcompressor` environment
3. Trains the model with QeRL using the `qerl` environment

**Usage:**

```bash
# Create logs directory
mkdir -p logs

# Recommended: Pull pre-built image from Docker Hub
DOCKER_REGISTRY="yourusername/qerl:latest" sbatch run_qerl_pipeline.sbatch

# Use existing local image
sbatch run_qerl_pipeline.sbatch

# Build Docker image locally (slower, not recommended for HPC)
BUILD_DOCKER=true sbatch run_qerl_pipeline.sbatch

# Use a different model
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" DOCKER_REGISTRY="yourusername/qerl:latest" sbatch run_qerl_pipeline.sbatch
```

**Environment Variables:**

Set these before running the script:

```bash
# Required for Weights & Biases logging
export WANDB_API_KEY="your_wandb_api_key"

# Required for downloading gated models from Hugging Face
export HF_TOKEN="your_huggingface_token"

# Recommended: Pull from Docker registry (e.g., Docker Hub)
export DOCKER_REGISTRY="yourusername/qerl:latest"

# Optional: Build Docker image locally (default: false)
export BUILD_DOCKER=false

# Optional: Use specific local image name (default: qerl:latest)
export DOCKER_IMAGE="qerl:latest"

# Optional: Change the base model to quantize
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

**Configuration:**

Edit the script to customize:
- SLURM resource requirements (`--gpus-per-node`, `--mem`, `--time`, etc.)
- Model selection (`BASE_MODEL`)
- Docker image name (`DOCKER_IMAGE`)
- Directory paths (`MODELS_DIR`, `CKPT_DIR`, etc.)

## Output

The script creates:
- **Logs**: `logs/qerl-pipeline-<job_id>.out` and `.err`
- **Quantized Models**: `models/<model_name>-NVFP4A16-GPTQ/`
- **Training Checkpoints**: `ckpt/<run_name>/`

## Monitoring

Check job status:
```bash
squeue -u $USER
```

View logs in real-time:
```bash
tail -f logs/qerl-pipeline-<job_id>.out
```

Cancel a job:
```bash
scancel <job_id>
```

## Requirements

- SLURM cluster with GPU nodes
- Docker installed and accessible on compute nodes
- NVIDIA Container Toolkit for GPU support in Docker
- Sufficient disk space for models and checkpoints

## Troubleshooting

**Docker permission errors:**
- Ensure your user has Docker access on compute nodes
- You may need to use `sudo` or be in the `docker` group

**GPU not detected:**
- Verify NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
- Check SLURM GPU allocation: `--gpus-per-node=1`

**Out of memory:**
- Increase memory allocation: `--mem=128G`
- Adjust batch size in training script
- Reduce `--vllm-gpu-memory-utilization` in training script

**Model download fails:**
- Set `HF_TOKEN` for gated models
- Check internet connectivity on compute nodes
- Pre-download models to shared storage

## Customization Examples

### Pull pre-built image from Docker Hub (Recommended for HPC):

This is the recommended approach for HPC clusters to avoid building on compute nodes:

```bash
DOCKER_REGISTRY="yourusername/qerl:v1.0.0" sbatch run_qerl_pipeline.sbatch
```

### Use existing local Docker image:

If the image is already built locally:

```bash
sbatch run_qerl_pipeline.sbatch
```

### Build Docker image locally:

Not recommended for HPC, but useful for development:

```bash
BUILD_DOCKER=true sbatch run_qerl_pipeline.sbatch
```

### Run only quantization:

Edit `run_qerl_pipeline.sbatch` and comment out Step 3 (Training).

### Run only training:

Comment out Step 2 (Quantization), ensuring the quantized model already exists:

```bash
DOCKER_REGISTRY="yourusername/qerl:latest" sbatch run_qerl_pipeline.sbatch
```

Then edit the script to skip quantization.

### Multi-GPU training:

Modify the SLURM parameters and training script to use multiple GPUs:
```bash
#SBATCH --gpus-per-node=4
```

And update the training script to use all GPUs.
