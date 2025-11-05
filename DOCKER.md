# Docker Guide for QeRL

This guide provides instructions for building and deploying QeRL using Docker.

The Docker image includes two conda environments:
- **qerl** (Python 3.10) - For training and inference with QeRL
- **llmcompressor** (Python 3.12) - For quantizing models to NVFP4 format

## Prerequisites

- Docker installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- NVIDIA Docker runtime for GPU support ([Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Docker Hub account (for pushing images)
- GPU with NVFP4 weight format support (e.g., RTX 5090, H100, B100)

## Building the Docker Image

### Basic Build

Build the Docker image with a tag:

```bash
docker build -t qerl:latest .
```

### Build with Custom Tag

```bash
docker build -t yourusername/qerl:v1.0.0 .
```

### Build Arguments

The Dockerfile is configured to work with default settings. You can customize the user ID and group ID for better integration with your host system:

```bash
# Build with custom user/group IDs (useful for rootless Docker)
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t qerl:latest .
```

This ensures files created inside the container have the same ownership as your host user, which is especially useful for rootless Docker setups.

## Rootless Docker Support

The Docker image is designed to be compatible with rootless Docker for enhanced security. The container runs as a non-root user (`qerluser` with UID 1000 by default).

### Benefits of Rootless Docker

- **Enhanced Security**: Container processes don't run as root
- **Better Isolation**: Reduced risk of privilege escalation
- **File Ownership**: Files created in mounted volumes have correct ownership

### Using with Rootless Docker

If you're using rootless Docker, you can match the container user to your host user:

```bash
# Build with your user/group IDs
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t qerl:latest .

# Run normally - no special flags needed
docker run --gpus all -it qerl:latest
```

For more information on setting up rootless Docker, see:
- [Docker Rootless Mode](https://docs.docker.com/engine/security/rootless/)
- [NVIDIA Container Toolkit with Rootless Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode)

## Running the Docker Container

### Basic Run with GPU Support

```bash
docker run --gpus all -it qerl:latest
```

### Run with Mounted Volumes

Mount your data, models, and checkpoints:

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/QeRL/data \
  -v $(pwd)/models:/workspace/QeRL/models \
  -v $(pwd)/ckpt:/workspace/QeRL/ckpt \
  qerl:latest
```

### Run with Environment Variables

```bash
docker run --gpus all -it \
  -e WANDB_API_KEY=your_wandb_key \
  -e CUDA_VISIBLE_DEVICES=0 \
  qerl:latest
```

### Interactive Development

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace/QeRL \
  --name qerl-dev \
  qerl:latest bash
```

## Uploading to Docker Hub

### 1. Log in to Docker Hub

```bash
docker login
```

Enter your Docker Hub username and password when prompted.

### 2. Tag Your Image

Tag the image with your Docker Hub username and repository name:

```bash
docker tag qerl:latest yourusername/qerl:latest
docker tag qerl:latest yourusername/qerl:v1.0.0
```

Replace `yourusername` with your actual Docker Hub username.

### 3. Push to Docker Hub

Push the tagged image(s):

```bash
# Push the latest tag
docker push yourusername/qerl:latest

# Push a specific version tag
docker push yourusername/qerl:v1.0.0
```

### 4. Verify Upload

Visit your Docker Hub repository at `https://hub.docker.com/r/yourusername/qerl` to verify the upload.

## Using the Published Image

Once uploaded to Docker Hub, others can pull and use your image:

```bash
# Pull the image
docker pull yourusername/qerl:latest

# Run the image
docker run --gpus all -it yourusername/qerl:latest
```

## Training with Docker

The Docker image includes two conda environments:
- **qerl** - For training and inference with QeRL
- **llmcompressor** - For quantizing models to NVFP4 format

### Example: Quantizing a Model with NVFP4

Before training, you may need to quantize your model to NVFP4 format.

1. Start the container with mounted volumes:

```bash
docker run --gpus all -it \
  -v $(pwd)/models:/workspace/QeRL/models \
  qerl:latest bash
```

2. Inside the container, activate the llmcompressor environment:

```bash
conda activate llmcompressor
```

3. Quantize your model:

```bash
cd llm-compressor
python quantize_nvfp4.py --model Qwen/Qwen2.5-7B-Instruct
```

The quantized model will be saved in the models directory.

### Example: Training Qwen2.5-7B with NVFP4

1. Start the container with mounted volumes:

```bash
docker run --gpus all -it \
  -v $(pwd)/models:/workspace/QeRL/models \
  -v $(pwd)/ckpt:/workspace/QeRL/ckpt \
  -e WANDB_API_KEY=your_key \
  qerl:latest bash
```

2. Inside the container, activate the qerl environment:

```bash
conda activate qerl
```

3. Run training (ensure you have a quantized model ready):

```bash
bash training/dapo_qwen2.5-7b_nvfp4_single_gpu.sh
```

## Using Docker Compose

Docker Compose provides a simpler way to manage the container configuration. A `docker-compose.yml` file is included in the repository.

### Setup

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your configuration (e.g., WANDB_API_KEY, HF_TOKEN):

```bash
nano .env  # or use your preferred editor
```

### Running with Docker Compose

Start the container:

```bash
docker-compose up -d
```

Execute commands in the running container:

```bash
docker-compose exec qerl bash
```

Inside the container, activate the environment:

```bash
conda activate qerl
bash training/dapo_qwen2.5-7b_nvfp4_single_gpu.sh
```

Stop the container:

```bash
docker-compose down
```

## Running on HPC with SLURM

For running QeRL workflows on HPC clusters with SLURM job scheduling, we provide batch scripts that automate the complete pipeline.

### SLURM Pipeline Script

The `run_qerl_pipeline.sbatch` script runs the complete workflow:
1. Pulls/builds the Docker image (configurable)
2. Quantizes a model to NVFP4 format
3. Trains the model with QeRL

**Quick Start (Recommended - Pull from Docker Hub):**

```bash
# Set required environment variables
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"
export DOCKER_REGISTRY="yourusername/qerl:latest"

# Submit the job
mkdir -p logs
sbatch run_qerl_pipeline.sbatch
```

**Alternative - Use Local Image:**

```bash
# If image already exists locally
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"

mkdir -p logs
sbatch run_qerl_pipeline.sbatch
```

**Customization:**

```bash
# Pull specific version from registry
DOCKER_REGISTRY="yourusername/qerl:v1.0.0" sbatch run_qerl_pipeline.sbatch

# Use a different model
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" DOCKER_REGISTRY="yourusername/qerl:latest" sbatch run_qerl_pipeline.sbatch

# Build locally (not recommended for HPC)
BUILD_DOCKER=true sbatch run_qerl_pipeline.sbatch
```

For detailed documentation, see [SLURM_GUIDE.md](SLURM_GUIDE.md).

## Optimizing Image Size

The Docker image is quite large due to CUDA toolkit, PyTorch, and other dependencies. To optimize:

### Multi-stage Build (Advanced)

You can create a multi-stage Dockerfile to reduce the final image size. This is left as an exercise for advanced users.

### Remove Unnecessary Files

After building, you can clean up the image:

```bash
docker build --rm -t qerl:latest .
```

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Container Toolkit is installed:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Adjust GPU memory utilization in your training scripts:

```bash
--vllm-gpu-memory-utilization 0.3
```

### Permission Issues

The Docker image runs as a non-root user by default. If you encounter permission issues with mounted volumes, ensure the build arguments match your host user:

```bash
# Rebuild with matching user/group IDs
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t qerl:latest .

# Then run normally
docker run --gpus all -it \
  -v $(pwd)/ckpt:/workspace/QeRL/ckpt \
  qerl:latest
```

Alternatively, you can override the user at runtime (not recommended as it may cause issues with conda environment):

```bash
docker run --gpus all -it --user $(id -u):$(id -g) \
  -v $(pwd)/ckpt:/workspace/QeRL/ckpt \
  qerl:latest
```

## Best Practices

1. **Version Tagging**: Always tag your images with version numbers for reproducibility
2. **Environment Variables**: Use `.env` files for sensitive data (never commit these to git)
3. **Volume Mounting**: Mount data and checkpoints as volumes to persist across container restarts
4. **Resource Limits**: Set memory and GPU limits for production deployments
5. **Regular Updates**: Keep your base images and dependencies updated
6. **Rootless Docker**: Use rootless Docker mode for enhanced security in production environments

## Security Considerations

- **Non-Root User**: The container runs as a non-root user (`qerluser`) by default for improved security
- **Rootless Docker**: Compatible with Docker rootless mode - see the Rootless Docker Support section
- Never include sensitive data (API keys, tokens) in the Docker image
- Use environment variables or secrets management for sensitive configuration
- Regularly scan images for vulnerabilities: `docker scan yourusername/qerl:latest`
- Use specific version tags rather than `latest` in production

## Additional Resources

- [QeRL GitHub Repository](https://github.com/NVlabs/QeRL)
- [QeRL Paper](https://arxiv.org/abs/2510.11696)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

## License

The Docker configuration is provided under the same Apache 2.0 License as the QeRL project.
