# Docker Guide for QeRL

This guide provides instructions for building and deploying QeRL using Docker.

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

The Dockerfile is configured to work with default settings. If you need to customize the build, you can modify the Dockerfile directly.

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

### Example: Training Qwen2.5-7B with NVFP4

1. Start the container with mounted volumes:

```bash
docker run --gpus all -it \
  -v $(pwd)/models:/workspace/QeRL/models \
  -v $(pwd)/ckpt:/workspace/QeRL/ckpt \
  -e WANDB_API_KEY=your_key \
  qerl:latest bash
```

2. Inside the container, activate the conda environment:

```bash
conda activate qerl
```

3. Prepare your quantized model (if not already done):

```bash
# This requires a separate llmcompressor environment
# See the main README.md for details on quantization
```

4. Run training:

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

If you encounter permission issues with mounted volumes:

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

## Security Considerations

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
