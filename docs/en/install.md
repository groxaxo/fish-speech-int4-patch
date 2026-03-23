## Requirements

- GPU Memory: **12 GB recommended** for the default BnB4 API deployment
- System: Linux, WSL
- Miniconda or Anaconda

## Recommended Install: RTX 3060 / 12 GB / BnB4

This fork now ships with a **default install-and-run path** for smaller GPUs. If you want the flagship `s2-pro` model with an OpenAI-compatible API, lazy loading, and an automatic idle shutdown, this is the path to use.

**Prerequisites**: Install system dependencies for audio processing:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### One-command setup

```bash
git clone https://github.com/groxaxo/fish-speech-int4-patch
cd fish-speech-int4-patch

./install_bnb4_3060.sh
```

What the installer does:

- creates a dedicated conda env named `fish-speech-bnb4`
- installs Fish Speech with `bitsandbytes` support
- downloads the official unquantized `fishaudio/s2-pro` checkpoint into `checkpoints/s2-pro`
- keeps the deployment local and self-contained

### Start the API server

```bash
./start_bnb4_3060.sh
```

By default this launches:

- `http://0.0.0.0:8880/v1`
- `--bnb4 --half`
- `--lazy-load`
- `--idle-timeout-seconds 300`
- `GPU_INDEX=0`

Useful overrides:

```bash
GPU_INDEX=3 PORT=8890 ./start_bnb4_3060.sh
ENV_NAME=fish-speech-bnb4 IDLE_TIMEOUT_SECONDS=900 ./start_bnb4_3060.sh
```

!!! note
    `--bnb4` should be paired with the official unquantized `s2-pro` checkpoint. Do not point the launcher at pre-quantized INT4/INT8 checkpoint folders.

## Alternative Installation Methods

Fish Audio S2 still supports the more general installation flows below if you want to customize the environment manually.

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU installation (choose your CUDA version: cu126, cu128, cu129)
pip install -e .[cu129]

# CPU-only installation
pip install -e .[cpu]

# Default installation (uses PyTorch default index)
pip install -e .

# If you encounter an error during installation due to pyaudio, consider using the following command:
# conda install pyaudio
# Then run pip install -e . again
```

### UV

UV provides faster dependency resolution and installation:

```bash
# GPU installation (choose your CUDA version: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU-only installation
uv sync --python 3.12 --extra cpu
```
### Intel Arc XPU support

For Intel Arc GPU users, install with XPU support:

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Install required C++ standard library
conda install libstdcxx -c conda-forge

# Install PyTorch with Intel XPU support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Install Fish Speech
pip install -e .
```

!!! warning
    The `compile` option is not supported on Windows and macOS. If you want to run with compile, you need to install Triton manually.


## Docker Setup

Fish Audio S2 series model provides multiple Docker deployment options to suit different needs. You can use pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images.

We provide Docker images for both WebUI and API server on both GPU (CUDA126 by default) and CPU. You can use the pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images. If you want to build locally, follow the instructions below. If you only want to use pre-built images, follow the [inference guide](inference.md).

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 12 GB GPU memory for the default BnB4 API path, or more for full-precision and custom deployments

# Use docker compose

For development or customization, you can use Docker Compose to build and run locally:

```bash
# Clone the repository first
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Start WebUI with CUDA
docker compose --profile webui up

# Start WebUI with compile optimization
COMPILE=1 docker compose --profile webui up

# Start API server
docker compose --profile server up

# Start API server with compile optimization  
COMPILE=1 docker compose --profile server up

# For CPU-only deployment
BACKEND=cpu docker compose --profile webui up
```

#### Environment Variables for Docker Compose

You can customize the deployment using environment variables:

```bash
# .env file example
BACKEND=cuda              # or cpu
COMPILE=1                 # Enable compile optimization
GRADIO_PORT=7860         # WebUI port
API_PORT=8080            # API server port
UV_VERSION=0.8.15        # UV package manager version
```

The command will build the image and run the container. You can access the WebUI at `http://localhost:7860` and the API server at `http://localhost:8080`.

### Manual Docker Build

For advanced users who want to customize the build process:

```bash
# Build WebUI image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

# Build API server image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# Build CPU-only images (supports multi-platform)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target webui \
    -t fish-speech-webui:cpu .

# Build development image
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Build Arguments

- `BACKEND`: `cuda` or `cpu` (default: `cuda`)
- `CUDA_VER`: CUDA version (default: `12.6.0`)
- `UV_EXTRA`: UV extra for CUDA (default: `cu126`)
- `UBUNTU_VER`: Ubuntu version (default: `24.04`)
- `PY_VER`: Python version (default: `3.12`)

### Volume Mounts

Both methods require mounting these directories:

- `./checkpoints:/app/checkpoints` - Model weights directory
- `./references:/app/references` - Reference audio files directory

### Environment Variables

- `COMPILE=1` - Enable torch.compile for faster inference (~10x speedup)
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI server host
- `GRADIO_SERVER_PORT=7860` - WebUI server port
- `API_SERVER_NAME=0.0.0.0` - API server host  
- `API_SERVER_PORT=8080` - API server port

!!! note
    The Docker containers expect model weights to be mounted at `/app/checkpoints`. Make sure to download the required model weights before starting the containers.

!!! warning
    GPU support requires NVIDIA Docker runtime. For CPU-only deployment, remove the `--gpus all` flag and use CPU images.
