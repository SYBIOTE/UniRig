# UniRig API — Deployment Guide

Deploy the UniRig auto-rigging API as a Docker service with GPU support.

**RunPod Serverless + shared network volume:** see [RUNPOD.md](RUNPOD.md).

## Prerequisites

- **NVIDIA GPU** with CUDA 12.x
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker Compose** v2+

## Quick Start

```bash
# 1. Download checkpoints (~2GB, one-time)
make ckpts

# 2. Build and run
make up
```

API: `http://localhost:8080`

## Manual Steps

### 1. Download checkpoints

Checkpoints are **not** in the image. Mount them via volume.

```bash
huggingface-cli download VAST-AI/UniRig \
  --include "skeleton/articulation-xl_quantization_256/model.ckpt" \
  --include "skeleton/rignet/model.ckpt" \
  --include "skin/articulation-xl/model.ckpt" \
  --local-dir ckpts
```

Verify:

```bash
ls ckpts/skeleton/articulation-xl_quantization_256/model.ckpt
ls ckpts/skeleton/rignet/model.ckpt
ls ckpts/skin/articulation-xl/model.ckpt
```

### 2. Build images

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t unirig-base:latest .
DOCKER_BUILDKIT=1 docker build -t unirig-api:latest .
```

First build: ~15–30 min (flash_attn, spconv, etc.). Later builds use cache.

### 3. Run

**Docker Compose (recommended):**

```bash
docker compose up -d
```

**Plain Docker:**

```bash
docker run --gpus all -p 8080:8080 \
  -v $(pwd)/ckpts:/app/experiments:ro \
  unirig-api:latest
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /rig` | Full pipeline (articulation-xl, higher quality) |
| `POST /rig/fast` | Full pipeline (rignet, faster) |
| `POST /skeleton` | Skeleton only (articulation-xl) |
| `POST /skeleton/fast` | Skeleton only (rignet) |
| `POST /skin` | Skin + merge (skeleton FBX + mesh → rigged) |

**Example:**

```bash
curl -X POST -F "file=@mesh.glb" -F "output_format=glb" http://localhost:8080/rig -o rigged.glb
```

## Configuration

| Env | Default | Description |
|-----|---------|-------------|
| `PORT` | 8080 | API port |
| `UNIRIG_COMPILE` | 0 | Set to 1 for torch.compile (faster after warmup, slower cold start) |
| `UNIRIG_APP_DIR` | /app | App root (for checkpoint paths) |

Copy `.env.example` to `.env` and adjust.

## Volume Options

**Bind mount (dev / single host):**

```yaml
volumes:
  - ./ckpts:/app/experiments:ro
```

**Named volume (persistent across restarts):**

```yaml
volumes:
  - unirig-ckpts:/app/experiments

volumes:
  unirig-ckpts:
```

Populate named volume once:

```bash
docker run --rm -v unirig-ckpts:/data -v $(pwd)/ckpts:/src alpine cp -r /src/. /data/
```

## Troubleshooting

**SIGBUS during build (WSL2):** Increase memory in `~/.wslconfig`:

```ini
[wsl2]
memory=8GB
swap=4GB
```

Then `wsl --shutdown` and restart.

**Checkpoint missing:** Ensure `ckpts/` has the three files and is mounted at `/app/experiments`.

**No GPU:** Run with `--gpus all`. Verify: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
