# UniRig API — Local Docker (dev)

For **RunPod Serverless** deployment, see [RUNPOD.md](RUNPOD.md).

## Prerequisites

- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Checkpoints in `ckpts/` (`make ckpts`)

## Build & run

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t sybiote/unirig-base:latest .
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=docker.io/sybiote/unirig-base:latest -f Dockerfile -t sybiote/unirig-api:latest .

docker run --gpus all -p 8080:8080 \
  -v "$(pwd)/ckpts:/app/experiments:ro" \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  sybiote/unirig-api:latest
```

## API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /ping`, `/health` | Liveness |
| `POST /rig` | Full pipeline JSON — skeleton + skin weights (articulation-xl) |
| `POST /skeleton` | Skeleton JSON only (articulation-xl) |

```bash
curl -X POST -F "file=@mesh.glb" http://localhost:8080/rig
```

## Configuration

| Env | Default | Description |
|-----|---------|-------------|
| `UNIRIG_APP_DIR` | `/app` | App root |
| `UNIRIG_COMPILE` | `0` | `1` = torch.compile (slower cold start) |
| `SKIP_CHECKPOINT_PREP` | `0` | `1` = skip entrypoint verify |
