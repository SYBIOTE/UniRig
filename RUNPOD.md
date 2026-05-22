# RunPod (Serverless) — UniRig API

Deploy UniRig on **[RunPod](https://www.runpod.io/)** Serverless with a **network volume** for checkpoints. This guide uses a **shared volume** with HY-Motion (`ckpts/` + `unirig/` prefixes).

## Layout

| File | Role |
|------|------|
| `Dockerfile.base` | CUDA + Blender/bpy + PyTorch + ML deps; checkpoints **not** baked in |
| `Dockerfile` | FastAPI API on top of `BASE_IMAGE` |
| `docker-entrypoint.sh` | Symlink volume → `/app/experiments`, verify checkpoints |
| `scripts/ensure_checkpoints.py` | Fail fast if weights are missing |

## Checkpoints (~6 GB)

Three files under `experiments/` (local `ckpts/` mirrors this):

```
skeleton/articulation-xl_quantization_256/model.ckpt   (~1.4 GB)
skeleton/rignet/model.ckpt                             (~593 MB)
skin/articulation-xl/model.ckpt                        (~4.1 GB)
```

Download locally (one-time): `make ckpts` — see [ckpts/README.md](ckpts/README.md).

## Shared network volume with HY-Motion

Recommended layout on one volume (e.g. `vpwhvs7cia` in **EU-RO-1**):

```
/runpod-volume/
  ckpts/          ← HY-Motion (existing)
    tencent/...
    Qwen3-8B/...
    clip-vit-large-patch14/...
  unirig/         ← UniRig
    skeleton/...
    skin/...
```

**Volume size:** ~25 GB used when both are seeded; provision **26–32 GB**.

### Seed UniRig weights via S3 API

From `UniRig/` (skip `.cache/`):

```bash
aws s3 cp ckpts/skeleton/articulation-xl_quantization_256/model.ckpt \
  s3://YOUR_VOLUME_ID/unirig/skeleton/articulation-xl_quantization_256/model.ckpt \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/

aws s3 cp ckpts/skeleton/rignet/model.ckpt \
  s3://YOUR_VOLUME_ID/unirig/skeleton/rignet/model.ckpt \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/

aws s3 cp ckpts/skin/articulation-xl/model.ckpt \
  s3://YOUR_VOLUME_ID/unirig/skin/articulation-xl/model.ckpt \
  --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io/
```

Verify the three `.ckpt` files exist under `unirig/skeleton/` and `unirig/skin/`.

## Build & push

Context must be **`UniRig/`** (monorepo):

```bash
cd UniRig
DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t YOUR_USER/unirig-base:latest .
docker push YOUR_USER/unirig-base:latest

docker build --build-arg BASE_IMAGE=YOUR_USER/unirig-base:latest \
  -f Dockerfile -t YOUR_USER/unirig-api:latest .
docker push YOUR_USER/unirig-api:latest
```

In RunPod Git/Docker build, set build arg **`BASE_IMAGE=docker.io/sybiote/unirig-base:latest`** (or your registry tag).

### Troubleshooting Git build: `unirig-base:latest: pull access denied`

RunPod is pulling `docker.io/library/unirig-base:latest` (no Docker Hub user). Fix:

1. **Build arg** in RunPod endpoint → Edit → Build → add:
   - Name: `BASE_IMAGE`
   - Value: `docker.io/sybiote/unirig-base:latest`
2. **Push base first:** `docker push sybiote/unirig-base:latest` (repo must be **public**, or add registry credentials in RunPod).
3. **Push updated `Dockerfile`** to the Git repo RunPod builds from (default is now `docker.io/sybiote/unirig-base:latest`, but older commits used bare `unirig-base:latest`).

## RunPod Serverless endpoint

1. Attach the **same network volume** as HY-Motion (same datacenter).
2. Worker image: **`unirig-api`**.
3. **Load balancer** endpoint (direct HTTP to FastAPI paths).

### Environment

| Variable | Example | Notes |
|----------|---------|--------|
| `UNIRIG_APP_DIR` | `/app` | App root |
| `UNIRIG_CKPTS_ROOT` | `/runpod-volume/unirig` | Optional; auto-detected if dir exists |
| `UNIRIG_CACHE_CKPTS` | `1` | Copy volume → `/tmp` at startup (faster load) |
| `UNIRIG_COMPILE` | `0` | Set `1` only after measuring cold-start impact |
| `NVIDIA_DRIVER_CAPABILITIES` | `graphics,compute,utility` | Required for Blender headless |

Entrypoint links `/runpod-volume/unirig` → `/app/experiments` automatically when present.

### Suggested endpoint settings

| Setting | Value |
|---------|--------|
| GPU | L4 24GB or RTX 4090 |
| Request timeout | 600s |
| Active workers | `0` (cost) or `1` (avoid 2–4 min cold start) |

### HTTP probes

- `GET /ping` — liveness (`{"status":"ok"}`)
- `GET /health` — same once models are loaded

Auth (load balancer): `Authorization: Bearer YOUR_RUNPOD_API_KEY`

```bash
curl -sS https://YOUR_ID.api.runpod.ai/ping \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## SaaS app

In `nirvana-animate-saas`:

```env
UNIRIG_API_URL=https://YOUR_ID.api.runpod.ai
RUNPOD_API_KEY=rpa_...
```

Push code and redeploy Vercel. The proxy adds RunPod auth when `RUNPOD_API_KEY` is set.

## References

- Local Docker: [DEPLOY.md](DEPLOY.md)
- HY-Motion RunPod (shared volume): [../HY-Motion-1.0/RUNPOD.md](../HY-Motion-1.0/RUNPOD.md)
