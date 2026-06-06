# UniRig Checkpoints

Download checkpoints here before running the API. See [DEPLOY.md](../DEPLOY.md) for full instructions.

**RunPod (shared network volume with HY-Motion):** see [RUNPOD.md](../RUNPOD.md) — upload `skeleton/` and `skin/` to `s3://VOLUME_ID/unirig/`.

```bash
huggingface-cli download VAST-AI/UniRig \
  --include "skeleton/articulation-xl_quantization_256/model.ckpt" \
  --include "skin/articulation-xl/model.ckpt" \
  --local-dir .
```

Or from repo root: `make ckpts`
