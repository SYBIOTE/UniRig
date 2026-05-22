# UniRig — RunPod API

[VAST-AI/UniRig](https://github.com/VAST-AI-Research/UniRig) · [Paper](https://arxiv.org/abs/2504.12451) · [Models (Hugging Face)](https://huggingface.co/VAST-AI/UniRig)

This tree is trimmed for **GPU inference on [RunPod](https://www.runpod.io/)** Serverless with a **network volume** for checkpoints. It ships a **FastAPI** service (`api.py`) and Docker images — no Cloud Build, training CLI, or shell launch scripts.

| Doc | Purpose |
|-----|---------|
| [RUNPOD.md](RUNPOD.md) | Build, push, env vars, shared volume layout |
| [DEPLOY.md](DEPLOY.md) | Local Docker dev |
| [ckpts/README.md](ckpts/README.md) | Checkpoint download / volume sync |

**License:** see [LICENSE](LICENSE).
