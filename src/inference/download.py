import os

from huggingface_hub import hf_hub_download

HF_MAP = {
    'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton/articulation-xl_quantization_256/model.ckpt',
    'experiments/skeleton/rignet/model.ckpt': 'skeleton/rignet/model.ckpt',
    'experiments/skin/articulation-xl/model.ckpt': 'skin/articulation-xl/model.ckpt',
}

REQUIRED_FOR_API = [
    'experiments/skeleton/articulation-xl_quantization_256/model.ckpt',
    'experiments/skeleton/rignet/model.ckpt',
    'experiments/skin/articulation-xl/model.ckpt',
]


def download(ckpt_name: str, base_dir: str | None = None) -> str:
    """Return path to checkpoint. Prefers local copy; falls back to HuggingFace download if UNIRIG_ALLOW_HF_DOWNLOAD=1."""
    if ckpt_name not in HF_MAP:
        raise FileNotFoundError(
            f"Unknown checkpoint: {ckpt_name}. Expected one of: {list(HF_MAP.keys())}"
        )

    # Check local path first (Docker: /app/experiments/... or local: cwd/experiments/...)
    for root in (base_dir, os.environ.get('UNIRIG_APP_DIR', ''), os.getcwd(), '/app'):
        if not root:
            continue
        local_path = os.path.join(root, ckpt_name)
        if os.path.isfile(local_path):
            return os.path.abspath(local_path)

    # Only attempt HF download if explicitly allowed (e.g. during docker build fallback)
    if os.environ.get('UNIRIG_ALLOW_HF_DOWNLOAD', '0') != '1':
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_name}. "
            "Pre-download checkpoints to ckpts/ before building, or set UNIRIG_ALLOW_HF_DOWNLOAD=1 to allow HuggingFace fallback. "
            "See ckpts/README.md for download instructions."
        )

    try:
        return hf_hub_download(
            repo_id='VAST-AI/UniRig',
            filename=HF_MAP[ckpt_name],
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download {ckpt_name} from HuggingFace: {e}"
        ) from e