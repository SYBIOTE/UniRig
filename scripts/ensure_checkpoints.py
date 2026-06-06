#!/usr/bin/env python3
"""Verify UniRig checkpoints under {UNIRIG_APP_DIR}/experiments.

Invoked at container start (via docker-entrypoint.sh) — verify only; exit 1 if missing.
See ckpts/README.md and RUNPOD.md for seeding instructions.
"""

from __future__ import annotations

import os
import sys

REQUIRED = [
    "experiments/skeleton/articulation-xl_quantization_256/model.ckpt",
    "experiments/skin/articulation-xl/model.ckpt",
]


def main() -> int:
    app_dir = os.environ.get("UNIRIG_APP_DIR", "/app")
    missing: list[str] = []
    for rel in REQUIRED:
        path = os.path.join(app_dir, rel)
        if not os.path.isfile(path):
            missing.append(rel)

    if missing:
        print("ERROR: UniRig checkpoint(s) missing:", file=sys.stderr)
        for rel in missing:
            print(f"  - {os.path.join(app_dir, rel)}", file=sys.stderr)
        print(
            "Seed ckpts/ on a network volume (see RUNPOD.md) or mount: "
            "-v ./ckpts:/app/experiments:ro",
            file=sys.stderr,
        )
        return 1

    print(">>> UniRig checkpoints OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
