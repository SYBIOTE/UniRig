#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${SKIP_CHECKPOINT_PREP:-0}" =~ ^(1|true|yes)$ ]]; then
  exec "$@"
fi

# RunPod shared volume: HY-Motion uses /runpod-volume/ckpts; UniRig uses /runpod-volume/unirig
CKPTS_SRC="${UNIRIG_CKPTS_ROOT:-}"
if [[ -z "$CKPTS_SRC" && -d /runpod-volume/unirig ]]; then
  CKPTS_SRC=/runpod-volume/unirig
fi
if [[ -n "$CKPTS_SRC" && -d "$CKPTS_SRC" ]]; then
  CKPTS_SRC="$(readlink -f "$CKPTS_SRC")"
  if [[ -L /app/experiments ]] && [[ "$(readlink -f /app/experiments)" == "$CKPTS_SRC" ]]; then
    echo ">>> /app/experiments already linked -> $CKPTS_SRC"
  else
    rm -rf /app/experiments
    ln -sfn "$CKPTS_SRC" /app/experiments
    echo ">>> Linked /app/experiments -> $CKPTS_SRC"
  fi
fi

if ! python scripts/ensure_checkpoints.py; then
  exit 1
fi

exec "$@"
