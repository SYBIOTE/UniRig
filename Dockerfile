# UniRig API (production)
# Tiny layer on top of the shared base — rebuilds in seconds when only app code changes.
#
# Build (standalone — if not using the base workflow):
#   DOCKER_BUILDKIT=1 docker build -f Dockerfile.base --target blender-base -t unirig-blender-base .
#   DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t unirig-base .
#   DOCKER_BUILDKIT=1 docker build -t unirig-api .
#
# Run:
#   docker run --gpus all -p 8080:8080 unirig-api
#
# Cloud Build:
#   gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=us-central1

# ── The base image tag is overridden by cloudbuild.yaml via --build-arg ──
ARG BASE_IMAGE=unirig-base:latest
FROM ${BASE_IMAGE}

# ── API-specific deps (fastapi, uvicorn — lightweight) ──
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements-api.txt

# ── App code ──
COPY api.py .
COPY scripts/ scripts/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# UNIRIG_COMPILE=1 enables torch.compile for faster inference after warmup, but increases cold-start time.
# Set to 1 only if you have measured acceptable startup latency.
ENV UNIRIG_COMPILE=0

EXPOSE 8080

# Requires GPU: run with docker run --gpus all -p 8080:8080 ...
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ping || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
