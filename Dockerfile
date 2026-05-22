# UniRig API (production) — RunPod Serverless
# Thin layer on Dockerfile.base: FastAPI + uvicorn + entrypoint.
#
# Build context must be UniRig/ (monorepo subdir).
#
#   DOCKER_BUILDKIT=1 docker build -f Dockerfile.base -t sybiote/unirig-base:latest .
#   DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=sybiote/unirig-base:latest -f Dockerfile -t sybiote/unirig-api:latest .
# runtime.py is copied in Dockerfile (overrides base). Rebuild base only when ML deps/src/configs change.

ARG BASE_IMAGE=docker.io/sybiote/unirig-base:latest
FROM ${BASE_IMAGE}

# ── API-specific deps (fastapi, uvicorn — lightweight) ──
COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements-api.txt

# ── App code (runtime.py here overrides base — avoids full base rebuild for runtime changes) ──
COPY api.py .
COPY runtime.py .
COPY scripts/ scripts/
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# UNIRIG_COMPILE=1 enables torch.compile for faster inference after warmup, but increases cold-start time.
# Set to 1 only if you have measured acceptable startup latency.
ENV UNIRIG_COMPILE=0
ENV UNIRIG_PRELOAD_RIGNET=0
ENV PORT=8080
ENV PORT_HEALTH=8080

EXPOSE 8080

# Requires GPU: run with docker run --gpus all -p 8080:8080 ...
# start-period: 3 models + GPU init can take several minutes on cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT_HEALTH:-${PORT:-8080}}/ping || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["sh", "-c", "exec python -m uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-access-log"]
