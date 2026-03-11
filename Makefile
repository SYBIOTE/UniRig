# UniRig — build and run
.PHONY: ckpts build-blender-base build-base build-api build up down logs

DOCKER_BUILD := docker build

ckpts:
	@echo ">>> Downloading checkpoints to ckpts/ (~2GB)..."
	@test -n "$${HF_TOKEN}" || (echo "Set HF_TOKEN (e.g. export HF_TOKEN=your_token)"; exit 1)
	HF_TOKEN=$${HF_TOKEN} hf download VAST-AI/UniRig \
		--include "skeleton/articulation-xl_quantization_256/model.ckpt" \
		--include "skeleton/rignet/model.ckpt" \
		--include "skin/articulation-xl/model.ckpt" \
		--local-dir ckpts
	@echo ">>> Done. Verify: ls ckpts/skeleton ckpts/skin"

build-blender-base:
	$(DOCKER_BUILD) -f Dockerfile.base --target blender-base -t unirig-blender-base:latest .

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t unirig-base:latest .

build-api: build-base
	$(DOCKER_BUILD) -t unirig-api:latest .

build: build-api

up: build
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f
