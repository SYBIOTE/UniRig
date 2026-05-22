# UniRig — build for RunPod / local Docker
.PHONY: ckpts build-base build-api build push-base push-api

DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
IMAGE_USER ?= sybiote

ckpts:
	@echo ">>> Downloading checkpoints to ckpts/ (~6GB)..."
	@test -n "$${HF_TOKEN}" || (echo "Set HF_TOKEN (e.g. export HF_TOKEN=your_token)"; exit 1)
	HF_TOKEN=$${HF_TOKEN} hf download VAST-AI/UniRig \
		--include "skeleton/articulation-xl_quantization_256/model.ckpt" \
		--include "skeleton/rignet/model.ckpt" \
		--include "skin/articulation-xl/model.ckpt" \
		--local-dir ckpts
	@echo ">>> Done. Verify: ls ckpts/skeleton ckpts/skin"

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t $(IMAGE_USER)/unirig-base:latest .

build-api: build-base
	$(DOCKER_BUILD) --build-arg BASE_IMAGE=$(IMAGE_USER)/unirig-base:latest \
		-f Dockerfile -t $(IMAGE_USER)/unirig-api:latest .

build: build-api

push-base:
	docker push $(IMAGE_USER)/unirig-base:latest

push-api:
	docker push $(IMAGE_USER)/unirig-api:latest

push: push-base push-api
