# UniRig API — Deploy to Google Cloud Run (with GCS volume for checkpoints)

Deploy UniRig to Cloud Run with checkpoints in a **Cloud Storage bucket** mounted at `/app/experiments`. No need to bake checkpoints into the image.

## Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI installed and logged in (`gcloud auth login`)
- **GPU quota** for Cloud Run in your region (request at [g.co/cloudrun/gpu-quota](https://g.co/cloudrun/gpu-quota) if needed)

---

## 1. One-time setup

Set variables (use your project and region):

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-east4
export BUCKET_NAME=your-unirig-ckpts-bucket
export SERVICE_NAME=unirig-api

gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com

# Artifact Registry repo for UniRig images
gcloud artifacts repositories create unirig \
  --repository-format=docker \
  --location=$REGION \
  --description="UniRig container images"

# GCS bucket for checkpoints (~2GB)
gcloud storage buckets create gs://$BUCKET_NAME --location=$REGION
```

---

## 2. Download checkpoints and upload to GCS

UniRig expects these paths under `/app/experiments` (mount point):

- `skeleton/articulation-xl_quantization_256/model.ckpt`
- `skeleton/rignet/model.ckpt`
- `skin/articulation-xl/model.ckpt`

Download locally, then upload so the **bucket root** mirrors that structure:

```bash
cd /path/to/UniRig

# Download checkpoints to local ckpts/ (one-time)
huggingface-cli download VAST-AI/UniRig \
  --include "skeleton/articulation-xl_quantization_256/model.ckpt" \
  --include "skeleton/rignet/model.ckpt" \
  --include "skin/articulation-xl/model.ckpt" \
  --local-dir ckpts

# Upload to GCS (bucket root = "experiments" content)
gsutil -m cp -r ckpts/* gs://$BUCKET_NAME/

# Verify
gsutil ls gs://$BUCKET_NAME/skeleton/
gsutil ls gs://$BUCKET_NAME/skin/
```

---

## 3. Grant Cloud Run access to the bucket

Cloud Run runs as a service identity. Grant it **Storage Object Viewer** on the bucket:

```bash
# Get the default Cloud Run service account (or use a custom one)
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
export RUN_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding gs://$BUCKET_NAME \
  --member="serviceAccount:${RUN_SERVICE_ACCOUNT}" \
  --role="roles/storage.objectViewer"
```

---

## 4. Build the images

From the **UniRig** directory:

```bash
cd /path/to/UniRig

gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_REGION=$REGION
```

First build can take 30–60+ minutes (base image with PyTorch, flash_attn, etc.). Later builds use cache.

---

## 5. Deploy to Cloud Run with volume mount

Deploy with the GCS bucket mounted at `/app/experiments`, GPU, and a long startup probe (UniRig loads models at startup):

```bash
gcloud run deploy $SERVICE_NAME \
  --image=$REGION-docker.pkg.dev/$PROJECT_ID/unirig/unirig-api:latest \
  --region=$REGION \
  --execution-environment=gen2 \
  --platform=managed \
  --memory=16Gi \
  --cpu=4 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --timeout=600 \
  --max-instances=2 \
  --min-instances=0 \
  --cpu-boost \
  --allow-unauthenticated \
  --add-volume=name=ckpts,type=cloud-storage,bucket=$BUCKET_NAME,readonly=true \
  --add-volume-mount=volume=ckpts,mount-path=/app/experiments \
  --startup-probe=tcpSocket.port=8080,initialDelaySeconds=0,failureThreshold=4,timeoutSeconds=60,periodSeconds=60
```

Notes:

- **Volume:** `ckpts` = GCS bucket mounted read-only at `/app/experiments` (where UniRig looks for checkpoints).
- **Memory:** 4 vCPU allows up to 16Gi; if you hit OOM, use `--cpu=8` and `--memory=32Gi`.
- **Startup:** Probe waits up to 4 minutes for the app to listen on 8080. If startup is still too slow, consider lazy-loading models (load on first request instead of in lifespan).

---

## 6. Test

```bash
export API_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

curl $API_URL/health

# Full pipeline (example)
curl -X POST -F "file=@mesh.glb" -F "output_format=glb" "$API_URL/rig" -o rigged.glb
```

---

## 7. Point the SaaS app at UniRig

In `nirvana-animate-saas`, set the UniRig API base URL (e.g. in `.env.local` or your API proxy config):

```bash
UNIRIG_API_URL=https://unirig-api-xxxxx-uc.a.run.app
```

Use the URL from `gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'`.

---

## Summary

| Step | Command / action |
|------|-------------------|
| 1 | One-time: enable APIs, create Artifact Registry repo `unirig`, create GCS bucket |
| 2 | Download checkpoints with `huggingface-cli`, upload to `gs://$BUCKET_NAME/` |
| 3 | Grant Cloud Run service account `roles/storage.objectViewer` on the bucket |
| 4 | `gcloud builds submit --config cloudbuild.yaml --substitutions=_REGION=$REGION` |
| 5 | `gcloud run deploy` with `--add-volume` and `--add-volume-mount` for `/app/experiments` |
| 6 | Test `/health` and `/rig` (or `/skeleton`, `/skin`) |
| 7 | Set `UNIRIG_API_URL` in the SaaS app |

## Troubleshooting

- **Startup probe timeout:** Model load can take 2–4+ minutes. If it still times out, implement lazy loading (load runtime on first request) so the container listens immediately.
- **OOM:** Increase to `--cpu=8` and `--memory=32Gi` (Cloud Run ties max memory to CPU).
- **Quota exceeded:** Reduce `--max-instances` or request a higher memory/GPU quota for the region.
- **Checkpoints not found:** Ensure bucket layout matches `skeleton/...` and `skin/...` under the bucket root, and the service account has `storage.objectViewer` on the bucket.
