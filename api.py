"""
UniRig microservice: auto-rig 3D meshes via file upload.
Exposes POST /rig, POST /rig/fast, POST /skeleton, POST /skeleton/fast (JSON only),
GET /health, and GET /ping (RunPod liveness).

Models load in a background thread: HTTP listens immediately (/ping 204),
then articulation-xl + skin load from the volume; rignet loads on first fast request.

Usage:
    python -m uvicorn api:app --host 0.0.0.0 --port 8080
"""

import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

ALLOWED_EXTENSIONS = {"obj", "fbx", "glb", "gltf", "vrm", "dae"}
DEFAULT_SEED = 12345

_runtime: object | None = None
_load_error: str | None = None


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validate_extension(filename: str) -> str:
    ext = _get_ext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '.{ext}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return ext


def _save_upload(upload: UploadFile, dest: str) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _get_runtime():
    if _load_error is not None:
        raise HTTPException(503, f"Runtime failed to load: {_load_error}")
    if _runtime is None:
        raise HTTPException(503, "Runtime not loaded yet — server is still starting")
    return _runtime


def _probe_response() -> Response | dict:
    """RunPod load balancer: 204 while initializing, 200 when ready."""
    if _load_error is not None:
        raise HTTPException(503, f"Runtime failed to load: {_load_error}")
    if _runtime is None:
        return Response(status_code=204)
    return {"status": "ok"}


def _load_runtime() -> None:
    global _runtime, _load_error
    from runtime import UniRigRuntime

    compile_models = os.environ.get("UNIRIG_COMPILE", "0") == "1"
    app_dir = os.environ.get("UNIRIG_APP_DIR", "/app")
    print(f">>> [api] Background model load started (app_dir={app_dir})")
    try:
        _runtime = UniRigRuntime(app_dir=app_dir, compile_models=compile_models)
        print(">>> [api] Background model load finished — /ping will return 200")
    except Exception as exc:
        print(f">>> [FATAL] Failed to load runtime: {exc}")
        _load_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _runtime, _load_error
    _runtime = None
    _load_error = None
    thread = threading.Thread(target=_load_runtime, daemon=True)
    thread.start()
    yield
    _runtime = None
    _load_error = None


app = FastAPI(title="UniRig API", version="2.1", lifespan=lifespan)


@app.get("/ping")
def ping():
    """RunPod load-balancer probe (204 while initializing, 200 when ready)."""
    return _probe_response()


@app.get("/health")
def health():
    return _probe_response()


@app.post("/rig/fast")
async def rig_mesh_fast(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
):
    """Full pipeline with rignet (faster). Returns skeleton + skin weights as JSON."""
    return await _rig_mesh_impl(file, seed, skeleton_model="rignet")


@app.post("/rig")
async def rig_mesh(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
):
    """Full pipeline with articulation-xl (higher quality). Returns skeleton + skin weights as JSON."""
    return await _rig_mesh_impl(file, seed, skeleton_model="articulation-xl")


async def _rig_mesh_impl(file: UploadFile, seed: int, skeleton_model: str):
    runtime = _get_runtime()
    filename = file.filename or "mesh.glb"
    ext = _validate_extension(filename)

    tmpdir = tempfile.mkdtemp(prefix="unirig_rig_")
    try:
        input_path = os.path.join(tmpdir, f"input.{ext}")
        _save_upload(file, input_path)
        data = runtime.generate_rig_data(
            input_path,
            seed=seed,
            npz_dir=os.path.join(tmpdir, "npz"),
            skeleton_model=skeleton_model,
        )
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Rigging failed: {e}")
    finally:
        _cleanup_dir(tmpdir)


@app.post("/skeleton/fast")
async def generate_skeleton_fast(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
):
    """Skeleton only with rignet (faster). Returns skeleton as JSON."""
    return await _generate_skeleton_impl(file, seed, skeleton_model="rignet")


@app.post("/skeleton")
async def generate_skeleton(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
):
    """Skeleton only with articulation-xl (higher quality). Returns skeleton as JSON."""
    return await _generate_skeleton_impl(file, seed, skeleton_model="articulation-xl")


async def _generate_skeleton_impl(file: UploadFile, seed: int, skeleton_model: str):
    runtime = _get_runtime()
    filename = file.filename or "mesh.glb"
    ext = _validate_extension(filename)

    tmpdir = tempfile.mkdtemp(prefix="unirig_skel_")
    try:
        input_path = os.path.join(tmpdir, f"input.{ext}")
        _save_upload(file, input_path)
        data = runtime.generate_skeleton_data(
            input_path, seed=seed, npz_dir=None, skeleton_model=skeleton_model
        )
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Skeleton generation failed: {e}")
    finally:
        _cleanup_dir(tmpdir)
