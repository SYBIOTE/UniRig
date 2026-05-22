"""
UniRig microservice: auto-rig 3D meshes via file upload.
Exposes POST /rig, POST /rig/fast, POST /skeleton, POST /skeleton/fast, POST /skin,
GET /health, and GET /ping (RunPod liveness).

Models are loaded in a background thread at startup so the HTTP server listens
immediately; /ping returns 204 until loading completes, then 200.

Usage:
    python -m uvicorn api:app --host 0.0.0.0 --port 8080
"""

import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from runtime import UniRigRuntime

ALLOWED_EXTENSIONS = {"obj", "fbx", "glb", "gltf", "vrm", "dae"}
ALLOWED_OUTPUT_FORMATS = {"glb", "fbx", "gltf"}
DEFAULT_SEED = 12345
DEFAULT_OUTPUT_FORMAT = "glb"

_runtime: UniRigRuntime | None = None
_load_error: str | None = None


def _get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validate_extension(filename: str) -> str:
    ext = _get_ext(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '.{ext}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return ext


def _validate_output_format(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(400, f"output_format must be one of: {', '.join(sorted(ALLOWED_OUTPUT_FORMATS))}")
    return fmt


SKELETON_OUTPUT_FORMATS = {"json", "fbx"}


def _validate_skeleton_output_format(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in SKELETON_OUTPUT_FORMATS:
        raise HTTPException(
            400,
            f"skeleton output_format must be one of: {', '.join(sorted(SKELETON_OUTPUT_FORMATS))}",
        )
    return fmt


def _save_upload(upload: UploadFile, dest: str) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _cleanup_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _get_runtime() -> UniRigRuntime:
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
    compile_models = os.environ.get("UNIRIG_COMPILE", "0") == "1"
    app_dir = os.environ.get("UNIRIG_APP_DIR", "/app")
    try:
        _runtime = UniRigRuntime(app_dir=app_dir, compile_models=compile_models)
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


app = FastAPI(title="UniRig API", version="2.0", lifespan=lifespan)


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
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    """Full pipeline with rignet (faster, smaller model). Returns the rigged model."""
    return await _rig_mesh_impl(file, seed, output_format, skeleton_model="rignet")


@app.post("/rig")
async def rig_mesh(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    """Full pipeline with articulation-xl (higher quality). Returns the rigged model."""
    return await _rig_mesh_impl(file, seed, output_format, skeleton_model="articulation-xl")


async def _rig_mesh_impl(
    file: UploadFile,
    seed: int,
    output_format: str,
    skeleton_model: str,
):
    runtime = _get_runtime()
    filename = file.filename or "mesh.glb"
    ext = _validate_extension(filename)
    output_format = _validate_output_format(output_format)

    tmpdir = tempfile.mkdtemp(prefix="unirig_rig_")
    try:
        input_path = os.path.join(tmpdir, f"input.{ext}")
        _save_upload(file, input_path)

        skeleton_path = os.path.join(tmpdir, "skeleton.fbx")
        skin_path = os.path.join(tmpdir, "skin.fbx")
        output_path = os.path.join(tmpdir, f"rigged.{output_format}")
        npz_dir = os.path.join(tmpdir, "npz")

        runtime.generate_skeleton(input_path, skeleton_path, seed=seed, npz_dir=npz_dir, skeleton_model=skeleton_model)
        runtime.generate_skin(skeleton_path, skin_path, seed=seed, npz_dir=npz_dir)
        runtime.merge(skin_path, input_path, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(500, "Rigging failed — no output file produced")

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"rigged.{output_format}",
            background=BackgroundTask(_cleanup_dir, tmpdir),
        )
    except HTTPException:
        _cleanup_dir(tmpdir)
        raise
    except Exception as e:
        _cleanup_dir(tmpdir)
        raise HTTPException(500, f"Rigging failed: {e}")


@app.post("/skeleton/fast")
async def generate_skeleton_fast(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
    output_format: str = Form("fbx"),
):
    """Skeleton only with rignet (faster). Returns skeleton as JSON or FBX."""
    return await _generate_skeleton_impl(
        file, seed, skeleton_model="rignet", output_format=output_format
    )


@app.post("/skeleton")
async def generate_skeleton(
    file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
    output_format: str = Form("fbx"),
):
    """Skeleton only with articulation-xl (higher quality). Returns skeleton as JSON or FBX."""
    return await _generate_skeleton_impl(
        file, seed, skeleton_model="articulation-xl", output_format=output_format
    )


async def _generate_skeleton_impl(
    file: UploadFile, seed: int, skeleton_model: str, output_format: str = "fbx"
):
    output_format = _validate_skeleton_output_format(output_format)
    runtime = _get_runtime()
    filename = file.filename or "mesh.glb"
    ext = _validate_extension(filename)

    tmpdir = tempfile.mkdtemp(prefix="unirig_skel_")
    try:
        input_path = os.path.join(tmpdir, f"input.{ext}")
        _save_upload(file, input_path)

        if output_format == "json":
            data = runtime.generate_skeleton_data(
                input_path, seed=seed, npz_dir=None, skeleton_model=skeleton_model
            )
            _cleanup_dir(tmpdir)
            return JSONResponse(content=data)

        skeleton_path = os.path.join(tmpdir, "skeleton.fbx")
        runtime.generate_skeleton(
            input_path, skeleton_path, seed=seed, skeleton_model=skeleton_model
        )

        if not os.path.exists(skeleton_path):
            raise HTTPException(500, "Skeleton generation failed — no output file produced")

        return FileResponse(
            skeleton_path,
            media_type="application/octet-stream",
            filename="skeleton.fbx",
            background=BackgroundTask(_cleanup_dir, tmpdir),
        )
    except HTTPException:
        _cleanup_dir(tmpdir)
        raise
    except Exception as e:
        _cleanup_dir(tmpdir)
        raise HTTPException(500, f"Skeleton generation failed: {e}")


@app.post("/skin")
async def generate_skin(
    skeleton_file: UploadFile = File(...),
    mesh_file: UploadFile = File(...),
    seed: int = Form(DEFAULT_SEED),
    output_format: str = Form(DEFAULT_OUTPUT_FORMAT),
):
    """Skin + merge: predict skinning weights on a skeleton, merge back into the original mesh."""
    runtime = _get_runtime()
    skel_filename = skeleton_file.filename or "skeleton.fbx"
    mesh_filename = mesh_file.filename or "mesh.glb"

    skel_ext = _get_ext(skel_filename)
    if skel_ext not in ("fbx",):
        raise HTTPException(400, "skeleton_file must be an FBX file")

    mesh_ext = _validate_extension(mesh_filename)
    output_format = _validate_output_format(output_format)

    tmpdir = tempfile.mkdtemp(prefix="unirig_skin_")
    try:
        skeleton_input = os.path.join(tmpdir, "skeleton_input.fbx")
        mesh_input = os.path.join(tmpdir, f"mesh_input.{mesh_ext}")
        _save_upload(skeleton_file, skeleton_input)
        _save_upload(mesh_file, mesh_input)

        skin_path = os.path.join(tmpdir, "skin.fbx")
        output_path = os.path.join(tmpdir, f"rigged.{output_format}")
        npz_dir = os.path.join(tmpdir, "npz")

        runtime.generate_skin(skeleton_input, skin_path, seed=seed, npz_dir=npz_dir)
        runtime.merge(skin_path, mesh_input, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(500, "Skinning failed — no output file produced")

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"rigged.{output_format}",
            background=BackgroundTask(_cleanup_dir, tmpdir),
        )
    except HTTPException:
        _cleanup_dir(tmpdir)
        raise
    except Exception as e:
        _cleanup_dir(tmpdir)
        raise HTTPException(500, f"Skinning failed: {e}")
