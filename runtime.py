"""
UniRig in-process runtime: loads AR (skeleton) and skin models once at startup,
reuses them across requests. Avoids the ~60-90s per-request cost of spawning
fresh python processes and reloading checkpoints from disk.

Extract (bpy mesh extraction) and merge (bpy mesh merging) still run via
subprocess since bpy has global state that conflicts with long-lived processes.
"""

import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import torch
import yaml
import lightning as L
from box import Box

from src.inference.download import download, REQUIRED_FOR_API
from src.data.extract import get_files
from src.tokenizer.spec import DetokenizeOutput


def _preflight(app_dir: str) -> None:
    """Fail fast with clear errors if runtime environment is not ready."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. UniRig requires a GPU. "
            "Run with: docker run --gpus all ..."
        )
    for ckpt_name in REQUIRED_FOR_API:
        local_path = os.path.join(app_dir, ckpt_name)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(
                f"Checkpoint missing: {ckpt_name}. "
                "Pre-download to ckpts/ before building. See ckpts/README.md."
            )
    # Ensure flash_attn (and thus torch libs) load correctly before model init
    try:
        from flash_attn.modules.mha import MHA  # noqa: F401
    except Exception as e:
        raise ImportError(
            f"Required native extensions failed to load: {e}. "
            "Ensure LD_LIBRARY_PATH includes PyTorch lib dir."
        ) from e


def _copy_checkpoints_to_local(app_dir: str) -> str:
    """Copy checkpoints from GCS mount to /tmp for faster torch.load.
    GCS FUSE reads are ~10x slower than local disk; copying first then loading
    from /tmp significantly reduces Cloud Run cold-start time.
    Set UNIRIG_CACHE_CKPTS=1 to enable (default on Cloud Run)."""
    if os.environ.get("UNIRIG_CACHE_CKPTS", "1") != "1":
        return app_dir
    cache_dir = "/tmp/unirig_ckpts"
    os.makedirs(cache_dir, exist_ok=True)

    def _copy_one(ckpt_name: str) -> None:
        src = os.path.join(app_dir, ckpt_name)
        dst = os.path.join(cache_dir, ckpt_name)
        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)
        print(f">>> [runtime] Copying {ckpt_name} to local cache...")
        shutil.copy2(src, dst)
        print(f">>> [runtime] Cached {ckpt_name}")

    with ThreadPoolExecutor(max_workers=3) as ex:
        ex.map(_copy_one, REQUIRED_FOR_API)

    return cache_dir


from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.datapath import Datapath
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

SKELETON_TASK_ARTICULATION_XL = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
SKELETON_TASK_RIGNET = "configs/task/quick_inference_skeleton_rignet.yaml"
SKIN_TASK = "configs/task/quick_inference_unirig_skin.yaml"
SHELL_TIMEOUT = 300


def _load_config(task: str, path: str) -> Box:
    if path.endswith(".yaml"):
        path = path.removesuffix(".yaml")
    path += ".yaml"
    return Box(yaml.safe_load(open(path, "r")))


def _load_skeleton_pipeline(app_dir: str, task_path: str, ckpt_base_dir: str | None = None) -> dict:
    """Load a skeleton (AR) pipeline from a task config. Returns dict with system, task, configs, etc."""
    ckpt_base = ckpt_base_dir or app_dir
    skel_task = _load_config("task", os.path.join(app_dir, task_path))
    skel_data_config = _load_config("data", os.path.join(app_dir, "configs/data", skel_task.components.data))
    skel_transform_config = _load_config("transform", os.path.join(app_dir, "configs/transform", skel_task.components.transform))

    skel_tokenizer_config = _load_config("tokenizer", os.path.join(app_dir, "configs/tokenizer", skel_task.components.tokenizer))
    skel_tokenizer_config = TokenizerConfig.parse(config=skel_tokenizer_config)
    skel_tokenizer = get_tokenizer(config=skel_tokenizer_config)

    skel_model_config = _load_config("model", os.path.join(app_dir, "configs/model", skel_task.components.model))
    skel_model = get_model(tokenizer=skel_tokenizer, **skel_model_config)

    skel_system_config = _load_config("system", os.path.join(app_dir, "configs/system", skel_task.components.system))
    skel_system = get_system(
        **skel_system_config,
        model=skel_model,
        optimizer_config=None,
        loss_config=None,
        scheduler_config=None,
        steps_per_epoch=1,
    )

    skel_ckpt = download(skel_task.resume_from_checkpoint, base_dir=ckpt_base)
    ckpt = torch.load(skel_ckpt, map_location="cpu")
    skel_system.load_state_dict(ckpt["state_dict"])
    del ckpt

    return {
        "system": skel_system,
        "task": skel_task,
        "predict_transform_config": TransformConfig.parse(config=skel_transform_config.predict_transform_config),
        "predict_dataset_config": DatasetConfig.parse(config=skel_data_config.predict_dataset_config),
        "tokenizer_config": skel_tokenizer_config,
        "process_fn": skel_model._process_fn,
        "data_name": skel_task.components.get("data_name", "raw_data.npz"),
        "writer_config": dict(skel_task.writer),
    }


def _load_skin_pipeline(app_dir: str, ckpt_base_dir: str | None = None) -> dict:
    """Load the skin pipeline. Returns dict with system, configs, etc."""
    ckpt_base = ckpt_base_dir or app_dir
    skin_task = _load_config("task", os.path.join(app_dir, SKIN_TASK))
    skin_data_config = _load_config("data", os.path.join(app_dir, "configs/data", skin_task.components.data))
    skin_transform_config = _load_config("transform", os.path.join(app_dir, "configs/transform", skin_task.components.transform))

    skin_model_config = _load_config("model", os.path.join(app_dir, "configs/model", skin_task.components.model))
    skin_model = get_model(**skin_model_config)

    skin_system_config = _load_config("system", os.path.join(app_dir, "configs/system", skin_task.components.system))
    skin_system = get_system(
        **skin_system_config,
        model=skin_model,
        optimizer_config=None,
        loss_config=None,
        scheduler_config=None,
        steps_per_epoch=1,
    )

    skin_ckpt = download(skin_task.resume_from_checkpoint, base_dir=ckpt_base)
    print(f">>> [runtime] Loading skin checkpoint: {skin_ckpt}")
    ckpt = torch.load(skin_ckpt, map_location="cpu")
    skin_system.load_state_dict(ckpt["state_dict"])
    del ckpt

    return {
        "system": skin_system,
        "task": skin_task,
        "predict_transform_config": TransformConfig.parse(config=skin_transform_config.predict_transform_config),
        "predict_dataset_config": DatasetConfig.parse(config=skin_data_config.predict_dataset_config),
        "process_fn": skin_model._process_fn,
        "data_name": skin_task.components.get("data_name", "predict_skeleton.npz"),
        "writer_config": dict(skin_task.writer),
    }


def _denormalize_joints(
    joints: np.ndarray,
    original_vertices: np.ndarray,
    normalize_into: tuple = (-1.0, 1.0),
) -> np.ndarray:
    """Reverse the AugmentAffine normalization applied during data preprocessing.

    The inference pipeline normalises mesh vertices (and therefore predicted
    joint positions) into a ``normalize_into`` cube centred at the origin.
    This helper maps them back into the original model coordinate space so the
    returned skeleton matches the input mesh.
    """
    bound_min = original_vertices.min(axis=0)
    bound_max = original_vertices.max(axis=0)
    center = (bound_max + bound_min) / 2.0
    extent = bound_max - bound_min
    denom = normalize_into[1] - normalize_into[0]  # 2.0 for [-1, 1]
    scale = np.max(extent / denom)
    bias = (normalize_into[0] + normalize_into[1]) / 2.0  # 0.0 for [-1, 1]
    return (joints - bias) * scale + center


def _zup_to_yup(points: np.ndarray) -> np.ndarray:
    """Rotate points from Z-up (Blender) to Y-up (glTF/Three.js).

    Equivalent to a -90° rotation around the X axis:
    (x, y, z) → (x, z, -y)
    """
    result = np.empty_like(points)
    result[:, 0] = points[:, 0]
    result[:, 1] = points[:, 2]
    result[:, 2] = -points[:, 1]
    return result


def _skeleton_to_json(
    detokenize_output,
    skeleton_model: str,
    original_vertices: "np.ndarray | None" = None,
) -> dict:
    """Convert DetokenizeOutput to frontend-friendly JSON schema.

    If ``original_vertices`` is provided the predicted joint positions are
    denormalized back into the original model coordinate space and converted
    from Blender Z-up to glTF/Three.js Y-up.
    """
    joints = detokenize_output.joints
    if hasattr(joints, "cpu"):
        joints = joints.detach().cpu().numpy()
    else:
        joints = np.asarray(joints)
    tails = detokenize_output.tails
    if tails is not None:
        if hasattr(tails, "cpu"):
            tails = tails.detach().cpu().numpy()
        else:
            tails = np.asarray(tails)
    parents = detokenize_output.parents
    names = detokenize_output.names or [f"bone_{i}" for i in range(len(joints))]

    if original_vertices is not None:
        joints = _denormalize_joints(joints, original_vertices)
        joints = _zup_to_yup(joints)
        if tails is not None:
            tails = _denormalize_joints(tails, original_vertices)
            tails = _zup_to_yup(tails)

    bones = []
    for i in range(len(joints)):
        if parents[i] is None:
            position = joints[i].tolist()
        else:
            p_idx = parents[i]
            delta = joints[i] - joints[p_idx]
            position = delta.tolist()
        parent_name = names[parents[i]] if parents[i] is not None else None
        tail = tails[i].tolist() if tails is not None and i < len(tails) else None
        bone = {
            "name": names[i],
            "parent": parent_name,
            "position": position,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
        if tail is not None:
            bone["tail"] = tail
        bones.append(bone)

    return {
        "name": "UniRig Skeleton",
        "bones": bones,
        "meta": {
            "model": skeleton_model,
        },
    }


def _run_shell(cmd: str, cwd: str = "/app", timeout: int = SHELL_TIMEOUT) -> None:
    print(f">>> [runtime] {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.stdout:
        print("[subprocess stdout]", result.stdout[-4000:])
    if result.stderr:
        print("[subprocess stderr]", result.stderr[-4000:])
    if result.returncode != 0:
        out = result.stdout[-2000:] if result.stdout else "(no stdout)"
        err = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{out}\n--- stderr ---\n{err}"
        )


class UniRigRuntime:
    """Persistent runtime that keeps both models loaded in memory."""

    def __init__(self, app_dir: str = "/app", compile_models: bool = False):
        self._app_dir = app_dir
        self._lock = threading.Lock()

        _preflight(app_dir)
        torch.set_float32_matmul_precision("high")

        # Copy checkpoints from GCS mount to /tmp for faster torch.load (~10x faster than FUSE)
        ckpt_base = _copy_checkpoints_to_local(app_dir)

        # Load all 3 models in parallel (skeleton x2 + skin)
        def load_articulation_xl():
            return "articulation-xl", _load_skeleton_pipeline(app_dir, SKELETON_TASK_ARTICULATION_XL, ckpt_base_dir=ckpt_base)

        def load_rignet():
            return "rignet", _load_skeleton_pipeline(app_dir, SKELETON_TASK_RIGNET, ckpt_base_dir=ckpt_base)

        def load_skin():
            return "skin", _load_skin_pipeline(app_dir, ckpt_base_dir=ckpt_base)

        self._skel_pipelines = {}
        skin_result = None
        with ThreadPoolExecutor(max_workers=3) as ex:
            for result in ex.map(lambda fn: fn(), [load_articulation_xl, load_rignet, load_skin]):
                name, data = result
                if name == "skin":
                    skin_result = data
                else:
                    self._skel_pipelines[name] = data
                    print(f">>> [runtime] Loaded skeleton model: {name}")

        assert skin_result is not None
        self._skin_predict_transform_config = skin_result["predict_transform_config"]
        self._skin_predict_dataset_config = skin_result["predict_dataset_config"]
        self._skin_process_fn = skin_result["process_fn"]
        self._skin_data_name = skin_result["data_name"]
        self._skin_writer_config = skin_result["writer_config"]
        self._skin_task = skin_result["task"]
        self._skin_system = skin_result["system"]
        print(">>> [runtime] Loaded skin model")

        # Optionally compile models for faster inference after warmup
        if compile_models:
            try:
                for name, pipeline in self._skel_pipelines.items():
                    pipeline["system"].model = torch.compile(pipeline["system"].model, mode="reduce-overhead")
                self._skin_system.model = torch.compile(self._skin_system.model, mode="reduce-overhead")
                print(">>> [runtime] torch.compile applied to all models")
            except Exception as e:
                print(f">>> [runtime] torch.compile failed (non-fatal): {e}")

        self._ready = True
        print(">>> [runtime] UniRigRuntime initialized")

    # ── Public API ──

    def generate_skeleton(
        self,
        input_path: str,
        output_path: str,
        seed: int = 12345,
        npz_dir: Union[str, None] = None,
        skeleton_model: str = "articulation-xl",
    ) -> str:
        """Extract mesh + predict skeleton. Returns path to skeleton FBX.
        skeleton_model: 'articulation-xl' (default, higher quality) or 'rignet' (faster, smaller)."""
        with self._lock:
            return self._generate_skeleton(input_path, output_path, seed, npz_dir, skeleton_model)

    def generate_skeleton_data(
        self,
        input_path: str,
        seed: int = 12345,
        npz_dir: Union[str, None] = None,
        skeleton_model: str = "articulation-xl",
    ) -> dict:
        """Extract mesh + predict skeleton. Returns structured skeleton data as a dict (no FBX)."""
        with self._lock:
            return self._generate_skeleton_data(input_path, seed, npz_dir, skeleton_model)

    def generate_skin(
        self,
        input_path: str,
        output_path: str,
        seed: int = 12345,
        npz_dir: Union[str, None] = None,
    ) -> str:
        """Extract mesh + predict skin weights. Returns path to skin FBX."""
        with self._lock:
            return self._generate_skin(input_path, output_path, seed, npz_dir)

    def merge(self, source_path: str, target_path: str, output_path: str) -> str:
        """Merge skeleton/skin FBX into the original mesh. Returns output path."""
        with self._lock:
            _run_shell(
                f"python -X faulthandler -m src.inference.merge"
                f" --require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm"
                f" --num_runs=1 --id=0"
                f" --source={source_path}"
                f" --target={target_path}"
                f" --output={output_path}",
                cwd=self._app_dir,
            )
            return output_path

    # ── Internal ──

    def _run_extract(self, input_path: str, npz_dir: str) -> None:
        print(f">>> [runtime] Running extract with input: {input_path} and output: {npz_dir}")
        """Run Blender mesh extraction via subprocess."""
        _run_shell(
            f"python -X faulthandler -m src.data.extract"
            f" --config=configs/data/quick_inference.yaml"
            f" --require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm"
            f" --force_override=true"
            f" --num_runs=1 --id=0"
            f" --time=api"
            f" --faces_target_count=50000"
            f" --input={input_path}"
            f" --output_dir={npz_dir}",
            cwd=self._app_dir,
        )

    def _generate_skeleton(
        self,
        input_path: str,
        output_path: str,
        seed: int,
        npz_dir: Union[str, None],
        skeleton_model: str = "articulation-xl",
    ) -> str:
        if skeleton_model not in self._skel_pipelines:
            raise ValueError(
                f"Unknown skeleton_model: {skeleton_model}. "
                f"Use one of: {list(self._skel_pipelines.keys())}"
            )
        pipeline = self._skel_pipelines[skeleton_model]

        if npz_dir is None:
            npz_dir = os.path.join(os.path.dirname(input_path), "npz")

        L.seed_everything(seed, workers=True)

        print(f">>> [runtime] Extracting mesh from {input_path} to {npz_dir}")
        # Step 1: extract mesh via bpy subprocess
        self._run_extract(input_path, npz_dir)

        # Step 2: build dataset from extracted npz
        files = get_files(
            data_name=pipeline["data_name"],
            inputs=input_path,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        datapath = Datapath(files=files)

        data = UniRigDatasetModule(
            process_fn=pipeline["process_fn"],
            predict_dataset_config=pipeline["predict_dataset_config"],
            predict_transform_config=pipeline["predict_transform_config"],
            tokenizer_config=pipeline["tokenizer_config"],
            debug=False,
            data_name=pipeline["data_name"],
            datapath=datapath,
        )

        # Step 3: create writer callback targeting the output path
        writer_cfg = dict(pipeline["writer_config"])
        writer_cfg["npz_dir"] = npz_dir
        writer_cfg["output_dir"] = None
        writer_cfg["output_name"] = output_path
        writer_cfg["user_mode"] = True
        writer = get_writer(
            **writer_cfg,
            order_config=pipeline["predict_transform_config"].order_config,
        )

        # Step 4: run prediction with the pre-loaded system
        trainer_config = dict(pipeline["task"].get("trainer", {}))
        trainer = L.Trainer(
            callbacks=[writer],
            logger=False,
            **trainer_config,
        )
        trainer.predict(pipeline["system"], datamodule=data, return_predictions=False)

        # Step 5: free GPU memory for the next stage
        pipeline["system"].cpu()
        torch.cuda.empty_cache()

        return output_path

    def _generate_skeleton_data(
        self,
        input_path: str,
        seed: int,
        npz_dir: Union[str, None],
        skeleton_model: str = "articulation-xl",
    ) -> dict:
        """Run skeleton inference and return structured data (no FBX export)."""
        if skeleton_model not in self._skel_pipelines:
            raise ValueError(
                f"Unknown skeleton_model: {skeleton_model}. "
                f"Use one of: {list(self._skel_pipelines.keys())}"
            )
        pipeline = self._skel_pipelines[skeleton_model]

        if npz_dir is None:
            npz_dir = os.path.join(os.path.dirname(input_path), "npz")

        L.seed_everything(seed, workers=True)

        self._run_extract(input_path, npz_dir)

        files = get_files(
            data_name=pipeline["data_name"],
            inputs=input_path,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        datapath = Datapath(files=files)

        data = UniRigDatasetModule(
            process_fn=pipeline["process_fn"],
            predict_dataset_config=pipeline["predict_dataset_config"],
            predict_transform_config=pipeline["predict_transform_config"],
            tokenizer_config=pipeline["tokenizer_config"],
            debug=False,
            data_name=pipeline["data_name"],
            datapath=datapath,
        )

        writer_cfg = dict(pipeline["writer_config"])
        writer_cfg["npz_dir"] = npz_dir
        writer_cfg["output_dir"] = None
        writer_cfg["output_name"] = None
        writer_cfg["user_mode"] = True
        writer_cfg["export_fbx"] = None
        writer = get_writer(
            **writer_cfg,
            order_config=pipeline["predict_transform_config"].order_config,
        )

        trainer_config = dict(pipeline["task"].get("trainer", {}))
        trainer = L.Trainer(
            callbacks=[writer],
            logger=False,
            **trainer_config,
        )
        predictions = trainer.predict(
            pipeline["system"], datamodule=data, return_predictions=True
        )

        pipeline["system"].cpu()
        torch.cuda.empty_cache()

        if not predictions or not predictions[0]:
            raise RuntimeError("Skeleton prediction returned no output")

        detokenize_output = predictions[0][0]
        if not isinstance(detokenize_output, DetokenizeOutput):
            raise RuntimeError(
                f"Expected DetokenizeOutput, got {type(detokenize_output)}"
            )

        # Diagnostic: print raw bone order and names for debugging name/order mismatch
        n = detokenize_output.num_bones
        names = detokenize_output.names or [f"bone_{i}" for i in range(n)]
        parents = detokenize_output.parents
        print("UniRig detokenize_output: cls=%s parts=%s num_bones=%s" % (detokenize_output.cls, getattr(detokenize_output, "parts", None), n))
        print("UniRig bone order (index -> parent_index, name):", [(i, parents[i] if parents and i < len(parents) else None, names[i] if i < len(names) else "?") for i in range(n)])
        print("UniRig names list (order):", names)

        # Load original (pre-normalisation) vertices so the skeleton can be
        # mapped back into the input model's coordinate space.
        original_vertices = None
        npz_path = os.path.join(files[0], pipeline["data_name"])
        if os.path.isfile(npz_path):
            raw = np.load(npz_path, allow_pickle=True)
            if "vertices" in raw:
                original_vertices = np.asarray(raw["vertices"], dtype=np.float32)

        return _skeleton_to_json(detokenize_output, skeleton_model, original_vertices)

    def _generate_skin(
        self, input_path: str, output_path: str, seed: int, npz_dir: Union[str, None]
    ) -> str:
        if npz_dir is None:
            npz_dir = os.path.join(os.path.dirname(input_path), "npz")

        L.seed_everything(seed, workers=True)

        # Step 1: extract mesh via bpy subprocess
        self._run_extract(input_path, npz_dir)

        # Step 2: build dataset from extracted npz
        files = get_files(
            data_name=self._skin_data_name,
            inputs=input_path,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        datapath = Datapath(files=files)

        data = UniRigDatasetModule(
            process_fn=self._skin_process_fn,
            predict_dataset_config=self._skin_predict_dataset_config,
            predict_transform_config=self._skin_predict_transform_config,
            debug=False,
            data_name=self._skin_data_name,
            datapath=datapath,
        )

        # Step 3: create writer callback targeting the output path
        writer_cfg = dict(self._skin_writer_config)
        writer_cfg["npz_dir"] = npz_dir
        writer_cfg["output_dir"] = None
        writer_cfg["output_name"] = output_path
        writer_cfg["user_mode"] = True
        writer = get_writer(
            **writer_cfg,
            order_config=self._skin_predict_transform_config.order_config,
        )

        # Step 4: run prediction with the pre-loaded system
        trainer_config = dict(self._skin_task.get("trainer", {}))
        trainer = L.Trainer(
            callbacks=[writer],
            logger=False,
            **trainer_config,
        )
        trainer.predict(self._skin_system, datamodule=data, return_predictions=False)

        # Step 5: free GPU memory for the next stage
        self._skin_system.cpu()
        torch.cuda.empty_cache()

        return output_path
