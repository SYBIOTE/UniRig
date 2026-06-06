"""
UniRig in-process runtime: loads AR (skeleton) and skin models once at startup,
reuses them across requests. Avoids the ~60-90s per-request cost of spawning
fresh python processes and reloading checkpoints from disk.

Mesh extraction (bpy) still runs via subprocess since bpy has global state
that conflicts with long-lived processes. Rig output is JSON only (no FBX/merge).
"""

import os
import shutil
import subprocess
import threading
import time
from typing import Union

import numpy as np
import torch
import yaml
import lightning as L
from box import Box

from src.inference.download import download
from src.data.extract import get_files
from src.tokenizer.spec import DetokenizeOutput


def _preflight() -> None:
    """Fail fast if GPU and native extensions are unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. UniRig requires a GPU. "
            "Run with: docker run --gpus all ..."
        )
    try:
        from flash_attn.modules.mha import MHA  # noqa: F401
    except Exception as e:
        raise ImportError(
            f"Required native extensions failed to load: {e}. "
            "Ensure LD_LIBRARY_PATH includes PyTorch lib dir."
        ) from e


def _load_checkpoint(path: str) -> dict:
    """Load a .ckpt from the network volume with minimal overhead."""
    t0 = time.perf_counter()
    try:
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f">>> [runtime] Loaded checkpoint in {time.perf_counter() - t0:.1f}s: {path}")
    return ckpt


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


def _load_skeleton_pipeline(app_dir: str, task_path: str) -> dict:
    """Load a skeleton (AR) pipeline from a task config. Returns dict with system, task, configs, etc."""
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

    skel_ckpt = download(skel_task.resume_from_checkpoint, base_dir=app_dir)
    ckpt = _load_checkpoint(skel_ckpt)
    skel_system.load_state_dict(ckpt["state_dict"])
    del ckpt
    skel_system.eval()

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


def _load_skin_pipeline(app_dir: str) -> dict:
    """Load the skin pipeline. Returns dict with system, configs, etc."""
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

    skin_ckpt = download(skin_task.resume_from_checkpoint, base_dir=app_dir)
    ckpt = _load_checkpoint(skin_ckpt)
    skin_system.load_state_dict(ckpt["state_dict"])
    del ckpt
    skin_system.eval()

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


def _rig_to_json(
    collected: dict,
    skeleton_model: str,
    original_vertices: "np.ndarray | None",
    group_per_vertex: int = 4,
) -> dict:
    """Build a combined skeleton + skin-weights JSON payload from the data
    captured by SkinWriter (collect_data mode).

    No Blender / FBX is involved. Bones, vertices and weights are all derived
    from the same skin batch so they share one coordinate space, then mapped
    back into the original input model space (denormalized + Z-up -> Y-up) so
    the client can nearest-neighbour transfer weights onto the full-res mesh.
    """
    names = list(collected["names"])
    parents_raw = collected["parents"]  # list[int], -1 == root
    joints = np.asarray(collected["joints"], dtype=np.float32)
    tails = np.asarray(collected["tails"], dtype=np.float32)
    vertices = np.asarray(collected["vertices"], dtype=np.float32)
    skin = np.asarray(collected["skin"], dtype=np.float32)
    J = joints.shape[0]
    parents = [None if int(p) < 0 else int(p) for p in parents_raw]

    if original_vertices is not None:
        joints = _zup_to_yup(_denormalize_joints(joints, original_vertices))
        tails = _zup_to_yup(_denormalize_joints(tails, original_vertices))
        vertices = _zup_to_yup(_denormalize_joints(vertices, original_vertices))

    bones = []
    for i in range(J):
        if parents[i] is None:
            position = joints[i].tolist()
        else:
            position = (joints[i] - joints[parents[i]]).tolist()
        bones.append({
            "name": names[i],
            "parent": names[parents[i]] if parents[i] is not None else None,
            "position": position,
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "tail": tails[i].tolist() if i < len(tails) else None,
        })

    # Sparse skin weights: keep the top-K influences per vertex and renormalize.
    num_bones = skin.shape[1]
    K = num_bones if group_per_vertex <= 0 else min(group_per_vertex, num_bones)
    topk_idx = np.argsort(-skin, axis=1)[:, :K]
    topk_w = np.take_along_axis(skin, topk_idx, axis=1)
    sums = topk_w.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    topk_w = topk_w / sums

    N = int(vertices.shape[0])
    return {
        "name": "UniRig Rig",
        "skeleton": {
            "name": "UniRig Skeleton",
            "bones": bones,
            "meta": {"model": skeleton_model},
        },
        "skin": {
            "vertexCount": N,
            "boneNames": names,
            "influencesPerVertex": int(K),
            "boneIndices": topk_idx.astype(np.int32).tolist(),
            "weights": topk_w.astype(np.float32).tolist(),
            "vertices": vertices.astype(np.float32).tolist(),
        },
        "meta": {
            "model": skeleton_model,
            "coordinateSystem": "y-up",
            "space": "original",
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
    """Persistent runtime that keeps models loaded in memory."""

    def __init__(self, app_dir: str = "/app", compile_models: bool = False):
        self._app_dir = app_dir
        self._lock = threading.Lock()
        self._model_init_lock = threading.Lock()
        self._preload_rignet = os.environ.get("UNIRIG_PRELOAD_RIGNET", "0") == "1"

        _preflight()
        print(">>> [runtime] Preflight OK — loading checkpoints from volume into memory...")
        torch.set_float32_matmul_precision("high")

        t0 = time.perf_counter()
        self._skel_pipelines = {
            "articulation-xl": _load_skeleton_pipeline(app_dir, SKELETON_TASK_ARTICULATION_XL),
        }
        print(f">>> [runtime] articulation-xl ready ({time.perf_counter() - t0:.1f}s elapsed)")

        t1 = time.perf_counter()
        skin_result = _load_skin_pipeline(app_dir)
        print(f">>> [runtime] skin ready ({time.perf_counter() - t1:.1f}s elapsed)")

        if self._preload_rignet:
            t2 = time.perf_counter()
            self._skel_pipelines["rignet"] = _load_skeleton_pipeline(app_dir, SKELETON_TASK_RIGNET)
            print(f">>> [runtime] rignet ready ({time.perf_counter() - t2:.1f}s elapsed)")
        else:
            print(">>> [runtime] rignet deferred (loads on first /rig/fast or /skeleton/fast request)")

        self._skin_predict_transform_config = skin_result["predict_transform_config"]
        self._skin_predict_dataset_config = skin_result["predict_dataset_config"]
        self._skin_process_fn = skin_result["process_fn"]
        self._skin_data_name = skin_result["data_name"]
        self._skin_writer_config = skin_result["writer_config"]
        self._skin_task = skin_result["task"]
        self._skin_system = skin_result["system"]

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
        print(f">>> [runtime] UniRigRuntime initialized ({time.perf_counter() - t0:.1f}s total)")

    def _ensure_skeleton_model(self, skeleton_model: str) -> None:
        if skeleton_model in self._skel_pipelines:
            return
        if skeleton_model != "rignet":
            raise ValueError(
                f"Unknown skeleton_model: {skeleton_model}. "
                f"Use one of: articulation-xl, rignet"
            )
        with self._model_init_lock:
            if skeleton_model in self._skel_pipelines:
                return
            print(">>> [runtime] Lazy-loading rignet...")
            t0 = time.perf_counter()
            self._skel_pipelines["rignet"] = _load_skeleton_pipeline(
                self._app_dir, SKELETON_TASK_RIGNET
            )
            print(f">>> [runtime] rignet ready ({time.perf_counter() - t0:.1f}s elapsed)")

    # ── Public API ──

    def generate_skeleton_data(
        self,
        input_path: str,
        seed: int = 12345,
        npz_dir: Union[str, None] = None,
        skeleton_model: str = "articulation-xl",
    ) -> dict:
        """Extract mesh + predict skeleton. Returns structured skeleton data as a dict (no FBX)."""
        self._ensure_skeleton_model(skeleton_model)
        with self._lock:
            return self._generate_skeleton_data(input_path, seed, npz_dir, skeleton_model)

    def generate_rig_data(
        self,
        input_path: str,
        seed: int = 12345,
        npz_dir: Union[str, None] = None,
        skeleton_model: str = "articulation-xl",
        group_per_vertex: int = 4,
    ) -> dict:
        """Full skeleton + skin pipeline returning JSON (no FBX/GLB, no merge).

        Returns a dict with the predicted skeleton and per-vertex skin weights
        (top-K influences) plus the corresponding vertex positions, all in the
        original model space (Y-up). Avoids the headless-Blender armature build
        entirely by passing the skeleton between stages via predict_skeleton.npz.
        """
        self._ensure_skeleton_model(skeleton_model)
        with self._lock:
            return self._generate_rig_data(
                input_path, seed, npz_dir, skeleton_model, group_per_vertex
            )

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

    def _generate_rig_data(
        self,
        input_path: str,
        seed: int,
        npz_dir: Union[str, None],
        skeleton_model: str = "articulation-xl",
        group_per_vertex: int = 4,
    ) -> dict:
        """FBX-free skeleton + skin pipeline that returns JSON.

        Stage 1 predicts the skeleton and writes predict_skeleton.npz (pure
        NumPy via raw_data.save, no Blender). Stage 2 predicts skin weights from
        that npz and captures them in memory (SkinWriter collect_data) instead
        of exporting an FBX. The skeleton, vertices and weights are then
        serialized together.
        """
        if skeleton_model not in self._skel_pipelines:
            raise ValueError(
                f"Unknown skeleton_model: {skeleton_model}. "
                f"Use one of: {list(self._skel_pipelines.keys())}"
            )
        pipeline = self._skel_pipelines[skeleton_model]

        if npz_dir is None:
            npz_dir = os.path.join(os.path.dirname(input_path), "npz")

        L.seed_everything(seed, workers=True)

        # ── Stage 1: extract mesh + predict skeleton → predict_skeleton.npz ──
        print(f">>> [runtime] Extracting mesh from {input_path} to {npz_dir}")
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

        # user_mode=False so the writer saves predict_skeleton.npz.
        writer_cfg = dict(pipeline["writer_config"])
        writer_cfg["npz_dir"] = npz_dir
        writer_cfg["output_dir"] = None
        writer_cfg["output_name"] = None
        writer_cfg["user_mode"] = False
        writer_cfg["export_npz"] = "predict_skeleton"
        writer = get_writer(
            **writer_cfg,
            order_config=pipeline["predict_transform_config"].order_config,
        )

        trainer = L.Trainer(
            callbacks=[writer],
            logger=False,
            **dict(pipeline["task"].get("trainer", {})),
        )
        trainer.predict(pipeline["system"], datamodule=data, return_predictions=False)

        pipeline["system"].cpu()
        torch.cuda.empty_cache()

        # ── Stage 2: predict skin from predict_skeleton.npz, capture in memory ──
        skin_files = get_files(
            data_name=self._skin_data_name,
            inputs=input_path,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        skin_files = [f[1] for f in skin_files]
        skin_datapath = Datapath(files=skin_files)

        skin_data = UniRigDatasetModule(
            process_fn=self._skin_process_fn,
            predict_dataset_config=self._skin_predict_dataset_config,
            predict_transform_config=self._skin_predict_transform_config,
            debug=False,
            data_name=self._skin_data_name,
            datapath=skin_datapath,
        )

        skin_writer_cfg = dict(self._skin_writer_config)
        skin_writer_cfg["npz_dir"] = npz_dir
        skin_writer_cfg["output_dir"] = None
        skin_writer_cfg["output_name"] = None
        skin_writer_cfg["user_mode"] = True
        skin_writer_cfg["export_npz"] = None
        skin_writer_cfg["collect_data"] = True
        skin_writer = get_writer(
            **skin_writer_cfg,
            order_config=self._skin_predict_transform_config.order_config,
        )

        skin_trainer = L.Trainer(
            callbacks=[skin_writer],
            logger=False,
            **dict(self._skin_task.get("trainer", {})),
        )
        skin_trainer.predict(
            self._skin_system, datamodule=skin_data, return_predictions=False
        )

        self._skin_system.cpu()
        torch.cuda.empty_cache()

        if not skin_writer.collected:
            raise RuntimeError("Skin prediction produced no data to serialize")
        collected = skin_writer.collected[0]

        # Original (pre-normalization) vertices so the rig maps back into the
        # input model's coordinate space, matching the /skeleton JSON output.
        original_vertices = None
        raw_npz_path = os.path.join(files[0], "raw_data.npz")
        if os.path.isfile(raw_npz_path):
            raw = np.load(raw_npz_path, allow_pickle=True)
            if "vertices" in raw:
                original_vertices = np.asarray(raw["vertices"], dtype=np.float32)

        return _rig_to_json(
            collected, skeleton_model, original_vertices, group_per_vertex
        )
