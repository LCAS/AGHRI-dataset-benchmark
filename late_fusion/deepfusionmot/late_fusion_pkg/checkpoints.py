"""Checkpoint metadata and lightweight verification helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


YOLO_GENERIC_PATH = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
    "early_fusion/camera_lidar_humble/checkpoints/generic/yolo11s.pt"
)
YOLO_GENERIC_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"

SECOND_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.1.0_models/second/"
    "second_hv_secfpn_8xb6-80e_kitti-3d-3class/"
    "second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"
)
POINTPILLARS_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)

SECOND_CONFIG = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/"
    "benchmarks/mmdetection3d/configs/models/second_kitti_pretrained_eval_aghri.py"
)
POINTPILLARS_CONFIG = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/"
    "benchmarks/mmdetection3d/configs/models/pointpillars_kitti_pretrained_eval_aghri.py"
)

CHECKPOINT_META_CLASS_MAPPING = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
LOCAL_ZERO_SHOT_CONFIG_CLASS_MAPPING = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
KITTI_CLASS_MAPPING = CHECKPOINT_META_CLASS_MAPPING
PEDESTRIAN_CLASS_ID = 0
CLASS_MAPPING_EVIDENCE = {
    "selected_mapping_source": "checkpoint_meta",
    "selected_mapping": CHECKPOINT_META_CLASS_MAPPING,
    "selected_pedestrian_class_id": PEDESTRIAN_CLASS_ID,
    "checkpoint_meta_evidence": (
        "The downloaded official SECOND and PointPillars checkpoints contain metadata "
        "with class order ['Pedestrian', 'Cyclist', 'Car']."
    ),
    "local_config_conflict": (
        "The local AGHRI zero-shot evaluation configs use pred_class_id=1 with a "
        "commented ['Car', 'Pedestrian', 'Cyclist'] ordering. Because mmdet3d is not "
        "available in this environment for an inference smoke test, the generic "
        "baseline follows the checkpoint metadata and records this conflict."
    ),
}


@dataclass(frozen=True)
class CheckpointRecord:
    model_name: str
    architecture: str
    training_dataset: str
    classes: list[str]
    official_source_url: str
    matching_config_file: str
    absolute_local_path: str
    file_size: int | None
    sha256: str | None
    download_date: str
    mmdetection3d_generation: str
    intended_use: str
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _record(
    model_name: str,
    architecture: str,
    training_dataset: str,
    classes: list[str],
    url: str,
    config: Path,
    path: Path,
    generation: str,
    intended_use: str,
) -> CheckpointRecord:
    exists = path.exists()
    return CheckpointRecord(
        model_name=model_name,
        architecture=architecture,
        training_dataset=training_dataset,
        classes=classes,
        official_source_url=url,
        matching_config_file=str(config),
        absolute_local_path=str(path.resolve()),
        file_size=file_size(path) if exists else None,
        sha256=sha256_file(path) if exists else None,
        download_date=datetime.now(timezone.utc).isoformat(),
        mmdetection3d_generation=generation,
        intended_use=intended_use,
        status="present" if exists else "missing",
    )


def build_checkpoint_manifest(checkpoint_dir: Path) -> list[CheckpointRecord]:
    second_path = checkpoint_dir / "second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"
    pp_path = checkpoint_dir / "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
    return [
        _record(
            "generic_yolo11s",
            "YOLO11s",
            "COCO-pretrained",
            ["person"],
            "",
            Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/config/aghri_zed_livox.yaml"),
            YOLO_GENERIC_PATH,
            "Ultralytics YOLO11",
            "generic YOLO camera baseline",
        ),
        _record(
            "second_hv_secfpn_8xb6_80e_kitti_3d_3class",
            "SECOND",
            "KITTI 3D 3-class",
            ["Pedestrian", "Cyclist", "Car"],
            SECOND_URL,
            SECOND_CONFIG,
            second_path,
            "MMDetection3D v1.1.0 model zoo",
            "generic SECOND LiDAR baseline",
        ),
        _record(
            "hv_pointpillars_secfpn_6x8_160e_kitti_3d_3class",
            "PointPillars",
            "KITTI 3D 3-class",
            ["Pedestrian", "Cyclist", "Car"],
            POINTPILLARS_URL,
            POINTPILLARS_CONFIG,
            pp_path,
            "MMDetection3D v1.0.0 model zoo",
            "downloaded PointPillars future baseline",
        ),
    ]


def write_manifest_files(records: list[CheckpointRecord], manifest_path: Path, sha_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )
    lines = []
    for record in records:
        if record.sha256:
            lines.append(f"{record.sha256}  {record.absolute_local_path}")
    sha_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def environment_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version,
    }
    for module_name in ("torch", "mmcv", "mmengine", "mmdet", "mmdet3d", "ultralytics"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import torch

        versions["cuda_available"] = bool(torch.cuda.is_available())
        versions["torch_cuda"] = torch.version.cuda
        versions["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        versions["cuda_available"] = False
    return versions


def torch_checkpoint_summary(path: Path) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict") or payload.get("model") or payload
            keys = list(state_dict.keys()) if hasattr(state_dict, "keys") else []
            meta = payload.get("meta", {})
        else:
            keys = []
            meta = {}
        return {
            "path": str(path),
            "readable": True,
            "state_dict_key_count": len(keys),
            "first_keys": keys[:10],
            "meta": meta,
        }
    except Exception as exc:
        return {"path": str(path), "readable": False, "error": f"{type(exc).__name__}: {exc}"}


def verify_mmdet3d_model(config: Path, checkpoint: Path) -> dict[str, Any]:
    """Try model construction in a subprocess so warnings can be captured."""
    code = f"""
import json
from pathlib import Path
try:
    from mmdet3d.apis import init_model
    model = init_model({str(config)!r}, {str(checkpoint)!r}, device='cpu')
    print(json.dumps({{'ok': True, 'model_class': model.__class__.__name__}}))
except Exception as exc:
    print(json.dumps({{'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "config": str(config),
        "checkpoint": str(checkpoint),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
