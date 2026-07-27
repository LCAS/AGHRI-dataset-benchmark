"""Live detector profile resolution for online ROS 2 inference.

This module intentionally does not read detector prediction caches.  It only
resolves model/config/checkpoint identities for live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from late_fusion_pkg.checkpoints import sha256_file


REPO_ROOT = Path("/home/prabuddhi/Desktop/late_fusion/deepfusionmot")

MODEL_CHOICES = ("generic", "finetuned")
LIDAR_DETECTOR_CHOICES = ("second", "pointpillars")
DATASET_PROFILES = ("aghri", "kitti_tracking", "kitti_object")

GENERIC_YOLO = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
    "early_fusion/camera_lidar_humble/checkpoints/generic/yolo11s.pt"
)
FINETUNED_YOLO = Path(
    "/home/prabuddhi/Desktop/Early_Fus/checkpoints/"
    "aghri_yolo11s_finetune/cluster_run/weights/best.pt"
)

GENERIC_SECOND = REPO_ROOT / "checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"
GENERIC_POINTPILLARS = (
    REPO_ROOT
    / "checkpoints/kitti_pretrained/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)
FINETUNED_SECOND = REPO_ROOT / "checkpoints/aghri_second_finetune/cluster_run/weights/best.pth"
FINETUNED_POINTPILLARS = REPO_ROOT / "checkpoints/aghri_pointpillars_finetune/cluster_run/weights/best.pth"
KITTI_FINETUNED_YOLO = (
    REPO_ROOT
    / "checkpoints/kitti_person_finetune/cluster_run/yolo11s_kitti_person_seed42_allframes_host/weights/best.pt"
)
KITTI_FINETUNED_SECOND = (
    REPO_ROOT
    / "checkpoints/kitti_person_finetune/cluster_run/second_kitti_person_seed42_full/"
    "best_kitti_bev_ap_mean_epoch_22.pth"
)
KITTI_FINETUNED_POINTPILLARS = (
    REPO_ROOT
    / "checkpoints/kitti_person_finetune/cluster_run/pointpillars_kitti_person_seed42_full/"
    "best_kitti_bev_ap_mean_epoch_46.pth"
)

GENERIC_SECOND_AGHRI_CONFIG = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
    "3d-detection/benchmarks/mmdetection3d/configs/models/second_kitti_pretrained_eval_aghri.py"
)
GENERIC_POINTPILLARS_AGHRI_CONFIG = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
    "3d-detection/benchmarks/mmdetection3d/configs/models/pointpillars_kitti_pretrained_eval_aghri.py"
)
GENERIC_SECOND_KITTI_CONFIG = Path(
    "/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/lib/python3.10/"
    "site-packages/mmdet3d/.mim/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py"
)
GENERIC_POINTPILLARS_KITTI_CONFIG = Path(
    "/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/lib/python3.10/"
    "site-packages/mmdet3d/.mim/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
)
FINETUNED_SECOND_CONFIG = REPO_ROOT / "checkpoints/aghri_second_finetune/cluster_run/configs/second_aghri_finetune_seed42.py"
FINETUNED_POINTPILLARS_CONFIG = (
    REPO_ROOT / "checkpoints/aghri_pointpillars_finetune/cluster_run/configs/pointpillars_aghri_finetune_seed42.py"
)
KITTI_FINETUNED_SECOND_CONFIG = REPO_ROOT / "config/kitti_person_finetune/second_kitti_person_finetune_seed42.py"
KITTI_FINETUNED_POINTPILLARS_CONFIG = (
    REPO_ROOT / "config/kitti_person_finetune/pointpillars_kitti_person_finetune_seed42.py"
)

EXPECTED_SHA256 = {
    str(GENERIC_YOLO): "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    str(FINETUNED_YOLO): "bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110",
    str(GENERIC_SECOND): "b086d0a369d0b955b3ff45b6849a4370c46dea2b95445671837a989839eb2bbd",
    str(FINETUNED_SECOND): "06e67ae24080f9eba67d745c71e8554ba950b2aa9d2886348ff260a257fe87ab",
    str(GENERIC_POINTPILLARS): "37dc242098f0d3dd7d870e0fe7cd4514f76fd85a04fb1377681279f963b3cbe5",
    str(FINETUNED_POINTPILLARS): "50a0dde7c5650a46e50785050c632a405dedcc24a8513713e07371e3666d6847",
    str(KITTI_FINETUNED_YOLO): "bd629aeaaa2dbf66ad2e1dd5acd45591ceed4e96e276079bc2f99ce5b5aeac2b",
    str(KITTI_FINETUNED_SECOND): "41207f5e2505b2ff6235e8aa90c01aca0ad687a13016414aeaf51adc74736e06",
    str(KITTI_FINETUNED_POINTPILLARS): "88aeb2fb1d1dee3a1380a61cfa899e8601b2d0c72d2a457d9fea81adbe3fc206",
}


@dataclass(frozen=True)
class LiveCameraProfile:
    dataset_profile: str
    camera_model: str
    label: str
    checkpoint_path: Path
    checkpoint_sha256: str
    person_class_id: int = 0


@dataclass(frozen=True)
class LiveLidarProfile:
    dataset_profile: str
    detector_family: str
    lidar_model: str
    label: str
    config_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    selected_class_id: int
    selected_class_name: str
    box_origin: str
    point_cloud_range: tuple[float, float, float, float, float, float]
    score_threshold: float


def _choice(value: str | None, default: str, choices: tuple[str, ...], name: str) -> str:
    selected = (value or "").strip() or default
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {selected!r}")
    return selected


def _expected_hash(path: Path) -> str:
    return EXPECTED_SHA256.get(str(path), "")


def verify_expected_hash(path: Path, expected_sha256: str | None = None, validate: bool = True) -> str:
    expected = expected_sha256 or _expected_hash(path)
    if validate:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint/config path does not exist: {path}")
        actual = sha256_file(path)
        if expected and actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
        return actual
    return expected or "unknown"


def resolve_live_camera_profile(
    dataset_profile: str = "aghri",
    camera_model: str = "generic",
    validate_paths: bool = True,
) -> LiveCameraProfile:
    dataset = _choice(dataset_profile, "aghri", DATASET_PROFILES, "dataset_profile")
    model = _choice(camera_model, "generic", MODEL_CHOICES, "camera_model")
    if model == "finetuned" and dataset.startswith("kitti"):
        checkpoint = KITTI_FINETUNED_YOLO
        label = "KITTI-person-fine-tuned YOLO11s"
    elif model == "finetuned":
        checkpoint = FINETUNED_YOLO
        label = "AGHRI-fine-tuned YOLO11s"
    else:
        checkpoint = GENERIC_YOLO
        label = "generic YOLO11s"
    sha = verify_expected_hash(checkpoint, validate=validate_paths)
    return LiveCameraProfile(dataset, model, label, checkpoint, sha)


def resolve_live_lidar_profile(
    dataset_profile: str = "aghri",
    lidar_detector: str = "pointpillars",
    lidar_model: str = "generic",
    validate_paths: bool = True,
    allow_cross_domain: bool = False,
) -> LiveLidarProfile:
    dataset = _choice(dataset_profile, "aghri", DATASET_PROFILES, "dataset_profile")
    detector = _choice(lidar_detector, "pointpillars", LIDAR_DETECTOR_CHOICES, "lidar_detector")
    model = _choice(lidar_model, "generic", MODEL_CHOICES, "lidar_model")
    if dataset not in {"aghri", "kitti_tracking", "kitti_object"} and model != "generic" and not allow_cross_domain:
        raise ValueError(
            f"{dataset} online LiDAR inference defaults to generic KITTI-pretrained models; "
            "AGHRI fine-tuned LiDAR models require allow_cross_domain:=true."
        )

    if detector == "second" and model == "finetuned" and dataset.startswith("kitti"):
        config = KITTI_FINETUNED_SECOND_CONFIG
        checkpoint = KITTI_FINETUNED_SECOND
        selected_class_id = 0
        selected_class_name = "person"
        box_origin = "bottom_center"
        label = "KITTI-person-fine-tuned SECOND"
        point_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
        score = 0.0
    elif detector == "pointpillars" and model == "finetuned" and dataset.startswith("kitti"):
        config = KITTI_FINETUNED_POINTPILLARS_CONFIG
        checkpoint = KITTI_FINETUNED_POINTPILLARS
        selected_class_id = 0
        selected_class_name = "person"
        box_origin = "bottom_center"
        label = "KITTI-person-fine-tuned PointPillars"
        point_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
        score = 0.0
    elif detector == "second" and model == "finetuned":
        config = FINETUNED_SECOND_CONFIG
        checkpoint = FINETUNED_SECOND
        selected_class_id = 0
        selected_class_name = "person"
        box_origin = "center"
        label = "AGHRI-fine-tuned SECOND"
        point_range = (-10.24, -10.24, -2.0, 25.6, 10.24, 3.0)
        score = 0.0
    elif detector == "pointpillars" and model == "finetuned":
        config = FINETUNED_POINTPILLARS_CONFIG
        checkpoint = FINETUNED_POINTPILLARS
        selected_class_id = 0
        selected_class_name = "person"
        box_origin = "center"
        label = "AGHRI-fine-tuned PointPillars"
        point_range = (-10.24, -10.24, -2.0, 25.6, 10.24, 3.0)
        score = 0.0
    elif detector == "second":
        config = GENERIC_SECOND_AGHRI_CONFIG if dataset == "aghri" else GENERIC_SECOND_KITTI_CONFIG
        checkpoint = GENERIC_SECOND
        selected_class_id = 0
        selected_class_name = "Pedestrian"
        box_origin = "bottom_center"
        label = "generic KITTI-pretrained SECOND"
        point_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
        score = 0.1 if dataset != "aghri" else 0.0
    else:
        config = GENERIC_POINTPILLARS_AGHRI_CONFIG if dataset == "aghri" else GENERIC_POINTPILLARS_KITTI_CONFIG
        checkpoint = GENERIC_POINTPILLARS
        selected_class_id = 0
        selected_class_name = "Pedestrian"
        box_origin = "bottom_center"
        label = "generic KITTI-pretrained PointPillars"
        point_range = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)
        score = 0.1 if dataset != "aghri" else 0.0

    sha = verify_expected_hash(checkpoint, validate=validate_paths)
    if validate_paths and not config.exists():
        raise FileNotFoundError(f"MMDetection3D config does not exist: {config}")
    return LiveLidarProfile(
        dataset_profile=dataset,
        detector_family=detector,
        lidar_model=model,
        label=label,
        config_path=config,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha,
        selected_class_id=selected_class_id,
        selected_class_name=selected_class_name,
        box_origin=box_origin,
        point_cloud_range=point_range,
        score_threshold=score,
    )


def live_startup_summary(camera: LiveCameraProfile, lidar: LiveLidarProfile) -> tuple[str, ...]:
    return (
        "ROS inference mode: LIVE",
        f"Dataset profile: {camera.dataset_profile}",
        f"Camera detector: {camera.label}",
        f"Camera checkpoint: {camera.checkpoint_path}",
        f"Camera checkpoint SHA256: {camera.checkpoint_sha256}",
        "Live camera inference: ENABLED",
        f"LiDAR detector family: {lidar.detector_family}",
        f"LiDAR model: {lidar.label}",
        f"LiDAR config: {lidar.config_path}",
        f"LiDAR checkpoint: {lidar.checkpoint_path}",
        f"LiDAR checkpoint SHA256: {lidar.checkpoint_sha256}",
        f"LiDAR box origin: {lidar.box_origin}",
        "Live LiDAR inference: ENABLED",
        "Cached camera replay: DISABLED",
        "Cached LiDAR replay: DISABLED",
        "Online projection: ENABLED",
        "Online association: ENABLED",
    )
