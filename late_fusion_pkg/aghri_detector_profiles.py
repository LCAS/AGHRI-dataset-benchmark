"""Cached AGHRI detector profile registry for ROS replay launch files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_ABS_REPO_ROOT = Path("/home/prabuddhi/Desktop/late_fusion/deepfusionmot")
REPO_ROOT = _ABS_REPO_ROOT if _ABS_REPO_ROOT.exists() else Path(__file__).resolve().parents[1]
COMPARISON_CACHE_ROOT = REPO_ROOT / "results" / "aghri_finetuned_detector_comparison" / "cache"
POINTPILLARS_CACHE_ROOT = REPO_ROOT / "results" / "aghri_pointpillars_detector_comparison" / "cache"

MODEL_CHOICES = ("generic", "finetuned")
LIDAR_DETECTOR_CHOICES = ("second", "pointpillars")
PROFILE_MODELS = {
    "generic_generic": ("generic", "generic"),
    "finetuned_generic": ("finetuned", "generic"),
    "generic_finetuned": ("generic", "finetuned"),
    "finetuned_finetuned": ("finetuned", "finetuned"),
}
PROFILE_CHOICES = tuple(PROFILE_MODELS) + ("custom",)


@dataclass(frozen=True)
class DetectorCacheSpec:
    """Static cache and metadata paths for one AGHRI detector model."""

    modality: str
    detector_family: str
    model: str
    cache_csv: Path
    metadata_json: Path
    box_origin: str = ""
    available: bool = True
    unavailable_reason: str = ""


@dataclass(frozen=True)
class ResolvedDetectorSpec:
    """Resolved cache path and provenance for one launched detector stream."""

    modality: str
    detector_family: str
    model: str
    cache_csv: Path
    metadata_json: Path | None
    checkpoint_path: str
    checkpoint_sha256: str
    source: str
    box_origin: str = ""


@dataclass(frozen=True)
class ResolvedDetectorSelection:
    """Resolved camera and LiDAR detector cache selection."""

    detector_profile: str
    camera: ResolvedDetectorSpec
    lidar: ResolvedDetectorSpec


CAMERA_CACHES = {
    "generic": DetectorCacheSpec(
        modality="camera",
        detector_family="yolo",
        model="generic",
        cache_csv=POINTPILLARS_CACHE_ROOT / "yolo_generic" / "detections.csv",
        metadata_json=POINTPILLARS_CACHE_ROOT / "yolo_generic" / "metadata.json",
    ),
    "finetuned": DetectorCacheSpec(
        modality="camera",
        detector_family="yolo",
        model="finetuned",
        cache_csv=POINTPILLARS_CACHE_ROOT / "yolo_finetuned" / "detections.csv",
        metadata_json=POINTPILLARS_CACHE_ROOT / "yolo_finetuned" / "metadata.json",
    ),
}
LIDAR_CACHES = {
    "second": {
        "generic": DetectorCacheSpec(
            modality="lidar",
            detector_family="second",
            model="generic",
            cache_csv=COMPARISON_CACHE_ROOT / "second_generic" / "detections.csv",
            metadata_json=COMPARISON_CACHE_ROOT / "second_generic" / "metadata.json",
            box_origin="bottom_center",
        ),
        "finetuned": DetectorCacheSpec(
            modality="lidar",
            detector_family="second",
            model="finetuned",
            cache_csv=COMPARISON_CACHE_ROOT / "second_finetuned" / "detections.csv",
            metadata_json=COMPARISON_CACHE_ROOT / "second_finetuned" / "metadata.json",
            box_origin="center",
        ),
    },
    "pointpillars": {
        "generic": DetectorCacheSpec(
            modality="lidar",
            detector_family="pointpillars",
            model="generic",
            cache_csv=POINTPILLARS_CACHE_ROOT / "pointpillars_generic" / "detections.csv",
            metadata_json=POINTPILLARS_CACHE_ROOT / "pointpillars_generic" / "metadata.json",
            box_origin="bottom_center",
        ),
        "finetuned": DetectorCacheSpec(
            modality="lidar",
            detector_family="pointpillars",
            model="finetuned",
            cache_csv=POINTPILLARS_CACHE_ROOT / "pointpillars_finetuned" / "detections.csv",
            metadata_json=POINTPILLARS_CACHE_ROOT / "pointpillars_finetuned" / "metadata.json",
            box_origin="center",
        ),
    },
}


def _normalise_choice(value: str | None, default: str, choices: tuple[str, ...], name: str) -> str:
    selected = (value or "").strip() or default
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {selected!r}")
    return selected


def _load_metadata(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _override_metadata_path(cache_csv: Path) -> Path | None:
    candidate = cache_csv.with_name("metadata.json")
    return candidate if candidate.exists() else None


def _resolve_spec(
    spec: DetectorCacheSpec,
    override_cache: str | None,
    validate_paths: bool,
) -> ResolvedDetectorSpec:
    override = (override_cache or "").strip()
    if not spec.available and not override:
        raise FileNotFoundError(spec.unavailable_reason or f"{spec.modality} cache is not available")
    if override:
        cache_csv = Path(override).expanduser()
        metadata_json = _override_metadata_path(cache_csv)
        source = "explicit_override"
    else:
        cache_csv = spec.cache_csv
        metadata_json = spec.metadata_json
        source = "registry"
    if validate_paths and not cache_csv.exists():
        raise FileNotFoundError(f"{spec.modality} cache does not exist: {cache_csv}")
    metadata = _load_metadata(metadata_json)
    return ResolvedDetectorSpec(
        modality=spec.modality,
        detector_family=spec.detector_family,
        model=spec.model,
        cache_csv=cache_csv,
        metadata_json=metadata_json,
        checkpoint_path=str(metadata.get("checkpoint_path", "unknown")),
        checkpoint_sha256=str(metadata.get("checkpoint_sha256", "unknown")),
        source=source,
        box_origin=spec.box_origin,
    )


def resolve_detector_selection(
    camera_model: str | None = "finetuned",
    lidar_detector: str | None = "second",
    lidar_model: str | None = "finetuned",
    detector_profile: str | None = "custom",
    yolo_cache_override: str | None = "",
    second_cache_override: str | None = "",
    validate_paths: bool = True,
) -> ResolvedDetectorSelection:
    """Resolve AGHRI cached detector model selectors into replay cache paths."""

    profile = _normalise_choice(detector_profile, "custom", PROFILE_CHOICES, "detector_profile")
    if profile == "custom":
        camera = _normalise_choice(camera_model, "finetuned", MODEL_CHOICES, "camera_model")
        lidar = _normalise_choice(lidar_model, "finetuned", MODEL_CHOICES, "lidar_model")
    else:
        camera, lidar = PROFILE_MODELS[profile]
    lidar_family = _normalise_choice(lidar_detector, "second", LIDAR_DETECTOR_CHOICES, "lidar_detector")
    return ResolvedDetectorSelection(
        detector_profile=profile,
        camera=_resolve_spec(CAMERA_CACHES[camera], yolo_cache_override, validate_paths),
        lidar=_resolve_spec(LIDAR_CACHES[lidar_family][lidar], second_cache_override, validate_paths),
    )


def selection_summary_lines(selection: ResolvedDetectorSelection) -> tuple[str, ...]:
    """Return concise launch log lines for a resolved detector selection."""

    camera_label = "fine-tuned YOLO11s" if selection.camera.model == "finetuned" else "generic YOLO11s"
    family_label = "PointPillars" if selection.lidar.detector_family == "pointpillars" else "SECOND"
    if selection.lidar.detector_family == "pointpillars" and selection.lidar.model == "finetuned":
        lidar_label = "fine-tuned AGHRI PointPillars"
    elif selection.lidar.detector_family == "pointpillars":
        lidar_label = "generic KITTI PointPillars"
    elif selection.lidar.model == "finetuned":
        lidar_label = "fine-tuned AGHRI SECOND"
    else:
        lidar_label = "generic KITTI SECOND"
    return (
        f"Camera detector: {camera_label}",
        f"LiDAR detector family: {family_label}",
        f"LiDAR model: {lidar_label}",
        "Playback mode: cached predictions",
        "Live detector inference: no",
        f"AGHRI detector_profile={selection.detector_profile}",
        (
            "AGHRI camera_model="
            f"{selection.camera.model} cache={selection.camera.cache_csv} "
            f"checkpoint_sha256={selection.camera.checkpoint_sha256} "
            f"checkpoint_path={selection.camera.checkpoint_path} source={selection.camera.source}"
        ),
        (
            "AGHRI lidar_model="
            f"{selection.lidar.model} lidar_detector={selection.lidar.detector_family} "
            f"cache={selection.lidar.cache_csv} "
            f"cache_box_origin={selection.lidar.box_origin} "
            f"checkpoint_sha256={selection.lidar.checkpoint_sha256} "
            f"checkpoint_path={selection.lidar.checkpoint_path} source={selection.lidar.source}"
        ),
    )
