#!/usr/bin/env python3
"""Offline AGHRI association ablation for ZED RGB + Livox fusion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_lidar_fusion.lidar_fusion import load_aghri_projection  # noqa: E402
from scripts.fusser import Fusser, centre_from_points, validate_association_config  # noqa: E402

TOOLS_SCRIPT = (
    Path("/home/prabuddhi/Desktop/agri-human-dataset-tools")
    / "ros2bag/check_and_make_rosbag2.py"
)


def load_tools_reader():
    """Load the dataset-tools PCD reader without modifying that repository."""
    spec = importlib.util.spec_from_file_location("aghri_rosbag_tools", TOOLS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.read_pcd_binary_xyz


READ_PCD_XYZ = load_tools_reader()
MATCH_IOU_THRESHOLD = 0.10
AMBIGUOUS_IOU_DELTA = 0.05
AMBIGUOUS_DISTANCE_DELTA_M = 0.50
BOOTSTRAP_SEED = 20260705


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["lidar_fusion"]["ros__parameters"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def file_sha256(path: Path) -> str:
    """Return a SHA256 digest for a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_occlusion_events(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Load optional AGHRI occlusion intervals keyed by sequence name."""
    if path is None or not path.exists():
        return {}
    events: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("Camera") != "cam_zed_rgb":
                continue
            try:
                start = int(row["Frame Start"])
                end = int(row["Frame End"])
            except (KeyError, TypeError, ValueError):
                continue
            events.setdefault(row["Sequence Name"], []).append({
                "start": start,
                "end": end,
                "type": row.get("Type of occlusion") or "unknown",
            })
    return events


def occlusion_category(
    events: dict[str, list[dict[str, Any]]],
    sequence: str,
    frame_index: int | None,
) -> str:
    """Return the occlusion category for a camera frame when available."""
    if frame_index is None:
        return "unknown"
    matched = [
        event["type"]
        for event in events.get(sequence, [])
        if event["start"] <= frame_index <= event["end"]
    ]
    if not matched:
        return "none"
    return "+".join(sorted(set(matched)))


def sequence_name(path: Path) -> str:
    return path.name


def infer_scene(name: str) -> str:
    if "footpath" in name:
        return "footpath"
    if "in_straw" in name:
        return "polytunnel_straw"
    if "in_vine" in name:
        return "polytunnel_vine"
    if "out_straw" in name:
        return "outdoor_straw"
    if "out_vine" in name:
        return "outdoor_vine"
    return "unknown"


def infer_motion(name: str) -> str:
    if "_mv_" in name or name.endswith("_mv_label"):
        return "moving"
    if "_st_" in name or name.endswith("_st_label"):
        return "stationary"
    return "unknown"


def timestamp(record: dict[str, Any]) -> float:
    return float(record["Timestamp"])


def nearest_record(records: list[dict[str, Any]], stamp: float, tolerance: float):
    if not records:
        return None, None
    best = min(records, key=lambda item: abs(timestamp(item) - stamp))
    delta = abs(timestamp(best) - stamp)
    if delta <= tolerance:
        return best, delta
    return None, delta


def nearest_indexed_record(
    records: list[dict[str, Any]],
    stamp: float,
    tolerance: float,
):
    """Return nearest record, its original list index, and timestamp delta."""
    if not records:
        return None, None, None
    best_index, best = min(
        enumerate(records),
        key=lambda item: abs(timestamp(item[1]) - stamp),
    )
    delta = abs(timestamp(best) - stamp)
    if delta <= tolerance:
        return best, best_index, delta
    return None, None, delta


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def labels_to_gt3d(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, label in enumerate(record.get("Labels") or []):
        box = label.get("BoundingBoxes") or []
        if len(box) < 9:
            continue
        center = np.asarray(box[:3], dtype=np.float32)
        size = np.asarray(box[3:6], dtype=np.float32)
        identity = str(label.get("Class") or idx)
        out.append({
            "id": identity,
            "class": label.get("Class"),
            "label_index": idx,
            "center_lidar": center,
            "size": size,
            "rotation_rpy": np.asarray(box[6:9], dtype=np.float32),
        })
    return out


def labels_to_gt2d(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    if record is None:
        return out
    for idx, label in enumerate(record.get("Labels") or []):
        box = label.get("BoundingBoxes") or []
        if len(box) < 4:
            continue
        x, y, w, h = [float(v) for v in box[:4]]
        identity = str(label.get("Class") or idx)
        out.append({
            "id": identity,
            "class": label.get("Class"),
            "label_index": idx,
            "box": np.asarray([x, y, x + w, y + h], dtype=np.float32),
        })
    return out


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    homo = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return (matrix @ homo.T).T[:, :3]


def image_border_status(box: np.ndarray, image_shape) -> str:
    """Classify a detection as image-border or image-centre using fixed bands."""
    height, width = int(image_shape[0]), int(image_shape[1])
    margin_x = 0.05 * width
    margin_y = 0.05 * height
    if (
        float(box[0]) <= margin_x or
        float(box[2]) >= width - margin_x or
        float(box[1]) <= margin_y or
        float(box[3]) >= height - margin_y
    ):
        return "image_border"
    return "image_centre"


def point_count_band(candidate_count: int) -> str:
    """Fixed LiDAR candidate-count bands defined before test evaluation."""
    if candidate_count < 20:
        return "low_lt_20"
    if candidate_count < 100:
        return "medium_20_99"
    return "high_ge_100"


def distance_band_from_depth(depth: float | None) -> str:
    """Fixed distance bands defined before test evaluation."""
    if depth is None or not math.isfinite(float(depth)):
        return "unknown"
    if depth < 2.0:
        return "near_lt_2m"
    if depth < 5.0:
        return "medium_2_5m"
    return "far_ge_5m"


def _match_errors(
    gt: dict[str, Any],
    centre_lidar: np.ndarray,
    centre_camera: np.ndarray,
    camera_from_lidar: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    gt_camera = transform_points(camera_from_lidar, gt["center_lidar"][None, :])[0]
    errors = {
        "lidar_center_error": float(np.linalg.norm(centre_lidar - gt["center_lidar"])),
        "camera_center_error": float(np.linalg.norm(centre_camera - gt_camera)),
        "camera_depth_abs_error": float(abs(centre_camera[2] - gt_camera[2])),
        "camera_lateral_error": float(centre_camera[0] - gt_camera[0]),
        "camera_vertical_error": float(centre_camera[1] - gt_camera[1]),
    }
    gt_match = {
        "id": gt["id"],
        "class": gt.get("class"),
        "label_index": gt.get("label_index"),
        "center_lidar": gt["center_lidar"].astype(float).tolist(),
        "center_camera": gt_camera.astype(float).tolist(),
    }
    return errors, gt_match


def match_ground_truth(
    det_box: np.ndarray,
    centre_lidar: np.ndarray | None,
    centre_camera: np.ndarray | None,
    camera_from_lidar: np.ndarray,
    gt3d: list[dict[str, Any]],
    gt2d: list[dict[str, Any]],
) -> dict[str, Any]:
    """Match a prediction to GT without using GT for point association."""
    result = {
        "gt_match": None,
        "errors": {},
        "ambiguous": False,
        "match_type": "none",
        "prediction_identity": None,
        "gt_identity": None,
        "matching_distance": None,
        "best_2d_gt_iou": None,
    }
    if centre_lidar is None or centre_camera is None or not gt3d:
        return result

    gt3d_by_id = {str(item["id"]): item for item in gt3d}
    iou_candidates = [
        (bbox_iou(det_box, item["box"]), item)
        for item in gt2d
    ]
    iou_candidates.sort(key=lambda item: item[0], reverse=True)
    if iou_candidates:
        result["best_2d_gt_iou"] = float(iou_candidates[0][0])
    if iou_candidates and iou_candidates[0][0] >= MATCH_IOU_THRESHOLD:
        best_iou, best_2d = iou_candidates[0]
        if (
            len(iou_candidates) > 1 and
            best_iou - iou_candidates[1][0] < AMBIGUOUS_IOU_DELTA
        ):
            result["ambiguous"] = True
            result["match_type"] = "ambiguous_2d_identity"
            result["prediction_identity"] = best_2d["id"]
            return result

        identity = str(best_2d["id"])
        result["prediction_identity"] = identity
        gt = gt3d_by_id.get(identity)
        if gt is not None:
            errors, gt_match = _match_errors(
                gt,
                centre_lidar,
                centre_camera,
                camera_from_lidar,
            )
            result.update({
                "gt_match": gt_match,
                "errors": errors,
                "match_type": "object_id_from_2d_gt",
                "gt_identity": identity,
                "matching_distance": errors["lidar_center_error"],
            })
            return result

    distances = [
        float(np.linalg.norm(centre_lidar - item["center_lidar"]))
        for item in gt3d
    ]
    if not distances:
        return result
    order = np.argsort(distances)
    best = int(order[0])
    if (
        len(order) > 1 and
        distances[int(order[1])] - distances[best] < AMBIGUOUS_DISTANCE_DELTA_M
    ):
        result["ambiguous"] = True
        result["match_type"] = "ambiguous_nearest_lidar_center"
        result["matching_distance"] = distances[best]
        return result

    gt = gt3d[best]
    errors, gt_match = _match_errors(gt, centre_lidar, centre_camera, camera_from_lidar)
    result.update({
        "gt_match": gt_match,
        "errors": errors,
        "match_type": "nearest_lidar_center",
        "gt_identity": gt["id"],
        "matching_distance": errors["lidar_center_error"],
    })
    return result


def cache_key(metadata: dict[str, Any]) -> str:
    """Return a stable cache key from verified detection inputs."""
    key = json.dumps(metadata, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def yolo_cache_metadata(
    image_path: Path,
    image: np.ndarray,
    image_timestamp: float,
    model_path: Path,
    model_sha256: str,
    conf: float,
    classes: list[int],
    imgsz: int,
    ultralytics_version: str,
) -> dict[str, Any]:
    """Build the identity metadata used to accept or reject cached YOLO boxes."""
    stat = image_path.stat()
    return {
        "image_path": str(image_path.resolve()),
        "image_timestamp": float(image_timestamp),
        "image_size_hw": [int(image.shape[0]), int(image.shape[1])],
        "image_mtime_ns": stat.st_mtime_ns,
        "image_size_bytes": stat.st_size,
        "image_sha256": file_sha256(image_path),
        "checkpoint_path": str(model_path.resolve()),
        "checkpoint_sha256": model_sha256,
        "confidence_threshold": float(conf),
        "classes": [int(item) for item in classes],
        "imgsz": int(imgsz),
        "ultralytics_version": ultralytics_version,
    }


def load_or_run_yolo(
    image_path: Path,
    image: np.ndarray,
    image_timestamp: float,
    model_holder: dict[str, Any],
    model_path: Path,
    cache_dir: Path,
    conf: float,
    classes: list[int],
    imgsz: int,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = yolo_cache_metadata(
        image_path,
        image,
        image_timestamp,
        model_path,
        model_holder["model_sha256"],
        conf,
        classes,
        imgsz,
        model_holder["ultralytics_version"],
    )
    key = cache_key(metadata)
    cache_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"
    if cache_path.exists() and meta_path.exists() and load_json(meta_path) == metadata:
        return np.load(cache_path)

    if model_holder.get("model") is None:
        model_holder["model"] = YOLO(str(model_path)).to(model_holder["device"])

    result = model_holder["model"](
        image,
        conf=conf,
        classes=classes,
        imgsz=imgsz,
        verbose=False,
    )
    boxes = result[0].boxes.data.detach().cpu().numpy().astype(np.float32)
    np.save(cache_path, boxes)
    write_json(meta_path, metadata)
    return boxes


def build_fusser(
    params: dict[str, Any],
    projection: np.ndarray,
    camera_from_lidar: np.ndarray,
    method: str,
    statistic: str,
    gap_m: float,
    min_points: int,
    trim: float,
    device: str,
    hybrid_min_cluster_points: int,
    hybrid_min_support_ratio: float,
    hybrid_max_depth_spread_m: float,
    hybrid_min_cluster_separation_m: float,
    hybrid_fallback_on_single_cluster: bool,
    hybrid_fallback_on_sparse_candidates: bool,
):
    fusser = Fusser.__new__(Fusser)
    selected_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    fusser.device = selected_device
    fusser.trafo_matrix = torch.tensor(projection, dtype=torch.float32, device=selected_device)
    fusser.lidar_to_camera_matrix = torch.tensor(
        camera_from_lidar,
        dtype=torch.float32,
        device=selected_device,
    )
    fusser.RF = float(params["reduction_factor"])
    fusser.points = False
    fusser.bboxes = False
    fusser.write = False
    fusser.point_depth_filter_mode = params["point_depth_filter_mode"]
    fusser.minimum_forward_distance = float(params["minimum_forward_distance"])
    fusser.output_coordinate_mode = params["output_coordinate_mode"]
    (
        fusser.association_method,
        fusser.centre_statistic,
        fusser.depth_cluster_gap_m,
        fusser.depth_cluster_min_points,
        fusser.box_bottom_trim_ratio,
        fusser.hybrid_min_cluster_points,
        fusser.hybrid_min_support_ratio,
        fusser.hybrid_max_depth_spread_m,
        fusser.hybrid_min_cluster_separation_m,
        fusser.hybrid_fallback_on_single_cluster,
        fusser.hybrid_fallback_on_sparse_candidates,
    ) = validate_association_config(
        method,
        statistic,
        gap_m,
        min_points,
        trim,
        hybrid_min_cluster_points,
        hybrid_min_support_ratio,
        hybrid_max_depth_spread_m,
        hybrid_min_cluster_separation_m,
        hybrid_fallback_on_single_cluster,
        hybrid_fallback_on_sparse_candidates,
    )
    fusser.last_association_metadata = []
    return fusser


def associate_detection(
    fusser: Fusser,
    det: torch.Tensor,
    available: torch.Tensor,
    camera_from_lidar: np.ndarray,
    gt3d: list[dict[str, Any]],
    gt2d: list[dict[str, Any]],
    image_shape,
) -> tuple[dict[str, Any], torch.Tensor]:
    started = time.perf_counter()
    full_box = det[:4]
    association_box = fusser.association_bbox(det)
    full_indices = fusser.legacy_association_indices(available, full_box)
    selected, association_metadata = fusser.select_association_indices_with_metadata(
        available,
        association_box,
    )
    association_time = time.perf_counter() - started

    selected_lidar = fusser.pc_velo[selected]
    selected_camera = fusser.pc_camera[selected]
    retained_lidar = torch.empty((0, 3), dtype=torch.float32, device=fusser.device)
    retained_camera = torch.empty((0, 3), dtype=torch.float32, device=fusser.device)
    valid = False
    if len(selected_lidar) > 2:
        inlier_mask = fusser.outlier_mask(selected_lidar)
        retained_lidar = selected_lidar[inlier_mask]
        retained_camera = selected_camera[inlier_mask]
        valid = len(retained_lidar) > 0

    centre_lidar = None
    centre_camera = None
    published = np.zeros(3, dtype=np.float32)
    if valid:
        centre_lidar_t = centre_from_points(retained_lidar, fusser.centre_statistic)
        centre_camera_t = centre_from_points(retained_camera, fusser.centre_statistic)
        centre_lidar = centre_lidar_t.detach().cpu().numpy()
        centre_camera = centre_camera_t.detach().cpu().numpy()
        coordinates = fusser.get_coordinates(retained_lidar, retained_camera)
        if coordinates is not None:
            published = coordinates.detach().cpu().numpy()

    det_box_np = det[:4].detach().cpu().numpy()
    match = match_ground_truth(
        det_box_np,
        centre_lidar,
        centre_camera,
        camera_from_lidar,
        gt3d,
        gt2d,
    )
    gt_match = match["gt_match"]
    errors = match["errors"]
    ambiguous = match["ambiguous"]

    selected_depths = (
        selected_camera[:, 2].detach().cpu().numpy()
        if len(selected_camera) else np.asarray([], dtype=np.float32)
    )
    retained_depths = (
        retained_camera[:, 2].detach().cpu().numpy()
        if len(retained_camera) else np.asarray([], dtype=np.float32)
    )
    status = "matched" if gt_match else "ambiguous" if ambiguous else "unmatched"
    if not valid:
        status = "no_valid_cluster" if len(selected) else "no_points"
    distance_reference = None
    if centre_camera is not None:
        distance_reference = float(centre_camera[2])
    elif gt_match is not None:
        distance_reference = float(gt_match["center_camera"][2])
    distance_band = distance_band_from_depth(distance_reference)
    candidate_count = int(association_metadata.get("candidate_point_count", 0))
    point_band = point_count_band(candidate_count)

    row = {
        "box": det[:4].detach().cpu().numpy().astype(float).tolist(),
        "confidence": float(det[4].detach().cpu()),
        "class": int(det[5].detach().cpu()),
        "association_box": association_box.detach().cpu().numpy().astype(float).tolist(),
        "association_method_requested": association_metadata.get(
            "association_method_requested",
        ),
        "association_path": association_metadata.get("association_path"),
        "association_reason": association_metadata.get("association_reason"),
        "fallback_occurred": bool(association_metadata.get("fallback_occurred")),
        "fallback_reason": association_metadata.get("fallback_reason"),
        "projected_points": int(len(fusser.points_2d)),
        "points_in_full_box": int(len(full_indices)),
        "candidate_points": candidate_count,
        "selected_point_count": int(len(selected)),
        "retained_points": int(len(retained_lidar)),
        "filtered_point_count": int(len(retained_lidar)),
        "candidate_retained_ratio": (
            float(len(retained_lidar) / len(selected)) if len(selected) else 0.0
        ),
        "cluster_count": int(association_metadata.get("cluster_count", 0)),
        "valid_cluster_count": int(
            association_metadata.get("valid_cluster_count", 0),
        ),
        "selected_depth_cluster_size": int(
            association_metadata.get("selected_cluster_size") or len(selected),
        ),
        "selected_cluster_support_ratio": association_metadata.get(
            "selected_cluster_support_ratio",
        ),
        "selected_cluster_separation": association_metadata.get(
            "selected_cluster_separation",
        ),
        "selected_depth_spread": (
            float(np.max(selected_depths) - np.min(selected_depths))
            if len(selected_depths) else None
        ),
        "selected_cluster_depth_spread_pre_filter": association_metadata.get(
            "selected_cluster_depth_spread",
        ),
        "retained_depth_spread": (
            float(np.max(retained_depths) - np.min(retained_depths))
            if len(retained_depths) else None
        ),
        "mean_lidar_xyz": None if centre_lidar is None else centre_lidar.astype(float).tolist(),
        "mean_camera_xyz": None if centre_camera is None else centre_camera.astype(float).tolist(),
        "predicted_lidar_xyz": None if centre_lidar is None else centre_lidar.astype(float).tolist(),
        "predicted_camera_xyz": None if centre_camera is None else centre_camera.astype(float).tolist(),
        "published_xyz": published.astype(float).tolist(),
        "valid": bool(valid),
        "ambiguous": bool(ambiguous),
        "gt_match": gt_match,
        "matched_gt_identity": match["gt_identity"],
        "prediction_identity": match["prediction_identity"],
        "matching_distance": match["matching_distance"],
        "match_type": match["match_type"],
        "matched_gt_lidar_xyz": None if gt_match is None else gt_match["center_lidar"],
        "matched_gt_camera_xyz": None if gt_match is None else gt_match["center_camera"],
        "errors": errors,
        "best_2d_gt_iou": match["best_2d_gt_iou"],
        "distance_band": distance_band,
        "point_count_band": point_band,
        "sparsity_band": point_band,
        "occlusion_category": "unknown",
        "image_border_status": image_border_status(det_box_np, image_shape),
        "status": status,
        "validity_failure_reason": None if valid else status,
        "association_time_sec": association_time,
        "selected_indices": selected.detach().cpu().numpy().astype(int).tolist(),
    }
    return row, selected


def draw_visual(
    out_path: Path,
    image: np.ndarray,
    det_row: dict[str, Any],
    fusser: Fusser,
    selected_indices: torch.Tensor,
) -> None:
    out = image.copy()
    for point in fusser.points_2d.detach().cpu().numpy().astype(int):
        cv2.circle(out, (int(point[0]), int(point[1])), 1, (160, 160, 160), -1)
    box = [int(round(v)) for v in det_row["box"]]
    abox = [int(round(v)) for v in det_row["association_box"]]
    cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.rectangle(out, (abox[0], abox[1]), (abox[2], abox[3]), (0, 255, 255), 1)
    for point in fusser.points_2d[selected_indices].detach().cpu().numpy().astype(int):
        cv2.circle(out, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
    err = det_row["errors"].get("lidar_center_error")
    label = (
        f"{det_row['status']} n={det_row['retained_points']} "
        f"err={err:.2f}" if err is not None else det_row["status"]
    )
    cv2.putText(out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def useful_pairs(recording: Path, frame_limit: int, image_slop: float):
    image_ann = load_json(recording / "annotations/cam_zed_rgb_ann.json")
    lidar_ann = load_json(recording / "annotations/lidar_ann.json")
    pairs = []
    for lidar_record in lidar_ann:
        if not (lidar_record.get("Labels") and lidar_record.get("File")):
            continue
        image_record, image_index, delta = nearest_indexed_record(
            image_ann,
            timestamp(lidar_record),
            image_slop,
        )
        if image_record is None or not image_record.get("File"):
            continue
        pairs.append((lidar_record, image_record, image_index, delta))
    if frame_limit <= 0 or len(pairs) <= frame_limit:
        return pairs
    indices = np.linspace(0, len(pairs) - 1, num=frame_limit, dtype=int)
    return [pairs[int(i)] for i in indices]


def evaluate_recording(
    recording: Path,
    split: str,
    config: dict[str, Any],
    args,
    model_holder: dict[str, Any],
    projection: np.ndarray,
    camera_from_lidar: np.ndarray,
    occlusion_events: dict[str, list[dict[str, Any]]],
):
    pairs = useful_pairs(recording, args.frame_limit, args.image_slop_sec)
    fusser = build_fusser(
        config,
        projection,
        camera_from_lidar,
        args.association_method,
        args.centre_statistic,
        args.depth_cluster_gap_m,
        args.depth_cluster_min_points,
        args.box_bottom_trim_ratio,
        args.device,
        args.hybrid_min_cluster_points,
        args.hybrid_min_support_ratio,
        args.hybrid_max_depth_spread_m,
        args.hybrid_min_cluster_separation_m,
        args.hybrid_fallback_on_single_cluster,
        args.hybrid_fallback_on_sparse_candidates,
    )
    rows = []
    frame_rows = []
    scene = infer_scene(sequence_name(recording))
    motion = infer_motion(sequence_name(recording))
    total_started = time.perf_counter()
    visual_count = 0

    for frame_number, (
        lidar_record,
        image_record,
        image_index,
        image_delta,
    ) in enumerate(pairs):
        image_path = recording / "sensor_data/cam_zed_rgb" / image_record["File"]
        lidar_path = recording / "sensor_data/lidar" / lidar_record["File"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        points = READ_PCD_XYZ(lidar_path)
        points = points[np.isfinite(points).all(axis=1)]
        frame_started = time.perf_counter()
        fusser.get_lidar_on_image_fov(points[:, :3], image)
        boxes = load_or_run_yolo(
            image_path,
            image,
            timestamp(image_record),
            model_holder,
            Path(args.yolo_model),
            Path(args.cache_dir),
            args.yolo_threshold,
            args.yolo_classes,
            args.yolo_imgsz,
        )
        gt3d = labels_to_gt3d(lidar_record)
        gt2d = labels_to_gt2d(image_record)
        available = torch.ones(
            len(fusser.points_2d),
            dtype=torch.bool,
            device=fusser.device,
        )
        detections = []
        for detection_index, box in enumerate(boxes):
            det = torch.tensor(box, dtype=torch.float32, device=fusser.device)
            det_row, selected = associate_detection(
                fusser,
                det,
                available,
                camera_from_lidar,
                gt3d,
                gt2d,
                image.shape,
            )
            if len(selected):
                available[selected] = False
            det_row.update({
                "recording": sequence_name(recording),
                "recording_path": str(recording),
                "split": split,
                "scene": scene,
                "motion": motion,
                "frame_number": frame_number,
                "camera_frame_index": image_index,
                "occlusion_category": occlusion_category(
                    occlusion_events,
                    sequence_name(recording),
                    image_index,
                ),
                "lidar_timestamp": timestamp(lidar_record),
                "image_timestamp": timestamp(image_record),
                "image_lidar_delta_sec": image_delta,
                "detection_index": detection_index,
            })
            detections.append(det_row)
            if visual_count < args.visual_limit and (
                det_row["status"] != "matched" or visual_count < 2
            ):
                visual_path = (
                    Path(args.output_dir) / "visuals"
                    / args.run_name
                    / f"{sequence_name(recording)}_{frame_number:03d}_{detection_index}.png"
                )
                draw_visual(visual_path, image, det_row, fusser, selected)
                visual_count += 1
        frame_time = time.perf_counter() - frame_started
        matched_gt_ids = sorted({
            str(row.get("matched_gt_identity"))
            for row in detections
            if row.get("matched_gt_identity") is not None
        })
        for det_row in detections:
            det_row["frame_processing_time_sec"] = frame_time
            det_row["scene_person_count_band"] = (
                "single_person" if len(gt3d) <= 1 else "multi_person"
            )
            rows.append(det_row)
        frame_rows.append({
            "recording": sequence_name(recording),
            "split": split,
            "scene": scene,
            "motion": motion,
            "frame_number": frame_number,
            "camera_frame_index": image_index,
            "occlusion_category": occlusion_category(
                occlusion_events,
                sequence_name(recording),
                image_index,
            ),
            "lidar_timestamp": timestamp(lidar_record),
            "image_timestamp": timestamp(image_record),
            "projected_points": int(len(fusser.points_2d)),
            "detections": len(detections),
            "gt3d_count": len(gt3d),
            "gt2d_count": len(gt2d),
            "matched_gt_ids": matched_gt_ids,
            "unmatched_gt_count": max(0, len(gt3d) - len(matched_gt_ids)),
            "frame_time_sec": frame_time,
        })

    return {
        "recording": sequence_name(recording),
        "recording_path": str(recording),
        "split": split,
        "scene": scene,
        "motion": motion,
        "usable_frames": len(pairs),
        "rows": rows,
        "frames": frame_rows,
        "total_time_sec": time.perf_counter() - total_started,
    }


def summarise_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str):
        out = []
        for row in rows:
            if key in row:
                out.append(row[key])
            elif row.get("errors") and key in row["errors"]:
                out.append(row["errors"][key])
        return [float(x) for x in out if x is not None and math.isfinite(float(x))]

    total = len(rows)
    valid = [row for row in rows if row["valid"]]
    matched = [row for row in rows if row["status"] == "matched"]
    no_points = [row for row in rows if row["status"] == "no_points"]
    no_valid = [row for row in rows if row["status"] == "no_valid_cluster"]
    ambiguous = [row for row in rows if row["ambiguous"]]
    lidar_errors = values("lidar_center_error")
    depth_errors = values("camera_depth_abs_error")
    lateral_errors = values("camera_lateral_error")
    vertical_errors = values("camera_vertical_error")
    assoc_times = values("association_time_sec")
    frame_times = values("frame_processing_time_sec")
    candidate_counts = values("candidate_points")
    retained_counts = values("retained_points")
    retained_ratios = values("candidate_retained_ratio")
    depth_spreads = values("selected_depth_spread")

    def mean(vals):
        return float(np.mean(vals)) if vals else None

    def median(vals):
        return float(np.median(vals)) if vals else None

    def percentile(vals, pct):
        return float(np.percentile(vals, pct)) if vals else None

    def counts(key: str):
        out = {}
        for row in rows:
            value = row.get(key)
            if value is None:
                value = "none"
            out[str(value)] = out.get(str(value), 0) + 1
        return out

    def grouped_mean(group_key: str, metric_key: str):
        groups = {}
        for row in rows:
            group = str(row.get(group_key) or "unknown")
            value = None
            if metric_key in row:
                value = row.get(metric_key)
            elif row.get("errors") and metric_key in row["errors"]:
                value = row["errors"][metric_key]
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                groups.setdefault(group, []).append(value)
        return {key: mean(vals) for key, vals in groups.items()}

    fallback_rows = [row for row in rows if row.get("fallback_occurred")]
    accepted_depth_rows = [
        row for row in rows
        if row.get("association_reason") == "accepted_depth_cluster"
    ]

    return {
        "total_yolo_person_detections": total,
        "detections_with_projected_lidar_points": len([
            r for r in rows if r["projected_points"] > 0
        ]),
        "detections_with_valid_selected_cluster": len(valid),
        "valid_association_rate": float(len(valid) / total) if total else None,
        "no_point_rate": float(len(no_points) / total) if total else None,
        "no_valid_cluster_rate": float(len(no_valid) / total) if total else None,
        "matched_prediction_count": len(matched),
        "unmatched_prediction_count": len([
            row for row in rows if row["status"] == "unmatched"
        ]),
        "ambiguous_match_count": len(ambiguous),
        "mean_lidar_center_error": mean(lidar_errors),
        "median_lidar_center_error": median(lidar_errors),
        "max_lidar_center_error": max(lidar_errors) if lidar_errors else None,
        "p75_lidar_center_error": percentile(lidar_errors, 75),
        "p90_lidar_center_error": percentile(lidar_errors, 90),
        "p95_lidar_center_error": percentile(lidar_errors, 95),
        "lidar_error_le_0_5m_count": len([
            value for value in lidar_errors if value <= 0.5
        ]),
        "lidar_error_le_0_25m_count": len([
            value for value in lidar_errors if value <= 0.25
        ]),
        "lidar_error_le_0_25m_rate": float(
            len([value for value in lidar_errors if value <= 0.25]) /
            len(lidar_errors)
        ) if lidar_errors else None,
        "lidar_error_le_0_5m_rate": float(
            len([value for value in lidar_errors if value <= 0.5]) /
            len(lidar_errors)
        ) if lidar_errors else None,
        "lidar_error_le_1_0m_count": len([
            value for value in lidar_errors if value <= 1.0
        ]),
        "lidar_error_le_1_0m_rate": float(
            len([value for value in lidar_errors if value <= 1.0]) /
            len(lidar_errors)
        ) if lidar_errors else None,
        "lidar_error_gt_1_0m_count": len([
            value for value in lidar_errors if value > 1.0
        ]),
        "lidar_error_gt_1_0m_rate": float(
            len([value for value in lidar_errors if value > 1.0]) /
            len(lidar_errors)
        ) if lidar_errors else None,
        "lidar_error_gt_2_0m_count": len([
            value for value in lidar_errors if value > 2.0
        ]),
        "lidar_error_gt_2_0m_rate": float(
            len([value for value in lidar_errors if value > 2.0]) /
            len(lidar_errors)
        ) if lidar_errors else None,
        "mean_camera_depth_error": mean(depth_errors),
        "median_camera_depth_error": median(depth_errors),
        "mean_lateral_error": mean(lateral_errors),
        "mean_vertical_error": mean(vertical_errors),
        "mean_candidate_point_count": mean(candidate_counts),
        "median_candidate_point_count": median(candidate_counts),
        "mean_retained_point_count": mean(retained_counts),
        "median_retained_point_count": median(retained_counts),
        "mean_candidate_retained_ratio": mean(retained_ratios),
        "mean_selected_depth_spread": mean(depth_spreads),
        "mean_association_time_sec": mean(assoc_times),
        "median_association_time_sec": median(assoc_times),
        "mean_frame_processing_time_sec": mean(frame_times),
        "median_frame_processing_time_sec": median(frame_times),
        "approx_fps": float(1.0 / mean(frame_times))
        if frame_times and mean(frame_times) and mean(frame_times) > 0 else None,
        "association_path_counts": counts("association_path"),
        "association_reason_counts": counts("association_reason"),
        "fallback_count": len(fallback_rows),
        "fallback_rate": float(len(fallback_rows) / total) if total else None,
        "fallback_reason_counts": counts("fallback_reason"),
        "accepted_depth_cluster_count": len(accepted_depth_rows),
        "accepted_depth_cluster_rate": float(len(accepted_depth_rows) / total)
        if total else None,
        "mean_lidar_error_by_distance_band": grouped_mean(
            "distance_band",
            "lidar_center_error",
        ),
        "mean_lidar_error_by_sparsity_band": grouped_mean(
            "sparsity_band",
            "lidar_center_error",
        ),
        "mean_lidar_error_by_image_border_status": grouped_mean(
            "image_border_status",
            "lidar_center_error",
        ),
        "mean_lidar_error_by_scene_person_count_band": grouped_mean(
            "scene_person_count_band",
            "lidar_center_error",
        ),
        "match_type_counts": counts("match_type"),
        "mean_lidar_error_by_scene": grouped_mean("scene", "lidar_center_error"),
        "mean_lidar_error_by_motion": grouped_mean("motion", "lidar_center_error"),
        "mean_lidar_error_by_occlusion_category": grouped_mean(
            "occlusion_category",
            "lidar_center_error",
        ),
        "occlusion_category_counts": counts("occlusion_category"),
    }


def summarise_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise frame-level coverage and unmatched GT counts."""
    total = len(frames)
    unmatched_gt = sum(int(frame.get("unmatched_gt_count", 0)) for frame in frames)
    gt3d = sum(int(frame.get("gt3d_count", 0)) for frame in frames)
    detections = sum(int(frame.get("detections", 0)) for frame in frames)
    frame_times = [
        float(frame["frame_time_sec"]) for frame in frames
        if frame.get("frame_time_sec") is not None
    ]
    return {
        "matched_image_lidar_pairs": total,
        "total_gt3d_instances": gt3d,
        "total_frame_detections": detections,
        "unmatched_gt_count": unmatched_gt,
        "mean_frame_time_sec": float(np.mean(frame_times)) if frame_times else None,
        "median_frame_time_sec": float(np.median(frame_times)) if frame_times else None,
        "approx_fps": float(1.0 / np.mean(frame_times))
        if frame_times and np.mean(frame_times) > 0 else None,
    }


def paired_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Key identifying the same YOLO detection across frozen methods."""
    return (
        row.get("recording"),
        row.get("frame_number"),
        row.get("detection_index"),
    )


def paired_comparison(
    legacy_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    equal_tolerance_m: float = 0.01,
    bootstrap_samples: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare valid paired prediction errors deterministically."""
    legacy_by_key = {paired_key(row): row for row in legacy_rows}
    hybrid_by_key = {paired_key(row): row for row in hybrid_rows}
    diffs = []
    pairs = []
    for key in sorted(set(legacy_by_key) & set(hybrid_by_key)):
        legacy_error = legacy_by_key[key].get("errors", {}).get("lidar_center_error")
        hybrid_error = hybrid_by_key[key].get("errors", {}).get("lidar_center_error")
        if legacy_error is None or hybrid_error is None:
            continue
        diff = float(hybrid_error) - float(legacy_error)
        diffs.append(diff)
        pairs.append({
            "key": key,
            "legacy_error": float(legacy_error),
            "hybrid_error": float(hybrid_error),
            "difference_hybrid_minus_legacy": diff,
            "legacy_status": legacy_by_key[key].get("status"),
            "hybrid_status": hybrid_by_key[key].get("status"),
            "hybrid_path": hybrid_by_key[key].get("association_path"),
            "hybrid_fallback_reason": hybrid_by_key[key].get("fallback_reason"),
        })
    if not diffs:
        return {
            "jointly_valid_pairs": 0,
            "pairs": [],
        }
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    improves = int(np.sum(diffs_arr < -equal_tolerance_m))
    worsens = int(np.sum(diffs_arr > equal_tolerance_m))
    equal = int(len(diffs_arr) - improves - worsens)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(diffs_arr, size=len(diffs_arr), replace=True)
        means.append(float(np.mean(sample)))
    return {
        "jointly_valid_pairs": int(len(diffs_arr)),
        "mean_paired_error_difference_hybrid_minus_legacy": float(np.mean(diffs_arr)),
        "median_paired_error_difference_hybrid_minus_legacy": float(np.median(diffs_arr)),
        "hybrid_improves_count": improves,
        "legacy_improves_count": worsens,
        "effectively_equal_count": equal,
        "hybrid_win_rate": float(improves / len(diffs_arr)),
        "legacy_win_rate": float(worsens / len(diffs_arr)),
        "equal_tolerance_m": equal_tolerance_m,
        "bootstrap_seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "mean_difference_bootstrap_ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "pairs": pairs,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "recording",
        "split",
        "scene",
        "motion",
        "frame_number",
        "camera_frame_index",
        "occlusion_category",
        "lidar_timestamp",
        "confidence",
        "class",
        "association_path",
        "association_reason",
        "fallback_occurred",
        "fallback_reason",
        "candidate_points",
        "selected_point_count",
        "cluster_count",
        "valid_cluster_count",
        "selected_depth_cluster_size",
        "selected_cluster_support_ratio",
        "selected_cluster_separation",
        "retained_points",
        "filtered_point_count",
        "valid",
        "status",
        "match_type",
        "prediction_identity",
        "matched_gt_identity",
        "matching_distance",
        "best_2d_gt_iou",
        "distance_band",
        "point_count_band",
        "image_border_status",
        "published_xyz",
        "lidar_center_error",
        "camera_depth_abs_error",
        "association_time_sec",
        "frame_processing_time_sec",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key) for key in fields},
                "published_xyz": row.get("published_xyz"),
                "lidar_center_error": row.get("errors", {}).get("lidar_center_error"),
                "camera_depth_abs_error": row.get("errors", {}).get("camera_depth_abs_error"),
            })


def load_split_names(split_dir: Path, split: str) -> list[str]:
    """Load official split names in file order."""
    path = split_dir / f"{split}.txt"
    with path.open("r", encoding="utf-8") as stream:
        return [
            line.strip() for line in stream
            if line.strip() and not line.startswith("#")
        ]


def dataset_recording_index(dataset_root: Path) -> dict[str, Path]:
    """Map AGHRI recording names to absolute paths."""
    out = {}
    for path in sorted(dataset_root.glob("dataset_part*/*_label")):
        out[path.name] = path.resolve()
    return out


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def expected_pair_count(recording: Path, image_slop: float) -> int:
    """Count deterministic labeled LiDAR/image pairs used by evaluation."""
    try:
        return len(useful_pairs(recording, frame_limit=0, image_slop=image_slop))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        return 0


def build_official_manifest(
    dataset_root: Path,
    split: str = "test",
    image_slop: float = 0.10,
    occlusion_log: Path | None = None,
) -> list[dict[str, Any]]:
    """Construct a complete official split manifest without omitting entries."""
    split_dir = dataset_root / "split_lists"
    names = load_split_names(split_dir, split)
    train_names = set(load_split_names(split_dir, "train"))
    val_names = set(load_split_names(split_dir, "val"))
    index = dataset_recording_index(dataset_root)
    occlusion_events = load_occlusion_events(occlusion_log)
    manifest = []
    for name in names:
        path = index.get(name)
        entry = {
            "split": split,
            "recording_name": name,
            "recording_path": None if path is None else str(path),
            "dataset_part": None if path is None else path.parent.name,
            "scene": infer_scene(name),
            "motion": infer_motion(name),
            "zed_rgb_available": False,
            "lidar_available": False,
            "annotation_3d_available": False,
            "occlusion_log_available": bool(occlusion_events.get(name)),
            "rgb_image_count": 0,
            "lidar_point_cloud_count": 0,
            "lidar_annotation_frame_count": 0,
            "expected_synchronised_frame_count": 0,
            "supported": False,
            "skip_reason": None,
            "train_val_leakage": name in train_names or name in val_names,
        }
        if path is None:
            entry["skip_reason"] = "recording_directory_missing"
            manifest.append(entry)
            continue

        image_dir = path / "sensor_data/cam_zed_rgb"
        lidar_dir = path / "sensor_data/lidar"
        image_ann = path / "annotations/cam_zed_rgb_ann.json"
        lidar_ann = path / "annotations/lidar_ann.json"
        entry["zed_rgb_available"] = image_dir.exists() and image_ann.exists()
        entry["lidar_available"] = lidar_dir.exists()
        entry["annotation_3d_available"] = lidar_ann.exists()
        entry["rgb_image_count"] = count_files(image_dir, "*.png")
        entry["lidar_point_cloud_count"] = count_files(lidar_dir, "*.pcd")
        if lidar_ann.exists():
            try:
                entry["lidar_annotation_frame_count"] = len(load_json(lidar_ann))
            except json.JSONDecodeError:
                entry["skip_reason"] = "lidar_annotation_json_invalid"
        entry["expected_synchronised_frame_count"] = expected_pair_count(
            path,
            image_slop,
        )
        missing = []
        if not entry["zed_rgb_available"]:
            missing.append("zed_rgb")
        if not entry["lidar_available"]:
            missing.append("lidar")
        if not entry["annotation_3d_available"]:
            missing.append("3d_annotations")
        if entry["expected_synchronised_frame_count"] == 0:
            missing.append("synchronised_labeled_pairs")
        if entry["train_val_leakage"]:
            missing.append("train_val_leakage")
        entry["supported"] = not missing and entry["skip_reason"] is None
        entry["skip_reason"] = entry["skip_reason"] or (
            None if entry["supported"] else ",".join(missing)
        )
        manifest.append(entry)
    return manifest


def write_manifest_outputs(csv_path: Path, json_path: Path, manifest: list[dict[str, Any]]):
    """Write manifest CSV and JSON outputs."""
    write_json(json_path, manifest)
    fields = [
        "split",
        "recording_name",
        "recording_path",
        "dataset_part",
        "scene",
        "motion",
        "zed_rgb_available",
        "lidar_available",
        "annotation_3d_available",
        "occlusion_log_available",
        "rgb_image_count",
        "lidar_point_cloud_count",
        "lidar_annotation_frame_count",
        "expected_synchronised_frame_count",
        "supported",
        "skip_reason",
        "train_val_leakage",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)


def read_manifest(path: Path) -> list[tuple[Path, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        first = stream.readline()
        stream.seek(0)
        if "recording_path" in first:
            for row in csv.DictReader(stream):
                if row.get("supported", "True").lower() == "false":
                    continue
                rows.append((Path(row["recording_path"]), row.get("split", "unknown")))
            return rows
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", maxsplit=1)
            if len(parts) == 1:
                rows.append((Path(parts[0]), "unknown"))
            else:
                rows.append((Path(parts[1]), parts[0]))
    return rows


def model_metadata(model_path: Path, args) -> dict[str, Any]:
    stat = model_path.stat()
    import ultralytics

    return {
        "model_path": str(model_path),
        "sha256": file_sha256(model_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ultralytics_version": ultralytics.__version__,
        "confidence_threshold": args.yolo_threshold,
        "classes": args.yolo_classes,
        "imgsz": args.yolo_imgsz,
        "device": args.device,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--manifest")
    parser.add_argument("--build-test-manifest", action="store_true")
    parser.add_argument(
        "--dataset-root",
        default="/media/prabuddhi/Backup2/Updated Dataset_PW",
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config/aghri_zed_livox.yaml"),
    )
    parser.add_argument("--output-dir", default="/tmp/aghri_association_ablation")
    parser.add_argument("--cache-dir", default="/tmp/aghri_association_ablation/cache")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--frame-limit", type=int, default=40)
    parser.add_argument("--image-slop-sec", type=float, default=0.10)
    parser.add_argument("--yolo-model")
    parser.add_argument("--yolo-threshold", type=float, default=0.1)
    parser.add_argument("--yolo-classes", type=int, nargs="+", default=[0])
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--association-method", default="legacy_box")
    parser.add_argument("--centre-statistic", default="mean")
    parser.add_argument("--depth-cluster-gap-m", type=float, default=0.35)
    parser.add_argument("--depth-cluster-min-points", type=int, default=5)
    parser.add_argument("--box-bottom-trim-ratio", type=float, default=0.0)
    parser.add_argument("--hybrid-min-cluster-points", type=int, default=5)
    parser.add_argument("--hybrid-min-support-ratio", type=float, default=0.35)
    parser.add_argument("--hybrid-max-depth-spread-m", type=float, default=0.35)
    parser.add_argument(
        "--hybrid-min-cluster-separation-m",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--hybrid-fallback-on-single-cluster",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-fallback-on-sparse-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--occlusion-log",
        default="/media/prabuddhi/Backup2/Updated Dataset_PW/occlusion_log.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visual-limit", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.build_test_manifest:
        manifest = build_official_manifest(
            Path(args.dataset_root),
            split="test",
            image_slop=args.image_slop_sec,
            occlusion_log=Path(args.occlusion_log),
        )
        write_manifest_outputs(
            output_dir / "test_manifest.csv",
            output_dir / "test_manifest.json",
            manifest,
        )
        print(json.dumps({
            "manifest_count": len(manifest),
            "supported_count": len([row for row in manifest if row["supported"]]),
            "unsupported_count": len([
                row for row in manifest if not row["supported"]
            ]),
            "train_val_leakage_count": len([
                row for row in manifest if row["train_val_leakage"]
            ]),
        }, indent=2))
        return

    if not args.yolo_model:
        raise ValueError("--yolo-model is required unless --build-test-manifest is used")
    validate_association_config(
        args.association_method,
        args.centre_statistic,
        args.depth_cluster_gap_m,
        args.depth_cluster_min_points,
        args.box_bottom_trim_ratio,
        args.hybrid_min_cluster_points,
        args.hybrid_min_support_ratio,
        args.hybrid_max_depth_spread_m,
        args.hybrid_min_cluster_separation_m,
        args.hybrid_fallback_on_single_cluster,
        args.hybrid_fallback_on_sparse_candidates,
    )
    config = load_yaml(Path(args.config))
    projection, camera_from_lidar, path, camera_frame = load_aghri_projection(
        config["calibration_zip_path"],
        config["intrinsics_member"],
        config["extrinsics_member"],
        config["camera_sensor_name"],
        config["lidar_frame_name"],
        config["camera_optical_frame_name"],
        config["camera_matrix_source"],
    )
    recordings = []
    if args.manifest:
        recordings.extend(read_manifest(Path(args.manifest)))
    for item in args.recording:
        recordings.append((Path(item), "unknown"))
    if not recordings:
        raise ValueError("Provide --recording or --manifest")
    occlusion_events = load_occlusion_events(Path(args.occlusion_log))

    if args.device == "auto":
        yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        yolo_device = args.device
    model_path = Path(args.yolo_model)
    import ultralytics
    model_holder = {
        "model": None,
        "device": yolo_device,
        "model_sha256": file_sha256(model_path),
        "ultralytics_version": ultralytics.__version__,
    }
    write_json(Path(args.cache_dir) / "yolo_metadata.json", model_metadata(model_path, args))

    all_rows = []
    all_frames = []
    recording_summaries = []
    failures = []
    run_dir = output_dir / "runs" / args.run_name
    for recording, split in recordings:
        try:
            result = evaluate_recording(
                recording,
                split,
                config,
                args,
                model_holder,
                projection,
                camera_from_lidar,
                occlusion_events,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should continue.
            failure = {
                "recording": sequence_name(recording),
                "recording_path": str(recording),
                "split": split,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            write_json(
                run_dir / "recordings" / sequence_name(recording) / "failure.json",
                failure,
            )
            continue
        all_rows.extend(result["rows"])
        all_frames.extend(result["frames"])
        recording_dir = run_dir / "recordings" / result["recording"]
        write_json(recording_dir / "detections.json", result["rows"])
        write_json(recording_dir / "frames.json", result["frames"])
        write_json(recording_dir / "summary.json", {
            "recording": result["recording"],
            "recording_path": result["recording_path"],
            "split": result["split"],
            "scene": result["scene"],
            "motion": result["motion"],
            "usable_frames": result["usable_frames"],
            "metrics": summarise_rows(result["rows"]),
            "frame_metrics": summarise_frames(result["frames"]),
        })
        rec_summary = {
            key: result[key]
            for key in ("recording", "recording_path", "split", "scene", "motion", "usable_frames")
        }
        rec_summary["metrics"] = summarise_rows(result["rows"])
        rec_summary["frame_metrics"] = summarise_frames(result["frames"])
        recording_summaries.append(rec_summary)

    summary = {
        "run_name": args.run_name,
        "association_method": args.association_method,
        "centre_statistic": args.centre_statistic,
        "depth_cluster_gap_m": args.depth_cluster_gap_m,
        "depth_cluster_min_points": args.depth_cluster_min_points,
        "box_bottom_trim_ratio": args.box_bottom_trim_ratio,
        "hybrid_min_cluster_points": args.hybrid_min_cluster_points,
        "hybrid_min_support_ratio": args.hybrid_min_support_ratio,
        "hybrid_max_depth_spread_m": args.hybrid_max_depth_spread_m,
        "hybrid_min_cluster_separation_m": args.hybrid_min_cluster_separation_m,
        "hybrid_fallback_on_single_cluster": (
            args.hybrid_fallback_on_single_cluster
        ),
        "hybrid_fallback_on_sparse_candidates": (
            args.hybrid_fallback_on_sparse_candidates
        ),
        "yolo_loaded_for_cache_misses": model_holder["model"] is not None,
        "occlusion_log": args.occlusion_log,
        "occlusion_sequences_loaded": len(occlusion_events),
        "camera_frame": camera_frame,
        "transform_path": path,
        "recordings": recording_summaries,
        "failures": failures,
        "overall": summarise_rows(all_rows),
        "frame_overall": summarise_frames(all_frames),
        "frame_count": len(all_frames),
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "detections.json", all_rows)
    write_json(run_dir / "frames.json", all_frames)
    write_csv(run_dir / "detections.csv", all_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
