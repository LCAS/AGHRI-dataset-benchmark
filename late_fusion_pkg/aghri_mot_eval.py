"""AGHRI MOT evaluation helpers using the vendored TrackEval metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_trackeval_importable() -> None:
    """Patch NumPy aliases required by the vendored TrackEval code."""

    np.float = float  # type: ignore[attr-defined]
    np.int = int  # type: ignore[attr-defined]


def bbox_xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box]
    return (x, y, x + max(0.0, w), y + max(0.0, h))


def bbox_iou_matrix(
    gt_boxes: list[tuple[float, float, float, float]],
    tracker_boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    if not gt_boxes or not tracker_boxes:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)
    gt = np.asarray(gt_boxes, dtype=float)
    trk = np.asarray(tracker_boxes, dtype=float)
    x1 = np.maximum(gt[:, None, 0], trk[None, :, 0])
    y1 = np.maximum(gt[:, None, 1], trk[None, :, 1])
    x2 = np.minimum(gt[:, None, 2], trk[None, :, 2])
    y2 = np.minimum(gt[:, None, 3], trk[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    gt_area = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    trk_area = np.maximum(0.0, trk[:, 2] - trk[:, 0]) * np.maximum(0.0, trk[:, 3] - trk[:, 1])
    denom = gt_area[:, None] + trk_area[None, :] - inter
    out = np.zeros_like(inter)
    valid = denom > 0.0
    out[valid] = inter[valid] / denom[valid]
    return out


def load_manifest_rows(manifest: Path, recording: str, limit_frames: int = 0, unique_camera_frames: bool = True) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("recording_name") != recording:
                continue
            if not row.get("image_path"):
                continue
            rows.append(row)
    rows.sort(key=lambda item: float(item.get("lidar_timestamp") or item.get("zed_rgb_timestamp") or 0.0))
    if unique_camera_frames:
        seen: set[str] = set()
        unique: list[dict] = []
        for row in rows:
            key = Path(row["image_path"]).name
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        rows = unique
    if limit_frames:
        rows = rows[:limit_frames]
    return rows


def find_recording_dir(dataset_root: Path, recording: str) -> Path:
    direct = list(dataset_root.glob(f"dataset_part*/{recording}"))
    if direct:
        return direct[0]
    matches = list(dataset_root.glob(f"dataset_part*/**/{recording}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find AGHRI recording under {dataset_root}: {recording}")


def load_camera_annotations(recording_dir: Path) -> dict[str, list[dict]]:
    path = recording_dir / "annotations" / "cam_zed_rgb_ann.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(frame["File"]): list(frame.get("Labels", [])) for frame in data}


def load_tracked_boxes(tracked_csv: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    with tracked_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frame_index = int(row["frame_index"])
            try:
                box = (
                    float(row["projected_x1"]),
                    float(row["projected_y1"]),
                    float(row["projected_x2"]),
                    float(row["projected_y2"]),
                )
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(box)):
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            out.setdefault(frame_index, []).append(
                {
                    "track_id": int(row["track_id"]),
                    "bbox_xyxy": box,
                    "score": float(row.get("score") or 1.0),
                    "state": row.get("track_state", ""),
                }
            )
    return out


def build_trackeval_sequence(
    *,
    manifest_rows: list[dict],
    annotations_by_file: dict[str, list[dict]],
    tracked_by_frame: dict[int, list[dict]],
    confirmed_only: bool = True,
) -> tuple[dict, dict]:
    gt_ids_raw: list[list[str]] = []
    tracker_ids_raw: list[list[int]] = []
    gt_boxes: list[list[tuple[float, float, float, float]]] = []
    tracker_boxes: list[list[tuple[float, float, float, float]]] = []

    for frame_index, row in enumerate(manifest_rows):
        file_name = Path(row["image_path"]).name
        labels = annotations_by_file.get(file_name, [])
        gt_ids_raw.append([str(label["Class"]) for label in labels])
        gt_boxes.append([bbox_xywh_to_xyxy(label["BoundingBoxes"]) for label in labels])
        tracker_rows = tracked_by_frame.get(frame_index, [])
        if confirmed_only:
            tracker_rows = [item for item in tracker_rows if item.get("state") == "confirmed"]
        tracker_ids_raw.append([int(item["track_id"]) for item in tracker_rows])
        tracker_boxes.append([item["bbox_xyxy"] for item in tracker_rows])

    gt_id_map = {value: idx for idx, value in enumerate(sorted({item for frame in gt_ids_raw for item in frame}))}
    tracker_id_map = {value: idx for idx, value in enumerate(sorted({item for frame in tracker_ids_raw for item in frame}))}
    gt_ids = [np.asarray([gt_id_map[item] for item in frame], dtype=int) for frame in gt_ids_raw]
    tracker_ids = [np.asarray([tracker_id_map[item] for item in frame], dtype=int) for frame in tracker_ids_raw]
    similarities = [bbox_iou_matrix(gt_frame, trk_frame) for gt_frame, trk_frame in zip(gt_boxes, tracker_boxes)]
    data = {
        "num_timesteps": len(manifest_rows),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
        "num_gt_dets": sum(len(frame) for frame in gt_ids),
        "num_tracker_dets": sum(len(frame) for frame in tracker_ids),
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
    }
    metadata = {
        "num_frames": len(manifest_rows),
        "num_gt_dets": data["num_gt_dets"],
        "num_tracker_dets": data["num_tracker_dets"],
        "num_gt_ids": data["num_gt_ids"],
        "num_tracker_ids": data["num_tracker_ids"],
        "gt_id_map": gt_id_map,
        "tracker_id_map": tracker_id_map,
    }
    return data, metadata


def _mean_array(value) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.mean(array)) if array.size else float(value)


def evaluate_sequence_raw(data: dict, clear_threshold: float = 0.5) -> dict:
    ensure_trackeval_importable()
    from late_fusion_pkg.vendor import trackeval

    hota = trackeval.metrics.HOTA().eval_sequence(data)
    clear = trackeval.metrics.CLEAR({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    identity = trackeval.metrics.Identity({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False}).eval_sequence(data)
    return {"HOTA": hota, "CLEAR": clear, "Identity": identity}


def summarise_metric_results(raw: dict) -> dict:
    hota = raw["HOTA"]
    clear = raw["CLEAR"]
    identity = raw["Identity"]
    return {
        "HOTA": _mean_array(hota["HOTA"]),
        "DetA": _mean_array(hota["DetA"]),
        "AssA": _mean_array(hota["AssA"]),
        "LocA": _mean_array(hota["LocA"]),
        "HOTA_0": float(hota["HOTA(0)"]),
        "MOTA": float(clear["MOTA"]),
        "MOTP": float(clear["MOTP"]),
        "IDSW": int(clear["IDSW"]),
        "MT": int(clear["MT"]),
        "PT": int(clear["PT"]),
        "ML": int(clear["ML"]),
        "Frag": int(clear["Frag"]),
        "CLR_TP": int(clear["CLR_TP"]),
        "CLR_FP": int(clear["CLR_FP"]),
        "CLR_FN": int(clear["CLR_FN"]),
        "IDF1": float(identity["IDF1"]),
        "IDP": float(identity["IDP"]),
        "IDR": float(identity["IDR"]),
        "IDTP": int(identity["IDTP"]),
        "IDFP": int(identity["IDFP"]),
        "IDFN": int(identity["IDFN"]),
    }


def evaluate_sequence(data: dict, clear_threshold: float = 0.5) -> dict:
    return summarise_metric_results(evaluate_sequence_raw(data, clear_threshold=clear_threshold))


def combine_sequence_metrics(raw_by_sequence: dict[str, dict], clear_threshold: float = 0.5) -> dict:
    ensure_trackeval_importable()
    from late_fusion_pkg.vendor import trackeval

    hota_metric = trackeval.metrics.HOTA()
    clear_metric = trackeval.metrics.CLEAR({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False})
    identity_metric = trackeval.metrics.Identity({"THRESHOLD": clear_threshold, "PRINT_CONFIG": False})
    combined = {
        "HOTA": hota_metric.combine_sequences({key: value["HOTA"] for key, value in raw_by_sequence.items()}),
        "CLEAR": clear_metric.combine_sequences({key: value["CLEAR"] for key, value in raw_by_sequence.items()}),
        "Identity": identity_metric.combine_sequences({key: value["Identity"] for key, value in raw_by_sequence.items()}),
    }
    return summarise_metric_results(combined)
