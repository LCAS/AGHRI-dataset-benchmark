#!/usr/bin/env python3
"""Evaluate cached AGHRI generic detector/fusion outputs against raw annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon


def _ap(tp_fp: list[tuple[float, bool]], total_gt: int) -> dict[str, float]:
    if total_gt <= 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    ordered = sorted(tp_fp, key=lambda item: item[0], reverse=True)
    tp = np.array([ok for _, ok in ordered], dtype=np.float32)
    fp = 1.0 - tp
    if len(tp) == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recalls = ctp / total_gt
    precisions = ctp / np.maximum(ctp + cfp, 1e-8)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    precision = float(precisions[-1])
    recall = float(recalls[-1])
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
    return {"ap": ap, "precision": precision, "recall": recall, "f1": f1}


def _iou2d(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / area) if area > 0 else 0.0


def _poly(box):
    x, y, _z, length, width, _height, yaw = [float(v) for v in box[:7]]
    corners = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    out = corners @ rot.T
    out[:, 0] += x
    out[:, 1] += y
    return Polygon(out)


def _bev_iou(a, b) -> float:
    pa, pb = _poly(a), _poly(b)
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def _iou3d(a, b) -> float:
    inter_area = _poly(a).intersection(_poly(b)).area
    if inter_area <= 0:
        return 0.0
    z1, z2 = float(a[2]), float(b[2])
    h1, h2 = float(a[5]), float(b[5])
    inter_h = max(0.0, min(z1 + h1, z2 + h2) - max(z1, z2))
    inter = inter_area * inter_h
    vol = float(a[3] * a[4] * a[5] + b[3] * b[4] * b[5] - inter)
    return float(inter / vol) if vol > 0 else 0.0


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fraction_above_1m": float((arr > 1.0).mean()),
    }


def _load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _ann_index(path: Path, key_suffix: str) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(item["File"]).stem + key_suffix: item.get("Labels", []) for item in data}


def _gt_for_manifest(rows: list[dict]) -> tuple[dict[str, list], dict[str, list]]:
    gt2d_cache = {}
    gt3d_cache = {}
    gt2d = {}
    gt3d = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not sample_id:
            continue
        if row.get("image_path"):
            ref = row["gt_2d_reference"]
            gt2d_cache.setdefault(ref, _ann_index(Path(ref), ".png"))
            labels = gt2d_cache[ref].get(Path(row["image_path"]).stem + ".png", [])
            boxes = []
            for label in labels:
                vals = [float(v) for v in label.get("BoundingBoxes", [])[:4]]
                if len(vals) == 4:
                    x, y, w, h = vals
                    boxes.append((x, y, x + w, y + h))
            gt2d[sample_id] = boxes
        if row.get("lidar_point_cloud_path"):
            ref = row["gt_3d_reference"]
            gt3d_cache.setdefault(ref, _ann_index(Path(ref), ".pcd"))
            labels = gt3d_cache[ref].get(Path(row["lidar_point_cloud_path"]).stem + ".pcd", [])
            boxes = []
            for label in labels:
                vals = [float(v) for v in label.get("BoundingBoxes", [])[:9]]
                if len(vals) >= 9:
                    boxes.append(vals[:6] + [vals[8]])
            gt3d[sample_id] = boxes
    return gt2d, gt3d


def _read_camera(path: Path) -> dict[str, list[tuple[float, tuple[float, float, float, float]]]]:
    out = defaultdict(list)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"}:
                continue
            out[row["sample_id"]].append((float(row["confidence"]), (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))))
    return out


def _normalise_lidar_box(box: list[float], box_origin: str) -> list[float]:
    if box_origin == "bottom_center":
        return box
    if box_origin == "center":
        out = list(box)
        out[2] -= out[5] / 2.0
        return out
    raise ValueError(f"Unsupported LiDAR box origin: {box_origin}")


def _read_lidar(path: Path, box_origin: str = "bottom_center") -> dict[str, list[tuple[float, list[float]]]]:
    out = defaultdict(list)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"}:
                continue
            if str(row.get("selected_pedestrian", "")).lower() not in {"true", "1", "yes"}:
                continue
            box = [float(row[key]) for key in ("x", "y", "z", "length", "width", "height", "yaw")]
            out[row["sample_id"]].append((float(row["score"]), _normalise_lidar_box(box, box_origin)))
    return out


def _read_matched_indices(path: Path) -> dict[str, set[int]]:
    out = defaultdict(set)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            out[row["sample_id"]].add(int(row["lidar_index"]))
    return out


def _eval_2d(preds, gt, threshold=0.5) -> dict:
    total_gt = sum(len(v) for v in gt.values())
    outcomes = []
    for sample_id, pred_boxes in preds.items():
        matched = [False] * len(gt.get(sample_id, []))
        for score, box in sorted(pred_boxes, key=lambda item: item[0], reverse=True):
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt.get(sample_id, [])):
                if matched[idx]:
                    continue
                iou = _iou2d(box, gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            ok = best_idx >= 0 and best_iou >= threshold
            if ok:
                matched[best_idx] = True
            outcomes.append((score, ok))
    result = _ap(outcomes, total_gt)
    result.update({"predictions": sum(len(v) for v in preds.values()), "ground_truth": total_gt})
    return result


def _eval_3d(preds, gt, iou_thresholds=(0.25, 0.5, 0.75)) -> dict:
    total_gt = sum(len(v) for v in gt.values())
    result = {"predictions": sum(len(v) for v in preds.values()), "ground_truth": total_gt}
    for mode, fn in (("bev", _bev_iou), ("3d", _iou3d)):
        for threshold in iou_thresholds:
            outcomes = []
            center_errors = []
            range_errors = []
            for sample_id, pred_boxes in preds.items():
                matched = [False] * len(gt.get(sample_id, []))
                for score, box in sorted(pred_boxes, key=lambda item: item[0], reverse=True):
                    best_iou, best_idx = 0.0, -1
                    for idx, gt_box in enumerate(gt.get(sample_id, [])):
                        if matched[idx]:
                            continue
                        iou = fn(box, gt_box)
                        if iou > best_iou:
                            best_iou, best_idx = iou, idx
                    ok = best_idx >= 0 and best_iou >= threshold
                    if ok:
                        matched[best_idx] = True
                        gt_box = gt[sample_id][best_idx]
                        center_errors.append(float(np.linalg.norm(np.asarray(box[:3]) - np.asarray(gt_box[:3]))))
                        range_errors.append(abs(float(np.linalg.norm(box[:3])) - float(np.linalg.norm(gt_box[:3]))))
                    outcomes.append((score, ok))
            metrics = _ap(outcomes, total_gt)
            prefix = f"{mode}_iou_{threshold:.2f}"
            result.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            result[f"{prefix}_center_error"] = _summary(center_errors)
            result[f"{prefix}_range_error"] = _summary(range_errors)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-detections", type=Path, required=True)
    parser.add_argument("--lidar-detections", type=Path, required=True)
    parser.add_argument("--associations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--lidar-box-origin", choices=("bottom_center", "center"), default="bottom_center")
    args = parser.parse_args()

    rows = _load_manifest(args.manifest)
    gt2d, gt3d = _gt_for_manifest(rows)
    camera = _read_camera(args.camera_detections)
    lidar_all = _read_lidar(args.lidar_detections, box_origin=args.lidar_box_origin)
    matched_idx = _read_matched_indices(args.associations)
    lidar_matched = {
        sample_id: [item for idx, item in enumerate(items) if idx in matched_idx.get(sample_id, set())]
        for sample_id, items in lidar_all.items()
    }
    metrics = {
        "camera": _eval_2d(camera, gt2d),
        "lidar_second": _eval_3d(lidar_all, gt3d),
        "late_fusion_matched_only": _eval_3d(lidar_matched, gt3d),
        "late_fusion_all3d": _eval_3d(lidar_all, gt3d),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flat = []
    for system, values in metrics.items():
        row = {"system": system}
        for key, value in values.items():
            row[key] = json.dumps(value) if isinstance(value, dict) else value
        flat.append(row)
    fields = sorted({key for row in flat for key in row})
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({k: {"predictions": v.get("predictions"), "ground_truth": v.get("ground_truth")} for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
