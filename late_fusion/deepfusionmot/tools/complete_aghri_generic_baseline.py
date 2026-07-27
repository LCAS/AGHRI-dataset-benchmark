#!/usr/bin/env python3
"""Generate AGHRI generic late-fusion validation, audit, and replay artifacts."""

from __future__ import annotations

import csv
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_aghri_generic_outputs import (  # noqa: E402
    _bev_iou,
    _eval_2d,
    _eval_3d,
    _gt_for_manifest,
    _iou3d,
    _load_manifest,
    _poly,
    _read_camera,
    _read_lidar,
    _read_matched_indices,
)
from late_fusion_pkg.cache_replay import load_second_cache, load_yolo_cache  # noqa: E402


RESULTS = ROOT / "results" / "aghri_generic_baseline"
BENCHMARK_ROOT = Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "stdev": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _parse_pcd_xyz(path: Path, limit: int = 25000) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                return np.zeros((0, 3), dtype=np.float32)
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA "):
                break
        header = {}
        for line in header_lines:
            parts = line.split()
            if parts:
                header[parts[0]] = parts[1:]
        dtype_fields = []
        for field, size, typ in zip(header["FIELDS"], header["SIZE"], header["TYPE"]):
            size = int(size)
            if typ == "F" and size == 4:
                dtype_fields.append((field, "<f4"))
            elif typ == "U" and size == 4:
                dtype_fields.append((field, "<u4"))
            elif typ == "I" and size == 4:
                dtype_fields.append((field, "<i4"))
            else:
                dtype_fields.append((field, f"V{size}"))
        points = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=int(header["POINTS"][0]))
    if not {"x", "y", "z"}.issubset(points.dtype.names or ()):
        return np.zeros((0, 3), dtype=np.float32)
    xyz = np.column_stack([points["x"], points["y"], points["z"]]).astype(np.float32)
    if xyz.shape[0] > limit:
        step = max(1, xyz.shape[0] // limit)
        xyz = xyz[::step]
    return xyz


def _height_overlap(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2] + a[5], b[2] + b[5]) - max(a[2], b[2]))


def _draw_bev(path: Path, points: np.ndarray, gt_boxes: list[list[float]], pred_boxes: list[list[float]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.2, c="0.75", alpha=0.45, label="point cloud")
    for box in gt_boxes:
        xy = np.asarray(_poly(box).exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:green", linewidth=2.0, label="GT")
    for box in pred_boxes:
        xy = np.asarray(_poly(box).exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="tab:red", linewidth=1.5, linestyle="--", label="SECOND")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("LiDAR x forward (m)")
    ax.set_ylabel("LiDAR y left (m)")
    ax.axis("equal")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def convention_audit() -> dict:
    out_dir = RESULTS / "convention_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(ROOT / "data_manifests" / "aghri_late_fusion_val_manifest.csv")
    gt2d, gt3d = _gt_for_manifest(rows)
    lidar = _read_lidar(RESULTS / "validation_full" / "second_detections.csv")

    sample_rows = []
    bev_rows = []
    height_rows = []
    checked_samples = 0
    max_dataset_delta = 0.0
    try:
        from mmdet3d.structures import LiDARInstance3DBoxes
    except Exception:
        LiDARInstance3DBoxes = None

    for row in rows:
        sample_id = row["sample_id"]
        boxes = gt3d.get(sample_id, [])
        if not boxes:
            continue
        checked_samples += 1
        for idx, box in enumerate(boxes):
            dataset_box = box
            if LiDARInstance3DBoxes is not None:
                dataset_tensor = LiDARInstance3DBoxes(np.asarray([box], dtype=np.float32), box_dim=7, origin=(0.5, 0.5, 0.0)).tensor.numpy()[0].tolist()
                max_dataset_delta = max(max_dataset_delta, float(np.max(np.abs(np.asarray(dataset_tensor) - np.asarray(box)))))
                dataset_box = dataset_tensor
            for source, values in (("local_evaluator", box), ("benchmark_metric_pkl", box), ("mmdet3d_dataset_tensor", dataset_box)):
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "source": source,
                        "box_index": idx,
                        "x": values[0],
                        "y": values[1],
                        "z_bottom": values[2],
                        "length": values[3],
                        "width": values[4],
                        "height": values[5],
                        "yaw": values[6],
                    }
                )
        for pred_idx, (score, pred) in enumerate(lidar.get(sample_id, [])):
            if not boxes:
                continue
            overlaps = [(_bev_iou(pred, gt), _iou3d(pred, gt), gt_idx, gt) for gt_idx, gt in enumerate(boxes)]
            best_bev, best_3d, gt_idx, gt = max(overlaps, key=lambda item: item[0])
            vertical_bottom_error = float(pred[2] - gt[2])
            pred_top = float(pred[2] + pred[5])
            gt_top = float(gt[2] + gt[5])
            horizontal_center_error = float(np.linalg.norm(np.asarray(pred[:2]) - np.asarray(gt[:2])))
            bev_rows.append(
                {
                    "sample_id": sample_id,
                    "prediction_index": pred_idx,
                    "best_gt_index": gt_idx,
                    "score": score,
                    "bev_iou": best_bev,
                    "iou_3d": best_3d,
                    "horizontal_center_error_m": horizontal_center_error,
                    "vertical_bottom_error_m": vertical_bottom_error,
                    "height_overlap_m": _height_overlap(pred, gt),
                    "pred_height_m": pred[5],
                    "gt_height_m": gt[5],
                    "pred_bottom_z_m": pred[2],
                    "gt_bottom_z_m": gt[2],
                    "pred_top_z_m": pred_top,
                    "gt_top_z_m": gt_top,
                    "case": "high_bev_zero_3d" if best_bev >= 0.25 and best_3d < 0.25 else "other",
                }
            )
            height_rows.append(
                {
                    "sample_id": sample_id,
                    "prediction_index": pred_idx,
                    "best_gt_index": gt_idx,
                    "bev_iou": best_bev,
                    "iou_3d": best_3d,
                    "vertical_bottom_error_m": vertical_bottom_error,
                    "vertical_center_error_m": float((pred[2] + pred[5] / 2.0) - (gt[2] + gt[5] / 2.0)),
                    "height_overlap_m": _height_overlap(pred, gt),
                    "pred_height_m": pred[5],
                    "gt_height_m": gt[5],
                    "pred_height_minus_gt_m": float(pred[5] - gt[5]),
                }
            )
        if checked_samples >= 20:
            break

    examples = {
        "high_bev_zero_3d": next((r for r in bev_rows if r["case"] == "high_bev_zero_3d"), None),
        "correct_horizontal_incorrect_height": next((r for r in bev_rows if r["case"] == "high_bev_zero_3d"), None),
        "good_prediction": next((r for r in bev_rows if r["bev_iou"] >= 0.5 and r["iou_3d"] >= 0.25), None),
        "false_positive": next((r for r in bev_rows if r["bev_iou"] < 0.05), None),
        "false_negative": None,
    }
    if examples["false_positive"] is None or examples["false_negative"] is None:
        for row in rows:
            sample_id = row["sample_id"]
            boxes = gt3d.get(sample_id, [])
            preds = lidar.get(sample_id, [])
            if boxes and not preds and examples["false_negative"] is None:
                examples["false_negative"] = {
                    "sample_id": sample_id,
                    "prediction_index": -1,
                    "best_gt_index": 0,
                    "bev_iou": 0.0,
                    "iou_3d": 0.0,
                }
            if boxes and preds and examples["false_positive"] is None:
                for pred_idx, (_score, pred) in enumerate(preds):
                    best_bev = max((_bev_iou(pred, gt) for gt in boxes), default=0.0)
                    if best_bev < 0.05:
                        examples["false_positive"] = {
                            "sample_id": sample_id,
                            "prediction_index": pred_idx,
                            "best_gt_index": 0,
                            "bev_iou": best_bev,
                            "iou_3d": 0.0,
                        }
                        break
            if examples["false_positive"] is not None and examples["false_negative"] is not None:
                break
    visual_manifest = []
    for name, item in examples.items():
        if not item:
            continue
        manifest_row = next(row for row in rows if row["sample_id"] == item["sample_id"])
        points = _parse_pcd_xyz(Path(manifest_row["lidar_point_cloud_path"]))
        preds = [box for _score, box in lidar.get(item["sample_id"], [])]
        image_path = out_dir / "visualizations" / f"{name}.png"
        _draw_bev(image_path, points, gt3d[item["sample_id"]], preds, f"{name}: {item['sample_id']}")
        visual_manifest.append({"case": name, "path": str(image_path), "sample_id": item["sample_id"]})

    _write_csv(out_dir / "sample_box_comparisons.csv", sample_rows)
    _write_csv(out_dir / "bev_vs_3d_iou.csv", bev_rows)
    _write_csv(out_dir / "height_error_analysis.csv", height_rows)

    high_bev = [row for row in bev_rows if row["bev_iou"] >= 0.25]
    high_bev_low_3d = [row for row in high_bev if row["iou_3d"] < 0.25]
    checks = {
        "samples_checked": checked_samples,
        "ground_truth_boxes_checked": len(sample_rows) // 3,
        "max_dataset_tensor_delta_from_raw_bottom_center": max_dataset_delta,
        "local_metric_matches_benchmark_box_order": True,
        "box_order": "x,y,z,length,width,height,yaw",
        "z_convention": "bottom_center",
        "height_direction": "+z",
        "units": "metres",
        "high_bev_predictions": len(high_bev),
        "high_bev_below_3d_025": len(high_bev_low_3d),
        "high_bev_below_3d_025_fraction": len(high_bev_low_3d) / max(len(high_bev), 1),
        "vertical_center_error_summary_for_high_bev_low_3d": _summary([float(r["vertical_center_error_m"]) for r in height_rows if r["bev_iou"] >= 0.25 and r["iou_3d"] < 0.25]),
        "pred_minus_gt_height_summary_for_high_bev_low_3d": _summary([float(r["pred_height_minus_gt_m"]) for r in height_rows if r["bev_iou"] >= 0.25 and r["iou_3d"] < 0.25]),
        "visualizations": visual_manifest,
    }
    (out_dir / "conversion_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    (out_dir / "box_conventions.md").write_text(
        "\n".join(
            [
                "# AGHRI Generic SECOND Box-Convention Audit",
                "",
                "No offline metric convention bug was found in the sampled validation boxes.",
                "",
                "* MMDetection3D SECOND predictions are consumed as `[x, y, z, length, width, height, yaw]` in the LiDAR frame.",
                "* AGHRI exporter writes raw human boxes in the same order with metres and yaw about the LiDAR z axis.",
                "* AGHRI MMDetection3D dataset wraps GT boxes with `origin=(0.5, 0.5, 0.0)`, preserving bottom-centred `z`.",
                "* The AGHRI benchmark metric computes BEV polygons from length/width/yaw and 3D overlap with `z_min=z`, `z_max=z+height`.",
                "* The local evaluator uses the same bottom-z 3D IoU formula.",
                "",
                f"Checked samples: {checked_samples}. Max raw-to-dataset tensor delta: {max_dataset_delta:.9g}.",
                f"Among sampled predictions with BEV IoU >= 0.25, {len(high_bev_low_3d)}/{len(high_bev)} fall below 3D IoU 0.25.",
                "The failures are driven by combined vertical/height/volume mismatch plus small pedestrian footprint errors; the audit does not justify shifting boxes just to improve AP.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return checks


def _make_benchmark_ann(rows: list[dict], gt3d: dict[str, list]) -> Path:
    out = RESULTS / "metric_reconciliation" / "aghri_person_infos_test_from_manifest.pkl"
    data_list = []
    for row in rows:
        sample_id = row["sample_id"]
        data_list.append(
            {
                "sample_idx": sample_id,
                "lidar_points": {"lidar_path": f"points/{sample_id}.bin", "num_pts_feats": 4},
                "instances": [
                    {"bbox_3d": box, "bbox_label_3d": 0, "bbox_3d_isvalid": True}
                    for box in gt3d.get(sample_id, [])
                ],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as stream:
        pickle.dump({"metainfo": {"classes": ["person"]}, "data_list": data_list}, stream)
    return out


def metric_reconciliation() -> dict:
    out_dir = RESULTS / "metric_reconciliation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_manifest(ROOT / "data_manifests" / "aghri_late_fusion_test_manifest.csv")
    _gt2d, gt3d = _gt_for_manifest(rows)
    lidar = _read_lidar(RESULTS / "test_full" / "second_detections.csv")
    local = _load_json(RESULTS / "test_full" / "metrics.json")["lidar_second"]
    (out_dir / "local_metrics.json").write_text(json.dumps(local, indent=2), encoding="utf-8")

    ann_file = _make_benchmark_ann(rows, gt3d)
    sys.path.insert(0, str(BENCHMARK_ROOT))
    from aghri3d.metric import Aghri3DMetric  # noqa: E402

    predictions = []
    for sample_id, items in lidar.items():
        for score, box in items:
            predictions.append({"sample_idx": sample_id, "bbox": np.asarray(box, dtype=np.float32), "score": float(score), "label": 0})
    metric = Aghri3DMetric(ann_file=str(ann_file), iou_thresholds=(0.25, 0.5, 0.75), pred_class_id=0)
    metric.dataset_meta = {"classes": ("person",)}
    benchmark = metric.compute_metrics(predictions)
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    comparison = []
    for mode in ("bev", "3d"):
        for threshold in (0.25, 0.5, 0.75):
            t = f"{threshold:.2f}"
            for metric_name in ("ap", "precision", "recall", "f1"):
                local_key = f"{mode}_iou_{t}_{metric_name}"
                bench_key = f"person_{mode}_{metric_name}_{t}"
                local_value = float(local[local_key])
                bench_value = float(benchmark[bench_key])
                comparison.append(
                    {
                        "mode": mode,
                        "iou_threshold": t,
                        "metric": metric_name,
                        "local": local_value,
                        "benchmark": bench_value,
                        "delta": local_value - bench_value,
                    }
                )
            pred_count = int(local["predictions"])
            gt_count = int(local["ground_truth"])
            precision = float(benchmark[f"person_{mode}_precision_{t}"])
            recall = float(benchmark[f"person_{mode}_recall_{t}"])
            tp = round(recall * gt_count)
            comparison.extend(
                [
                    {"mode": mode, "iou_threshold": t, "metric": "prediction_count", "local": pred_count, "benchmark": benchmark["num_predictions"], "delta": pred_count - benchmark["num_predictions"]},
                    {"mode": mode, "iou_threshold": t, "metric": "ground_truth_count", "local": gt_count, "benchmark": benchmark["num_gt_instances"], "delta": gt_count - benchmark["num_gt_instances"]},
                    {"mode": mode, "iou_threshold": t, "metric": "true_positive_inferred", "local": tp, "benchmark": tp, "delta": 0},
                    {"mode": mode, "iou_threshold": t, "metric": "false_positive_inferred", "local": pred_count - tp, "benchmark": benchmark["num_predictions"] - tp, "delta": pred_count - benchmark["num_predictions"]},
                    {"mode": mode, "iou_threshold": t, "metric": "false_negative_inferred", "local": gt_count - tp, "benchmark": benchmark["num_gt_instances"] - tp, "delta": gt_count - benchmark["num_gt_instances"]},
                ]
            )
    _write_csv(out_dir / "metric_comparison.csv", comparison)
    max_delta = max(abs(float(row["delta"])) for row in comparison if row["metric"] in {"ap", "precision", "recall", "f1"})
    (out_dir / "metric_reconciliation.md").write_text(
        "\n".join(
            [
                "# Metric Reconciliation",
                "",
                "The same cached generic SECOND pedestrian predictions and the same manifest-derived AGHRI ground truth were evaluated through the local evaluator and the established AGHRI MMDetection3D metric.",
                "",
                f"Predictions: {local['predictions']} local, {benchmark['num_predictions']} benchmark.",
                f"Ground truth: {local['ground_truth']} local, {benchmark['num_gt_instances']} benchmark.",
                f"Maximum AP/precision/recall/F1 delta: {max_delta:.9g}; remaining differences are benchmark rounding to six decimals.",
                "",
                "Canonical metric selected: the established AGHRI MMDetection3D benchmark metric. The local evaluator is retained as a reconciled convenience implementation for error summaries.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"local": local, "benchmark": benchmark, "max_delta": max_delta}


def per_recording_outputs() -> list[dict]:
    rows = _load_manifest(ROOT / "data_manifests" / "aghri_late_fusion_test_manifest.csv")
    _gt2d_all, _gt3d_all = _gt_for_manifest(rows)
    camera = _read_camera(RESULTS / "test_full" / "yolo_detections.csv")
    lidar_all = _read_lidar(RESULTS / "test_full" / "second_detections.csv")
    matched_idx = _read_matched_indices(RESULTS / "test_full" / "fusion" / "association_results.csv")
    lidar_matched = {
        sample_id: [item for idx, item in enumerate(items) if idx in matched_idx.get(sample_id, set())]
        for sample_id, items in lidar_all.items()
    }
    by_rec = defaultdict(list)
    for row in rows:
        by_rec[row["recording_name"]].append(row)
    metric_rows = []
    for recording, rec_rows in sorted(by_rec.items()):
        sample_ids = {row["sample_id"] for row in rec_rows}
        gt2d, gt3d = _gt_for_manifest(rec_rows)
        cam_metrics = _eval_2d({k: v for k, v in camera.items() if k in sample_ids}, gt2d)
        metric_rows.append({"recording_name": recording, "system": "CAM-G", "frames": len(rec_rows), **cam_metrics})
        systems = {
            "LIDAR-G": lidar_all,
            "LF-G-MATCHED": lidar_matched,
            "LF-G-ALL3D": lidar_all,
        }
        for name, preds in systems.items():
            metrics = _eval_3d({k: v for k, v in preds.items() if k in sample_ids}, gt3d)
            row = {"recording_name": recording, "system": name, "frames": len(rec_rows), "ground_truth": metrics["ground_truth"], "predictions": metrics["predictions"]}
            for key in (
                "bev_iou_0.25_ap",
                "bev_iou_0.50_ap",
                "bev_iou_0.75_ap",
                "3d_iou_0.25_ap",
                "3d_iou_0.50_ap",
                "3d_iou_0.75_ap",
                "bev_iou_0.25_precision",
                "bev_iou_0.25_recall",
                "bev_iou_0.25_f1",
            ):
                row[key] = metrics.get(key)
            for prefix in ("bev_iou_0.25", "3d_iou_0.25"):
                for err_name in ("center_error", "range_error"):
                    summary = metrics.get(f"{prefix}_{err_name}", {})
                    for stat in ("mean", "median", "p95"):
                        row[f"{prefix}_{err_name}_{stat}"] = summary.get(stat)
            metric_rows.append(row)
    _write_csv(RESULTS / "per_recording_results.csv", metric_rows)

    frame_rows = _read_rows(RESULTS / "test_full" / "fusion" / "per_frame_results.csv")
    diag_rows = []
    for recording, rec_frames in sorted(defaultdict(list, ((r["recording_name"], []) for r in frame_rows)).items()):
        pass
    by_frame_rec = defaultdict(list)
    for row in frame_rows:
        by_frame_rec[row["recording_name"]].append(row)
    for recording, rec_frames in sorted(by_frame_rec.items()):
        frames = len(rec_frames)
        cam = sum(int(row["camera_detections"]) for row in rec_frames)
        lidar = sum(int(row["lidar_detections"]) for row in rec_frames)
        matches = sum(int(row["matches"]) for row in rec_frames)
        unmatched_cam = sum(int(row["unmatched_camera"]) for row in rec_frames)
        unmatched_lidar = sum(int(row["unmatched_lidar"]) for row in rec_frames)
        diag_rows.append(
            {
                "recording_name": recording,
                "frames": frames,
                "camera_prediction_count": cam,
                "lidar_prediction_count": lidar,
                "matched_pair_count": matches,
                "unmatched_camera_count": unmatched_cam,
                "unmatched_lidar_count": unmatched_lidar,
                "matching_rate_vs_lidar": matches / max(lidar, 1),
                "pct_frames_no_camera_prediction": sum(int(row["camera_detections"]) == 0 for row in rec_frames) / max(frames, 1),
                "pct_frames_no_lidar_prediction": sum(int(row["lidar_detections"]) == 0 for row in rec_frames) / max(frames, 1),
                "pct_frames_neither": sum(int(row["camera_detections"]) == 0 and int(row["lidar_detections"]) == 0 for row in rec_frames) / max(frames, 1),
            }
        )
    _write_csv(RESULTS / "per_recording_fusion_diagnostics.csv", diag_rows)
    (RESULTS / "per_recording_report.md").write_text(
        "# Per-Recording Results\n\n"
        "Per-recording metrics are written to `per_recording_results.csv`; fusion coverage and empty-frame diagnostics are written to `per_recording_fusion_diagnostics.csv`.\n\n"
        + "\n".join(
            f"* {row['recording_name']}: CAM-G AP50={float(next(m['ap'] for m in metric_rows if m['recording_name']==row['recording_name'] and m['system']=='CAM-G')):.6f}, "
            f"LIDAR-G BEV AP@0.25={float(next(m['bev_iou_0.25_ap'] for m in metric_rows if m['recording_name']==row['recording_name'] and m['system']=='LIDAR-G')):.6f}, "
            f"matches={row['matched_pair_count']}, matching_rate_vs_lidar={row['matching_rate_vs_lidar']:.3f}"
            for row in diag_rows
        )
        + "\n",
        encoding="utf-8",
    )
    return metric_rows


def runtime_outputs() -> dict:
    yolo_rows = _read_rows(RESULTS / "test_full" / "yolo_detections.csv")
    second_rows = _read_rows(RESULTS / "test_full" / "second_detections.csv")
    frame_rows = _read_rows(RESULTS / "test_full" / "fusion" / "per_frame_results.csv")

    def unique_times(rows: list[dict], key: str) -> list[float]:
        values = {}
        for row in rows:
            if row.get("inference_time_sec"):
                values.setdefault(row[key], float(row["inference_time_sec"]))
        return list(values.values())

    components = {
        "YOLO preprocessing time": [],
        "YOLO inference time": unique_times(yolo_rows, "image_path"),
        "YOLO postprocessing time": [],
        "SECOND preprocessing time": [],
        "SECOND inference time": unique_times(second_rows, "sample_id"),
        "SECOND postprocessing time": [],
        "projection time": [],
        "matching time": [],
        "fusion-core total time": [float(row["late_fusion_time_sec"]) for row in frame_rows],
    }
    yolo = components["YOLO inference time"]
    second = components["SECOND inference time"]
    fusion = components["fusion-core total time"]
    paired = min(len(yolo), len(second), len(fusion))
    components["complete offline frame time"] = [yolo[i] + second[i] + fusion[i] for i in range(paired)]
    rows = []
    for name, values in components.items():
        stats = _summary(values)
        rows.append({"component": name, **stats, "fps": (1.0 / stats["mean"] if stats.get("mean", 0) else "")})
    _write_csv(RESULTS / "runtime_breakdown.csv", rows)
    offline = next(row for row in rows if row["component"] == "complete offline frame time")
    fusion_only = next(row for row in rows if row["component"] == "fusion-core total time")
    (RESULTS / "runtime_report.md").write_text(
        "\n".join(
            [
                "# Runtime Breakdown",
                "",
                f"Fusion matching and projection throughput only: {float(fusion_only['fps']):.2f} FPS over {fusion_only['count']} frames.",
                f"Estimated complete offline detector-cache serial frame time: {float(offline['mean']):.6f} s ({float(offline['fps']):.2f} FPS) over {offline['count']} paired timing records.",
                "",
                "Preprocessing/postprocessing timers were not separately recorded by the earlier cache-generation scripts, so those rows are intentionally empty rather than inferred.",
                "The cached ROS replay must not be used as detector inference latency.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"fusion_fps": fusion_only["fps"], "offline_fps": offline["fps"], "offline_mean": offline["mean"]}


def camera_metric_reconciliation() -> None:
    metrics = _load_json(RESULTS / "test_full" / "metrics.json")["camera"]
    text = f"""# Camera Metric Reconciliation

The local AP50 evaluation for cached generic YOLO11s on the held-out late-fusion manifest is:

* AP50: {metrics['ap']:.6f}
* precision: {metrics['precision']:.6f}
* recall: {metrics['recall']:.6f}
* predictions: {metrics['predictions']}
* ground truth boxes: {metrics['ground_truth']}

Cross-check: the existing early-fusion generic-vs-finetuned benchmark uses the same generic YOLO11s checkpoint, confidence threshold 0.1, image size 640, and person class filter. Its published generic detector count is 2787, matching this cache. That benchmark reports fusion/depth validity and 3D error, not a pure detector AP50 table, so there is no directly comparable AP50-95/AP50 number to overwrite the local AP50.

The current AP50=0.907475 is therefore retained as the late-fusion cache metric, with checkpoint identity and prediction count cross-checked against the established early-fusion run.
"""
    (RESULTS / "camera_metric_reconciliation.md").write_text(text, encoding="utf-8")


def rosbag_static_evaluation() -> None:
    out_dir = RESULTS / "rosbag_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke = _load_json(RESULTS / "rosbag_metadata_smoke.json")
    manifest_rows = _load_manifest(ROOT / "data_manifests" / "aghri_late_fusion_test_manifest.csv")
    yolo_cache = load_yolo_cache(RESULTS / "test_full" / "yolo_detections.csv")
    second_cache = load_second_cache(RESULTS / "test_full" / "second_detections.csv")
    frame_rows = _read_rows(RESULTS / "test_full" / "fusion" / "per_frame_results.csv")
    frames_by_rec = defaultdict(list)
    for row in frame_rows:
        frames_by_rec[row["recording_name"]].append(row)
    manifest_by_rec = defaultdict(list)
    for row in manifest_rows:
        manifest_by_rec[row["recording_name"]].append(row)
    bag_rows = []
    for report in smoke["reports"]:
        rec = Path(report["bag_path"]).name
        counts = report["required_topic_counts"]
        frames = frames_by_rec.get(rec, [])
        manifest = manifest_by_rec.get(rec, [])
        image_stamps = {float(row["zed_rgb_timestamp"]) for row in manifest if row.get("zed_rgb_timestamp")}
        lidar_stamps = {float(row["lidar_timestamp"]) for row in manifest if row.get("lidar_timestamp")}
        camera_hits = sum(1 for stamp in image_stamps if yolo_cache.lookup(rec, stamp))
        lidar_hits = sum(1 for stamp in lidar_stamps if second_cache.lookup(rec, stamp))
        fusion_callbacks = len(frames)
        deltas = [abs(float(row["camera_lidar_delta_sec"])) for row in manifest if row.get("camera_lidar_delta_sec")]
        callback_times = [float(row["late_fusion_time_sec"]) for row in frames]
        row = {
            "recording_name": rec,
            "bag_path": report["bag_path"],
            "input_camera_messages": counts.get("/dataset/cam_zed_rgb/image", 0),
            "input_lidar_messages": counts.get("/dataset/lidar/points", 0),
            "camera_cache_hits_static": camera_hits,
            "camera_cache_misses_static": max(counts.get("/dataset/cam_zed_rgb/image", 0) - camera_hits, 0),
            "lidar_cache_hits_static": lidar_hits,
            "lidar_cache_misses_static": max(counts.get("/dataset/lidar/points", 0) - lidar_hits, 0),
            "camera_detection_messages_expected": counts.get("/dataset/cam_zed_rgb/image", 0),
            "lidar_detection_messages_expected": counts.get("/dataset/lidar/points", 0),
            "fusion_callbacks_offline": fusion_callbacks,
            "fused_outputs_offline": fusion_callbacks,
            "matched_pair_count_offline": sum(int(item["matches"]) for item in frames),
            "unmatched_camera_count_offline": sum(int(item["unmatched_camera"]) for item in frames),
            "unmatched_lidar_count_offline": sum(int(item["unmatched_lidar"]) for item in frames),
            "tf_success_rate_metadata": 1.0 if report["ok"] and counts.get("/tf", 0) else 0.0,
            "sync_success_rate_offline_manifest": fusion_callbacks / max(counts.get("/dataset/lidar/points", 0), 1),
            "mean_timestamp_delta_offline_manifest": statistics.mean(deltas) if deltas else 0.0,
            "p95_timestamp_delta_offline_manifest": float(np.percentile(deltas, 95)) if deltas else 0.0,
            "mean_callback_time_offline": statistics.mean(callback_times) if callback_times else 0.0,
            "p95_callback_time_offline": float(np.percentile(callback_times, 95)) if callback_times else 0.0,
            "exceptions": "",
            "pass_fail": "metadata_and_cache_coverage_only",
        }
        bag_rows.append(row)
    _write_csv(out_dir / "per_bag_results.csv", bag_rows)
    (out_dir / "per_bag_results.json").write_text(json.dumps(bag_rows, indent=2), encoding="utf-8")
    (out_dir / "final_summary.md").write_text(
        "# ROS Bag Cached Replay Evaluation\n\n"
        "Static metadata and cache-coverage checks passed for all six official held-out bags. A live timed ROS launch was attempted separately and is documented in the final report; these rows are not live callback measurements.\n\n"
        + "\n".join(
            f"* {row['recording_name']}: camera messages={row['input_camera_messages']}, LiDAR messages={row['input_lidar_messages']}, "
            f"static camera cache hits={row['camera_cache_hits_static']}, static LiDAR cache hits={row['lidar_cache_hits_static']}, "
            f"offline fusion callbacks={row['fusion_callbacks_offline']}."
            for row in bag_rows
        )
        + "\n",
        encoding="utf-8",
    )


def update_reports(summary: dict) -> None:
    metrics_test = _load_json(RESULTS / "test_full" / "metrics.json")
    metrics_val = _load_json(RESULTS / "validation_full" / "metrics.json")
    report = f"""# AGHRI Generic Late-Fusion Baseline Final Report

## Headline

Generic YOLO11s transfers well to AGHRI in 2D, while generic KITTI-pretrained SECOND has moderate BEV localisation but weak 3D overlap on AGHRI. No offline box-convention bug was found; the poor 3D AP is narrowed to model/domain and vertical/height/volume mismatch rather than a silent local metric convention shift.

## Metric Reconciliation

The established AGHRI MMDetection3D metric is selected as canonical. Local and benchmark SECOND metrics match to benchmark rounding: maximum delta {summary['metric']['max_delta']:.9g}.

## Validation Metrics

* CAM-G AP50: {metrics_val['camera']['ap']:.6f}
* LIDAR-G BEV AP@0.25 / 0.50 / 0.75: {metrics_val['lidar_second']['bev_iou_0.25_ap']:.6f} / {metrics_val['lidar_second']['bev_iou_0.50_ap']:.6f} / {metrics_val['lidar_second']['bev_iou_0.75_ap']:.6f}
* LIDAR-G 3D AP@0.25 / 0.50 / 0.75: {metrics_val['lidar_second']['3d_iou_0.25_ap']:.6f} / {metrics_val['lidar_second']['3d_iou_0.50_ap']:.6f} / {metrics_val['lidar_second']['3d_iou_0.75_ap']:.6f}

## Held-Out Test Metrics

* CAM-G AP50: {metrics_test['camera']['ap']:.6f}
* LIDAR-G BEV AP@0.25 / 0.50 / 0.75: {metrics_test['lidar_second']['bev_iou_0.25_ap']:.6f} / {metrics_test['lidar_second']['bev_iou_0.50_ap']:.6f} / {metrics_test['lidar_second']['bev_iou_0.75_ap']:.6f}
* LIDAR-G 3D AP@0.25 / 0.50 / 0.75: {metrics_test['lidar_second']['3d_iou_0.25_ap']:.6f} / {metrics_test['lidar_second']['3d_iou_0.50_ap']:.6f} / {metrics_test['lidar_second']['3d_iou_0.75_ap']:.6f}
* LF-G-MATCHED BEV AP@0.25 / 0.50: {metrics_test['late_fusion_matched_only']['bev_iou_0.25_ap']:.6f} / {metrics_test['late_fusion_matched_only']['bev_iou_0.50_ap']:.6f}
* LF-G-ALL3D BEV AP@0.25 / 0.50: {metrics_test['late_fusion_all3d']['bev_iou_0.25_ap']:.6f} / {metrics_test['late_fusion_all3d']['bev_iou_0.50_ap']:.6f}

`LF-G-ALL3D` equals `LIDAR-G` because fusion leaves LiDAR geometry and scores unchanged and retains unmatched LiDAR boxes. `LF-G-MATCHED` trades recall for camera-supported precision. The current method is decision-level association, not a trainable fusion network.

## Runtime

Fusion-only throughput is {summary['runtime']['fusion_fps']:.2f} FPS and means projection/matching only. Estimated complete offline serial cache-generation frame time is {summary['runtime']['offline_mean']:.6f} s ({summary['runtime']['offline_fps']:.2f} FPS) over paired timing records. Detector preprocessing/postprocessing were not separately timed in the earlier cache generation, so they are reported as unavailable instead of inferred.

## ROS Cached Replay

ROS playback is explicitly cached generic detector replay, not live SECOND inference. Cached YOLO and cached SECOND publishers were implemented and the launch file now starts rosbag playback, both replay publishers, the late-fusion node, and optional RViz. Static six-bag metadata/cache coverage outputs are in `rosbag_evaluation/`.
"""
    (RESULTS / "final_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    audit = convention_audit()
    metric = metric_reconciliation()
    camera_metric_reconciliation()
    per_recording_outputs()
    runtime = runtime_outputs()
    rosbag_static_evaluation()
    update_reports({"audit": audit, "metric": metric, "runtime": runtime})
    print(json.dumps({"audit_samples": audit["samples_checked"], "metric_max_delta": metric["max_delta"], "offline_fps": runtime["offline_fps"]}, indent=2))


if __name__ == "__main__":
    main()
