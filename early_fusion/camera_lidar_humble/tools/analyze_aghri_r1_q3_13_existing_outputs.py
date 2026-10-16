#!/usr/bin/env python3
"""R1-Q3-13 end-to-end analysis from frozen AGHRI early-fusion outputs.

This script does not run YOLO, does not redo LiDAR association, and does not
modify the saved evaluator outputs. It reads the existing legacy-association
Generic YOLO11s and AGHRI fine-tuned YOLO11s result JSON files and adds a
GT-centric localisation summary with recording-level bootstrap uncertainty.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys
import zipfile
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT.parent / "results/aghri_detector_fusion_2x2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/r1_q3_13_existing_output_eval"
DEFAULT_CONFIG = REPO_ROOT / "config/aghri_zed_livox.yaml"
DEFAULT_BOOTSTRAP_SEED = 20261006
MODEL_RUNS = {
    "Generic YOLO11s": "generic_legacy",
    "AGHRI fine-tuned YOLO11s": "finetuned_legacy",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            row for row in csv.DictReader(stream)
            if row.get("supported", "True").lower() != "false"
        ]


def timestamp(record: dict[str, Any]) -> float:
    return float(record["Timestamp"])


def nearest_indexed_record(records: list[dict[str, Any]], stamp: float, tolerance: float):
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


def useful_pairs(recording: Path, image_slop: float):
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
    return pairs


def load_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as stream:
            return json.loads(stream.read().decode("utf-8"))


def quat_to_rotation_matrix(rotation: dict[str, Any]) -> np.ndarray:
    x = float(rotation["x"])
    y = float(rotation["y"])
    z = float(rotation["z"])
    w = float(rotation["w"])
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError("zero-length quaternion in calibration")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_to_matrix(edge: dict[str, Any]) -> np.ndarray:
    transform = edge["transform"]
    translation = transform["translation"]
    rotation = transform["rotation"]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quat_to_rotation_matrix(rotation)
    matrix[:3, 3] = [
        float(translation["x"]),
        float(translation["y"]),
        float(translation["z"]),
    ]
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inverse


def transform_between_frames(
    extrinsics: dict[str, Any],
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    adjacency: dict[str, list[str]] = {}
    transform_map: dict[tuple[str, str], np.ndarray] = {}
    for edge in extrinsics["transforms"]:
        parent = (edge.get("header") or {}).get("frame_id")
        child = edge.get("child_frame_id")
        if not parent or not child:
            continue
        matrix = transform_to_matrix(edge)
        adjacency.setdefault(parent, []).append(child)
        adjacency.setdefault(child, []).append(parent)
        transform_map[(child, parent)] = matrix
        transform_map[(parent, child)] = invert_transform(matrix)

    queue = deque([[source_frame]])
    path = None
    while queue:
        candidate = queue.popleft()
        if candidate[-1] == target_frame:
            path = candidate
            break
        for neighbor in adjacency.get(candidate[-1], []):
            if neighbor not in candidate:
                queue.append(candidate + [neighbor])
    if path is None:
        raise ValueError(f"No transform path from {source_frame} to {target_frame}")

    transform = np.eye(4, dtype=float)
    for source, target in zip(path[:-1], path[1:]):
        transform = transform_map[(source, target)] @ transform
    return transform


def load_camera_from_lidar(config_path: Path) -> np.ndarray:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["lidar_fusion"]["ros__parameters"]
    extrinsics = load_json_from_zip(
        Path(params["calibration_zip_path"]),
        params["extrinsics_member"],
    )
    return transform_between_frames(
        extrinsics,
        params["lidar_frame_name"],
        params["camera_optical_frame_name"],
    )


def distance_band_from_depth(depth: float | None) -> str:
    if depth is None or not math.isfinite(float(depth)):
        return "unknown"
    if depth < 2.0:
        return "near_lt_2m"
    if depth < 5.0:
        return "medium_2_5m"
    return "far_ge_5m"


def gt_identity(label: dict[str, Any], index: int) -> str:
    value = label.get("Class")
    return str(value if value is not None else index)


def transform_point(matrix: np.ndarray, xyz: list[float]) -> np.ndarray:
    point = np.asarray([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=float)
    return (matrix @ point)[:3]


def build_gt_instances(
    manifest: list[dict[str, Any]],
    frames_by_recording: dict[str, list[dict[str, Any]]],
    camera_from_lidar: np.ndarray,
    image_slop: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    gt_rows: list[dict[str, Any]] = []
    frame_lookup = {
        recording: {int(frame["frame_number"]): frame for frame in frames}
        for recording, frames in frames_by_recording.items()
    }
    for entry in manifest:
        recording = entry["recording_name"]
        recording_path = Path(entry["recording_path"])
        pairs = useful_pairs(recording_path, image_slop)
        saved_frames = frame_lookup.get(recording, {})
        if len(pairs) != len(saved_frames):
            warnings.append(
                f"{recording}: reconstructed {len(pairs)} pairs but saved "
                f"frames contain {len(saved_frames)} rows"
            )
        for frame_number, (lidar_record, _image_record, image_index, _delta) in enumerate(pairs):
            frame = saved_frames.get(frame_number)
            if frame is None:
                continue
            labels = lidar_record.get("Labels") or []
            if len(labels) != int(frame.get("gt3d_count", 0)):
                warnings.append(
                    f"{recording} frame {frame_number}: annotation has {len(labels)} "
                    f"GT labels but saved frame has {frame.get('gt3d_count')}"
                )
            matched_ids = {str(item) for item in frame.get("matched_gt_ids", [])}
            for label_index, label in enumerate(labels):
                box = label.get("BoundingBoxes") or []
                if len(box) < 3:
                    continue
                identity = gt_identity(label, label_index)
                camera_xyz = transform_point(camera_from_lidar, box[:3])
                gt_rows.append({
                    "recording": recording,
                    "scene": entry.get("scene"),
                    "motion": frame.get("motion") or entry.get("motion"),
                    "frame_number": frame_number,
                    "camera_frame_index": image_index,
                    "gt_identity": identity,
                    "camera_z_m": float(camera_xyz[2]),
                    "distance_band": distance_band_from_depth(float(camera_xyz[2])),
                    "occlusion_category": frame.get("occlusion_category", "unknown"),
                    "localized": identity in matched_ids,
                })
    return gt_rows, warnings


def error_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        float(row.get("errors", {}).get("lidar_center_error"))
        for row in rows
        if row.get("errors", {}).get("lidar_center_error") is not None
    ]
    if not errors:
        return {
            "conditional_error_n": 0,
            "mean_3d_error_m": None,
            "median_3d_error_m": None,
            "p95_3d_error_m": None,
            "error_gt_1m_rate": None,
        }
    arr = np.asarray(errors, dtype=float)
    return {
        "conditional_error_n": int(len(arr)),
        "mean_3d_error_m": float(np.mean(arr)),
        "median_3d_error_m": float(np.median(arr)),
        "p95_3d_error_m": float(np.percentile(arr, 95)),
        "error_gt_1m_rate": float(np.mean(arr > 1.0)),
    }


def model_counts(
    detections: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> dict[str, int]:
    total_gt = sum(int(frame.get("gt3d_count", 0)) for frame in frames)
    localized = sum(len(set(str(x) for x in frame.get("matched_gt_ids", []))) for frame in frames)
    detections_count = len(detections)
    valid = sum(1 for row in detections if row.get("valid"))
    invalid = detections_count - valid
    missed = total_gt - localized
    false_localizations = valid - localized
    return {
        "total_gt_persons": total_gt,
        "uniquely_localised_gt_persons": localized,
        "missed_gt_persons": missed,
        "camera_detections": detections_count,
        "total_valid_fused_outputs": valid,
        "false_localisations": false_localizations,
        "invalid_fusion_outputs": invalid,
    }


def add_rates(counts: dict[str, Any]) -> dict[str, Any]:
    out = dict(counts)
    out["gt_person_localisation_recall"] = safe_div(
        out["uniquely_localised_gt_persons"],
        out["total_gt_persons"],
    )
    out["miss_rate"] = safe_div(out["missed_gt_persons"], out["total_gt_persons"])
    out["one_to_one_localisation_precision"] = safe_div(
        out["uniquely_localised_gt_persons"],
        out["total_valid_fused_outputs"],
    )
    out["false_localisation_rate"] = safe_div(
        out["false_localisations"],
        out["total_valid_fused_outputs"],
    )
    out["invalid_output_rate"] = safe_div(
        out["invalid_fusion_outputs"],
        out["camera_detections"],
    )
    return out


def aggregate_recordings(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "total_gt_persons",
        "uniquely_localised_gt_persons",
        "missed_gt_persons",
        "camera_detections",
        "total_valid_fused_outputs",
        "false_localisations",
        "invalid_fusion_outputs",
    ]
    counts = {key: sum(int(record[key]) for record in records) for key in keys}
    return add_rates(counts)


def bootstrap_cis(
    recording_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, tuple[float | None, float | None]]:
    metrics = [
        "gt_person_localisation_recall",
        "miss_rate",
        "one_to_one_localisation_precision",
        "false_localisation_rate",
        "invalid_output_rate",
    ]
    if not recording_rows:
        return {metric: (None, None) for metric in metrics}
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in metrics}
    records = list(recording_rows)
    for _ in range(samples):
        indices = rng.integers(0, len(records), size=len(records))
        aggregate = aggregate_recordings([records[int(index)] for index in indices])
        for metric in metrics:
            value = aggregate[metric]
            if value is not None and math.isfinite(float(value)):
                values[metric].append(float(value))
    return {
        metric: (
            float(np.percentile(vals, 2.5)) if vals else None,
            float(np.percentile(vals, 97.5)) if vals else None,
        )
        for metric, vals in values.items()
    }


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return dict(groups)


def gt_stratified_rows(
    model_label: str,
    gt_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    out = []
    gt_groups = group_rows(gt_rows, key)
    det_groups = group_rows(detection_rows, key)
    for stratum in sorted(set(gt_groups) | set(det_groups)):
        gt_group = gt_groups.get(stratum, [])
        det_group = det_groups.get(stratum, [])
        total_gt = len(gt_group)
        localized = sum(1 for row in gt_group if row["localized"])
        stats = error_stats([row for row in det_group if row.get("status") == "matched"])
        if total_gt == 0 and stats["conditional_error_n"] == 0:
            continue
        out.append({
            "model": model_label,
            "stratum": stratum,
            "total_gt_persons": total_gt,
            "uniquely_localised_gt_persons": localized,
            "gt_person_localisation_recall": safe_div(localized, total_gt),
            "missed_gt_persons": total_gt - localized,
            "miss_rate": safe_div(total_gt - localized, total_gt),
            **stats,
        })
    return out


def point_density_rows(model_label: str, detection_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for stratum, rows in sorted(group_rows(detection_rows, "point_count_band").items()):
        total = len(rows)
        valid = sum(1 for row in rows if row.get("valid"))
        matched = [row for row in rows if row.get("status") == "matched"]
        ambiguous = sum(1 for row in rows if row.get("ambiguous"))
        stats = error_stats(matched)
        out.append({
            "model": model_label,
            "stratum": stratum,
            "camera_detections": total,
            "total_valid_fused_outputs": valid,
            "invalid_fusion_outputs": total - valid,
            "invalid_output_rate": safe_div(total - valid, total),
            "matched_detection_outputs": len(matched),
            "ambiguous_valid_outputs": ambiguous,
            **stats,
        })
    return out


def recording_rows(
    model_label: str,
    detections: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detections_by_recording = group_rows(detections, "recording")
    frames_by_recording = group_rows(frames, "recording")
    out = []
    for recording in sorted(set(detections_by_recording) | set(frames_by_recording)):
        counts = model_counts(
            detections_by_recording.get(recording, []),
            frames_by_recording.get(recording, []),
        )
        out.append({
            "model": model_label,
            "recording": recording,
            **add_rates(counts),
        })
    return out


def build_overall_row(
    model_label: str,
    run_name: str,
    detections: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    summary: dict[str, Any],
    cis: dict[str, tuple[float | None, float | None]],
) -> dict[str, Any]:
    overall = summary["overall"]
    row = {
        "model": model_label,
        "source_run": run_name,
        **add_rates(model_counts(detections, frames)),
        "conditional_matched_detection_outputs": overall.get("matched_prediction_count"),
        "ambiguous_valid_outputs": overall.get("ambiguous_match_count"),
        "conditional_mean_3d_error_m": overall.get("mean_lidar_center_error"),
        "conditional_median_3d_error_m": overall.get("median_lidar_center_error"),
        "conditional_p95_3d_error_m": overall.get("p95_lidar_center_error"),
        "conditional_error_gt_1m_rate": overall.get("lidar_error_gt_1_0m_rate"),
    }
    for metric, (low, high) in cis.items():
        row[f"{metric}_ci95_low"] = low
        row[f"{metric}_ci95_high"] = high
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    overall_rows: list[dict[str, Any]],
    warnings: list[str],
    args: argparse.Namespace,
) -> None:
    cols = [
        "model",
        "total_gt_persons",
        "uniquely_localised_gt_persons",
        "gt_person_localisation_recall",
        "miss_rate",
        "total_valid_fused_outputs",
        "one_to_one_localisation_precision",
        "false_localisation_rate",
        "invalid_output_rate",
        "conditional_mean_3d_error_m",
        "conditional_median_3d_error_m",
        "conditional_p95_3d_error_m",
        "conditional_error_gt_1m_rate",
    ]
    ci_rows = []
    for row in overall_rows:
        ci_rows.append({
            "model": row["model"],
            "recall_ci95": (
                f"[{fmt(row.get('gt_person_localisation_recall_ci95_low'))}, "
                f"{fmt(row.get('gt_person_localisation_recall_ci95_high'))}]"
            ),
            "precision_ci95": (
                f"[{fmt(row.get('one_to_one_localisation_precision_ci95_low'))}, "
                f"{fmt(row.get('one_to_one_localisation_precision_ci95_high'))}]"
            ),
            "miss_rate_ci95": (
                f"[{fmt(row.get('miss_rate_ci95_low'))}, "
                f"{fmt(row.get('miss_rate_ci95_high'))}]"
            ),
            "false_rate_ci95": (
                f"[{fmt(row.get('false_localisation_rate_ci95_low'))}, "
                f"{fmt(row.get('false_localisation_rate_ci95_high'))}]"
            ),
            "invalid_rate_ci95": (
                f"[{fmt(row.get('invalid_output_rate_ci95_low'))}, "
                f"{fmt(row.get('invalid_output_rate_ci95_high'))}]"
            ),
        })
    ci_cols = [
        "model",
        "recall_ci95",
        "precision_ci95",
        "miss_rate_ci95",
        "false_rate_ci95",
        "invalid_rate_ci95",
    ]
    text = f"""# R1-Q3-13 Existing-Output Early-Fusion Evaluation

This report was generated from saved AGHRI early-fusion outputs. It does not
retrain YOLO11s, rerun detection, alter fusion thresholds, or modify the
original result files.

## Current Matching Algorithm

The frozen evaluator first computes a fused output for each YOLO person
detection. Ground truth is used only after fusion. The primary match is in 2D
image space: the YOLO box is compared with ZED RGB ground-truth boxes using IoU,
with a match threshold of `IoU >= 0.10`. If the top two 2D GT candidates differ
by less than `0.05` IoU, the output is marked ambiguous. If no 2D identity match
is available, the evaluator falls back to nearest 3D LiDAR GT centre matching.
The 3D fallback is marked ambiguous if the second-nearest GT is within `0.50 m`
of the nearest GT.

The original conditional localisation errors are computed only for valid fused
outputs that receive a GT match. Missed GT persons, invalid fusion outputs, and
false/duplicate fused outputs are reported separately here rather than being
assigned artificial large errors.

## End-to-End Definitions

For GT-centric localisation, a ground-truth person can contribute at most once
per frame. If multiple valid detections match the same GT identity in the same
frame, that GT contributes one true localisation and the extra valid fused
outputs count against one-to-one precision as false localisations. Ambiguous
valid outputs are not counted as unique GT localisations.

- GT-person localisation recall = uniquely localised GT persons / total GT persons.
- Miss rate = missed GT persons / total GT persons.
- One-to-one localisation precision = uniquely localised GT persons / valid fused outputs.
- False-localisation rate = false localisations / valid fused outputs.
- Invalid-output rate = invalid fusion outputs / all YOLO person detections.

## Overall Results

{markdown_table(overall_rows, cols)}

## Uncertainty

Confidence intervals use recording-level bootstrap over the six held-out test
recordings. Each bootstrap sample resamples six recordings with replacement and
recomputes metrics from summed numerators and denominators. Bootstrap samples:
`{args.bootstrap_samples}`. Fixed seed: `{args.seed}`.

{markdown_table(ci_rows, ci_cols)}

## Stratification Notes

Distance-stratified GT recall is reconstructed from the AGHRI 3D annotation
centres transformed into the ZED RGB camera frame, using the same `<2 m`,
`2-5 m`, and `>=5 m` bands used by the evaluator. Motion is recording-level.
LiDAR point-density bins are detection-side bins from the saved evaluator
outputs, so they support invalid-output and conditional-error analysis but not a
GT-person denominator for missed people. Occlusion is included only as the
existing ZED RGB event/frame-level category, not as a new per-person occlusion
label.

## Limitations

The saved evaluator performs per-detection matching rather than a global
Hungarian assignment. This script therefore uses the saved per-frame unique
matched GT identities for GT-centric recall and precision accounting, while
preserving the original conditional error distribution for valid matched
detections. Distance/occlusion strata should be read with the denominator notes
above.

## Outputs

- `overall_r1_q3_13.csv`
- `stratified_distance.csv`
- `stratified_motion.csv`
- `stratified_point_density.csv`
- `stratified_occlusion_frame_event.csv`
- `recording_level_counts.csv`
- `bootstrap_ci.json`

## Command

```bash
{sys.executable} tools/analyze_aghri_r1_q3_13_existing_outputs.py \\
  --source-root {args.source_root} \\
  --manifest {args.manifest} \\
  --output-dir {args.output_dir} \\
  --config {args.config} \\
  --bootstrap-samples {args.bootstrap_samples} \\
  --seed {args.seed}
```
"""
    if warnings:
        text += "\n## Validation Warnings\n\n"
        text += "\n".join(f"- {warning}" for warning in warnings)
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute R1-Q3-13 end-to-end metrics from saved AGHRI outputs.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-slop-sec", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.manifest is None:
        candidate = args.source_root / "manifest/test_manifest.csv"
        fallback = REPO_ROOT / "results/aghri_generic_vs_finetuned/manifest/test_manifest.csv"
        args.manifest = candidate if candidate.exists() else fallback

    manifest = load_manifest(args.manifest)
    camera_from_lidar = load_camera_from_lidar(args.config)

    loaded: dict[str, dict[str, Any]] = {}
    any_frames = None
    for model_label, run_name in MODEL_RUNS.items():
        run_dir = args.source_root / "runs" / run_name
        detections = load_json(run_dir / "detections.json")
        frames = load_json(run_dir / "frames.json")
        summary = load_json(run_dir / "summary.json")
        loaded[model_label] = {
            "run_name": run_name,
            "detections": detections,
            "frames": frames,
            "summary": summary,
        }
        if any_frames is None:
            any_frames = frames

    if any_frames is None:
        raise RuntimeError("No model runs loaded")

    frames_by_recording = group_rows(any_frames, "recording")
    gt_rows, warnings = build_gt_instances(
        manifest,
        frames_by_recording,
        camera_from_lidar,
        args.image_slop_sec,
    )
    expected_gt = sum(int(frame.get("gt3d_count", 0)) for frame in any_frames)
    if len(gt_rows) != expected_gt:
        warnings.append(
            f"Reconstructed GT rows ({len(gt_rows)}) differ from saved frame GT "
            f"count ({expected_gt})"
        )

    overall_rows = []
    recording_count_rows = []
    bootstrap_output = {}
    distance_rows = []
    motion_rows = []
    point_density_output_rows = []
    occlusion_rows = []

    for model_label, data in loaded.items():
        frames = data["frames"]
        detections = data["detections"]
        run_name = data["run_name"]

        frame_lookup = {
            (frame["recording"], int(frame["frame_number"])): {
                str(item) for item in frame.get("matched_gt_ids", [])
            }
            for frame in frames
        }
        model_gt_rows = [
            {
                **row,
                "localized": row["gt_identity"] in frame_lookup.get(
                    (row["recording"], int(row["frame_number"])),
                    set(),
                ),
            }
            for row in gt_rows
        ]

        per_recording = recording_rows(model_label, detections, frames)
        recording_count_rows.extend(per_recording)
        cis = bootstrap_cis(per_recording, args.bootstrap_samples, args.seed)
        bootstrap_output[model_label] = {
            metric: {"ci95_low": low, "ci95_high": high}
            for metric, (low, high) in cis.items()
        }
        overall_rows.append(build_overall_row(
            model_label,
            run_name,
            detections,
            frames,
            data["summary"],
            cis,
        ))
        distance_rows.extend(gt_stratified_rows(
            model_label,
            model_gt_rows,
            detections,
            "distance_band",
        ))
        motion_rows.extend(gt_stratified_rows(
            model_label,
            model_gt_rows,
            detections,
            "motion",
        ))
        occlusion_rows.extend(gt_stratified_rows(
            model_label,
            model_gt_rows,
            detections,
            "occlusion_category",
        ))
        point_density_output_rows.extend(point_density_rows(model_label, detections))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall_r1_q3_13.csv", overall_rows)
    write_csv(args.output_dir / "recording_level_counts.csv", recording_count_rows)
    write_csv(args.output_dir / "stratified_distance.csv", distance_rows)
    write_csv(args.output_dir / "stratified_motion.csv", motion_rows)
    write_csv(args.output_dir / "stratified_point_density.csv", point_density_output_rows)
    write_csv(args.output_dir / "stratified_occlusion_frame_event.csv", occlusion_rows)
    write_json(args.output_dir / "bootstrap_ci.json", bootstrap_output)
    write_report(args.output_dir / "R1_Q3_13_existing_output_report.md", overall_rows, warnings, args)

    print(f"Wrote R1-Q3-13 analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
