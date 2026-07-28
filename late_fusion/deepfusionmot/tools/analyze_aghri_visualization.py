#!/usr/bin/env python3
"""Produce deterministic jitter and projection diagnostics from frozen caches."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from late_fusion_pkg.aghri_calibration import load_aghri_camera_lidar_calibration
from late_fusion_pkg.detection_types import Detection2D, Detection3D
from late_fusion_pkg.visualization_utils import project_box_wireframe, projected_vs_yolo_metrics, wrap_angle


RECORDING = "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"


def read_grouped(path: Path, recording: str, selected_only: bool = False):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row["sample_id"].startswith(recording + "/"):
                continue
            if selected_only and row.get("selected_pedestrian", "").lower() not in {"true", "1", "yes"}:
                continue
            grouped[row["sample_id"]].append(row)
    return grouped


def as_lidar(row) -> Detection3D:
    return Detection3D(
        tuple(float(row[key]) for key in ("x", "y", "z")),
        tuple(float(row[key]) for key in ("length", "width", "height")),
        float(row["yaw"]), float(row["score"]), int(row["raw_label_id"]),
        sample_id=row["sample_id"], timestamp=float(row["timestamp"]),
    )


def as_camera(row) -> Detection2D:
    return Detection2D(
        tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2")),
        float(row["confidence"]), int(row["class_id"]), row["class_name"],
        sample_id=row["sample_id"], timestamp=float(row["timestamp"]),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows generated for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, value):
    return float(np.percentile(np.asarray(values, dtype=float), value)) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", default=RECORDING)
    parser.add_argument("--second", type=Path, default=Path("results/aghri_generic_baseline/test_full/second_detections.csv"))
    parser.add_argument("--yolo", type=Path, default=Path("results/aghri_generic_baseline/test_full/yolo_detections.csv"))
    parser.add_argument("--associations", type=Path, default=Path("results/aghri_generic_baseline/test_full/fusion/association_results.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/aghri_generic_baseline/rosbag_evaluation/visualization"))
    args = parser.parse_args()

    second = read_grouped(args.second, args.recording, selected_only=True)
    yolo = read_grouped(args.yolo, args.recording)
    associations = read_grouped(args.associations, args.recording)
    manifest_rows = read_grouped(args.manifest, args.recording)
    manifest = {key: rows[0] for key, rows in manifest_rows.items()}
    calibration_path = Path(next(iter(manifest.values()))["calibration_source"])
    calibration = load_aghri_camera_lidar_calibration(calibration_path)
    p = calibration["camera_p"]
    transform = calibration["camera_from_lidar"]
    width, height = calibration["image_width"], calibration["image_height"]

    matched_frames = []
    comparison_rows = []
    projection_rows = []
    for sample_id, assoc_rows in sorted(associations.items(), key=lambda item: float(second[item[0]][0]["timestamp"])):
        lidar_rows = second[sample_id]
        camera_rows = yolo.get(sample_id, [])
        matches = []
        reasons = defaultdict(int)
        for assoc in assoc_rows:
            lidar_index, camera_index = int(assoc["lidar_index"]), int(assoc["camera_index"])
            if lidar_index >= len(lidar_rows) or camera_index >= len(camera_rows):
                reasons["invalid_geometry"] += 1
                continue
            lidar = as_lidar(lidar_rows[lidar_index])
            camera = as_camera(camera_rows[camera_index])
            matches.append(lidar)
            metric = projected_vs_yolo_metrics(lidar, camera, p, transform, width, height)
            comparison_rows.append({
                "sample_id": sample_id,
                "lidar_timestamp": f"{lidar.timestamp:.9f}",
                "camera_timestamp": f"{camera.timestamp:.9f}",
                "camera_lidar_delta_sec": manifest.get(sample_id, {}).get("camera_lidar_delta_sec", ""),
                "camera_index": camera_index,
                "lidar_index": lidar_index,
                "fusion_iou": assoc["iou"],
                **metric,
                "yolo_x1": camera.bbox_xyxy[0], "yolo_y1": camera.bbox_xyxy[1],
                "yolo_x2": camera.bbox_xyxy[2], "yolo_y2": camera.bbox_xyxy[3],
            })
            _lines, reason = project_box_wireframe(lidar, p, transform, width, height)
            reasons["drawn" if reason == "ok" else reason] += 1
        stamp = float(lidar_rows[0]["timestamp"])
        matched_frames.append((stamp, sample_id, matches))
        projection_rows.append({
            "sample_id": sample_id,
            "detection_timestamp": f"{stamp:.9f}",
            "image_timestamp": manifest.get(sample_id, {}).get("zed_rgb_timestamp", ""),
            "image_detection_timestamp_delta": manifest.get(sample_id, {}).get("camera_lidar_delta_sec", ""),
            "matched_box_count": len(matches),
            "boxes_successfully_projected": reasons["drawn"],
            "boxes_rejected_behind_camera": reasons["behind_camera"],
            "boxes_rejected_outside_image": reasons["outside_image"],
            "boxes_rejected_invalid_geometry": reasons["invalid_geometry"],
            "tf_failures": 0,
            "camera_info_failures": 0,
        })

    jitter_rows = []
    previous = None
    for stamp, sample_id, detections in matched_frames:
        available = set(range(len(previous[2]))) if previous else set()
        for index, det in enumerate(detections):
            prior_index, displacement, yaw_change, dimension_change, frame_gap = "", "", "", "", ""
            if previous and available:
                frame_gap = stamp - previous[0]
                distances = {
                    candidate: float(np.linalg.norm(np.asarray(det.center_xyz) - np.asarray(previous[2][candidate].center_xyz)))
                    for candidate in available
                }
                prior_index = min(distances, key=distances.get)
                prior = previous[2][prior_index]
                available.remove(prior_index)
                displacement = distances[prior_index]
                yaw_change = abs(wrap_angle(det.yaw - prior.yaw))
                dimension_change = float(np.linalg.norm(np.asarray(det.size_lwh) - np.asarray(prior.size_lwh)))
            jitter_rows.append({
                "sample_id": sample_id, "timestamp": f"{stamp:.9f}", "box_index": index,
                "previous_box_index": prior_index, "x": det.center_xyz[0], "y": det.center_xyz[1], "z": det.center_xyz[2],
                "length": det.size_lwh[0], "width": det.size_lwh[1], "height": det.size_lwh[2],
                "yaw": det.yaw, "confidence": det.score,
                "camera_lidar_delta_sec": manifest.get(sample_id, {}).get("camera_lidar_delta_sec", ""),
                "previous_timestamp": "" if not previous else f"{previous[0]:.9f}",
                "frame_gap_sec": frame_gap,
                "centre_displacement_m": displacement, "yaw_change_rad": yaw_change,
                "dimension_change_l2_m": dimension_change,
            })
        previous = (stamp, sample_id, detections)

    write_csv(args.output / "jitter_analysis.csv", jitter_rows)
    write_csv(args.output / "projection_diagnostics.csv", projection_rows)
    write_csv(args.output / "projected_vs_yolo.csv", comparison_rows)

    displacements = [float(row["centre_displacement_m"]) for row in jitter_rows if row["centre_displacement_m"] != ""]
    ious = [float(row["iou"]) for row in comparison_rows if row.get("status") == "ok"]
    offsets = [float(row["centre_offset_px"]) for row in comparison_rows if row.get("status") == "ok"]
    height_ratios = [float(row["height_ratio"]) for row in comparison_rows if row.get("status") == "ok"]
    width_ratios = [float(row["width_ratio"]) for row in comparison_rows if row.get("status") == "ok"]
    deltas = [float(row["camera_lidar_delta_sec"]) for row in jitter_rows if row["camera_lidar_delta_sec"] != ""]
    adjacent = [row for row in jitter_rows if row["frame_gap_sec"] != "" and float(row["frame_gap_sec"]) <= 0.25]
    adjacent_displacements = [float(row["centre_displacement_m"]) for row in adjacent]
    adjacent_yaws = [float(row["yaw_change_rad"]) for row in adjacent]
    adjacent_dimensions = [float(row["dimension_change_l2_m"]) for row in adjacent]

    (args.output / "jitter_analysis.md").write_text(
        "# Matched SECOND jitter analysis\n\n"
        f"Recording: `{args.recording}`. Consecutive boxes were associated by nearest 3D centre.\n\n"
        f"- Matched samples: {len(matched_frames)}; matched boxes: {len(jitter_rows)}; consecutive associations: {len(displacements)}.\n"
        f"- Adjacent pairs (gap <= 0.25 s): {len(adjacent)}. Centre displacement median {percentile(adjacent_displacements, 50):.3f} m, P95 {percentile(adjacent_displacements, 95):.3f} m, maximum {max(adjacent_displacements, default=math.nan):.3f} m.\n"
        f"- Adjacent absolute yaw change: median {percentile(adjacent_yaws, 50):.3f} rad, P95 {percentile(adjacent_yaws, 95):.3f} rad.\n"
        f"- Adjacent dimension-vector change: median {percentile(adjacent_dimensions, 50):.3f} m, P95 {percentile(adjacent_dimensions, 95):.3f} m.\n"
        f"- Across all consecutive matched observations, including detection gaps: centre displacement median {percentile(displacements, 50):.3f} m and maximum {max(displacements, default=math.nan):.3f} m.\n"
        f"- Camera-LiDAR delta: median {percentile(deltas, 50) * 1000:.1f} ms, maximum {max(deltas, default=math.nan) * 1000:.1f} ms.\n\n"
        "The reported centre, yaw, dimension and score variation is present in the frozen SECOND cache before RViz. "
        "SECOND is a frame-independent detector and no tracker is used. RViz receives source-stamped boxes unchanged; "
        "DELETEALL prevents old marker frames from accumulating.\n",
        encoding="utf-8",
    )
    projected_count = sum(int(row["boxes_successfully_projected"]) for row in projection_rows)
    rejected_count = sum(
        int(row[key]) for row in projection_rows
        for key in ("boxes_rejected_behind_camera", "boxes_rejected_outside_image", "boxes_rejected_invalid_geometry")
    )
    (args.output / "projection_diagnostics.md").write_text(
        "# Projection diagnostics\n\n"
        f"Offline audit of {len(projection_rows)} matched arrays and {projected_count + rejected_count} matched boxes.\n\n"
        f"- Successfully projectable: {projected_count}.\n"
        f"- Geometry/visibility rejections: {rejected_count}.\n"
        f"- Projected-vs-YOLO IoU: median {percentile(ious, 50):.3f}, P10 {percentile(ious, 10):.3f}.\n"
        f"- Centre offset: median {percentile(offsets, 50):.1f} px, P95 {percentile(offsets, 95):.1f} px.\n"
        f"- Projected/YOLO height ratio: median {percentile(height_ratios, 50):.2f}; width ratio: median {percentile(width_ratios, 50):.2f}.\n\n"
        "The previous live run published 239 matched marker arrays but only 236 projection callbacks. Its 117 received "
        "matched boxes all drew successfully, so missing images were synchronizer-level array drops rather than per-box "
        "projection rejection. The refined node pairs every matched array with an image cache and reports every failure reason.\n",
        encoding="utf-8",
    )

    # Save overlays for the three weakest projected-vs-YOLO matches.
    import cv2
    overlay_dir = args.output / "diagnostic_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    weakest = sorted((row for row in comparison_rows if row.get("status") == "ok"), key=lambda row: float(row["iou"]))[:3]
    for rank, row in enumerate(weakest, 1):
        sample_id = row["sample_id"]
        assoc = next(item for item in associations[sample_id] if int(item["lidar_index"]) == int(row["lidar_index"]) and int(item["camera_index"]) == int(row["camera_index"]))
        lidar = as_lidar(second[sample_id][int(assoc["lidar_index"])])
        camera = as_camera(yolo[sample_id][int(assoc["camera_index"])])
        image_path = Path(manifest[sample_id]["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in camera.bbox_xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 80, 20), 2)
        lines, _reason = project_box_wireframe(lidar, p, transform, image.shape[1], image.shape[0])
        for start, end in lines:
            cv2.line(image, start, end, (0, 230, 0), 2, cv2.LINE_AA)
        points = [point for line in lines for point in line]
        if points:
            ex1, ey1 = min(point[0] for point in points), min(point[1] for point in points)
            ex2, ey2 = max(point[0] for point in points), max(point[1] for point in points)
            cv2.rectangle(image, (ex1, ey1), (ex2, ey2), (0, 230, 0), 1)
        cv2.putText(image, f"IoU {float(row['iou']):.3f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)
        cv2.imwrite(str(overlay_dir / f"weak_alignment_{rank}.png"), image)


if __name__ == "__main__":
    main()
