#!/usr/bin/env python3
"""Deterministic offline generic late-fusion evaluator.

This entry point consumes cached YOLO and SECOND CSV detections. If required
inputs are unavailable, it writes the requested result skeleton plus a clear
blocker in the reproducibility manifest.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.checkpoints import build_checkpoint_manifest, sha256_file
from late_fusion_pkg.detection_types import Detection2D, Detection3D
from late_fusion_pkg.fusion_core import fuse_detections


def _read_camera(path: Path) -> dict[str, list[Detection2D]]:
    if not path.exists():
        return {}
    out: dict[str, list[Detection2D]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"}:
                continue
            det = Detection2D(
                bbox_xyxy=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                score=float(row["confidence"]),
                label_id=int(row["class_id"]),
                label_name=row.get("class_name", "person"),
                sample_id=row["sample_id"],
                timestamp=float(row["timestamp"]) if row.get("timestamp") else None,
            )
            out.setdefault(row["sample_id"], []).append(det)
    return out


def _normalise_lidar_z(z: float, height: float, box_origin: str) -> tuple[float, str]:
    if box_origin == "bottom_center":
        return z, "bottom_center"
    if box_origin == "center":
        return z - height / 2.0, "bottom_center"
    raise ValueError(f"Unsupported LiDAR box origin: {box_origin}")


def _read_lidar(path: Path, box_origin: str = "bottom_center") -> dict[str, list[Detection3D]]:
    if not path.exists():
        return {}
    out: dict[str, list[Detection3D]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"}:
                continue
            if str(row.get("selected_pedestrian", "")).lower() not in {"true", "1", "yes"}:
                continue
            height = float(row["height"])
            z, normalised_origin = _normalise_lidar_z(float(row["z"]), height, box_origin)
            det = Detection3D(
                center_xyz=(float(row["x"]), float(row["y"]), z),
                size_lwh=(float(row["length"]), float(row["width"]), height),
                yaw=float(row["yaw"]),
                score=float(row["score"]),
                label_id=int(row["raw_label_id"]),
                label_name=row.get("raw_label_name", "Pedestrian"),
                sample_id=row["sample_id"],
                timestamp=float(row["timestamp"]) if row.get("timestamp") else None,
                box_origin=normalised_origin,
            )
            out.setdefault(row["sample_id"], []).append(det)
    return out


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _runtime_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "p95": float(np.percentile(values, 95)),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--camera-detections", type=Path, default=Path("results/aghri_generic_baseline/cache/yolo_generic/detections.csv"))
    parser.add_argument("--lidar-detections", type=Path, default=Path("results/aghri_generic_baseline/cache/second_generic/detections.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/aghri_generic_baseline"))
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--confidence-policy", default="lidar_score")
    parser.add_argument("--image-width", type=int, default=672)
    parser.add_argument("--image-height", type=int, default=376)
    parser.add_argument("--lidar-box-origin", choices=("bottom_center", "center"), default="bottom_center")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    blockers = []
    if not args.manifest.exists():
        blockers.append(f"manifest missing: {args.manifest}")
    if not args.camera_detections.exists():
        blockers.append(f"camera detections missing: {args.camera_detections}")
    if not args.lidar_detections.exists():
        blockers.append(f"LiDAR detections missing: {args.lidar_detections}")

    # Fallback projection from verified bag metadata: camera x=-y_lidar, y=-z_lidar, z=x_lidar plus translation.
    camera_p = np.array(
        [
            [477.29998779296875, 0.0, 339.74749755859375, 0.0],
            [0.0, 477.4949951171875, 191.94400024414062, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    camera_from_lidar = np.array(
        [
            [0.0, -1.0, 0.0, -0.22],
            [0.0, 0.0, -1.0, 0.295],
            [1.0, 0.0, 0.0, 0.075],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    camera = _read_camera(args.camera_detections)
    lidar = _read_lidar(args.lidar_detections, box_origin=args.lidar_box_origin)
    per_frame = []
    associations = []
    runtimes = []
    if not blockers:
        with args.manifest.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                sample_id = row["sample_id"]
                if not sample_id:
                    continue
                start = time.perf_counter()
                result = fuse_detections(
                    camera.get(sample_id, []),
                    lidar.get(sample_id, []),
                    camera_p,
                    camera_from_lidar,
                    args.image_width,
                    args.image_height,
                    iou_threshold=args.iou_threshold,
                    confidence_policy=args.confidence_policy,
                )
                elapsed = time.perf_counter() - start
                runtimes.append(elapsed)
                per_frame.append(
                    {
                        "sample_id": sample_id,
                        "recording_name": row["recording_name"],
                        "camera_detections": len(camera.get(sample_id, [])),
                        "lidar_detections": len(lidar.get(sample_id, [])),
                        "matches": len(result.matches),
                        "unmatched_camera": len(result.unmatched_camera),
                        "unmatched_lidar": len(result.unmatched_lidar),
                        "final_3d": len(result.final_3d),
                        "late_fusion_time_sec": elapsed,
                    }
                )
                for match in result.matches:
                    associations.append(
                        {
                            "sample_id": sample_id,
                            "camera_index": match.camera_index,
                            "lidar_index": match.lidar_index,
                            "iou": match.iou,
                            "fused_score": match.fused_score,
                        }
                    )

    _write_csv(
        args.output_root / "per_frame_results.csv",
        per_frame,
        ["sample_id", "recording_name", "camera_detections", "lidar_detections", "matches", "unmatched_camera", "unmatched_lidar", "final_3d", "late_fusion_time_sec"],
    )
    _write_csv(
        args.output_root / "association_results.csv",
        associations,
        ["sample_id", "camera_index", "lidar_index", "iou", "fused_score"],
    )
    runtime_summary = _runtime_summary(runtimes)
    _write_csv(args.output_root / "runtime_results.csv", [runtime_summary], list(runtime_summary.keys()))
    overall = {
        "system": "LF-G-ALL3D",
        "frames": len(per_frame),
        "camera_predictions": sum(int(r["camera_detections"]) for r in per_frame),
        "lidar_predictions": sum(int(r["lidar_detections"]) for r in per_frame),
        "matched_pairs": sum(int(r["matches"]) for r in per_frame),
        "final_3d_predictions": sum(int(r["final_3d"]) for r in per_frame),
        "blocked": bool(blockers),
    }
    _write_csv(args.output_root / "overall_results.csv", [overall], list(overall.keys()))
    _write_csv(args.output_root / "per_recording_results.csv", [], ["recording_name"])

    checkpoint_records = [record.__dict__ for record in build_checkpoint_manifest(Path("checkpoints/kitti_pretrained"))]
    (args.output_root / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_records, indent=2), encoding="utf-8")
    config = {
        "iou_threshold": args.iou_threshold,
        "confidence_policy": args.confidence_policy,
        "canonical_output_policy": "matched_plus_unmatched_lidar",
        "image_width": args.image_width,
        "image_height": args.image_height,
        "lidar_box_origin": args.lidar_box_origin,
        "internal_lidar_box_origin": "bottom_center",
    }
    (args.output_root / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    repro = {
        "date": datetime.now(timezone.utc).isoformat(),
        "command_line": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "git": {
            "late_fusion_commit": _git(Path("."), ["rev-parse", "HEAD"]),
            "late_fusion_status": _git(Path("."), ["status", "--short"]),
        },
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest) if args.manifest.exists() else None,
        "threshold_policy": "fixed from existing late-fusion default, not tuned on test",
        "output_policy": "matched plus unmatched LiDAR is canonical final 3D output",
        "confidence_policy": args.confidence_policy,
        "lidar_box_origin": args.lidar_box_origin,
        "internal_lidar_box_origin": "bottom_center",
        "blockers": blockers,
    }
    (args.output_root / "reproducibility_manifest.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    report = [
        "# AGHRI Generic Late-Fusion Baseline Report",
        "",
        f"Blocked: {bool(blockers)}",
        "",
        "## Blockers",
        *(f"- {item}" for item in blockers),
        "",
        "## Overall",
        json.dumps(overall, indent=2),
        "",
        "## Runtime",
        json.dumps(runtime_summary, indent=2),
    ]
    (args.output_root / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "blockers": blockers, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
