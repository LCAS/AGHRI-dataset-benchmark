#!/usr/bin/env python3
"""Run generic KITTI-pretrained PointPillars on AGHRI manifest point clouds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.mmdet3d_runner import run_mmdet3d_pointclouds


DEFAULT_CONFIG = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/"
    "mmdetection3d/configs/models/pointpillars_kitti_pretrained_eval_aghri.py"
)
DEFAULT_CHECKPOINT = Path(
    "checkpoints/kitti_pretrained/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
)


def read_pointclouds(manifest: Path, limit: int = 0):
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        count = 0
        for row in csv.DictReader(stream):
            raw_path = row.get("lidar_point_cloud_path", "")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_file() or path.suffix.lower() not in {".bin", ".pcd"}:
                continue
            count += 1
            ts = float(row["lidar_timestamp"]) if row.get("lidar_timestamp") else None
            yield row["sample_id"], ts, path
            if limit and count >= limit:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, default=Path("results/aghri_pointpillars_detector_comparison/cache/pointpillars_generic/detections.csv"))
    parser.add_argument("--output-metadata", type=Path, default=Path("results/aghri_pointpillars_detector_comparison/cache/pointpillars_generic/metadata.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--selected-class-id", type=int, default=0)
    parser.add_argument("--selected-class-name", default="Pedestrian")
    parser.add_argument("--class-mapping-json", default='{"0": "Pedestrian", "1": "Cyclist", "2": "Car"}')
    parser.add_argument("--model-type", default="generic")
    parser.add_argument("--model-point-cloud-range", default="0.0,-39.68,-3.0,69.12,39.68,1.0")
    parser.add_argument("--voxel-size", default="")
    parser.add_argument("--nms-settings-json", default="{}")
    parser.add_argument("--box-origin", default="bottom_center")
    args = parser.parse_args()
    raw_mapping = json.loads(args.class_mapping_json)
    class_mapping = {int(key): str(value) for key, value in raw_mapping.items()}
    point_cloud_range = tuple(float(value) for value in args.model_point_cloud_range.split(","))
    if len(point_cloud_range) != 6:
        raise SystemExit("--model-point-cloud-range must contain six comma-separated floats")
    voxel_size = tuple(float(value) for value in args.voxel_size.split(",")) if args.voxel_size else None
    if voxel_size is not None and len(voxel_size) != 3:
        raise SystemExit("--voxel-size must contain three comma-separated floats")
    metadata = run_mmdet3d_pointclouds(
        read_pointclouds(args.manifest, limit=args.limit),
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        output_metadata=args.output_metadata,
        device=args.device,
        score_threshold=args.score_threshold,
        selected_class_id=args.selected_class_id,
        selected_class_name=args.selected_class_name,
        class_mapping=class_mapping,
        model_point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        nms_settings=json.loads(args.nms_settings_json),
        manifest_path=args.manifest,
        model_type=args.model_type,
        box_frame="front_lidar_link",
        detector_family="pointpillars",
        box_origin=args.box_origin,
    )
    print(metadata)


if __name__ == "__main__":
    main()
