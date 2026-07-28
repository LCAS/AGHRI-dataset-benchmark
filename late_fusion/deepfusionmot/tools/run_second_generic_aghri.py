#!/usr/bin/env python3
"""Run generic KITTI-pretrained SECOND on rows from an AGHRI late-fusion manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.mmdet3d_runner import run_mmdet3d_pointclouds


def read_pointclouds(manifest: Path, limit: int = 0):
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_path = row.get("lidar_point_cloud_path", "")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_file() or path.suffix.lower() not in {".bin", ".pcd"}:
                continue
            ts = float(row["lidar_timestamp"]) if row.get("lidar_timestamp") else None
            yield row["sample_id"], ts, path
            limit -= 1
            if limit == 0:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d/configs/models/second_kitti_pretrained_eval_aghri.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/aghri_generic_baseline/cache/second_generic/detections.csv"))
    parser.add_argument("--output-metadata", type=Path, default=Path("results/aghri_generic_baseline/cache/second_generic/metadata.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--selected-class-id", type=int, default=0)
    parser.add_argument("--selected-class-name", default="Pedestrian")
    parser.add_argument("--class-mapping", default="0:Pedestrian,1:Cyclist,2:Car")
    parser.add_argument("--model-point-cloud-range", default="0,-39.68,-3,69.12,39.68,1")
    args = parser.parse_args()
    class_mapping = {}
    for item in args.class_mapping.split(","):
        if not item.strip():
            continue
        key, value = item.split(":", 1)
        class_mapping[int(key)] = value
    point_range = tuple(float(value) for value in args.model_point_cloud_range.split(","))
    meta = run_mmdet3d_pointclouds(
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
        model_point_cloud_range=point_range,
    )
    print(meta)


if __name__ == "__main__":
    main()
