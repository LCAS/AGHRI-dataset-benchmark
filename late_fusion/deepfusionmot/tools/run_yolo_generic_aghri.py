#!/usr/bin/env python3
"""Run generic YOLO11s on rows from an AGHRI late-fusion manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.yolo_runner import run_yolo_images


def read_images(manifest: Path, limit: int = 0):
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw_path = row.get("image_path", "")
            if not raw_path:
                continue
            image_path = Path(raw_path)
            if not image_path.is_file():
                continue
            ts = float(row["zed_rgb_timestamp"]) if row.get("zed_rgb_timestamp") else None
            yield row["sample_id"], ts, image_path
            limit -= 1
            if limit == 0:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/checkpoints/generic/yolo11s.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/aghri_generic_baseline/cache/yolo_generic/detections.csv"))
    parser.add_argument("--output-metadata", type=Path, default=Path("results/aghri_generic_baseline/cache/yolo_generic/metadata.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    meta = run_yolo_images(
        read_images(args.manifest, limit=args.limit),
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        output_metadata=args.output_metadata,
        device=args.device,
    )
    print(
        {
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "frames": meta.get("frames"),
            "detections": meta.get("detections"),
            "device": meta.get("device"),
            "output_csv": str(args.output_csv),
            "output_metadata": str(args.output_metadata),
        }
    )


if __name__ == "__main__":
    main()
