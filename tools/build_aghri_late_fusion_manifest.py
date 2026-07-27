#!/usr/bin/env python3
"""Build the deterministic AGHRI late-fusion image/LiDAR manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.aghri_manifest import build_late_fusion_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/results/aghri_generic_vs_finetuned/manifest/test_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"),
    )
    parser.add_argument("--max-delta-sec", type=float, default=0.10)
    parser.add_argument("--calibration-source", default="/media/prabuddhi/Backup2/Updated Dataset_PW/calibration.zip")
    args = parser.parse_args()
    summary = build_late_fusion_manifest(
        official_manifest=args.official_manifest,
        output_csv=args.output,
        max_delta_sec=args.max_delta_sec,
        calibration_source=args.calibration_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
