#!/usr/bin/env python3
"""Write checkpoint manifests and lightweight verification reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.checkpoints import (
    CLASS_MAPPING_EVIDENCE,
    PEDESTRIAN_CLASS_ID,
    build_checkpoint_manifest,
    environment_versions,
    torch_checkpoint_summary,
    verify_mmdet3d_model,
    write_manifest_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/kitti_pretrained"))
    parser.add_argument("--manifest", type=Path, default=Path("checkpoints/checkpoint_manifest.json"))
    parser.add_argument("--sha256sums", type=Path, default=Path("checkpoints/SHA256SUMS.txt"))
    parser.add_argument("--verification-report", type=Path, default=Path("checkpoints/checkpoint_verification_report.json"))
    parser.add_argument("--skip-mmdet3d-load", action="store_true")
    args = parser.parse_args()

    records = build_checkpoint_manifest(args.checkpoint_dir)
    write_manifest_files(records, args.manifest, args.sha256sums)

    report = {
        "environment": environment_versions(),
        "pedestrian_class_mapping": CLASS_MAPPING_EVIDENCE | {"pedestrian_class_id": PEDESTRIAN_CLASS_ID},
        "checkpoints": [],
    }
    for record in records:
        item = {
            "record": record.__dict__,
            "torch_summary": torch_checkpoint_summary(Path(record.absolute_local_path))
            if record.status == "present"
            else None,
        }
        if (
            not args.skip_mmdet3d_load
            and record.status == "present"
            and record.architecture in {"SECOND", "PointPillars"}
        ):
            item["mmdet3d_load"] = verify_mmdet3d_model(
                Path(record.matching_config_file),
                Path(record.absolute_local_path),
            )
        report["checkpoints"].append(item)

    args.verification_report.parent.mkdir(parents=True, exist_ok=True)
    args.verification_report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "verification_report": str(args.verification_report)}, indent=2))


if __name__ == "__main__":
    main()
