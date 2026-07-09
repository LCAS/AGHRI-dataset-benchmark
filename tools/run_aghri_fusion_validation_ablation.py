#!/usr/bin/env python3
"""Run isolated validation-only association ablations for AGHRI fusion.

The script writes under the paper-assets directory by default and never updates
the frozen held-out test benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
DEFAULT_OUTPUT = Path("/home/prabuddhi/Desktop/Early_Fus/results/aghri_fusion_paper_assets/derived_data/validation_ablation")
DEFAULT_MODEL = (
    REPO_ROOT / "checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--frame-limit", type=int, default=0, help="0 means all synchronized validation frames.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_validation_manifest(dataset_root: Path, out_dir: Path) -> Path:
    split_file = dataset_root / "split_lists" / "val.txt"
    if not split_file.exists():
        raise FileNotFoundError(split_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "val_manifest.csv"
    rows = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        matches = sorted(dataset_root.glob(f"dataset_part*/{name}"))
        if not matches:
            rows.append({"recording_path": str(dataset_root / name), "split": "val", "status": "missing"})
        else:
            rows.append({"recording_path": str(matches[0]), "split": "val", "status": "found"})
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["recording_path", "split", "status"])
        writer.writeheader()
        for row in rows:
            if row["status"] == "found":
                writer.writerow({"recording_path": row["recording_path"], "split": row["split"], "status": row["status"]})
    (out_dir / "val_manifest_full_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return manifest


def evaluator_command(args: argparse.Namespace, manifest: Path, run_name: str, trim: float) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_aghri_association.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(args.output_dir),
        "--cache-dir",
        str(args.output_dir / "cache"),
        "--run-name",
        run_name,
        "--frame-limit",
        str(args.frame_limit),
        "--image-slop-sec",
        "0.10",
        "--yolo-model",
        str(args.yolo_model),
        "--yolo-threshold",
        "0.1",
        "--yolo-classes",
        "0",
        "--yolo-imgsz",
        "640",
        "--association-method",
        "legacy_box",
        "--centre-statistic",
        "mean",
        "--box-bottom-trim-ratio",
        f"{trim:.3f}",
        "--visual-limit",
        "0",
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_validation_manifest(args.dataset_root, args.output_dir)
    trims = [0.40, 0.30, 0.20, 0.10, 0.0]  # equivalent retained box heights: 0.60 ... 1.00
    commands = []
    for trim in trims:
        retained = 1.0 - trim
        commands.append({"retained_box_height": retained, "cmd": evaluator_command(args, manifest, f"val_box_retention_{retained:.2f}", trim)})
    (args.output_dir / "ablation_commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    for item in commands:
        log = args.output_dir / f"{item['retained_box_height']:.2f}.log"
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(item["cmd"], check=True, stdout=stream, stderr=subprocess.STDOUT)
    print(f"Validation ablation outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
