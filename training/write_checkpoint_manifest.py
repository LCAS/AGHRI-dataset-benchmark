#!/usr/bin/env python3
"""Write a provenance manifest for an AGHRI YOLO11s checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(path: Path) -> str:
    """Hash a file with SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_results(run_dir: Path) -> Dict[str, Any]:
    """Read the best available metrics from Ultralytics results.csv."""
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return {}
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def clean_key(key: str) -> str:
        return key.strip().replace(" ", "_").replace("/", "_")

    best_row = max(
        rows,
        key=lambda row: float(row.get("metrics/mAP50-95(B)", row.get("metrics_mAP50-95(B)", 0)) or 0),
    )
    final_row = rows[-1]
    return {
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "training_duration_sec": None,
        "validation_precision": float(best_row.get("metrics/precision(B)", 0) or 0),
        "validation_recall": float(best_row.get("metrics/recall(B)", 0) or 0),
        "validation_ap50": float(best_row.get("metrics/mAP50(B)", 0) or 0),
        "validation_ap50_95": float(best_row.get("metrics/mAP50-95(B)", 0) or 0),
        "final_epoch": int(float(final_row.get("epoch", 0))),
        "raw_best_row": {clean_key(k): v for k, v in best_row.items()},
    }


def read_yaml(path: Path) -> Dict[str, Any]:
    """Read YAML if available."""
    if not path.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def load_model_metadata(best_checkpoint: Path) -> Dict[str, Any]:
    """Verify Ultralytics can load the checkpoint and record model metadata."""
    from ultralytics import YOLO

    model = YOLO(str(best_checkpoint))
    names = getattr(model, "names", None)
    return {
        "ultralytics_model_class": type(model.model).__name__,
        "names": names,
        "person_class_is_zero": bool(names and names.get(0) == "person"),
    }


def package_versions() -> Dict[str, Any]:
    """Record package and accelerator metadata."""
    versions: Dict[str, Any] = {}
    try:
        import torch

        versions.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None,
            }
        )
    except Exception as error:
        versions["torch_metadata_error"] = str(error)
    try:
        import ultralytics

        versions["ultralytics_version"] = ultralytics.__version__
    except Exception as error:
        versions["ultralytics_metadata_error"] = str(error)
    return versions


def manifest_for_checkpoint(
    best_checkpoint: Path,
    last_checkpoint: Optional[Path],
    generic_checkpoint: Path,
    dataset_yaml: Path,
    train_manifest: Path,
    val_manifest: Path,
    output_path: Path,
    notes: str = "",
) -> Dict[str, Any]:
    """Build and write a checkpoint provenance manifest."""
    run_dir = best_checkpoint.parent.parent
    manifest = {
        "created_unix_time": time.time(),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "best_checkpoint_size_bytes": best_checkpoint.stat().st_size,
        "last_checkpoint": str(last_checkpoint.resolve()) if last_checkpoint else None,
        "last_checkpoint_sha256": sha256_file(last_checkpoint)
        if last_checkpoint and last_checkpoint.is_file()
        else None,
        "generic_initialization_checkpoint": str(generic_checkpoint.resolve()),
        "generic_initialization_sha256": sha256_file(generic_checkpoint),
        "checkpoint_differs_from_generic": sha256_file(best_checkpoint)
        != sha256_file(generic_checkpoint),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(train_manifest),
        "val_manifest": str(val_manifest.resolve()),
        "val_manifest_sha256": sha256_file(val_manifest),
        "training_args": read_yaml(run_dir / "args.yaml"),
        "training_metrics": read_training_results(run_dir),
        "model_metadata": load_model_metadata(best_checkpoint),
        "environment": package_versions(),
        "notes": notes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--last-checkpoint", default=None)
    parser.add_argument(
        "--generic-checkpoint",
        default=(
            "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/"
            "benchmarks/ultralytics/yolo11s.pt"
        ),
    )
    parser.add_argument(
        "--dataset-yaml",
        default="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/data.yaml",
    )
    parser.add_argument(
        "--train-manifest",
        default="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/manifests/train.csv",
    )
    parser.add_argument(
        "--val-manifest",
        default="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/manifests/val.csv",
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/prabuddhi/Desktop/Early_Fus/checkpoints/"
            "aghri_yolo11s_finetune/checkpoint_manifest.json"
        ),
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    manifest = manifest_for_checkpoint(
        best_checkpoint=Path(args.best_checkpoint).resolve(),
        last_checkpoint=Path(args.last_checkpoint).resolve()
        if args.last_checkpoint
        else None,
        generic_checkpoint=Path(args.generic_checkpoint).resolve(),
        dataset_yaml=Path(args.dataset_yaml).resolve(),
        train_manifest=Path(args.train_manifest).resolve(),
        val_manifest=Path(args.val_manifest).resolve(),
        output_path=Path(args.output).resolve(),
        notes=args.notes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
