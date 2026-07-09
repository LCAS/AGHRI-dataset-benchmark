#!/usr/bin/env python3
"""Canonical AGHRI ZED RGB generic-versus-fine-tuned fusion comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RESULTS = Path("/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2")
DEFAULT_OUTPUT_ROOT = Path("/home/prabuddhi/Desktop/Early_Fus/results/aghri_generic_vs_finetuned")
DEFAULT_MANIFEST = DEFAULT_SOURCE_RESULTS / "manifest/test_manifest.csv"
DEFAULT_DATASET_ROOT = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
DEFAULT_GENERIC = REPO_ROOT / "checkpoints/generic/yolo11s.pt"
DEFAULT_FINETUNED = (
    REPO_ROOT / "checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt"
)
GENERIC_SHA = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
FINETUNED_SHA = "bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_manifest(path: Path, dataset_root: Path) -> dict[str, Any]:
    rows = manifest_rows(path)
    if len(rows) != 6:
        raise SystemExit(f"Expected six official test recordings, found {len(rows)}")
    leakage = [row["recording_name"] for row in rows if row.get("train_val_leakage", "").lower() == "true"]
    if leakage:
        raise SystemExit(f"Manifest includes train/validation leakage: {leakage}")
    unsupported = [row["recording_name"] for row in rows if row.get("supported", "").lower() != "true"]
    if unsupported:
        raise SystemExit(f"Manifest includes unsupported recordings: {unsupported}")
    missing = []
    for row in rows:
        rec = Path(row["recording_path"])
        if not str(rec).startswith(str(dataset_root)):
            raise SystemExit(f"Recording path is outside dataset root: {rec}")
        for rel in [
            "sensor_data/cam_zed_rgb",
            "sensor_data/lidar",
            "annotations/cam_zed_rgb_ann.json",
            "annotations/lidar_ann.json",
        ]:
            if not (rec / rel).exists():
                missing.append(str(rec / rel))
    if missing:
        raise SystemExit("Manifest references missing paths:\n" + "\n".join(missing[:20]))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "recording_count": len(rows),
        "expected_synchronised_frame_count": sum(
            int(row.get("expected_synchronised_frame_count") or 0) for row in rows
        ),
        "train_val_leakage_count": 0,
        "recordings": [row["recording_name"] for row in rows],
    }


def summary_row(label: str, summary: dict[str, Any], checkpoint: Path, sha: str) -> dict[str, Any]:
    overall = summary["overall"]
    frame = summary["frame_overall"]
    return {
        "model": label,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha,
        "detections": overall["total_yolo_person_detections"],
        "matched_valid_results": overall["matched_prediction_count"],
        "valid_fusion_rate": overall["valid_association_rate"],
        "mean_3d_error_m": overall["mean_lidar_center_error"],
        "median_3d_error_m": overall["median_lidar_center_error"],
        "p95_3d_error_m": overall["p95_lidar_center_error"],
        "error_rate_above_1m": overall["lidar_error_gt_1_0m_rate"],
        "detector_runtime_sec": None,
        "fusion_runtime_sec": overall["mean_association_time_sec"],
        "fps": frame["approx_fps"],
        "processed_frames": summary["frame_count"],
        "gt_instances": frame["total_gt3d_instances"],
    }


def per_recording_rows(label: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rec in summary["recordings"]:
        metrics = rec["metrics"]
        frame = rec["frame_metrics"]
        rows.append({
            "model": label,
            "recording": rec["recording"],
            "scene": rec["scene"],
            "motion": rec["motion"],
            "usable_frames": rec["usable_frames"],
            "gt_instances": frame["total_gt3d_instances"],
            "detections": metrics["total_yolo_person_detections"],
            "matched_valid_results": metrics["matched_prediction_count"],
            "valid_fusion_rate": metrics["valid_association_rate"],
            "mean_3d_error_m": metrics["mean_lidar_center_error"],
            "median_3d_error_m": metrics["median_lidar_center_error"],
            "p95_3d_error_m": metrics["p95_lidar_center_error"],
            "error_rate_above_1m": metrics["lidar_error_gt_1_0m_rate"],
            "fps": frame["approx_fps"],
        })
    return rows


def repo_status(path: Path) -> dict[str, str]:
    return {
        "branch": subprocess.check_output(["git", "-C", str(path), "branch", "--show-current"], text=True).strip(),
        "status_short": subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True),
    }


def make_report(output_root: Path, rows: list[dict[str, Any]], comparison: dict[str, Any]) -> None:
    generic, finetuned = rows
    report = f"""# AGHRI ZED RGB Generic Versus Fine-Tuned Fusion Evaluation

This is the canonical two-checkpoint held-out test summary for the fixed camera-LiDAR fusion workflow.

| Model | Detections | Matched/valid | Valid fusion rate | Mean 3D error | Median 3D error | P95 3D error | Error >1 m | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Generic YOLO11s | {generic['detections']} | {generic['matched_valid_results']} | {generic['valid_fusion_rate']:.4f} | {generic['mean_3d_error_m']:.4f} | {generic['median_3d_error_m']:.4f} | {generic['p95_3d_error_m']:.4f} | {generic['error_rate_above_1m']:.4f} | {generic['fps']:.2f} |
| AGHRI-fine-tuned YOLO11s | {finetuned['detections']} | {finetuned['matched_valid_results']} | {finetuned['valid_fusion_rate']:.4f} | {finetuned['mean_3d_error_m']:.4f} | {finetuned['median_3d_error_m']:.4f} | {finetuned['p95_3d_error_m']:.4f} | {finetuned['error_rate_above_1m']:.4f} | {finetuned['fps']:.2f} |

GT-centric comparison: fine-tuned minus generic mean 3D error on jointly valid GT instances is `{comparison.get('mean_finetuned_minus_generic_error_m')}` m.
"""
    (output_root / "final_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--generic-checkpoint", type=Path, default=DEFAULT_GENERIC)
    parser.add_argument("--finetuned-checkpoint", type=Path, default=DEFAULT_FINETUNED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generic_sha = check_hash(args.generic_checkpoint, GENERIC_SHA)
    finetuned_sha = check_hash(args.finetuned_checkpoint, FINETUNED_SHA)
    manifest = validate_manifest(args.test_manifest, args.dataset_root)

    generic_summary_path = args.source_results / "runs/generic_legacy/summary.json"
    finetuned_summary_path = args.source_results / "runs/finetuned_legacy/summary.json"
    comparison_path = args.source_results / "comparisons/primary_checkpoint_comparison.json"
    for path in [generic_summary_path, finetuned_summary_path, comparison_path]:
        if not path.exists():
            raise SystemExit(f"Required verified source result is missing: {path}")

    generic_summary = load_json(generic_summary_path)
    finetuned_summary = load_json(finetuned_summary_path)
    comparison = load_json(comparison_path)
    if generic_summary["frame_count"] != manifest["expected_synchronised_frame_count"]:
        raise SystemExit("Generic source result frame count does not match the manifest")
    if finetuned_summary["frame_count"] != manifest["expected_synchronised_frame_count"]:
        raise SystemExit("Fine-tuned source result frame count does not match the manifest")

    if args.verify_only:
        print(json.dumps({
            "generic_sha256": generic_sha,
            "finetuned_sha256": finetuned_sha,
            "manifest": manifest,
            "source_results_verified": True,
        }, indent=2))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    canonical_manifest = args.output_root / "manifest/test_manifest.csv"
    canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.test_manifest, canonical_manifest)
    rows = [
        summary_row("Generic YOLO11s", generic_summary, args.generic_checkpoint, generic_sha),
        summary_row("AGHRI-fine-tuned YOLO11s", finetuned_summary, args.finetuned_checkpoint, finetuned_sha),
    ]
    per_rec = (
        per_recording_rows("Generic YOLO11s", generic_summary)
        + per_recording_rows("AGHRI-fine-tuned YOLO11s", finetuned_summary)
    )
    write_csv(args.output_root / "overall_results.csv", rows)
    write_json(args.output_root / "overall_results.json", rows)
    write_csv(args.output_root / "per_recording_results.csv", per_rec)
    write_json(args.output_root / "per_recording_results.json", per_rec)
    write_json(args.output_root / "checkpoint_comparison.json", comparison)
    write_json(args.output_root / "reproducibility_manifest.json", {
        "dataset_root": str(args.dataset_root),
        "test_manifest": manifest,
        "canonical_manifest_copy": str(canonical_manifest),
        "generic_checkpoint": rows[0],
        "finetuned_checkpoint": rows[1],
        "source_results": str(args.source_results),
        "device": args.device,
        "reuse_cache": args.reuse_cache,
        "repo_status": repo_status(REPO_ROOT),
    })
    make_report(args.output_root, rows, comparison)
    print(json.dumps({"output_root": str(args.output_root), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
