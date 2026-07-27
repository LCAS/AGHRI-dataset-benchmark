#!/usr/bin/env python3
"""Sweep AGHRI detector-cleanup tracking policies for FT YOLO + FT PointPillars."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = (
    REPO_ROOT
    / "results"
    / "aghri_deepfusionmot_tracking"
    / "threshold_tuning_val_ft_yolo_ft_pointpillars"
    / "cache"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "aghri_deepfusionmot_tracking"
    / "policy_sweep_val_ft_yolo_ft_pointpillars"
)
DEFAULT_CAMERA_CACHE = DEFAULT_CACHE_ROOT / "yolo_finetuned" / "detections.csv"
DEFAULT_LIDAR_CACHE = DEFAULT_CACHE_ROOT / "pointpillars_finetuned" / "detections.csv"
DEFAULT_CALIBRATION = Path("/media/prabuddhi/Backup2/Updated Dataset_PW/calibration")


def _parse_values(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def _recordings_from_manifest(manifest: Path) -> list[str]:
    recordings: list[str] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            recording = str(row.get("recording_name") or "")
            if recording and recording not in seen:
                seen.add(recording)
                recordings.append(recording)
    return recordings


def _fmt_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _config_id(lidar_score: float, nms: float, initiation: float) -> str:
    return f"lidar{_fmt_float(lidar_score)}_nms{_fmt_float(nms)}_init{_fmt_float(initiation)}"


def _run_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _read_combined_p4(metrics_json: Path) -> dict:
    rows = json.loads(metrics_json.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("combination_id") == "P4":
            return row
    raise RuntimeError(f"No P4 row found in {metrics_json}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else 0.0


def _int(row: dict, field: str) -> int:
    value = row.get(field, "")
    return int(float(value)) if value not in {"", None} else 0


def _write_summary(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    ranked = sorted(rows, key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1"), -_int(row, "IDSW")), reverse=True)
    best = ranked[0] if ranked else {}
    lines = [
        "# AGHRI Validation Detector-Cleanup Policy Sweep",
        "",
        "- Detector combination: `P4` / FT YOLO + FT PointPillars.",
        f"- Manifest: `{args.manifest}`",
        f"- Output root: `{args.output_root}`",
        "- GT annotations are not modified, corrected, skipped, or filtered.",
        "- Fixed tracker config: min_hits=3, max_age_frames=10, 3D gate=0.030, 2D/support gate=0.400.",
        f"- Camera score threshold fixed at {args.camera_score:.2f}.",
        f"- Configurations evaluated: {len(rows)}",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best By HOTA",
                "",
                f"- Config ID: `{best['config_id']}`",
                f"- LiDAR score threshold: {float(best['min_lidar_score']):.2f}",
                f"- LiDAR NMS IoU threshold: {float(best['lidar_nms_iou_threshold']):.2f}",
                f"- LiDAR-only initiation score threshold: {float(best['lidar_only_initiation_score_threshold']):.2f}",
                f"- HOTA: {float(best['HOTA']):.4f}",
                f"- DetA: {float(best['DetA']):.4f}",
                f"- AssA: {float(best['AssA']):.4f}",
                f"- IDF1: {float(best['IDF1']):.4f}",
                f"- MOTA: {float(best['MOTA']):.4f}",
                f"- IDSW: {int(best['IDSW'])}",
                f"- Tracker detections: {int(best['tracker_dets'])}",
                f"- Tracker IDs: {int(best['tracker_ids'])}",
                f"- CLEAR FP/FN: {int(best['CLR_FP'])}/{int(best['CLR_FN'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top 12 By HOTA",
            "",
            "| Rank | Config | LiDAR Score | NMS IoU | LiDAR-Only Init | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Trk IDs | Trk Dets |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked[:12], start=1):
        lines.append(
            "| {rank} | `{config_id}` | {min_lidar_score:.2f} | {lidar_nms_iou_threshold:.2f} | "
            "{lidar_only_initiation_score_threshold:.2f} | {HOTA:.4f} | {DetA:.4f} | {AssA:.4f} | "
            "{IDF1:.4f} | {MOTA:.4f} | {IDSW} | {CLR_FP} | {CLR_FN} | {tracker_ids} | {tracker_dets} |".format(
                rank=rank,
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data_manifests" / "aghri_late_fusion_val_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-cache", type=Path, default=DEFAULT_CAMERA_CACHE)
    parser.add_argument("--lidar-cache", type=Path, default=DEFAULT_LIDAR_CACHE)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--ego-motion-mode", default="aghri_odom")
    parser.add_argument("--camera-score", type=float, default=0.10)
    parser.add_argument("--lidar-score-thresholds", default="0.10,0.20")
    parser.add_argument("--lidar-nms-thresholds", default="0.00,0.05,0.10,0.20,0.30")
    parser.add_argument("--lidar-only-initiation-thresholds", default="0.00,0.15,0.20,0.30")
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recordings = args.recording or _recordings_from_manifest(args.manifest)
    lidar_scores = _parse_values(args.lidar_score_thresholds, float)
    nms_values = _parse_values(args.lidar_nms_thresholds, float)
    initiation_values = _parse_values(args.lidar_only_initiation_thresholds, float)
    configs = list(itertools.product(lidar_scores, nms_values, initiation_values))
    if args.max_configs:
        configs = configs[: args.max_configs]

    results: list[dict] = []
    for index, (lidar_score, nms_threshold, initiation_threshold) in enumerate(configs, start=1):
        config_id = _config_id(lidar_score, nms_threshold, initiation_threshold)
        config_root = args.output_root / "runs" / config_id
        batch_root = config_root / "batch"
        p4_root = batch_root / "P4"
        metrics_root = config_root / "metrics"
        print(f"[{index}/{len(configs)}] {config_id}")

        for recording in recordings:
            summary = p4_root / recording / "tracking_summary.json"
            if summary.exists() and not args.rerun:
                continue
            command = [
                sys.executable,
                "tools/run_aghri_deepfusionmot_tracking.py",
                "--manifest",
                str(args.manifest),
                "--recording",
                recording,
                "--camera-cache",
                str(args.camera_cache),
                "--lidar-cache",
                str(args.lidar_cache),
                "--output-dir",
                str(p4_root),
                "--combination-id",
                "P4",
                "--camera-model",
                "finetuned",
                "--lidar-detector",
                "pointpillars",
                "--lidar-model",
                "finetuned",
                "--limit-frames",
                "0",
                "--visual-frames",
                "0",
                "--ego-motion-mode",
                args.ego_motion_mode,
                "--calibration-path",
                str(args.calibration_path),
                "--deepfusionmot-strict",
                "--tune-min-hits",
                "3",
                "--tune-max-age-frames",
                "10",
                "--tune-association-3d-threshold",
                "0.03",
                "--tune-association-2d-threshold",
                "0.40",
                "--tune-support-2d-to-3d-threshold",
                "0.40",
                "--tune-output-mode",
                "deepfusionmot",
                "--min-camera-score",
                str(args.camera_score),
                "--min-lidar-score",
                str(lidar_score),
                "--lidar-nms-iou-threshold",
                str(nms_threshold),
                "--lidar-only-initiation-score-threshold",
                str(initiation_threshold),
            ]
            _run_command(command, args.dry_run)

        metrics_json = metrics_root / "deepfusionmot_metrics.json"
        if not metrics_json.exists() or args.rerun:
            command = [
                sys.executable,
                "tools/evaluate_aghri_deepfusionmot_metrics.py",
                "--manifest",
                str(args.manifest),
                "--dataset-root",
                str(args.dataset_root),
                "--batch-root",
                str(batch_root),
                "--output-dir",
                str(metrics_root),
            ]
            _run_command(command, args.dry_run)
        if args.dry_run:
            continue

        row = _read_combined_p4(metrics_json)
        row.update(
            {
                "config_id": config_id,
                "min_camera_score": args.camera_score,
                "min_lidar_score": lidar_score,
                "lidar_nms_iou_threshold": nms_threshold,
                "lidar_only_initiation_score_threshold": initiation_threshold,
                "run_dir": str(batch_root),
                "metrics_dir": str(metrics_root),
            }
        )
        results.append(row)
        _write_csv(args.output_root / "policy_sweep_results.csv", results)
        (args.output_root / "policy_sweep_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        _write_summary(args.output_root / "policy_sweep_summary.md", results, args)

    if not args.dry_run:
        ranked = sorted(results, key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1"), -_int(row, "IDSW")), reverse=True)
        _write_csv(args.output_root / "top_12_policy_sweep_by_hota.csv", ranked[:12])
        print(f"summary={args.output_root / 'policy_sweep_summary.md'}")
        if ranked:
            print(f"best_config={ranked[0]['config_id']}")
            print(f"best_HOTA={float(ranked[0]['HOTA']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
