#!/usr/bin/env python3
"""Sweep AGHRI DeepFusionMOT tracking thresholds on a manifest split."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "aghri_deepfusionmot_tracking"
    / "threshold_tuning_val_ft_yolo_ft_pointpillars"
)
DEFAULT_CAMERA_CACHE = DEFAULT_OUTPUT_ROOT / "cache" / "yolo_finetuned" / "detections.csv"
DEFAULT_LIDAR_CACHE = DEFAULT_OUTPUT_ROOT / "cache" / "pointpillars_finetuned" / "detections.csv"
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
    return f"{value:.3f}".replace(".", "p")


def _config_id(min_hits: int, max_age: int, association_3d: float, association_2d: float) -> str:
    return (
        f"mh{min_hits}_ma{max_age}_"
        f"a3d{_fmt_float(association_3d)}_a2d{_fmt_float(association_2d)}"
    )


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
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    ranked = sorted(rows, key=lambda row: (float(row["HOTA"]), float(row["IDF1"]), -int(row["IDSW"])), reverse=True)
    best = ranked[0] if ranked else {}
    lines = [
        "# AGHRI Validation Threshold Sweep: FT YOLO + FT PointPillars",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Output root: `{args.output_root}`",
        f"- Camera cache: `{args.camera_cache}`",
        f"- LiDAR cache: `{args.lidar_cache}`",
        f"- Ego-motion mode: `{args.ego_motion_mode}`",
        f"- Calibration path: `{args.calibration_path}`",
        f"- Configurations evaluated: {len(rows)}",
        "- Detector combination: `P4` / fine-tuned YOLO + fine-tuned PointPillars",
        "- `support_2d_to_3d_threshold` is tied to the swept `association_2d_threshold`.",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Best By HOTA",
                "",
                f"- Config ID: `{best['config_id']}`",
                f"- min_hits: {best['min_hits']}",
                f"- max_age_frames: {best['max_age_frames']}",
                f"- association_3d_threshold: {best['association_3d_threshold']}",
                f"- association_2d_threshold: {best['association_2d_threshold']}",
                f"- HOTA: {float(best['HOTA']):.4f}",
                f"- IDF1: {float(best['IDF1']):.4f}",
                f"- MOTA: {float(best['MOTA']):.4f}",
                f"- IDSW: {int(best['IDSW'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top 10 By HOTA",
            "",
            "| Rank | Config | min_hits | max_age | 3D gate | 2D gate | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Trk IDs | Trk Dets |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked[:10], start=1):
        lines.append(
            "| {rank} | `{config_id}` | {min_hits} | {max_age_frames} | {association_3d_threshold:.3f} | "
            "{association_2d_threshold:.3f} | {HOTA:.4f} | {DetA:.4f} | {AssA:.4f} | "
            "{IDF1:.4f} | {MOTA:.4f} | {IDSW} | {tracker_ids} | {tracker_dets} |".format(
                rank=rank,
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data_manifests" / "aghri_late_fusion_val_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-cache", type=Path, default=DEFAULT_CAMERA_CACHE)
    parser.add_argument("--lidar-cache", type=Path, default=DEFAULT_LIDAR_CACHE)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--ego-motion-mode", default="aghri_odom")
    parser.add_argument("--min-hits", default="1,2,3")
    parser.add_argument("--max-age-frames", default="10,15,25,35")
    parser.add_argument("--association-3d-thresholds", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--association-2d-thresholds", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recordings = args.recording or _recordings_from_manifest(args.manifest)
    min_hits_values = _parse_values(args.min_hits, int)
    max_age_values = _parse_values(args.max_age_frames, int)
    association_3d_values = _parse_values(args.association_3d_thresholds, float)
    association_2d_values = _parse_values(args.association_2d_thresholds, float)
    configs = list(itertools.product(min_hits_values, max_age_values, association_3d_values, association_2d_values))
    if args.max_configs:
        configs = configs[: args.max_configs]

    results: list[dict] = []
    for index, (min_hits, max_age, association_3d, association_2d) in enumerate(configs, start=1):
        config_id = _config_id(min_hits, max_age, association_3d, association_2d)
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
                str(min_hits),
                "--tune-max-age-frames",
                str(max_age),
                "--tune-association-3d-threshold",
                str(association_3d),
                "--tune-association-2d-threshold",
                str(association_2d),
                "--tune-support-2d-to-3d-threshold",
                str(association_2d),
            ]
            _run_command(command, args.dry_run)

        metrics_json = metrics_root / "deepfusionmot_metrics.json"
        if not metrics_json.exists() or args.rerun:
            command = [
                sys.executable,
                "tools/evaluate_aghri_deepfusionmot_metrics.py",
                "--manifest",
                str(args.manifest),
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
                "min_hits": min_hits,
                "max_age_frames": max_age,
                "association_3d_threshold": association_3d,
                "association_2d_threshold": association_2d,
                "support_2d_to_3d_threshold": association_2d,
                "run_dir": str(batch_root),
                "metrics_dir": str(metrics_root),
            }
        )
        results.append(row)
        _write_csv(args.output_root / "threshold_sweep_results.csv", results)
        (args.output_root / "threshold_sweep_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        _write_summary(args.output_root / "threshold_sweep_summary.md", results, args)

    if not args.dry_run:
        ranked = sorted(results, key=lambda row: (float(row["HOTA"]), float(row["IDF1"]), -int(row["IDSW"])), reverse=True)
        _write_csv(args.output_root / "top_10_by_hota.csv", ranked[:10])
        print(f"summary={args.output_root / 'threshold_sweep_summary.md'}")
        if ranked:
            print(f"best_config={ranked[0]['config_id']}")
            print(f"best_HOTA={float(ranked[0]['HOTA']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
