#!/usr/bin/env python3
"""Run focused AGHRI tracking diagnostics for FT YOLO + FT PointPillars."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD_ROOT = (
    REPO_ROOT
    / "results"
    / "aghri_deepfusionmot_tracking"
    / "threshold_tuning_val_ft_yolo_ft_pointpillars"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "aghri_deepfusionmot_tracking"
    / "diagnostics_val_ft_yolo_ft_pointpillars"
)
DEFAULT_CAMERA_CACHE = DEFAULT_THRESHOLD_ROOT / "cache" / "yolo_finetuned" / "detections.csv"
DEFAULT_LIDAR_CACHE = DEFAULT_THRESHOLD_ROOT / "cache" / "pointpillars_finetuned" / "detections.csv"
DEFAULT_CALIBRATION = Path("/media/prabuddhi/Backup2/Updated Dataset_PW/calibration")
DEFAULT_BEST_CONFIG = {
    "config_id": "mh3_ma10_a3d0p030_a2d0p400",
    "min_hits": 3,
    "max_age_frames": 10,
    "association_3d_threshold": 0.03,
    "association_2d_threshold": 0.40,
    "support_2d_to_3d_threshold": 0.40,
}


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


def _sweep_config_id(camera_score: float, lidar_score: float, output_mode: str) -> str:
    return f"cam{_fmt_float(camera_score)}_lidar{_fmt_float(lidar_score)}_{output_mode}"


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
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict, field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else 0.0


def _int(row: dict, field: str) -> int:
    value = row.get(field, "")
    return int(float(value)) if value not in {"", None} else 0


def _load_best_threshold_config(threshold_root: Path) -> dict:
    path = threshold_root / "threshold_sweep_results.csv"
    if not path.exists():
        return dict(DEFAULT_BEST_CONFIG)
    rows = _read_csv_rows(path)
    if not rows:
        return dict(DEFAULT_BEST_CONFIG)
    best = max(rows, key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1"), -_int(row, "IDSW")))
    return {
        "config_id": best["config_id"],
        "min_hits": _int(best, "min_hits"),
        "max_age_frames": _int(best, "max_age_frames"),
        "association_3d_threshold": _float(best, "association_3d_threshold"),
        "association_2d_threshold": _float(best, "association_2d_threshold"),
        "support_2d_to_3d_threshold": _float(best, "support_2d_to_3d_threshold"),
    }


def _diagnose_sequence_rows(source: Path) -> list[dict]:
    rows = _read_csv_rows(source) if source.exists() else []
    diagnostics = []
    for row in rows:
        gt_dets = _int(row, "gt_dets")
        tracker_dets = _int(row, "tracker_dets")
        clr_fp = _int(row, "CLR_FP")
        clr_fn = _int(row, "CLR_FN")
        idsw = _int(row, "IDSW")
        diagnosis = "mixed"
        if gt_dets == 0 and tracker_dets:
            diagnosis = "pure_false_positive_sequence"
        elif clr_fn > clr_fp and clr_fn > idsw:
            diagnosis = "missed_detections_dominant"
        elif clr_fp > clr_fn and clr_fp > idsw:
            diagnosis = "false_positives_dominant"
        elif idsw:
            diagnosis = "association_errors_visible"
        diagnostics.append(
            {
                "recording": row["recording"],
                "frames_evaluated": _int(row, "frames_evaluated"),
                "gt_dets": gt_dets,
                "tracker_dets": tracker_dets,
                "gt_ids": _int(row, "gt_ids"),
                "tracker_ids": _int(row, "tracker_ids"),
                "HOTA": f"{_float(row, 'HOTA'):.6f}",
                "DetA": f"{_float(row, 'DetA'):.6f}",
                "AssA": f"{_float(row, 'AssA'):.6f}",
                "MOTA": f"{_float(row, 'MOTA'):.6f}",
                "IDF1": f"{_float(row, 'IDF1'):.6f}",
                "IDSW": idsw,
                "CLR_TP": _int(row, "CLR_TP"),
                "CLR_FP": clr_fp,
                "CLR_FN": clr_fn,
                "diagnosis": diagnosis,
            }
        )
    diagnostics.sort(key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1")))
    return diagnostics


def _copy_sequence_breakdown(threshold_root: Path, output_root: Path, best: dict) -> list[dict]:
    source = threshold_root / "runs" / str(best["config_id"]) / "metrics" / "deepfusionmot_metrics_per_recording.csv"
    diagnostics = _diagnose_sequence_rows(source)
    _write_csv(output_root / "per_sequence_metric_breakdown.csv", diagnostics)
    _write_sequence_md(
        output_root / "per_sequence_metric_breakdown.md",
        diagnostics,
        "AGHRI Validation Per-Sequence Tracking Diagnostics",
        [
            "- Detector combination: `P4` / FT YOLO + FT PointPillars.",
            f"- Source threshold sweep: `{threshold_root}`",
            f"- Tracker config: `{best['config_id']}`",
            f"- min_hits={best['min_hits']}, max_age_frames={best['max_age_frames']}, "
            f"3D gate={best['association_3d_threshold']:.3f}, 2D/support gate={best['association_2d_threshold']:.3f}.",
            "- Rows are sorted from lowest HOTA to highest HOTA.",
        ],
    )
    return diagnostics


def _write_sequence_md(path: Path, rows: list[dict], title: str, bullets: list[str]) -> None:
    lines = [
        f"# {title}",
        "",
        *bullets,
        "",
        "| Recording | Frames | GT | Trk | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | Diagnosis |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {recording} | {frames_evaluated} | {gt_dets} | {tracker_dets} | {HOTA:.4f} | "
            "{DetA:.4f} | {AssA:.4f} | {IDF1:.4f} | {MOTA:.4f} | {IDSW} | {CLR_FP} | {CLR_FN} | {diagnosis} |".format(
                **{
                    **row,
                    "HOTA": _float(row, "HOTA"),
                    "DetA": _float(row, "DetA"),
                    "AssA": _float(row, "AssA"),
                    "IDF1": _float(row, "IDF1"),
                    "MOTA": _float(row, "MOTA"),
                }
            )
        )
    if rows:
        worst = rows[0]
        total_fp = sum(_int(row, "CLR_FP") for row in rows)
        total_fn = sum(_int(row, "CLR_FN") for row in rows)
        total_idsw = sum(_int(row, "IDSW") for row in rows)
        lines.extend(
            [
                "",
                "## Reading",
                "",
                f"- Lowest-HOTA sequence: `{worst['recording']}` with HOTA {_float(worst, 'HOTA'):.4f} and IDF1 {_float(worst, 'IDF1'):.4f}.",
                f"- Total CLEAR false positives across validation sequences: {total_fp}.",
                f"- Total CLEAR false negatives across validation sequences: {total_fn}.",
                f"- Total ID switches across validation sequences: {total_idsw}.",
                "- This separates detector-count problems from association problems before expanding back to all eight detector combinations.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_best_sweep_sequence_breakdown(output_root: Path, best_row: dict) -> None:
    source = Path(str(best_row["metrics_dir"])) / "deepfusionmot_metrics_per_recording.csv"
    rows = _diagnose_sequence_rows(source)
    _write_csv(output_root / "best_detector_output_per_sequence_breakdown.csv", rows)
    _write_sequence_md(
        output_root / "best_detector_output_per_sequence_breakdown.md",
        rows,
        "AGHRI Best Confidence/Output Per-Sequence Diagnostics",
        [
            "- Detector combination: `P4` / FT YOLO + FT PointPillars.",
            f"- Source sweep config: `{best_row['config_id']}`",
            f"- Camera score threshold: {float(best_row['min_camera_score']):.2f}",
            f"- LiDAR score threshold: {float(best_row['min_lidar_score']):.2f}",
            f"- Output mode: `{best_row['output_mode']}`",
            f"- Combined HOTA={float(best_row['HOTA']):.4f}, IDF1={float(best_row['IDF1']):.4f}, "
            f"MOTA={float(best_row['MOTA']):.4f}.",
            "- Rows are sorted from lowest HOTA to highest HOTA.",
        ],
    )


def _score_stats(cache_path: Path, score_field: str, thresholds: list[float]) -> dict:
    values = []
    per_recording: dict[str, int] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("empty_frame") == "True":
                continue
            try:
                score = float(row[score_field])
            except (KeyError, TypeError, ValueError):
                continue
            values.append(score)
            recording = str(row.get("sample_id") or "").split("/", 1)[0]
            if recording:
                per_recording[recording] = per_recording.get(recording, 0) + 1
    values.sort()
    counts = {f">={threshold:.2f}": sum(1 for value in values if value >= threshold) for threshold in thresholds}
    return {
        "count": len(values),
        "min": values[0] if values else 0.0,
        "median": values[len(values) // 2] if values else 0.0,
        "max": values[-1] if values else 0.0,
        "counts_by_threshold": counts,
        "detections_by_recording": per_recording,
    }


def _write_sweep_summary(path: Path, rows: list[dict], args: argparse.Namespace, best: dict, score_stats: dict) -> None:
    ranked = sorted(rows, key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1"), -_int(row, "IDSW")), reverse=True)
    best_row = ranked[0] if ranked else {}
    lines = [
        "# AGHRI Validation Confidence/Output-Mode Diagnostics",
        "",
        "- Detector combination: `P4` / FT YOLO + FT PointPillars.",
        f"- Manifest: `{args.manifest}`",
        f"- Output root: `{args.output_root}`",
        f"- Camera cache: `{args.camera_cache}`",
        f"- LiDAR cache: `{args.lidar_cache}`",
        f"- Fixed tracker config: `{best['config_id']}`",
        f"- min_hits={best['min_hits']}, max_age_frames={best['max_age_frames']}, "
        f"3D gate={best['association_3d_threshold']:.3f}, 2D/support gate={best['association_2d_threshold']:.3f}.",
        f"- Configurations evaluated: {len(rows)}",
        "",
        "## Cache Score Coverage",
        "",
        f"- YOLO detections: {score_stats['camera']['count']} "
        f"(min={score_stats['camera']['min']:.3f}, median={score_stats['camera']['median']:.3f}, max={score_stats['camera']['max']:.3f}).",
        f"- PointPillars detections: {score_stats['lidar']['count']} "
        f"(min={score_stats['lidar']['min']:.3f}, median={score_stats['lidar']['median']:.3f}, max={score_stats['lidar']['max']:.3f}).",
        "",
    ]
    if best_row:
        lines.extend(
            [
                "## Best By HOTA",
                "",
                f"- Config ID: `{best_row['config_id']}`",
                f"- Camera score threshold: {float(best_row['min_camera_score']):.2f}",
                f"- LiDAR score threshold: {float(best_row['min_lidar_score']):.2f}",
                f"- Output mode: `{best_row['output_mode']}`",
                f"- HOTA: {float(best_row['HOTA']):.4f}",
                f"- DetA: {float(best_row['DetA']):.4f}",
                f"- AssA: {float(best_row['AssA']):.4f}",
                f"- IDF1: {float(best_row['IDF1']):.4f}",
                f"- MOTA: {float(best_row['MOTA']):.4f}",
                f"- IDSW: {int(best_row['IDSW'])}",
                f"- Tracker detections: {int(best_row['tracker_dets'])}",
                f"- Tracker IDs: {int(best_row['tracker_ids'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top 12 By HOTA",
            "",
            "| Rank | Config | Cam Score | LiDAR Score | Output | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Trk IDs | Trk Dets |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(ranked[:12], start=1):
        lines.append(
            "| {rank} | `{config_id}` | {min_camera_score:.2f} | {min_lidar_score:.2f} | `{output_mode}` | "
            "{HOTA:.4f} | {DetA:.4f} | {AssA:.4f} | {IDF1:.4f} | {MOTA:.4f} | {IDSW} | {tracker_ids} | {tracker_dets} |".format(
                rank=rank,
                **row,
            )
        )
    if ranked:
        deep = [row for row in ranked if row["output_mode"] == "deepfusionmot"]
        broad = [row for row in ranked if row["output_mode"] == "all"]
        lines.extend(["", "## Output-Mode Check", ""])
        if deep:
            row = deep[0]
            lines.append(
                f"- Best `deepfusionmot`: HOTA {float(row['HOTA']):.4f}, IDF1 {float(row['IDF1']):.4f}, "
                f"MOTA {float(row['MOTA']):.4f} at cam {float(row['min_camera_score']):.2f}, lidar {float(row['min_lidar_score']):.2f}."
            )
        if broad:
            row = broad[0]
            lines.append(
                f"- Best `all`: HOTA {float(row['HOTA']):.4f}, IDF1 {float(row['IDF1']):.4f}, "
                f"MOTA {float(row['MOTA']):.4f} at cam {float(row['min_camera_score']):.2f}, lidar {float(row['min_lidar_score']):.2f}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data_manifests" / "aghri_late_fusion_val_manifest.csv")
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW"))
    parser.add_argument("--threshold-root", type=Path, default=DEFAULT_THRESHOLD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--camera-cache", type=Path, default=DEFAULT_CAMERA_CACHE)
    parser.add_argument("--lidar-cache", type=Path, default=DEFAULT_LIDAR_CACHE)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--ego-motion-mode", default="aghri_odom")
    parser.add_argument("--camera-score-thresholds", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--lidar-score-thresholds", default="0.00,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--output-modes", default="deepfusionmot,all")
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    best = _load_best_threshold_config(args.threshold_root)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        _copy_sequence_breakdown(args.threshold_root, args.output_root, best)

    recordings = args.recording or _recordings_from_manifest(args.manifest)
    camera_scores = _parse_values(args.camera_score_thresholds, float)
    lidar_scores = _parse_values(args.lidar_score_thresholds, float)
    output_modes = _parse_values(args.output_modes, str)
    configs = list(itertools.product(camera_scores, lidar_scores, output_modes))
    if args.max_configs:
        configs = configs[: args.max_configs]

    results: list[dict] = []
    for index, (camera_score, lidar_score, output_mode) in enumerate(configs, start=1):
        config_id = _sweep_config_id(camera_score, lidar_score, output_mode)
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
                str(best["min_hits"]),
                "--tune-max-age-frames",
                str(best["max_age_frames"]),
                "--tune-association-3d-threshold",
                str(best["association_3d_threshold"]),
                "--tune-association-2d-threshold",
                str(best["association_2d_threshold"]),
                "--tune-support-2d-to-3d-threshold",
                str(best["support_2d_to_3d_threshold"]),
                "--tune-output-mode",
                output_mode,
                "--min-camera-score",
                str(camera_score),
                "--min-lidar-score",
                str(lidar_score),
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
                "min_camera_score": camera_score,
                "min_lidar_score": lidar_score,
                "output_mode": output_mode,
                "min_hits": best["min_hits"],
                "max_age_frames": best["max_age_frames"],
                "association_3d_threshold": best["association_3d_threshold"],
                "association_2d_threshold": best["association_2d_threshold"],
                "support_2d_to_3d_threshold": best["support_2d_to_3d_threshold"],
                "run_dir": str(batch_root),
                "metrics_dir": str(metrics_root),
            }
        )
        results.append(row)
        _write_csv(args.output_root / "detector_output_sweep_results.csv", results)
        (args.output_root / "detector_output_sweep_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        score_stats = {
            "camera": _score_stats(args.camera_cache, "confidence", camera_scores),
            "lidar": _score_stats(args.lidar_cache, "score", lidar_scores),
        }
        _write_sweep_summary(args.output_root / "detector_output_sweep_summary.md", results, args, best, score_stats)

    if not args.dry_run:
        ranked = sorted(results, key=lambda row: (_float(row, "HOTA"), _float(row, "IDF1"), -_int(row, "IDSW")), reverse=True)
        if ranked:
            _write_best_sweep_sequence_breakdown(args.output_root, ranked[0])
        _write_csv(args.output_root / "top_12_detector_output_by_hota.csv", ranked[:12])
        print(f"sequence_breakdown={args.output_root / 'per_sequence_metric_breakdown.md'}")
        print(f"best_sequence_breakdown={args.output_root / 'best_detector_output_per_sequence_breakdown.md'}")
        print(f"sweep_summary={args.output_root / 'detector_output_sweep_summary.md'}")
        if ranked:
            print(f"best_config={ranked[0]['config_id']}")
            print(f"best_HOTA={float(ranked[0]['HOTA']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
