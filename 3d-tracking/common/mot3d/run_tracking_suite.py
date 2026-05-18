"""
3D tracking benchmark suite -- run all supported trackers across multiple scenes
and collect MOTA / MOTP / IDF1 / HOTA metrics into a single summary table.

Supported input sources (--sources):
  gt          Use GT annotations as perfect detections (upper-bound baseline).
              Requires --raw-root pointing to the labelled_dataset directory.
  detections  Use real model detections from --detections-dir.
              The directory must contain one sub-folder per scene, each holding
              per-frame JSON files in the format: [{x,y,z,l,w,h,yaw,score}, ...]

Pipeline per source x scene x tracker:
  1. (gt only) Convert raw lidar_ann.json -> per-frame detection JSONs
  2. (gt only) Convert raw lidar_ann.json -> GT MOT3D CSV
  3. Run tracker on detection JSONs       -> per-scene predicted MOT3D CSV
  4. Evaluate predicted vs GT             -> per-scene + overall metrics JSON / CSV
  5. Append row to aggregate summary

Default scenes (the three benchmark scenarios in this project):
  - footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label
  - in_straw_3pick_diff_st_10_24_2024_5_a_label
  - out_vine_4swap+walk_st_ly_11_06_2024_2_label

Usage:
    python common/mot3d/run_tracking_suite.py
    python common/mot3d/run_tracking_suite.py --sources gt
    python common/mot3d/run_tracking_suite.py --sources detections ^
        --detections-dir D:/AOC/agri-human-dataset-benchmark/3d-detection/reports/detections/pointpillars_aghri
    python common/mot3d/run_tracking_suite.py --sources gt detections ^
        --detections-dir D:/AOC/agri-human-dataset-benchmark/3d-detection/reports/detections/pointpillars_aghri ^
        --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

TRACKING_ROOT = Path(__file__).resolve().parents[2]   # 3d-tracking/
COMMON_ROOT   = Path(__file__).resolve().parents[0]   # 3d-tracking/common/mot3d/

sys.path.insert(0, str(COMMON_ROOT))
from gt_as_detections import convert_scene_to_detections, convert_scene_to_gt_mot3d

RAW_ROOT_DEFAULT = Path(r"D:\AOC\datasets\agri-human-sensing\labelled_dataset")

DEFAULT_SCENES = [
    "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label",
    "in_straw_3pick_diff_st_10_24_2024_5_a_label",
    "out_vine_4swap+walk_st_ly_11_06_2024_2_label",
]


@dataclass(frozen=True)
class TrackerSpec:
    key: str
    run_script: Path        # relative to TRACKING_ROOT
    config_path: Path       # relative to TRACKING_ROOT


TRACKER_SPECS: Dict[str, TrackerSpec] = {
    "ab3dmot": TrackerSpec(
        key="ab3dmot",
        run_script=Path("benchmarks/ab3dmot/src/run_tracker.py"),
        config_path=Path("benchmarks/ab3dmot/configs/tracking/aghri_ab3dmot.yaml"),
    ),
    "simpletrack": TrackerSpec(
        key="simpletrack",
        run_script=Path("benchmarks/simpletrack/src/run_tracker.py"),
        config_path=Path("benchmarks/simpletrack/configs/tracking/aghri_simpletrack.yaml"),
    ),
    "centerpoint": TrackerSpec(
        key="centerpoint",
        run_script=Path("benchmarks/centerpoint/src/run_tracker.py"),
        config_path=Path("benchmarks/centerpoint/configs/tracking/aghri_centerpoint.yaml"),
    ),
}

EVAL_SCRIPT = COMMON_ROOT / "evaluate_mot3d.py"


def _run(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=TRACKING_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _prepare_gt(
    scene_dirs: List[Path],
    gt_det_root: Path,
    gt_mot_root: Path,
    skip_existing: bool,
) -> None:
    """Convert raw annotations for all scenes (detection JSONs + GT MOT3D CSVs)."""
    print("-- Preparing GT detections and GT MOT3D files --")
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        det_out = gt_det_root / scene_name
        gt_csv = gt_mot_root / f"{scene_name}.csv"

        if skip_existing and det_out.exists() and gt_csv.exists():
            print(f"  [skip] {scene_name}")
            continue

        frames = convert_scene_to_detections(scene_dir, det_out)
        frame_count, ann_count = convert_scene_to_gt_mot3d(scene_dir, gt_csv)
        print(f"  {scene_name}: {frames} det frames, {ann_count} GT annotations")


def _build_tracker_command(
    spec: TrackerSpec,
    detections_dir: Path,
    mot_output_dir: Path,
    runtime_json: Path,
) -> List[str]:
    return [
        sys.executable,
        str(TRACKING_ROOT / spec.run_script),
        "--config",           str(TRACKING_ROOT / spec.config_path),
        "--detections-dir",   str(detections_dir),
        "--mot-output-dir",   str(mot_output_dir),
        "--runtime-json",     str(runtime_json),
    ]


def _build_eval_command(
    gt_dir: Path,
    pred_dir: Path,
    out_csv: Path,
    out_json: Path,
    iou_threshold: float,
) -> List[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--gt-dir",        str(gt_dir),
        "--pred-dir",      str(pred_dir),
        "--iou-threshold", str(iou_threshold),
        "--out-csv",       str(out_csv),
        "--out-json",      str(out_json),
    ]


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tracker(
    spec: TrackerSpec,
    det_root: Path,
    runs_dir: Path,
    summary_dir: Path,
    gt_mot_root: Path,
    iou_threshold: float,
    skip_existing: bool,
    source_label: str,
) -> List[Dict]:
    """Run one tracker over all scenes and evaluate. Returns list of result rows."""
    pred_dir     = runs_dir / spec.key
    runtime_json = summary_dir / f"{spec.key}_runtime.json"
    metrics_csv  = summary_dir / f"{spec.key}_metrics.csv"
    metrics_json = summary_dir / f"{spec.key}_metrics.json"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n-- Tracker: {spec.key} (source: {source_label}) --")

    # ---- tracking step ----
    if skip_existing and pred_dir.exists() and any(pred_dir.glob("*.csv")) and runtime_json.exists():
        print(f"  [skip tracking] existing outputs found in {pred_dir}")
    else:
        cmd = _build_tracker_command(spec, det_root, pred_dir, runtime_json)
        result = _run(cmd)
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  [FAIL] tracking: {result.stderr.strip() or result.stdout.strip()}")
            return [{
                "source": source_label,
                "tracker": spec.key, "scene": "ALL", "status": "tracking_failed",
                "error": result.stderr.strip() or result.stdout.strip(),
            }]

    # ---- evaluation step ----
    if skip_existing and metrics_json.exists():
        print(f"  [skip evaluation] {metrics_json.name} already exists")
    else:
        cmd = _build_eval_command(gt_mot_root, pred_dir, metrics_csv, metrics_json, iou_threshold)
        result = _run(cmd)
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  [FAIL] evaluation: {result.stderr.strip() or result.stdout.strip()}")
            return [{
                "source": source_label,
                "tracker": spec.key, "scene": "ALL", "status": "evaluation_failed",
                "error": result.stderr.strip() or result.stdout.strip(),
            }]

    # ---- collect per-scene rows ----
    rows = []
    if metrics_json.exists():
        metric_rows = _load_json(metrics_json)
        runtime = _load_json(runtime_json) if runtime_json.exists() else {}
        total_frames = runtime.get("total_frames", 0)
        tracking_time_s = runtime.get("tracking_time_seconds", 0.0)
        tracking_time_per_frame_ms = (
            tracking_time_s * 1000.0 / total_frames if total_frames > 0 else 0.0
        )
        for m in metric_rows:
            scene_label = m.get("index", "unknown")
            if scene_label == "OVERALL":
                continue
            rows.append({
                "source": source_label,
                "tracker": spec.key,
                "scene": scene_label,
                "status": "completed",
                # file paths for traceability
                "detections_dir": str(det_root / scene_label),
                "gt_mot_dir": str(gt_mot_root),
                "pred_csv": str(pred_dir / f"{scene_label}.csv"),
                "runtime_json": str(runtime_json),
                "metrics_csv": str(metrics_csv),
                "metrics_json": str(metrics_json),
                # runtime (aggregated over all scenes in this run)
                **{k: v for k, v in runtime.items() if k not in ("tracker",)},
                "tracking_time_per_frame_ms": round(tracking_time_per_frame_ms, 4),
                # per-scene metrics
                **{k: v for k, v in m.items() if k != "index"},
            })

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 3D tracking benchmark suite across all trackers and scenes.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("gt", "detections"),
        default=["gt"],
        help=(
            "Input source(s) to run.\n"
            "  gt         -- GT annotations as perfect detections (requires --raw-root)\n"
            "  detections -- real model detections (requires --detections-dir)"
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT_DEFAULT,
        help="Raw labelled_dataset root directory (used for 'gt' source).",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        default=None,
        help=(
            "Directory with per-scene detection sub-folders (used for 'detections' source).\n"
            "Each sub-folder name must match a scene name and contain per-frame JSON files."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene directory names to benchmark.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=sorted(TRACKER_SPECS),
        default=sorted(TRACKER_SPECS),
        help="Trackers to run.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=TRACKING_ROOT / "reports" / "runs" / "tracker_suite",
        help="Root output directory for all tracker prediction CSVs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=TRACKING_ROOT / "reports" / "summary" / "tracker_suite",
        help="Directory for per-tracker metrics files and the aggregate summary.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.25,
        help="Minimum BEV IoU to count as a true positive match (default: 0.25).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tracker runs whose prediction CSVs and metrics already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs_dir    = args.runs_dir
    summary_dir = args.summary_dir

    # GT files are shared across all sources -- always derived from raw annotations.
    gt_det_root = runs_dir / "_gt_detections"
    gt_mot_root = runs_dir / "_gt_mot3d"

    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Validate scene directories for gt source.
    scene_dirs = []
    if "gt" in args.sources:
        for scene_name in args.scenes:
            scene_dir = args.raw_root / scene_name
            if not scene_dir.is_dir():
                print(f"[WARN] Scene directory not found, skipping: {scene_dir}")
                continue
            scene_dirs.append(scene_dir)

        if not scene_dirs:
            print("[ERROR] No valid scene directories found. Check --raw-root and --scenes.")
            sys.exit(1)

        gt_det_root.mkdir(parents=True, exist_ok=True)
        gt_mot_root.mkdir(parents=True, exist_ok=True)
        _prepare_gt(scene_dirs, gt_det_root, gt_mot_root, args.skip_existing)

    # For detections source, GT MOT3D files must already exist (from a prior gt run).
    if "detections" in args.sources and not "gt" in args.sources:
        if not gt_mot_root.exists() or not any(gt_mot_root.glob("*.csv")):
            print(
                "[ERROR] GT MOT3D evaluation files not found.\n"
                f"        Expected CSV files in: {gt_mot_root}\n"
                "        Run with '--sources gt' first to generate them."
            )
            sys.exit(1)

    all_results: List[Dict] = []

    for source in args.sources:
        if source == "gt":
            det_root   = gt_det_root
            source_label = "gt"
        else:  # detections
            if args.detections_dir is None:
                print("[WARN] --detections-dir not set; skipping 'detections' source.")
                continue
            det_root     = args.detections_dir
            source_label = det_root.name   # e.g. "pointpillars_aghri"

        source_runs_dir    = runs_dir    / source_label
        source_summary_dir = summary_dir / source_label
        source_runs_dir.mkdir(parents=True, exist_ok=True)
        source_summary_dir.mkdir(parents=True, exist_ok=True)

        for tracker_key in args.trackers:
            spec = TRACKER_SPECS[tracker_key]
            tracker_summary_dir = source_summary_dir / tracker_key
            tracker_summary_dir.mkdir(parents=True, exist_ok=True)
            rows = _run_tracker(
                spec,
                det_root,
                source_runs_dir,
                tracker_summary_dir,
                gt_mot_root,
                args.iou_threshold,
                args.skip_existing,
                source_label,
            )
            all_results.extend(rows)

    # Write aggregate summary.
    summary_json = summary_dir / "suite_summary.json"
    summary_csv  = summary_dir / "suite_summary.csv"

    summary_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    fieldnames: List[str] = []
    for row in all_results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    completed = sum(1 for r in all_results if r.get("status") == "completed")
    failed    = sum(1 for r in all_results if "failed" in str(r.get("status", "")))
    print(f"\n-- Suite finished --")
    print(f"  Completed rows : {completed}")
    print(f"  Failed rows    : {failed}")
    print(f"  Summary CSV    : {summary_csv}")
    print(f"  Summary JSON   : {summary_json}")


if __name__ == "__main__":
    main()
