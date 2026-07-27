#!/usr/bin/env python3
"""Run offline AGHRI tracking for the eight detector combinations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from run_aghri_deepfusionmot_tracking import DEFAULT_RECORDING, _run  # noqa: E402


@dataclass(frozen=True)
class TrackingCombination:
    combination_id: str
    legacy_combination_id: str
    label: str
    camera_model: str
    lidar_detector: str
    lidar_model: str


COMBINATIONS: tuple[TrackingCombination, ...] = (
    TrackingCombination("S1", "cam_g_lidar_g", "Generic YOLO + Generic SECOND", "generic", "second", "generic"),
    TrackingCombination("S2", "cam_ft_lidar_g", "Fine-tuned YOLO + Generic SECOND", "finetuned", "second", "generic"),
    TrackingCombination("S3", "cam_g_lidar_ft", "Generic YOLO + Fine-tuned SECOND", "generic", "second", "finetuned"),
    TrackingCombination("S4", "cam_ft_lidar_ft", "Fine-tuned YOLO + Fine-tuned SECOND", "finetuned", "second", "finetuned"),
    TrackingCombination("P1", "cam_g_pointpillars_g", "Generic YOLO + Generic PointPillars", "generic", "pointpillars", "generic"),
    TrackingCombination("P2", "cam_ft_pointpillars_g", "Fine-tuned YOLO + Generic PointPillars", "finetuned", "pointpillars", "generic"),
    TrackingCombination("P3", "cam_g_pointpillars_ft", "Generic YOLO + Fine-tuned PointPillars", "generic", "pointpillars", "finetuned"),
    TrackingCombination("P4", "cam_ft_pointpillars_ft", "Fine-tuned YOLO + Fine-tuned PointPillars", "finetuned", "pointpillars", "finetuned"),
)


SUMMARY_FIELDS = [
    "combination_id",
    "legacy_combination_id",
    "combination_label",
    "camera_model",
    "lidar_detector",
    "lidar_model",
    "recording",
    "frames_processed",
    "camera_detections",
    "lidar_detections",
    "score_filtered_lidar_detections",
    "nms_suppressed_lidar_detections",
    "fused_detections",
    "tracks_created",
    "confirmed_tracks",
    "mean_confirmed_track_length",
    "maximum_track_length",
    "prediction_only_updates",
    "deleted_tracks",
    "invalid_detections_rejected",
    "tracker_only_fps",
    "cached_pipeline_fps",
    "visual_frame_count",
    "visual_video_created",
    "tracker_preset",
    "deepfusionmot_strict",
    "min_camera_score",
    "min_lidar_score",
    "lidar_nms_iou_threshold",
    "lidar_only_initiation_score_threshold",
    "output_dir",
    "camera_cache",
    "lidar_cache",
]


def _selected_combinations(value: str) -> list[TrackingCombination]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(COMBINATIONS)
    by_id = {combo.combination_id: combo for combo in COMBINATIONS}
    by_legacy = {combo.legacy_combination_id: combo for combo in COMBINATIONS}
    selected: list[TrackingCombination] = []
    for item in requested:
        combo = by_id.get(item) or by_legacy.get(item)
        if combo is None:
            choices = ", ".join(combo.combination_id for combo in COMBINATIONS)
            raise ValueError(f"Unknown combination {item!r}; choose from {choices} or all")
        selected.append(combo)
    return selected


def _single_run_command(args: argparse.Namespace, combo: TrackingCombination, combo_output_dir: Path) -> str:
    min_camera_score = getattr(args, "min_camera_score", 0.0)
    min_lidar_score = getattr(args, "min_lidar_score", 0.0)
    lidar_nms_iou_threshold = getattr(args, "lidar_nms_iou_threshold", 0.0)
    lidar_only_initiation_score_threshold = getattr(args, "lidar_only_initiation_score_threshold", 0.0)
    parts = [
        "PYTHONPATH=.",
        "/usr/bin/python3",
        "tools/run_aghri_deepfusionmot_tracking.py",
        f"--recording {args.recording}",
        f"--output-dir {combo_output_dir}",
        f"--combination-id {combo.combination_id}",
        f"--camera-model {combo.camera_model}",
        f"--lidar-detector {combo.lidar_detector}",
        f"--lidar-model {combo.lidar_model}",
        f"--limit-frames {args.limit_frames}",
        f"--ego-motion-mode {args.ego_motion_mode}",
        f"--ego-odom-max-delta-sec {args.ego_odom_max_delta_sec}",
        f"--max-age {args.max_age}",
        f"--max-age-frames {args.max_age_frames}",
        f"--min-hits {args.min_hits}",
        f"--cache-tolerance-sec {args.cache_tolerance_sec}",
        f"--min-camera-score {min_camera_score}",
        f"--min-lidar-score {min_lidar_score}",
        f"--lidar-nms-iou-threshold {lidar_nms_iou_threshold}",
        f"--lidar-only-initiation-score-threshold {lidar_only_initiation_score_threshold}",
        f"--iou-threshold {args.iou_threshold}",
        f"--visual-frames {args.visual_frames}",
        f"--visual-video-fps {args.visual_video_fps}",
    ]
    if args.calibration_path:
        parts.append(f"--calibration-path {args.calibration_path}")
    if args.deepfusionmot_strict:
        parts.append("--deepfusionmot-strict")
    if args.association_config:
        parts.append(f"--association-config {args.association_config}")
    return " ".join(str(part) for part in parts)


def _single_args(args: argparse.Namespace, combo: TrackingCombination) -> SimpleNamespace:
    combo_output_dir = args.output_dir / combo.combination_id
    return SimpleNamespace(
        manifest=args.manifest,
        recording=args.recording,
        camera_cache=None,
        lidar_cache=None,
        output_dir=combo_output_dir,
        combination_id=combo.combination_id,
        camera_model=combo.camera_model,
        lidar_detector=combo.lidar_detector,
        lidar_model=combo.lidar_model,
        max_age=args.max_age,
        max_age_frames=args.max_age_frames,
        min_hits=args.min_hits,
        association_config=args.association_config,
        ego_motion_mode=args.ego_motion_mode,
        ego_odom_max_delta_sec=args.ego_odom_max_delta_sec,
        calibration_path=args.calibration_path,
        deepfusionmot_strict=args.deepfusionmot_strict,
        limit_frames=args.limit_frames,
        cache_tolerance_sec=args.cache_tolerance_sec,
        min_camera_score=getattr(args, "min_camera_score", 0.0),
        min_lidar_score=getattr(args, "min_lidar_score", 0.0),
        lidar_nms_iou_threshold=getattr(args, "lidar_nms_iou_threshold", 0.0),
        lidar_only_initiation_score_threshold=getattr(args, "lidar_only_initiation_score_threshold", 0.0),
        iou_threshold=args.iou_threshold,
        visual_frames=args.visual_frames,
        visual_video_fps=args.visual_video_fps,
        run_command=_single_run_command(args, combo, combo_output_dir),
    )


def _summary_path(args: argparse.Namespace, combo: TrackingCombination) -> Path:
    return args.output_dir / combo.combination_id / args.recording / "tracking_summary.json"


def _read_summary(args: argparse.Namespace, combo: TrackingCombination) -> dict:
    path = _summary_path(args, combo)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["legacy_combination_id"] = combo.legacy_combination_id
    summary["output_dir"] = str(path.parent)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _write_markdown(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    header = [
        "# AGHRI DeepFusionMOT Eight-Combination Batch",
        "",
        f"- Recording: `{args.recording}`",
        f"- Frames per combination: {args.limit_frames}",
        f"- Ego motion mode: `{args.ego_motion_mode}`",
        f"- Tracker preset: `{'deepfusionmot_strict' if args.deepfusionmot_strict else 'aghri_default'}`",
        f"- Camera score threshold: {args.min_camera_score:.3f}",
        f"- LiDAR score threshold: {args.min_lidar_score:.3f}",
        f"- LiDAR NMS IoU threshold: {args.lidar_nms_iou_threshold:.3f}",
        f"- LiDAR-only initiation score threshold: {args.lidar_only_initiation_score_threshold:.3f}",
        f"- Calibration path override: `{args.calibration_path or ''}`",
        f"- Output root: `{args.output_dir}`",
        "- This file reports tracking-count diagnostics; HOTA, IDF1, ID switches and MOTA are written by `tools/evaluate_aghri_deepfusionmot_metrics.py` when AGHRI annotation identities are available.",
        "",
        "| ID | Combination | Frames | Fused | Tracks | Confirmed | Max Len | Pred-only | Deleted | Tracker FPS | Pipeline FPS | Visuals |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = []
    for row in rows:
        table.append(
            "| {combination_id} | {combination_label} | {frames_processed} | {fused_detections} | "
            "{tracks_created} | {confirmed_tracks} | {maximum_track_length} | {prediction_only_updates} | "
            "{deleted_tracks} | {tracker_only_fps:.2f} | {cached_pipeline_fps:.2f} | {visual_frame_count} |".format(
                **row
            )
        )
    notes = [
        "",
        "## Combination Map",
        "",
        "- S1: Generic YOLO + Generic SECOND",
        "- S2: Fine-tuned YOLO + Generic SECOND",
        "- S3: Generic YOLO + Fine-tuned SECOND",
        "- S4: Fine-tuned YOLO + Fine-tuned SECOND",
        "- P1: Generic YOLO + Generic PointPillars",
        "- P2: Fine-tuned YOLO + Generic PointPillars",
        "- P3: Generic YOLO + Fine-tuned PointPillars",
        "- P4: Fine-tuned YOLO + Fine-tuned PointPillars",
        "",
        "## Limitations",
        "",
        "- This is an offline cached-detector tracking batch, not ROS 2 tracking.",
        "- `ego_motion_mode=aghri_odom` uses per-recording AGHRI `metadata/odom_global.jsonl`; `none` disables ego compensation.",
        "- `--deepfusionmot-strict` adopts DeepFusionMOT Pedestrian tracker settings only; it does not add detector fine-tuning, detector NMS tuning, or non-DeepFusionMOT heuristics.",
        "- Track IDs are algorithmic IDs scoped to each recording/combination; metric evaluation compares them against AGHRI annotation identities.",
        "- Visual continuity is a smoke-test proxy only.",
    ]
    path.write_text("\n".join(header + table + notes) + "\n", encoding="utf-8")


def run_batch(args: argparse.Namespace) -> int:
    selected = _selected_combinations(args.combinations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for combo in selected:
        print(f"[{combo.combination_id}] Running {combo.label}")
        _run(_single_args(args, combo))
        summaries.append(_read_summary(args, combo))

    _write_csv(args.output_dir / "tracking_batch_summary.csv", summaries)
    (args.output_dir / "tracking_batch_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_markdown(args.output_dir / "tracking_batch_summary.md", summaries, args)
    (args.output_dir / "run_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(f"batch_output_dir={args.output_dir}")
    print(f"combinations={len(summaries)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aghri_deepfusionmot_tracking/eight_combo_batch"))
    parser.add_argument("--combinations", default="all", help="Comma-separated S1..S4/P1..P4 list, legacy IDs, or all.")
    parser.add_argument("--max-age", type=float, default=0.75)
    parser.add_argument("--max-age-frames", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--association-config", default="")
    parser.add_argument("--ego-motion-mode", default="none")
    parser.add_argument("--ego-odom-max-delta-sec", type=float, default=0.25)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument(
        "--deepfusionmot-strict",
        action="store_true",
        help="Use DeepFusionMOT Pedestrian-style tracker lifecycle, gates, IDs, and output filtering.",
    )
    parser.add_argument("--limit-frames", type=int, default=80)
    parser.add_argument("--cache-tolerance-sec", type=float, default=0.08)
    parser.add_argument("--min-camera-score", type=float, default=0.0)
    parser.add_argument("--min-lidar-score", type=float, default=0.0)
    parser.add_argument("--lidar-nms-iou-threshold", type=float, default=0.0)
    parser.add_argument("--lidar-only-initiation-score-threshold", type=float, default=0.0)
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument("--visual-frames", type=int, default=10)
    parser.add_argument("--visual-video-fps", type=float, default=5.0)
    args = parser.parse_args()
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
