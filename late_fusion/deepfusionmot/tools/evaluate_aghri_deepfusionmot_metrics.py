#!/usr/bin/env python3
"""Evaluate AGHRI tracking outputs with DeepFusionMOT/TrackEval MOT metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.aghri_mot_eval import (  # noqa: E402
    build_trackeval_sequence,
    combine_sequence_metrics,
    evaluate_sequence,
    evaluate_sequence_raw,
    find_recording_dir,
    load_camera_annotations,
    load_manifest_rows,
    load_tracked_boxes,
)


METRIC_FIELDS = [
    "combination_id",
    "combination_label",
    "recording",
    "frames_evaluated",
    "gt_dets",
    "tracker_dets",
    "gt_ids",
    "tracker_ids",
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "HOTA_0",
    "MOTA",
    "MOTP",
    "IDSW",
    "IDF1",
    "IDP",
    "IDR",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_TP",
    "CLR_FP",
    "CLR_FN",
    "IDTP",
    "IDFP",
    "IDFN",
    "tracker_only_fps",
    "cached_pipeline_fps",
    "tracking_output_dir",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def evaluate_tracking_dir(args: argparse.Namespace, tracking_dir: Path) -> dict:
    result, _raw = evaluate_tracking_dir_with_raw(args, tracking_dir)
    return result


def evaluate_tracking_dir_with_raw(args: argparse.Namespace, tracking_dir: Path) -> tuple[dict, dict]:
    summary_path = tracking_dir / "tracking_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    recording = args.recording or summary.get("recording")
    if not recording:
        raise ValueError(f"No recording supplied and no tracking_summary.json recording found in {tracking_dir}")
    rows = load_manifest_rows(
        args.manifest,
        recording,
        limit_frames=args.limit_frames or int(summary.get("frames_processed") or 0),
        unique_camera_frames=args.unique_camera_frames,
    )
    recording_dir = find_recording_dir(args.dataset_root, recording)
    annotations = load_camera_annotations(recording_dir)
    tracked = load_tracked_boxes(tracking_dir / "tracked_3d_results.csv")
    data, metadata = build_trackeval_sequence(
        manifest_rows=rows,
        annotations_by_file=annotations,
        tracked_by_frame=tracked,
        confirmed_only=not args.include_tentative,
    )
    raw_metrics = evaluate_sequence_raw(data, clear_threshold=args.threshold)
    metrics = evaluate_sequence(data, clear_threshold=args.threshold)
    result = {
        "combination_id": summary.get("combination_id", tracking_dir.parents[0].name),
        "combination_label": summary.get("combination_label", ""),
        "recording": recording,
        "frames_evaluated": metadata["num_frames"],
        "gt_dets": metadata["num_gt_dets"],
        "tracker_dets": metadata["num_tracker_dets"],
        "gt_ids": metadata["num_gt_ids"],
        "tracker_ids": metadata["num_tracker_ids"],
        "tracker_only_fps": summary.get("tracker_only_fps", ""),
        "cached_pipeline_fps": summary.get("cached_pipeline_fps", ""),
        "tracking_output_dir": str(tracking_dir),
        **metrics,
    }
    out_json = tracking_dir / "deepfusionmot_metrics.json"
    out_md = tracking_dir / "deepfusionmot_metrics.md"
    out_json.write_text(json.dumps({"metrics": result, "metadata": metadata}, indent=2), encoding="utf-8")
    lines = [
        "# AGHRI DeepFusionMOT/TrackEval Metrics",
        "",
        f"- Combination: `{result['combination_id']}` {result['combination_label']}",
        f"- Recording: `{recording}`",
        f"- Frames evaluated: {result['frames_evaluated']}",
        f"- GT detections / IDs: {result['gt_dets']} / {result['gt_ids']}",
        f"- Tracker detections / IDs: {result['tracker_dets']} / {result['tracker_ids']}",
        f"- Evaluation box space: ZED RGB 2D boxes; tracker boxes are projected tracked 3D boxes.",
        f"- CLEAR/Identity IoU threshold: {args.threshold}",
        "",
        "| HOTA | DetA | AssA | IDSW | MOTP | MOTA | IDF1 | FPS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {HOTA:.4f} | {DetA:.4f} | {AssA:.4f} | {IDSW} | {MOTP:.4f} | {MOTA:.4f} | {IDF1:.4f} | {tracker_only_fps} |".format(
            **result
        ),
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, raw_metrics


def _tracking_dirs_from_batch(root: Path) -> list[Path]:
    dirs = []
    for summary in sorted(root.glob("*/**/tracking_summary.json")):
        dirs.append(summary.parent)
    return dirs


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_float(row.get(field, "")) for field in METRIC_FIELDS})


def _batch_ego_motion_note(batch_root: Path) -> str:
    modes = set()
    applied = 0
    transitions = 0
    for path in batch_root.glob("*/*/tracking_summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mode = str(summary.get("ego_motion_mode", ""))
        if mode:
            modes.add(mode)
        applied += int(summary.get("ego_motion_applied_frames") or 0)
        transitions += max(0, int(summary.get("frames_processed") or 0) - 1)
    if not modes:
        return ""
    clean_modes = ", ".join(sorted(modes))
    if "aghri_odom" in modes:
        return f"- Ego motion mode: `{clean_modes}`; AGHRI odom compensation was applied on {applied}/{transitions} possible frame transitions."
    return f"- Ego motion mode: `{clean_modes}`."


def _write_batch_md(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    ego_note = _batch_ego_motion_note(args.batch_root)
    lines = [
        "# AGHRI DeepFusionMOT/TrackEval Metric Summary",
        "",
        f"- Batch root: `{args.batch_root}`",
        f"- Dataset root: `{args.dataset_root}`",
        f"- CLEAR/Identity IoU threshold: {args.threshold}",
        "- Metrics are computed with the bundled TrackEval implementation used by DeepFusionMOT.",
        "- Evaluation uses ZED RGB 2D annotation identities and projected tracked 3D boxes.",
    ]
    if ego_note:
        lines.append(ego_note)
    lines.extend(
        [
            "",
            "| ID | Combination | Frames | GT IDs | Trk IDs | HOTA | DetA | AssA | IDSW | MOTP | MOTA | IDF1 | FPS |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {combination_id} | {combination_label} | {frames_evaluated} | {gt_ids} | {tracker_ids} | "
            "{HOTA:.4f} | {DetA:.4f} | {AssA:.4f} | {IDSW} | {MOTP:.4f} | {MOTA:.4f} | {IDF1:.4f} | {tracker_only_fps:.2f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are MOT identity metrics, unlike the previous track-count diagnostics.",
            "- They are not KITTI-camera metrics; they are AGHRI ZED RGB identity metrics using the same TrackEval metric families.",
            "- `Class` values in AGHRI annotation JSON files are treated as ground-truth person IDs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combined_rows(per_sequence_rows: list[dict], raw_by_combo: dict[str, dict[str, dict]], args: argparse.Namespace) -> list[dict]:
    combined: list[dict] = []
    rows_by_combo: dict[str, list[dict]] = {}
    for row in per_sequence_rows:
        rows_by_combo.setdefault(str(row["combination_id"]), []).append(row)
    combo_order = ["S1", "S2", "S3", "S4", "P1", "P2", "P3", "P4"]
    for combo_id in combo_order:
        if combo_id not in rows_by_combo:
            continue
        group = rows_by_combo[combo_id]
        metrics = combine_sequence_metrics(raw_by_combo[combo_id], clear_threshold=args.threshold)
        weighted_fps_num = sum(float(row.get("tracker_only_fps") or 0.0) * int(row["frames_evaluated"]) for row in group)
        weighted_pipe_num = sum(float(row.get("cached_pipeline_fps") or 0.0) * int(row["frames_evaluated"]) for row in group)
        frame_count = sum(int(row["frames_evaluated"]) for row in group)
        combined.append(
            {
                "combination_id": combo_id,
                "combination_label": group[0].get("combination_label", ""),
                "recording": "COMBINED",
                "frames_evaluated": frame_count,
                "gt_dets": sum(int(row["gt_dets"]) for row in group),
                "tracker_dets": sum(int(row["tracker_dets"]) for row in group),
                "gt_ids": sum(int(row["gt_ids"]) for row in group),
                "tracker_ids": sum(int(row["tracker_ids"]) for row in group),
                "tracker_only_fps": weighted_fps_num / frame_count if frame_count else 0.0,
                "cached_pipeline_fps": weighted_pipe_num / frame_count if frame_count else 0.0,
                "tracking_output_dir": str(args.batch_root),
                **metrics,
            }
        )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW"))
    parser.add_argument("--tracking-dir", type=Path, default=None)
    parser.add_argument("--batch-root", type=Path, default=Path("results/aghri_deepfusionmot_tracking/eight_combo_batch"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/aghri_deepfusionmot_tracking/eight_combo_batch_metrics"))
    parser.add_argument("--recording", default="")
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-tentative", action="store_true")
    parser.add_argument("--unique-camera-frames", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.tracking_dir:
        result = evaluate_tracking_dir(args, args.tracking_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(args.output_dir / "deepfusionmot_metrics.csv", [result])
        _write_batch_md(args.output_dir / "deepfusionmot_metrics.md", [result], args)
        print(json.dumps(result, indent=2))
        return 0

    rows: list[dict] = []
    raw_by_combo: dict[str, dict[str, dict]] = {}
    for path in _tracking_dirs_from_batch(args.batch_root):
        result, raw = evaluate_tracking_dir_with_raw(args, path)
        rows.append(result)
        raw_by_combo.setdefault(str(result["combination_id"]), {})[str(result["recording"])] = raw
    combined_rows = _combined_rows(rows, raw_by_combo, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "deepfusionmot_metrics_per_recording.csv", rows)
    (args.output_dir / "deepfusionmot_metrics_per_recording.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "deepfusionmot_metrics.csv", combined_rows)
    (args.output_dir / "deepfusionmot_metrics.json").write_text(json.dumps(combined_rows, indent=2), encoding="utf-8")
    _write_batch_md(args.output_dir / "deepfusionmot_metrics.md", combined_rows, args)
    print(f"metrics_output_dir={args.output_dir}")
    print(f"evaluated_runs={len(rows)}")
    print(f"combined_combinations={len(combined_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
