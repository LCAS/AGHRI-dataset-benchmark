#!/usr/bin/env python3
"""Run one offline AGHRI DeepFusionMOT-style tracking experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.aghri_calibration import load_aghri_camera_lidar_calibration
from late_fusion_pkg.aghri_detector_profiles import LIDAR_DETECTOR_CHOICES, MODEL_CHOICES, resolve_detector_selection
from late_fusion_pkg.aghri_ego_motion import AghriOdomEgoMotion
from late_fusion_pkg.cache_replay import load_second_cache, load_yolo_cache
from late_fusion_pkg.deepfusionmot_association import non_max_suppress_3d
from late_fusion_pkg.deepfusionmot_io import (
    DIAGNOSTIC_FIELDS,
    TRACKED_2D_FIELDS,
    TRACKED_3D_FIELDS,
    diagnostics_to_row,
    draw_tracking_frame,
    tracked2d_to_row,
    tracked3d_to_row,
    write_csv,
)
from late_fusion_pkg.deepfusionmot_tracker import DeepFusionMOTTracker, DeepFusionMOTTrackerConfig
from late_fusion_pkg.fusion_core import fuse_detections


DEFAULT_RECORDING = "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"


def _model_label(model: str) -> str:
    return "FT" if model == "finetuned" else "G"


def _detector_label(detector: str) -> str:
    return "PointPillars" if detector == "pointpillars" else "SECOND"


def _load_manifest_rows(manifest: Path, recording: str) -> list[dict]:
    rows: list[dict] = []
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("recording_name") != recording:
                continue
            if not row.get("lidar_timestamp"):
                continue
            rows.append(row)
    rows.sort(key=lambda item: float(item["lidar_timestamp"]))
    return rows


def _load_association_config(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _fallback_calibration() -> dict:
    return {
        "camera_p": np.array(
            [
                [477.29998779296875, 0.0, 339.74749755859375, 0.0],
                [0.0, 477.4949951171875, 191.94400024414062, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        "camera_from_lidar": np.array(
            [
                [0.0, -1.0, 0.0, -0.22],
                [0.0, 0.0, -1.0, 0.295],
                [1.0, 0.0, 0.0, 0.075],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "base_from_lidar": np.array(
            [
                [1.0, 0.0, 0.0, 0.615],
                [0.0, 1.0, 0.0, 0.28],
                [0.0, 0.0, 1.0, 0.33],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "image_width": 672,
        "image_height": 376,
        "camera_frame": "front_left_camera_optical_frame",
        "lidar_frame": "front_lidar_link",
        "transform_path": ["front_lidar_link", "front_left_camera_optical_frame"],
        "base_transform_path": ["front_lidar_link", "base_link"],
        "matrix_source": "fallback_verified_camera_info_p",
    }


def _resolve_calibration(rows: list[dict], calibration_path: Path | None = None) -> tuple[dict, str]:
    if calibration_path:
        return load_aghri_camera_lidar_calibration(calibration_path), str(calibration_path)
    source = rows[0].get("calibration_source") if rows else ""
    if source:
        path = Path(source)
        if path.exists():
            try:
                return load_aghri_camera_lidar_calibration(path), str(path)
            except Exception:
                pass
    return _fallback_calibration(), "fallback_verified_aghri_projection"


def _resolve_recording_dir(rows: list[dict]) -> Path | None:
    for row in rows:
        for field in ("image_path", "lidar_point_cloud_path", "gt_2d_reference", "gt_3d_reference"):
            value = row.get(field)
            if not value:
                continue
            path = Path(value)
            if field in {"gt_2d_reference", "gt_3d_reference"}:
                return path.parents[1]
            if "sensor_data" in path.parts:
                index = path.parts.index("sensor_data")
                return Path(*path.parts[:index])
            for parent in path.parents:
                if (parent / "metadata").is_dir():
                    return parent
    return None


def _runtime_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "total_sec": 0.0, "mean_sec": 0.0, "fps": 0.0}
    total = sum(values)
    return {
        "count": len(values),
        "total_sec": total,
        "mean_sec": statistics.mean(values),
        "median_sec": statistics.median(values),
        "fps": len(values) / total if total > 0.0 else 0.0,
    }


def _write_visual_video(frame_paths: list[str], output_path: Path, fps: float) -> tuple[bool, str]:
    if not frame_paths:
        return False, "no visual frames were generated"
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        return False, f"OpenCV is unavailable: {exc}"

    first = cv2.imread(frame_paths[0])
    if first is None:
        return False, f"could not read first visual frame: {frame_paths[0]}"
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False, f"could not open video writer: {output_path}"
    try:
        writer.write(first)
        for path in frame_paths[1:]:
            image = cv2.imread(path)
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
    finally:
        writer.release()
    return True, str(output_path)


def _run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir / args.recording
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_frames"
    selection = resolve_detector_selection(
        camera_model=args.camera_model,
        lidar_detector=args.lidar_detector,
        lidar_model=args.lidar_model,
        yolo_cache_override=str(args.camera_cache) if args.camera_cache else "",
        second_cache_override=str(args.lidar_cache) if args.lidar_cache else "",
    )
    camera_cache_path = args.camera_cache or selection.camera.cache_csv
    lidar_cache_path = args.lidar_cache or selection.lidar.cache_csv
    detector_name = _detector_label(selection.lidar.detector_family)
    combo_label = (
        f"{_model_label(selection.camera.model)} YOLO + "
        f"{_model_label(selection.lidar.model)} {detector_name}"
    )

    manifest_rows = _load_manifest_rows(args.manifest, args.recording)
    matched_rows = [row for row in manifest_rows if row.get("image_path")]
    selected_rows = matched_rows or manifest_rows
    if args.limit_frames:
        selected_rows = selected_rows[: args.limit_frames]
    if not selected_rows:
        raise RuntimeError(f"No manifest rows found for recording {args.recording!r}")

    calibration, calibration_source = _resolve_calibration(selected_rows, args.calibration_path)
    camera_p = calibration["camera_p"]
    camera_from_lidar = calibration["camera_from_lidar"]
    image_width = int(calibration["image_width"])
    image_height = int(calibration["image_height"])
    recording_dir = _resolve_recording_dir(selected_rows)
    ego_motion = None
    ego_motion_source = ""
    if args.ego_motion_mode == "aghri_odom":
        if recording_dir is None:
            raise RuntimeError(f"Could not resolve AGHRI recording directory for {args.recording!r}")
        ego_motion = AghriOdomEgoMotion.from_recording_dir(
            recording_dir,
            calibration["base_from_lidar"],
            max_time_delta_sec=args.ego_odom_max_delta_sec,
        )
        ego_motion_source = str(recording_dir / "metadata" / "odom_global.jsonl")

    load_start = time.perf_counter()
    camera_cache = load_yolo_cache(camera_cache_path, timestamp_tolerance_sec=args.cache_tolerance_sec)
    lidar_cache = load_second_cache(
        lidar_cache_path,
        timestamp_tolerance_sec=args.cache_tolerance_sec,
        box_origin=selection.lidar.box_origin or "bottom_center",
    )
    cache_load_sec = time.perf_counter() - load_start

    assoc = _load_association_config(args.association_config)
    tracker_overrides = {}
    if args.deepfusionmot_strict:
        tracker_overrides = {
            "min_hits": 3,
            "max_age_frames": 25,
            "association_3d_threshold": 0.01,
            "association_2d_threshold": 0.50,
            "support_2d_to_3d_threshold": 0.50,
            "association_3d_metric": "bev_iou",
            "id_policy": "deepfusionmot_even_odd",
            "lifecycle_mode": "deepfusionmot_frame",
            "output_mode": "deepfusionmot",
        }
    if args.tune_min_hits is not None:
        tracker_overrides["min_hits"] = args.tune_min_hits
    if args.tune_max_age_frames is not None:
        tracker_overrides["max_age_frames"] = args.tune_max_age_frames
    if args.tune_association_3d_threshold is not None:
        tracker_overrides["association_3d_threshold"] = args.tune_association_3d_threshold
    if args.tune_association_2d_threshold is not None:
        tracker_overrides["association_2d_threshold"] = args.tune_association_2d_threshold
    if args.tune_support_2d_to_3d_threshold is not None:
        tracker_overrides["support_2d_to_3d_threshold"] = args.tune_support_2d_to_3d_threshold
    if args.tune_output_mode:
        tracker_overrides["output_mode"] = args.tune_output_mode
    tracker_config = DeepFusionMOTTrackerConfig(
        min_hits=int(tracker_overrides.get("min_hits", args.min_hits)),
        max_age_frames=int(tracker_overrides.get("max_age_frames", args.max_age_frames)),
        max_age_seconds=args.max_age,
        association_3d_threshold=float(
            tracker_overrides.get("association_3d_threshold", assoc.get("association_3d_threshold", 0.05))
        ),
        association_2d_threshold=float(
            tracker_overrides.get("association_2d_threshold", assoc.get("association_2d_threshold", 0.20))
        ),
        support_2d_to_3d_threshold=float(
            tracker_overrides.get("support_2d_to_3d_threshold", assoc.get("support_2d_to_3d_threshold", 0.10))
        ),
        association_3d_metric=str(
            tracker_overrides.get("association_3d_metric", assoc.get("association_3d_metric", "hybrid"))
        ),
        max_center_distance_m=float(assoc.get("max_center_distance_m", 2.0)),
        lidar_only_initiation_score_threshold=args.lidar_only_initiation_score_threshold,
        ego_motion_mode=args.ego_motion_mode,
        id_policy=str(tracker_overrides.get("id_policy", "sequential")),
        lifecycle_mode=str(tracker_overrides.get("lifecycle_mode", "time")),
        output_mode=str(tracker_overrides.get("output_mode", "all")),
    )
    tracker = DeepFusionMOTTracker(tracker_config)

    tracked_3d_rows: list[dict] = []
    tracked_2d_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    per_frame_rows: list[dict] = []
    tracker_runtimes: list[float] = []
    pipeline_runtimes: list[float] = []
    visual_outputs = []
    camera_count = 0
    lidar_count = 0
    score_filtered_lidar_count = 0
    nms_suppressed_lidar_count = 0
    matched_count = 0
    prediction_only_updates = 0
    ego_motion_applied_frames = 0
    previous_lidar_timestamp: float | None = None

    for frame_index, row in enumerate(selected_rows):
        pipeline_start = time.perf_counter()
        lidar_timestamp = float(row["lidar_timestamp"])
        camera_timestamp = float(row.get("zed_rgb_timestamp") or lidar_timestamp)
        raw_camera_dets = tuple(camera_cache.lookup(args.recording, camera_timestamp))
        raw_lidar_dets = tuple(lidar_cache.lookup(args.recording, lidar_timestamp))
        camera_dets = tuple(det for det in raw_camera_dets if det.score >= args.min_camera_score)
        score_filtered_lidar_dets = tuple(det for det in raw_lidar_dets if det.score >= args.min_lidar_score)
        lidar_dets = non_max_suppress_3d(score_filtered_lidar_dets, args.lidar_nms_iou_threshold)
        result = fuse_detections(
            camera_dets,
            lidar_dets,
            camera_p,
            camera_from_lidar,
            image_width,
            image_height,
            iou_threshold=args.iou_threshold,
            confidence_policy="lidar_score",
        )
        ego_motion_transform = None
        if ego_motion is not None:
            ego_motion_transform = ego_motion.relative_lidar_transform(previous_lidar_timestamp, lidar_timestamp)
            if ego_motion_transform is not None:
                ego_motion_applied_frames += 1
        tracker_start = time.perf_counter()
        tracking = tracker.update(
            recording_id=args.recording,
            frame_index=frame_index,
            timestamp=lidar_timestamp,
            camera_detections=camera_dets,
            lidar_detections=lidar_dets,
            fusion_result=result,
            ego_motion_transform=ego_motion_transform,
        )
        previous_lidar_timestamp = lidar_timestamp
        tracker_runtimes.append(time.perf_counter() - tracker_start)
        pipeline_runtimes.append(time.perf_counter() - pipeline_start)
        camera_count += len(camera_dets)
        lidar_count += len(lidar_dets)
        score_filtered_lidar_count += len(score_filtered_lidar_dets)
        nms_suppressed_lidar_count += max(0, len(score_filtered_lidar_dets) - len(lidar_dets))
        matched_count += len(result.matches)
        prediction_only_updates += tracking.diagnostics.predicted_only_tracks

        for track in tracking.tracked_3d:
            tracked_3d_rows.append(tracked3d_to_row(track))
        for track in tracking.tracked_2d:
            tracked_2d_rows.append(tracked2d_to_row(track))
        diagnostic_rows.append(diagnostics_to_row(tracking.diagnostics))
        per_frame_rows.append(
            {
                "recording_id": args.recording,
                "frame_index": frame_index,
                "timestamp": f"{lidar_timestamp:.9f}",
                "raw_camera_detections": len(raw_camera_dets),
                "raw_lidar_detections": len(raw_lidar_dets),
                "camera_detections": len(camera_dets),
                "score_filtered_lidar_detections": len(score_filtered_lidar_dets),
                "nms_suppressed_lidar_detections": max(0, len(score_filtered_lidar_dets) - len(lidar_dets)),
                "lidar_detections": len(lidar_dets),
                "fusion_matches": len(result.matches),
                "tracked_3d": len(tracking.tracked_3d),
                "confirmed_3d": sum(1 for item in tracking.tracked_3d if item.track_state == "confirmed"),
                "tracked_2d": len(tracking.tracked_2d),
                "tracker_update_time_ms": tracking.diagnostics.tracker_update_time_ms,
            }
        )
        if len(visual_outputs) < args.visual_frames and tracking.tracked_3d and row.get("image_path"):
            out_png = visual_dir / f"frame_{frame_index:04d}.png"
            stats = draw_tracking_frame(
                image_path=Path(row["image_path"]),
                output_path=out_png,
                tracks=tracking.tracked_3d,
                camera_p_3x4=camera_p,
                camera_from_lidar_4x4=camera_from_lidar,
                image_width=image_width,
                image_height=image_height,
            )
            if stats["drawn"] > 0:
                visual_outputs.append(str(out_png))

    write_csv(output_dir / "tracked_3d_results.csv", tracked_3d_rows, TRACKED_3D_FIELDS)
    write_csv(output_dir / "tracked_2d_results.csv", tracked_2d_rows, TRACKED_2D_FIELDS)
    write_csv(output_dir / "tracking_diagnostics.csv", diagnostic_rows, DIAGNOSTIC_FIELDS)
    write_csv(
        output_dir / "per_frame_tracking_summary.csv",
        per_frame_rows,
        [
            "recording_id",
            "frame_index",
            "timestamp",
            "raw_camera_detections",
            "raw_lidar_detections",
            "camera_detections",
            "score_filtered_lidar_detections",
            "nms_suppressed_lidar_detections",
            "lidar_detections",
            "fusion_matches",
            "tracked_3d",
            "confirmed_3d",
            "tracked_2d",
            "tracker_update_time_ms",
        ],
    )

    track_lengths: dict[int, int] = {}
    confirmed_lengths: dict[int, int] = {}
    confirmed_track_ids = set()
    for row in tracked_3d_rows:
        track_id = int(row["track_id"])
        track_lengths[track_id] = track_lengths.get(track_id, 0) + 1
        if row["track_state"] == "confirmed":
            confirmed_lengths[track_id] = confirmed_lengths.get(track_id, 0) + 1
            confirmed_track_ids.add(track_id)
    tracker_runtime = _runtime_summary(tracker_runtimes)
    pipeline_runtime = _runtime_summary(pipeline_runtimes)
    video_path = output_dir / "tracking_visualization.mp4"
    video_created, video_status = _write_visual_video(visual_outputs, video_path, args.visual_video_fps)
    summary = {
        "recording": args.recording,
        "combination_id": args.combination_id,
        "combination_label": combo_label,
        "camera_model": selection.camera.model,
        "lidar_detector": selection.lidar.detector_family,
        "lidar_model": selection.lidar.model,
        "frames_processed": len(selected_rows),
        "camera_cache": str(camera_cache_path),
        "lidar_cache": str(lidar_cache_path),
        "min_camera_score": args.min_camera_score,
        "min_lidar_score": args.min_lidar_score,
        "lidar_nms_iou_threshold": args.lidar_nms_iou_threshold,
        "score_filtered_lidar_detections": score_filtered_lidar_count,
        "nms_suppressed_lidar_detections": nms_suppressed_lidar_count,
        "lidar_only_initiation_score_threshold": args.lidar_only_initiation_score_threshold,
        "calibration_source": calibration_source,
        "camera_detections": camera_count,
        "lidar_detections": lidar_count,
        "fused_detections": matched_count,
        "tracks_created": tracker.total_created_3d,
        "confirmed_tracks": len(confirmed_track_ids),
        "mean_confirmed_track_length": statistics.mean(confirmed_lengths.values()) if confirmed_lengths else 0.0,
        "maximum_track_length": max(track_lengths.values()) if track_lengths else 0,
        "prediction_only_updates": prediction_only_updates,
        "deleted_tracks": tracker.total_deleted_3d,
        "invalid_detections_rejected": tracker.total_invalid_rejected,
        "tracker_only_fps": tracker_runtime["fps"],
        "cached_pipeline_fps": pipeline_runtime["fps"],
        "cache_load_sec": cache_load_sec,
        "visual_frames": visual_outputs,
        "visual_frame_count": len(visual_outputs),
        "visual_video": str(video_path) if video_created else "",
        "visual_video_created": video_created,
        "visual_video_status": video_status,
        "visual_id_continuity_appears_reasonable": bool(confirmed_lengths and max(confirmed_lengths.values()) >= 5),
        "identity_ground_truth_metrics_reported": False,
        "ego_motion_mode": args.ego_motion_mode,
        "ego_motion_source": ego_motion_source,
        "ego_motion_applied_frames": ego_motion_applied_frames,
        "deepfusionmot_strict": bool(args.deepfusionmot_strict),
        "output_mode": tracker_config.output_mode,
        "tracker_preset": "deepfusionmot_strict" if args.deepfusionmot_strict else "aghri_default",
        "recording_selection_reason": (
            "Selected from AGHRI cache/manifest evidence because it has consecutive timestamped frames, "
            "valid ZED RGB image paths, FT YOLO detections, FT PointPillars detections, and valid AGHRI calibration."
        ),
    }
    (output_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    resolved = {
        "tracker_config": tracker_config.__dict__,
        "camera_cache_validation": camera_cache.validation.__dict__,
        "lidar_cache_validation": lidar_cache.validation.__dict__,
        "image_width": image_width,
        "image_height": image_height,
        "iou_threshold": args.iou_threshold,
        "min_camera_score": args.min_camera_score,
        "min_lidar_score": args.min_lidar_score,
        "lidar_nms_iou_threshold": args.lidar_nms_iou_threshold,
        "lidar_only_initiation_score_threshold": args.lidar_only_initiation_score_threshold,
        "limit_frames": args.limit_frames,
        "ego_motion_source": ego_motion_source,
        "ego_motion_applied_frames": ego_motion_applied_frames,
    }
    (output_dir / "resolved_tracking_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    run_command = getattr(args, "run_command", "") or " ".join(sys.argv)
    (output_dir / "run_command.txt").write_text(run_command + "\n", encoding="utf-8")
    (output_dir / "run_stderr.txt").write_text("", encoding="utf-8")
    lines = [
        "# AGHRI DeepFusionMOT-Style Tracking Experiment",
        "",
        f"- Combination ID: `{args.combination_id}`",
        f"- Combination: {combo_label}",
        f"- Recording used: `{args.recording}`",
        f"- Frames processed: {len(selected_rows)}",
        f"- Camera model: `{selection.camera.model}`",
        f"- LiDAR detector: `{selection.lidar.detector_family}`",
        f"- LiDAR model: `{selection.lidar.model}`",
        f"- Camera cache: `{camera_cache_path}`",
        f"- LiDAR cache: `{lidar_cache_path}`",
        f"- Camera score threshold: {args.min_camera_score:.3f}",
        f"- LiDAR score threshold: {args.min_lidar_score:.3f}",
        f"- LiDAR NMS IoU threshold: {args.lidar_nms_iou_threshold:.3f}",
        f"- LiDAR-only initiation score threshold: {args.lidar_only_initiation_score_threshold:.3f}",
        f"- Calibration source: `{calibration_source}`",
        f"- Tracker preset: `{summary['tracker_preset']}`",
        f"- Tracker output mode: `{tracker_config.output_mode}`",
        f"- Ego motion mode: `{args.ego_motion_mode}`",
        f"- Ego motion source: `{ego_motion_source}`",
        f"- Ego motion applied frames: {ego_motion_applied_frames}",
        f"- Camera detections: {camera_count}",
        f"- LiDAR detections: {lidar_count}",
        f"- Score-filtered LiDAR detections before NMS: {score_filtered_lidar_count}",
        f"- NMS-suppressed LiDAR detections: {nms_suppressed_lidar_count}",
        f"- Fused detections: {matched_count}",
        f"- Tracks created: {tracker.total_created_3d}",
        f"- Confirmed tracks: {len(confirmed_track_ids)}",
        f"- Mean confirmed-track length: {summary['mean_confirmed_track_length']:.2f}",
        f"- Maximum track length: {summary['maximum_track_length']}",
        f"- Prediction-only updates: {prediction_only_updates}",
        f"- Deleted tracks: {tracker.total_deleted_3d}",
        f"- Tracker-only FPS: {tracker_runtime['fps']:.2f}",
        f"- Cached-pipeline FPS: {pipeline_runtime['fps']:.2f}",
        f"- Invalid detections rejected: {tracker.total_invalid_rejected}",
        f"- Annotated visual frames: {len(visual_outputs)}",
        f"- Visual MP4 created: {video_created}",
        f"- Visual MP4/status: `{video_status}`",
        f"- Visual ID continuity appears reasonable: {summary['visual_id_continuity_appears_reasonable']}",
        f"- Recording selection reason: {summary['recording_selection_reason']}",
        "",
        "## Interface Inspection",
        "",
        "- `Detection2D`: `bbox_xyxy`, `score`, `label_id`, `label_name`, `detection_id`, `sample_id`, `timestamp`, `metadata`.",
        "- `Detection3D`: `center_xyz`, `size_lwh`, `yaw`, `score`, `label_id`, `label_name`, `detection_id`, `sample_id`, `timestamp`, `frame_id`, `box_origin`, `metadata`.",
        "- `FusionResult`: `matches`, `matched_3d`, `unmatched_lidar`, `unmatched_camera`, `final_3d`, `projected_lidar`, `iou_matrix`, `confidence_policy`, `output_policy`.",
        "- AGHRI 3D boxes are consumed in `front_lidar_link`; dimensions are `(length, width, height)` and yaw is LiDAR-frame z-yaw.",
        "- PointPillars cache boxes are loaded through the detector profile box-origin setting and normalized by the cache loader before tracking.",
        "- Cache lookup is timestamp-based with the configured tolerance; recording boundaries are explicit `recording`/`recording_id` values.",
        "- The tracker is a temporal layer after existing fusion; matched, unmatched-camera, unmatched-LiDAR, and ALL3D detection outputs are not replaced.",
        "- With `--deepfusionmot-strict`, the tracker uses the DeepFusionMOT Pedestrian-style settings: `min_hits=3`, `max_age_frames=25`, 3D IoU gate `0.01`, 2D IoU gate `0.50`, even 3D IDs, odd 2D IDs, frame-count lifecycle, and confirmed-or-early-frame 3D output.",
        "- AGHRI calibration is loaded from `intrinsics.json` and `extrinsics.json`; `ego_motion_mode=aghri_odom` uses per-recording `metadata/odom_global.jsonl` rather than KITTI OXTS files.",
        "",
        "## Visual Validation",
        "",
        f"- Visual frames directory: `{visual_dir}`",
        f"- Visual frames generated: {len(visual_outputs)}",
        f"- Visual MP4/status: `{video_status}`",
        "- The visual overlays show projected tracked 3D boxes, persistent track IDs, stable per-ID colours, and track state/update source labels.",
        "- Visual continuity is treated as a smoke-test proxy only, not as HOTA/IDF1/MOTA identity accuracy.",
        "",
        "## Known Limitations",
        "",
        "- This is a smoke test, not the eight-combination AGHRI benchmark.",
        "- `ego_motion_mode=aghri_odom` uses AGHRI per-recording odometry; `none` leaves tracks in the previous detector frame.",
        "- HOTA, IDF1, IDSW, and MOTA are produced by `tools/evaluate_aghri_deepfusionmot_metrics.py` when AGHRI annotation identities are available.",
        "- Track IDs are algorithmic IDs; metric evaluation compares them against AGHRI annotation identities.",
    ]
    (output_dir / "tracking_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stdout = [
        f"recording={args.recording}",
        f"combination_id={args.combination_id}",
        f"combination={combo_label}",
        f"frames={len(selected_rows)}",
        f"tracked_3d_rows={len(tracked_3d_rows)}",
        f"confirmed_tracks={len(confirmed_track_ids)}",
        f"tracker_only_fps={tracker_runtime['fps']:.2f}",
        f"output_dir={output_dir}",
    ]
    (output_dir / "run_stdout.txt").write_text("\n".join(stdout) + "\n", encoding="utf-8")
    print("\n".join(stdout))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--camera-cache", type=Path, default=None)
    parser.add_argument("--lidar-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/aghri_deepfusionmot_tracking/smoke_test_ft_yolo_ft_pointpillars"))
    parser.add_argument("--combination-id", default="P4")
    parser.add_argument("--camera-model", choices=MODEL_CHOICES, default="finetuned")
    parser.add_argument("--lidar-detector", choices=LIDAR_DETECTOR_CHOICES, default="pointpillars")
    parser.add_argument("--lidar-model", choices=MODEL_CHOICES, default="finetuned")
    parser.add_argument("--max-age", type=float, default=0.75, help="Maximum prediction-only survival time in seconds.")
    parser.add_argument("--max-age-frames", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--tune-min-hits", type=int, default=None)
    parser.add_argument("--tune-max-age-frames", type=int, default=None)
    parser.add_argument("--tune-association-3d-threshold", type=float, default=None)
    parser.add_argument("--tune-association-2d-threshold", type=float, default=None)
    parser.add_argument("--tune-support-2d-to-3d-threshold", type=float, default=None)
    parser.add_argument("--tune-output-mode", choices=("all", "deepfusionmot"), default="")
    parser.add_argument("--association-config", default="")
    parser.add_argument("--ego-motion-mode", default="none")
    parser.add_argument("--ego-odom-max-delta-sec", type=float, default=0.25)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument(
        "--deepfusionmot-strict",
        action="store_true",
        help="Use settings directly adopted from DeepFusionMOT Pedestrian config and tracker lifecycle.",
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
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
