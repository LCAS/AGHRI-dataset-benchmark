"""I/O and visual evidence helpers for offline AGHRI tracking."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from late_fusion_pkg.tracking_types import TrackedDetection2D, TrackedDetection3D, TrackingDiagnostics
from late_fusion_pkg.visualization_utils import project_box_wireframe


TRACKED_3D_FIELDS = [
    "recording_id",
    "frame_index",
    "timestamp",
    "track_id",
    "track_state",
    "update_source",
    "x",
    "y",
    "z",
    "length",
    "width",
    "height",
    "yaw",
    "score",
    "age",
    "hits",
    "time_since_update",
    "projected_x1",
    "projected_y1",
    "projected_x2",
    "projected_y2",
]

TRACKED_2D_FIELDS = [
    "recording_id",
    "frame_index",
    "timestamp",
    "track_id",
    "track_state",
    "update_source",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "age",
    "hits",
    "time_since_update",
]

DIAGNOSTIC_FIELDS = [
    "recording_id",
    "frame_index",
    "timestamp",
    "camera_detection_count",
    "lidar_detection_count",
    "matched_detection_count",
    "camera_only_count",
    "lidar_only_count",
    "active_2d_tracks",
    "active_3d_tracks",
    "new_tracks",
    "confirmed_tracks",
    "predicted_only_tracks",
    "deleted_tracks",
    "invalid_detections_rejected",
    "association_time_ms",
    "tracker_update_time_ms",
    "dt",
    "reset",
]


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tracked3d_to_row(track: TrackedDetection3D) -> dict:
    det = track.detection
    projected = track.projected_bbox_xyxy or ("", "", "", "")
    return {
        "recording_id": track.recording_id,
        "frame_index": track.frame_index,
        "timestamp": "" if track.timestamp is None else f"{track.timestamp:.9f}",
        "track_id": track.track_id,
        "track_state": track.track_state,
        "update_source": track.update_source,
        "x": det.center_xyz[0],
        "y": det.center_xyz[1],
        "z": det.center_xyz[2],
        "length": det.size_lwh[0],
        "width": det.size_lwh[1],
        "height": det.size_lwh[2],
        "yaw": det.yaw,
        "score": det.score,
        "age": track.age,
        "hits": track.hits,
        "time_since_update": track.time_since_update,
        "projected_x1": projected[0],
        "projected_y1": projected[1],
        "projected_x2": projected[2],
        "projected_y2": projected[3],
    }


def tracked2d_to_row(track: TrackedDetection2D) -> dict:
    x1, y1, x2, y2 = track.bbox_xyxy
    return {
        "recording_id": track.recording_id,
        "frame_index": track.frame_index,
        "timestamp": "" if track.timestamp is None else f"{track.timestamp:.9f}",
        "track_id": track.track_id,
        "track_state": track.track_state,
        "update_source": track.update_source,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": track.score,
        "age": track.age,
        "hits": track.hits,
        "time_since_update": track.time_since_update,
    }


def diagnostics_to_row(diagnostics: TrackingDiagnostics) -> dict:
    row = asdict(diagnostics)
    row.pop("extra", None)
    row["timestamp"] = "" if diagnostics.timestamp is None else f"{diagnostics.timestamp:.9f}"
    return row


def color_for_track(track_id: int) -> tuple[int, int, int]:
    palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
    return tuple(int((channel * (track_id * track_id - track_id + 1)) % 255) for channel in palette)


def draw_tracking_frame(
    *,
    image_path: Path | None,
    output_path: Path,
    tracks: Iterable[TrackedDetection3D],
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    if image_path is not None and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        image = Image.new("RGB", (int(image_width), int(image_height)), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    stats = {"drawn": 0, "not_drawn": 0}
    width, height = image.size
    for track in tracks:
        lines, reason = project_box_wireframe(
            track.detection,
            camera_p_3x4,
            camera_from_lidar_4x4,
            width,
            height,
        )
        color = color_for_track(track.track_id)
        if not lines:
            stats["not_drawn"] += 1
            continue
        for start, end in lines:
            draw.line([start, end], fill=color, width=2)
        points = [point for line in lines for point in line]
        x1 = min(point[0] for point in points)
        y1 = min(point[1] for point in points)
        label = f"track {track.track_id} {track.track_state} {track.update_source}"
        draw.text((x1, max(0, y1 - 14)), label, fill=color)
        stats["drawn"] += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return stats

