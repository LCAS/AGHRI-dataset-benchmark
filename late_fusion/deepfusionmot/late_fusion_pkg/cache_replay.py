"""Timestamp-indexed detector cache loading for offline ROS replay."""

from __future__ import annotations

import csv
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

from late_fusion_pkg.detection_types import Detection2D, Detection3D


@dataclass(frozen=True)
class CacheValidation:
    path: str
    rows: int
    recordings: tuple[str, ...]
    unique_timestamps: int
    duplicate_rows: int
    invalid_rows: int
    finite_scores: bool
    frame_ids: tuple[str, ...]


@dataclass
class TimestampCache:
    detections: dict[tuple[str, float], tuple[Detection2D | Detection3D, ...]]
    validation: CacheValidation
    timestamp_tolerance_sec: float = 0.02

    def __post_init__(self) -> None:
        by_recording: dict[str, list[float]] = {}
        for recording, stamp in self.detections:
            by_recording.setdefault(recording, []).append(stamp)
        self._timestamps = {
            recording: tuple(sorted(stamps))
            for recording, stamps in by_recording.items()
        }

    def lookup(self, recording: str, timestamp: float) -> tuple[Detection2D | Detection3D, ...]:
        stamps = self._timestamps.get(recording, ())
        if not stamps:
            return ()
        pos = bisect_left(stamps, float(timestamp))
        candidates = []
        if pos < len(stamps):
            candidates.append(stamps[pos])
        if pos > 0:
            candidates.append(stamps[pos - 1])
        if not candidates:
            return ()
        best = min(candidates, key=lambda item: abs(item - float(timestamp)))
        if abs(best - float(timestamp)) > self.timestamp_tolerance_sec:
            return ()
        return self.detections.get((recording, best), ())

    def contains_timestamp(self, recording: str, timestamp: float) -> bool:
        stamps = self._timestamps.get(recording, ())
        if not stamps:
            return False
        pos = bisect_left(stamps, float(timestamp))
        candidates = stamps[max(0, pos - 1) : min(len(stamps), pos + 1)]
        return bool(candidates) and min(abs(item - float(timestamp)) for item in candidates) <= self.timestamp_tolerance_sec


def _recording_from_sample(sample_id: str) -> str:
    parts = str(sample_id).split("/")
    if len(parts) >= 4 and parts[0] == "kitti_tracking":
        return "/".join(parts[:3])
    return str(sample_id).split("/", 1)[0]


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _dedupe(items):
    seen = set()
    out = []
    duplicates = 0
    for item in items:
        if item[0] in seen:
            duplicates += 1
            continue
        seen.add(item[0])
        out.append(item[1])
    return tuple(out), duplicates


def load_yolo_cache(path: str | Path, timestamp_tolerance_sec: float = 0.02) -> TimestampCache:
    cache_path = Path(path)
    grouped: dict[tuple[str, float], list[tuple[tuple, Detection2D]]] = {}
    recordings = set()
    invalid_rows = 0
    finite_scores = True
    rows = 0
    duplicate_rows = 0
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            try:
                recording = _recording_from_sample(row["sample_id"])
                timestamp = float(row["timestamp"])
                if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"} or row.get("class_name") == "__EMPTY__":
                    recordings.add(recording)
                    grouped.setdefault((recording, timestamp), [])
                    continue
                score = float(row["confidence"])
                values = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
                if not _finite(values + [score, timestamp]):
                    raise ValueError("non-finite cache value")
            except Exception:
                invalid_rows += 1
                finite_scores = False
                continue
            recordings.add(recording)
            key = (
                round(values[0], 4),
                round(values[1], 4),
                round(values[2], 4),
                round(values[3], 4),
                round(score, 6),
                int(row.get("class_id") or 0),
            )
            det = Detection2D(
                bbox_xyxy=tuple(values),
                score=score,
                label_id=int(row.get("class_id") or 0),
                label_name=row.get("class_name") or "person",
                detection_id=f"cached_yolo_{rows}",
                sample_id=row.get("sample_id", ""),
                timestamp=timestamp,
                metadata={"source": "cached_yolo", "image_path": row.get("image_path", "")},
            )
            grouped.setdefault((recording, timestamp), []).append((key, det))
    detections = {}
    for key, items in grouped.items():
        deduped, count = _dedupe(items)
        duplicate_rows += count
        detections[key] = deduped
    validation = CacheValidation(
        path=str(cache_path),
        rows=rows,
        recordings=tuple(sorted(recordings)),
        unique_timestamps=len(detections),
        duplicate_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        finite_scores=finite_scores,
        frame_ids=("source_image_header",),
    )
    return TimestampCache(detections=detections, validation=validation, timestamp_tolerance_sec=timestamp_tolerance_sec)


def load_second_cache(
    path: str | Path,
    timestamp_tolerance_sec: float = 0.02,
    box_origin: str = "bottom_center",
    frame_id: str = "front_lidar_link",
) -> TimestampCache:
    if box_origin not in {"bottom_center", "center"}:
        raise ValueError(f"Unsupported SECOND cache box_origin: {box_origin}")
    cache_path = Path(path)
    grouped: dict[tuple[str, float], list[tuple[tuple, Detection3D]]] = {}
    recordings = set()
    invalid_rows = 0
    finite_scores = True
    rows = 0
    duplicate_rows = 0
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            if str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"} or row.get("raw_label_name") == "__EMPTY__":
                try:
                    recording = _recording_from_sample(row["sample_id"])
                    timestamp = float(row["timestamp"])
                except Exception:
                    invalid_rows += 1
                    finite_scores = False
                    continue
                recordings.add(recording)
                grouped.setdefault((recording, timestamp), [])
                continue
            if str(row.get("selected_pedestrian", "")).lower() not in {"true", "1", "yes"}:
                continue
            try:
                recording = _recording_from_sample(row["sample_id"])
                timestamp = float(row["timestamp"])
                score = float(row["score"])
                center = [float(row[key]) for key in ("x", "y", "z")]
                size = [float(row[key]) for key in ("length", "width", "height")]
                yaw = float(row["yaw"])
                if not _finite(center + size + [yaw, score, timestamp]) or min(size) <= 0:
                    raise ValueError("invalid SECOND cache value")
            except Exception:
                invalid_rows += 1
                finite_scores = False
                continue
            recordings.add(recording)
            key = tuple(round(value, 5) for value in center + size + [yaw, score])
            det = Detection3D(
                center_xyz=(center[0], center[1], center[2] - size[2] / 2.0 if box_origin == "center" else center[2]),
                size_lwh=tuple(size),
                yaw=yaw,
                score=score,
                label_id=int(row.get("raw_label_id") or 0),
                label_name=row.get("raw_label_name") or "Pedestrian",
                detection_id=f"cached_second_{rows}",
                sample_id=row.get("sample_id", ""),
                timestamp=timestamp,
                frame_id=frame_id,
                box_origin="bottom_center",
                metadata={"source": "cached_second", "pointcloud_path": row.get("pointcloud_path", "")},
            )
            grouped.setdefault((recording, timestamp), []).append((key, det))
    detections = {}
    for key, items in grouped.items():
        deduped, count = _dedupe(items)
        duplicate_rows += count
        detections[key] = deduped
    validation = CacheValidation(
        path=str(cache_path),
        rows=rows,
        recordings=tuple(sorted(recordings)),
        unique_timestamps=len(detections),
        duplicate_rows=duplicate_rows,
        invalid_rows=invalid_rows,
        finite_scores=finite_scores,
        frame_ids=(frame_id,),
    )
    return TimestampCache(detections=detections, validation=validation, timestamp_tolerance_sec=timestamp_tolerance_sec)
