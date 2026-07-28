"""Typed records for the native offline DeepFusionMOT-style tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from late_fusion_pkg.detection_types import Detection2D, Detection3D


class TrackState(str, Enum):
    """Lifecycle state for a 2D or 3D track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"
    REACTIVATE = "reactivate"


class UpdateSource(str, Enum):
    """Source of the most recent track update."""

    MATCHED_CAMERA_LIDAR = "matched_camera_lidar"
    LIDAR_ONLY = "lidar_only"
    CAMERA_ONLY_SUPPORT = "camera_only_support"
    PREDICTION_ONLY = "prediction_only"


@dataclass
class Track2D:
    track_id: int
    recording_id: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    timestamp: float | None
    state: TrackState = TrackState.TENTATIVE
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    time_since_update: float = 0.0
    update_source: UpdateSource = UpdateSource.CAMERA_ONLY_SUPPORT
    motion: Any = None
    detection: Detection2D | None = None


@dataclass
class Track3D:
    track_id: int
    recording_id: str
    detection: Detection3D
    timestamp: float | None
    state: TrackState = TrackState.TENTATIVE
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    time_since_update: float = 0.0
    update_source: UpdateSource = UpdateSource.LIDAR_ONLY
    projected_bbox_xyxy: tuple[float, float, float, float] | None = None
    motion: Any = None


@dataclass(frozen=True)
class TrackedDetection2D:
    recording_id: str
    frame_index: int
    timestamp: float | None
    track_id: int
    track_state: str
    update_source: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    age: int
    hits: int
    time_since_update: float


@dataclass(frozen=True)
class TrackedDetection3D:
    recording_id: str
    frame_index: int
    timestamp: float | None
    track_id: int
    track_state: str
    update_source: str
    detection: Detection3D
    age: int
    hits: int
    time_since_update: float
    projected_bbox_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class TrackingDiagnostics:
    recording_id: str
    frame_index: int
    timestamp: float | None
    camera_detection_count: int
    lidar_detection_count: int
    matched_detection_count: int
    camera_only_count: int
    lidar_only_count: int
    active_2d_tracks: int
    active_3d_tracks: int
    new_tracks: int
    confirmed_tracks: int
    predicted_only_tracks: int
    deleted_tracks: int
    invalid_detections_rejected: int
    association_time_ms: float
    tracker_update_time_ms: float
    dt: float
    reset: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackingFrameResult:
    recording_id: str
    frame_index: int
    timestamp: float | None
    tracked_3d: tuple[TrackedDetection3D, ...]
    tracked_2d: tuple[TrackedDetection2D, ...]
    diagnostics: TrackingDiagnostics

