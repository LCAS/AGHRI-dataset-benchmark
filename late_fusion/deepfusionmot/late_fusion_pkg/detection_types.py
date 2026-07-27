"""Typed detection records shared by offline and ROS late-fusion code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Stamp:
    """ROS-like timestamp represented without importing ROS."""

    sec: int = 0
    nanosec: int = 0

    @classmethod
    def from_seconds(cls, value: float) -> "Stamp":
        sec = int(value)
        return cls(sec=sec, nanosec=int(round((value - sec) * 1_000_000_000)))

    @property
    def seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) / 1_000_000_000.0


@dataclass(frozen=True)
class Header:
    stamp: Stamp = field(default_factory=Stamp)
    frame_id: str = ""


@dataclass(frozen=True)
class Detection2D:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    label_id: int
    label_name: str = "person"
    detection_id: str = ""
    sample_id: str = ""
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Detection3D:
    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw: float
    score: float
    label_id: int
    label_name: str = "Pedestrian"
    detection_id: str = ""
    sample_id: str = ""
    timestamp: float | None = None
    frame_id: str = "front_lidar_link"
    box_origin: str = "bottom_center"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectedDetection3D:
    detection: Detection3D
    bbox_xyxy: tuple[float, float, float, float]
    valid: bool
    reason: str = ""
    corners_camera: tuple[tuple[float, float, float], ...] = ()
    corners_image: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class Match:
    camera_index: int
    lidar_index: int
    iou: float
    fused_score: float


@dataclass(frozen=True)
class FusionResult:
    matches: tuple[Match, ...]
    matched_3d: tuple[Detection3D, ...]
    unmatched_lidar: tuple[Detection3D, ...]
    unmatched_camera: tuple[Detection2D, ...]
    final_3d: tuple[Detection3D, ...]
    projected_lidar: tuple[ProjectedDetection3D, ...]
    iou_matrix: tuple[tuple[float, ...], ...]
    confidence_policy: str
    output_policy: str

