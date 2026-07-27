"""Small conversion helpers for live ROS detector nodes."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueueStats:
    received: int = 0
    processed: int = 0
    dropped: int = 0
    failures: int = 0


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def finite_xyzi(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def pointcloud2_to_xyzi(cloud_msg) -> tuple[np.ndarray, int, int]:
    """Convert sensor_msgs/PointCloud2 to finite float32 XYZI points."""

    from sensor_msgs_py import point_cloud2 as pc2

    field_names = [field.name for field in cloud_msg.fields]
    names = ["x", "y", "z"]
    intensity_name = "intensity" if "intensity" in field_names else ("reflectivity" if "reflectivity" in field_names else None)
    if intensity_name:
        names.append(intensity_name)
    raw = pc2.read_points_numpy(cloud_msg, field_names=tuple(names), skip_nans=False)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, len(names))
    raw_count = int(arr.shape[0])
    if intensity_name is None:
        intensity = np.zeros((arr.shape[0], 1), dtype=np.float32)
        arr = np.column_stack([arr[:, :3], intensity])
    else:
        arr = arr[:, :4]
    finite = finite_xyzi(arr)
    return finite, raw_count, int(finite.shape[0])


def encode_points(points: np.ndarray) -> str:
    pts = finite_xyzi(points)
    return base64.b64encode(pts.astype(np.float32, copy=False).tobytes()).decode("ascii")


def decode_points(encoded: str, point_count: int) -> np.ndarray:
    raw = base64.b64decode(encoded.encode("ascii"))
    pts = np.frombuffer(raw, dtype=np.float32)
    if pts.size != int(point_count) * 4:
        raise ValueError(f"Point payload size mismatch: got {pts.size} floats for {point_count} points")
    return pts.reshape(int(point_count), 4).copy()


def latency_ms(start: float, end: float | None = None) -> float:
    end = time.perf_counter() if end is None else end
    return float((end - start) * 1000.0)
