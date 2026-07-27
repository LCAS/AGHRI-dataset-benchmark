"""AGHRI odometry-based ego-motion compensation helpers."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AghriPose:
    timestamp: float
    map_from_base: np.ndarray


def _quat_to_rot(q: dict) -> np.ndarray:
    x = float(q["x"])
    y = float(q["y"])
    z = float(q["z"])
    w = float(q["w"])
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero-norm odometry quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_from_pose(row: dict) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_rot(row["q"])
    p = row["p"]
    matrix[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return matrix


def _inverse(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -(rot.T @ trans)
    return out


class AghriOdomEgoMotion:
    """Timestamped AGHRI map-from-LiDAR poses from per-recording odometry."""

    def __init__(self, poses: list[AghriPose], base_from_lidar: np.ndarray, *, max_time_delta_sec: float = 0.25):
        if not poses:
            raise ValueError("AGHRI odometry contains no poses")
        self.poses = sorted(poses, key=lambda item: item.timestamp)
        self.timestamps = [pose.timestamp for pose in self.poses]
        self.base_from_lidar = np.asarray(base_from_lidar, dtype=float)
        self.max_time_delta_sec = float(max_time_delta_sec)

    @classmethod
    def from_recording_dir(
        cls,
        recording_dir: Path,
        base_from_lidar: np.ndarray,
        *,
        odom_file: str = "odom_global.jsonl",
        max_time_delta_sec: float = 0.25,
    ) -> "AghriOdomEgoMotion":
        path = recording_dir / "metadata" / odom_file
        if not path.exists():
            raise FileNotFoundError(f"AGHRI odometry file not found: {path}")
        poses: list[AghriPose] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                poses.append(AghriPose(timestamp=float(row["t"]), map_from_base=_matrix_from_pose(row)))
        return cls(poses, base_from_lidar, max_time_delta_sec=max_time_delta_sec)

    def nearest_pose(self, timestamp: float) -> AghriPose | None:
        if not self.poses:
            return None
        timestamp = float(timestamp)
        idx = bisect_left(self.timestamps, timestamp)
        candidates = []
        if idx < len(self.poses):
            candidates.append(self.poses[idx])
        if idx > 0:
            candidates.append(self.poses[idx - 1])
        if not candidates:
            return None
        pose = min(candidates, key=lambda item: abs(item.timestamp - timestamp))
        if abs(pose.timestamp - timestamp) > self.max_time_delta_sec:
            return None
        return pose

    def map_from_lidar(self, timestamp: float) -> np.ndarray | None:
        pose = self.nearest_pose(timestamp)
        if pose is None:
            return None
        return pose.map_from_base @ self.base_from_lidar

    def relative_lidar_transform(self, previous_timestamp: float | None, current_timestamp: float | None) -> np.ndarray | None:
        if previous_timestamp is None or current_timestamp is None:
            return None
        previous = self.map_from_lidar(float(previous_timestamp))
        current = self.map_from_lidar(float(current_timestamp))
        if previous is None or current is None:
            return None
        return _inverse(current) @ previous
