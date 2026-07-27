"""Timestamp-aware constant-velocity motion models for AGHRI tracking."""

from __future__ import annotations

import math

import numpy as np

from late_fusion_pkg.detection_types import Detection2D, Detection3D


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def angle_delta(current: float, previous: float) -> float:
    """Return the shortest signed angle from previous to current."""

    return wrap_angle(float(current) - float(previous))


def safe_dt(timestamp: float | None, previous_timestamp: float | None, *, max_dt: float = 1.0) -> float:
    """Return a deterministic non-negative dt for duplicate/missing/large gaps."""

    if timestamp is None or previous_timestamp is None:
        return 0.0
    delta = float(timestamp) - float(previous_timestamp)
    if not math.isfinite(delta) or delta <= 0.0:
        return 0.0
    return min(delta, float(max_dt))


def detection2d_measurement(det: Detection2D) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in det.bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return np.asarray([x1 + width / 2.0, y1 + height / 2.0, width, height], dtype=float)


def measurement_to_xyxy(measurement: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, width, height = [float(value) for value in measurement[:4]]
    width = max(0.0, width)
    height = max(0.0, height)
    return (cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def detection3d_measurement(det: Detection3D) -> np.ndarray:
    return np.asarray(
        [
            float(det.center_xyz[0]),
            float(det.center_xyz[1]),
            float(det.center_xyz[2]),
            float(det.size_lwh[0]),
            float(det.size_lwh[1]),
            float(det.size_lwh[2]),
            wrap_angle(det.yaw),
        ],
        dtype=float,
    )


def measurement_to_detection3d(measurement: np.ndarray, template: Detection3D, timestamp: float | None) -> Detection3D:
    values = [float(value) for value in measurement[:7]]
    return Detection3D(
        center_xyz=(values[0], values[1], values[2]),
        size_lwh=(max(1e-6, values[3]), max(1e-6, values[4]), max(1e-6, values[5])),
        yaw=wrap_angle(values[6]),
        score=float(template.score),
        label_id=template.label_id,
        label_name=template.label_name,
        detection_id=template.detection_id,
        sample_id=template.sample_id,
        timestamp=timestamp,
        frame_id=template.frame_id,
        box_origin=template.box_origin,
        metadata=dict(template.metadata),
    )


class ConstantVelocityKalman2D:
    """Linear Kalman filter over [cx, cy, w, h] and their velocities."""

    def __init__(self, detection: Detection2D):
        measurement = detection2d_measurement(detection)
        self.x = np.zeros(8, dtype=float)
        self.x[:4] = measurement
        self.p = np.eye(8, dtype=float)
        self.p[4:, 4:] *= 100.0
        self.q = 0.05
        self.r = np.diag([8.0, 8.0, 12.0, 12.0])

    def predict(self, dt: float) -> tuple[float, float, float, float]:
        dt = max(0.0, float(dt))
        f = np.eye(8, dtype=float)
        for idx in range(4):
            f[idx, idx + 4] = dt
        q = np.eye(8, dtype=float) * self.q * max(dt, 1e-3)
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q
        self.x[2:4] = np.maximum(self.x[2:4], 1e-6)
        return measurement_to_xyxy(self.x)

    def update(self, detection: Detection2D) -> tuple[float, float, float, float]:
        z = detection2d_measurement(detection)
        h = np.zeros((4, 8), dtype=float)
        h[:4, :4] = np.eye(4, dtype=float)
        innovation = z - h @ self.x
        s = h @ self.p @ h.T + self.r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.p = (np.eye(8, dtype=float) - k @ h) @ self.p
        self.x[2:4] = np.maximum(self.x[2:4], 1e-6)
        return measurement_to_xyxy(self.x)


class ConstantVelocityKalman3D:
    """Linear Kalman filter over AGHRI LiDAR boxes and centre/yaw velocity."""

    def __init__(self, detection: Detection3D):
        measurement = detection3d_measurement(detection)
        self.x = np.zeros(11, dtype=float)
        self.x[:7] = measurement
        self.p = np.eye(11, dtype=float)
        self.p[7:, 7:] *= 100.0
        self.q = 0.02
        self.r = np.diag([0.08, 0.08, 0.08, 0.06, 0.06, 0.10, 0.20])

    def predict(self, dt: float, template: Detection3D, timestamp: float | None) -> Detection3D:
        dt = max(0.0, float(dt))
        f = np.eye(11, dtype=float)
        f[0, 7] = dt
        f[1, 8] = dt
        f[2, 9] = dt
        f[6, 10] = dt
        q = np.eye(11, dtype=float) * self.q * max(dt, 1e-3)
        self.x = f @ self.x
        self.x[6] = wrap_angle(self.x[6])
        self.p = f @ self.p @ f.T + q
        self.x[3:6] = np.maximum(self.x[3:6], 1e-6)
        return measurement_to_detection3d(self.x, template, timestamp)

    def apply_ego_motion(self, transform: np.ndarray, template: Detection3D, timestamp: float | None) -> Detection3D:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("ego-motion transform must have shape (4, 4)")
        center = np.asarray([self.x[0], self.x[1], self.x[2], 1.0], dtype=float)
        transformed = matrix @ center
        rotation = matrix[:3, :3]
        yaw_delta = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        self.x[:3] = transformed[:3]
        self.x[6] = wrap_angle(self.x[6] + yaw_delta)
        self.x[7:10] = rotation @ self.x[7:10]
        return measurement_to_detection3d(self.x, template, timestamp)

    def update(self, detection: Detection3D) -> Detection3D:
        z = detection3d_measurement(detection)
        h = np.zeros((7, 11), dtype=float)
        h[:7, :7] = np.eye(7, dtype=float)
        prediction = h @ self.x
        innovation = z - prediction
        innovation[6] = angle_delta(z[6], prediction[6])
        s = h @ self.p @ h.T + self.r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innovation
        self.x[6] = wrap_angle(self.x[6])
        self.x[3:6] = np.maximum(self.x[3:6], 1e-6)
        self.p = (np.eye(11, dtype=float) - k @ h) @ self.p
        return measurement_to_detection3d(self.x, detection, detection.timestamp)


def valid_detection3d(det: Detection3D) -> bool:
    values = [*det.center_xyz, *det.size_lwh, det.yaw, det.score]
    return all(math.isfinite(float(value)) for value in values) and min(float(value) for value in det.size_lwh) > 0.0


def valid_detection2d(det: Detection2D) -> bool:
    x1, y1, x2, y2 = [float(value) for value in det.bbox_xyxy]
    values = [x1, y1, x2, y2, det.score]
    return all(math.isfinite(float(value)) for value in values) and x2 > x1 and y2 > y1
