"""Helpers for AGHRI late-fusion image and marker visualisation."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable

import numpy as np

from late_fusion_pkg.detection_types import Detection2D, Detection3D
from late_fusion_pkg.projection import lidar_box_corners, project_camera_points, transform_points


CUBOID_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def stable_marker_id(index: int, text: bool = False) -> int:
    return int(index) * 2 + (1 if text else 0)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def interpolate_yaw(previous: float, current: float, alpha: float) -> float:
    """Interpolate along the shortest angular path."""
    return wrap_angle(float(previous) + float(alpha) * wrap_angle(float(current) - float(previous)))


class VisualBoxSmoother:
    """Nearest-centre EMA used only by visual outputs, never detection topics."""

    def __init__(self, alpha: float = 0.35, max_gap_sec: float = 0.20, max_distance_m: float = 1.5):
        self.alpha = float(alpha)
        self.max_gap_sec = float(max_gap_sec)
        self.max_distance_m = float(max_distance_m)
        self._previous: list[Detection3D] = []
        self._stamp: float | None = None

    def reset(self) -> None:
        self._previous = []
        self._stamp = None

    def update(self, detections: list[Detection3D], stamp: float) -> list[Detection3D]:
        stamp = float(stamp)
        if self._stamp is None or stamp <= self._stamp or stamp - self._stamp > self.max_gap_sec:
            self._previous = list(detections)
            self._stamp = stamp
            return list(detections)

        available = set(range(len(self._previous)))
        smoothed: list[Detection3D] = []
        for current in detections:
            best = None
            best_distance = math.inf
            current_center = np.asarray(current.center_xyz, dtype=float)
            for index in available:
                distance = float(np.linalg.norm(current_center - np.asarray(self._previous[index].center_xyz, dtype=float)))
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is None or best_distance > self.max_distance_m:
                smoothed.append(current)
                continue
            available.remove(best)
            previous = self._previous[best]
            alpha = self.alpha
            center = tuple((1.0 - alpha) * np.asarray(previous.center_xyz) + alpha * np.asarray(current.center_xyz))
            size = tuple((1.0 - alpha) * np.asarray(previous.size_lwh) + alpha * np.asarray(current.size_lwh))
            smoothed.append(
                replace(
                    current,
                    center_xyz=tuple(float(value) for value in center),
                    size_lwh=tuple(float(value) for value in size),
                    yaw=interpolate_yaw(previous.yaw, current.yaw, alpha),
                )
            )
        self._previous = smoothed
        self._stamp = stamp
        return list(smoothed)


def color_rgba(name: str) -> tuple[float, float, float, float]:
    colors = {
        "blue": (0.1, 0.35, 1.0, 0.95),
        "green": (0.0, 0.9, 0.25, 0.95),
        "yellow": (1.0, 0.82, 0.0, 0.95),
        "cyan": (0.0, 0.9, 1.0, 0.9),
        "red": (1.0, 0.05, 0.05, 0.95),
        "magenta": (1.0, 0.0, 1.0, 0.95),
    }
    return colors[name]


def _set_color(color_msg, rgba: Iterable[float]) -> None:
    color_msg.r, color_msg.g, color_msg.b, color_msg.a = [float(v) for v in rgba]


def _set_lifetime(marker, lifetime_sec: float) -> None:
    sec = int(max(0.0, float(lifetime_sec)))
    marker.lifetime.sec = sec
    marker.lifetime.nanosec = int(round((max(0.0, float(lifetime_sec)) - sec) * 1_000_000_000))


def delete_all_marker_array(marker_array_cls, marker_cls, header=None):
    arr = marker_array_cls()
    marker = marker_cls()
    if header is not None:
        marker.header = header
    marker.action = marker_cls.DELETEALL
    arr.markers.append(marker)
    return arr


def detections3d_to_marker_array(
    detections: list[Detection3D],
    header,
    marker_array_cls,
    marker_cls,
    point_cls,
    *,
    box_namespace: str,
    label_namespace: str,
    rgba: tuple[float, float, float, float],
    line_width: float = 0.04,
    lifetime_sec: float = 0.25,
    draw_labels: bool = True,
    label_prefix: str = "person",
):
    arr = marker_array_cls()
    clear = marker_cls()
    clear.header = header
    clear.action = marker_cls.DELETEALL
    arr.markers.append(clear)
    for idx, det in enumerate(detections):
        corners = lidar_box_corners(det)
        marker = marker_cls()
        marker.header = header
        marker.ns = box_namespace
        marker.id = stable_marker_id(idx)
        marker.type = marker_cls.LINE_LIST
        marker.action = marker_cls.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(line_width)
        _set_color(marker.color, rgba)
        _set_lifetime(marker, lifetime_sec)
        for a, b in CUBOID_EDGES:
            for corner_idx in (a, b):
                point = point_cls()
                point.x, point.y, point.z = [float(v) for v in corners[corner_idx]]
                marker.points.append(point)
        arr.markers.append(marker)
        if draw_labels:
            label = marker_cls()
            label.header = header
            label.ns = label_namespace
            label.id = stable_marker_id(idx, text=True)
            label.type = marker_cls.TEXT_VIEW_FACING
            label.action = marker_cls.ADD
            label.pose.position.x = float(det.center_xyz[0])
            label.pose.position.y = float(det.center_xyz[1])
            label.pose.position.z = float(det.center_xyz[2] + det.size_lwh[2] + 0.25)
            label.pose.orientation.w = 1.0
            label.scale.z = 0.35
            _set_color(label.color, rgba)
            _set_lifetime(label, lifetime_sec)
            if label_prefix == "matched":
                label.text = f"matched#{idx + 1}:{float(det.score):.2f}"
            elif label_prefix == "track":
                label.text = f"track#{det.detection_id or idx + 1}:{float(det.score):.2f}"
            else:
                label.text = f"{label_prefix}: {float(det.score):.2f}"
            arr.markers.append(label)
    return arr


def clip_point_to_image(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    x = int(round(max(0.0, min(float(width - 1), float(point[0])))))
    y = int(round(max(0.0, min(float(height - 1), float(point[1])))))
    return x, y


def _clip_line_to_image(
    start: tuple[float, float], end: tuple[float, float], width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Liang-Barsky clip, preserving crossing segments and rejecting off-screen ones."""
    x0, y0 = [float(value) for value in start]
    x1, y1 = [float(value) for value in end]
    if not np.isfinite([x0, y0, x1, y1]).all():
        return None
    dx, dy = x1 - x0, y1 - y0
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, width - 1 - x0), (-dy, y0), (dy, height - 1 - y0)):
        if abs(p) < 1e-12:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    clipped_start = (x0 + lower * dx, y0 + lower * dy)
    clipped_end = (x0 + upper * dx, y0 + upper * dy)
    p0 = clip_point_to_image(clipped_start, width, height)
    p1 = clip_point_to_image(clipped_end, width, height)
    return None if p0 == p1 else (p0, p1)


def project_box_wireframe(
    det: Detection3D,
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    image_width: int,
    image_height: int,
    min_camera_depth: float = 1e-3,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], str]:
    try:
        corners_lidar = lidar_box_corners(det)
    except (ValueError, TypeError):
        return [], "invalid_geometry"
    if not np.isfinite(corners_lidar).all() or min(det.size_lwh) <= 0.0:
        return [], "invalid_geometry"
    corners_camera = transform_points(corners_lidar, camera_from_lidar_4x4)
    lines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if np.all(corners_camera[:, 2] <= min_camera_depth):
        return lines, "behind_camera"
    for a, b in CUBOID_EDGES:
        start = corners_camera[a].copy()
        end = corners_camera[b].copy()
        if start[2] <= min_camera_depth and end[2] <= min_camera_depth:
            continue
        if start[2] <= min_camera_depth or end[2] <= min_camera_depth:
            behind, ahead = (start, end) if start[2] <= min_camera_depth else (end, start)
            fraction = (min_camera_depth - behind[2]) / (ahead[2] - behind[2])
            clipped = behind + fraction * (ahead - behind)
            clipped[2] = min_camera_depth
            if start[2] <= min_camera_depth:
                start = clipped
            else:
                end = clipped
        uv = project_camera_points(np.vstack([start, end]), camera_p_3x4)
        segment = _clip_line_to_image(tuple(uv[0]), tuple(uv[1]), image_width, image_height)
        if segment is not None:
            lines.append(segment)
    return lines, "ok" if lines else "outside_image"


def projected_vs_yolo_metrics(
    lidar: Detection3D,
    camera: Detection2D,
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, float | str]:
    from late_fusion_pkg.fusion_core import iou_2d
    from late_fusion_pkg.projection import project_lidar_box

    projected = project_lidar_box(lidar, camera_p_3x4, camera_from_lidar_4x4, image_width, image_height)
    if not projected.valid:
        return {"status": projected.reason}
    px1, py1, px2, py2 = projected.bbox_xyxy
    yx1, yy1, yx2, yy2 = camera.bbox_xyxy
    projected_width, projected_height = px2 - px1, py2 - py1
    yolo_width, yolo_height = yx2 - yx1, yy2 - yy1
    centre_offset = math.hypot((px1 + px2 - yx1 - yx2) / 2.0, (py1 + py2 - yy1 - yy2) / 2.0)
    corners_camera = np.asarray(projected.corners_camera, dtype=float)
    return {
        "status": "ok",
        "iou": iou_2d(projected.bbox_xyxy, camera.bbox_xyxy),
        "centre_offset_px": centre_offset,
        "height_ratio": projected_height / yolo_height if yolo_height > 0.0 else math.nan,
        "width_ratio": projected_width / yolo_width if yolo_width > 0.0 else math.nan,
        "camera_depth_m": float(np.mean(corners_camera[:, 2])),
        "lidar_range_m": math.hypot(float(lidar.center_xyz[0]), float(lidar.center_xyz[1])),
        "projected_x1": px1,
        "projected_y1": py1,
        "projected_x2": px2,
        "projected_y2": py2,
    }


def draw_detection2d_array(image_bgr: np.ndarray, detections: list[Detection2D], color_bgr: tuple[int, int, int], *, draw_labels: bool = True, line_width: int = 2):
    import cv2

    out = image_bgr.copy()
    height, width = out.shape[:2]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy
        p1 = clip_point_to_image((x1, y1), width, height)
        p2 = clip_point_to_image((x2, y2), width, height)
        cv2.rectangle(out, p1, p2, color_bgr, int(line_width))
        if draw_labels:
            label = f"{det.label_name or det.label_id}: {float(det.score):.2f} #{idx}"
            cv2.putText(out, label, (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
    return out


def draw_projected_wireframes(
    image_bgr: np.ndarray,
    detections: list[Detection3D],
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    color_bgr: tuple[int, int, int],
    *,
    draw_labels: bool = True,
    draw_envelope: bool = False,
    timestamp_delta_sec: float | None = None,
    line_width: int = 2,
    label_prefix: str = "matched",
) -> tuple[np.ndarray, dict[str, int]]:
    import cv2

    out = image_bgr.copy()
    height, width = out.shape[:2]
    stats: dict[str, int] = {
        "drawn": 0,
        "behind_camera": 0,
        "outside_image": 0,
        "invalid_geometry": 0,
    }
    for idx, det in enumerate(detections):
        lines, reason = project_box_wireframe(det, camera_p_3x4, camera_from_lidar_4x4, width, height)
        if not lines:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        for p1, p2 in lines:
            cv2.line(out, p1, p2, color_bgr, int(line_width), cv2.LINE_AA)
        points = [point for line in lines for point in line]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        envelope = (min(x_values), min(y_values), max(x_values), max(y_values))
        if draw_envelope:
            x1, y1, x2, y2 = envelope
            dash = 8
            for x in range(x1, x2 + 1, dash * 2):
                cv2.line(out, (x, y1), (min(x + dash, x2), y1), color_bgr, 1)
                cv2.line(out, (x, y2), (min(x + dash, x2), y2), color_bgr, 1)
            for y in range(y1, y2 + 1, dash * 2):
                cv2.line(out, (x1, y), (x1, min(y + dash, y2)), color_bgr, 1)
                cv2.line(out, (x2, y), (x2, min(y + dash, y2)), color_bgr, 1)
        if draw_labels:
            p1 = (envelope[0], envelope[1])
            if label_prefix == "track":
                label = f"track #{det.detection_id or idx + 1}: {float(det.score):.2f}"
            else:
                label = f"{label_prefix} #{idx + 1}: {float(det.score):.2f}"
            cv2.putText(out, label, (p1[0], max(12, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
        stats["drawn"] += 1
    return out, stats
