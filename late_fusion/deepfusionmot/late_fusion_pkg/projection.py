"""Camera projection and LiDAR 3D-box geometry utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from late_fusion_pkg.detection_types import Detection3D, ProjectedDetection3D


def yaw_rotation_z(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def lidar_box_corners(box: Detection3D) -> np.ndarray:
    """Return 8 LiDAR-frame corners for a box with explicit dimension order.

    `size_lwh` is always `(length, width, height)`. For `bottom_center`,
    `center_xyz[2]` is the bottom face z. For `center`, it is the geometric
    center z.
    """
    length, width, height = [float(v) for v in box.size_lwh]
    l2 = length / 2.0
    w2 = width / 2.0
    if box.box_origin == "bottom_center":
        z_values = (0.0, height)
    elif box.box_origin == "center":
        z_values = (-height / 2.0, height / 2.0)
    else:
        raise ValueError(f"Unsupported box_origin: {box.box_origin}")

    local = np.array(
        [
            [-l2, -w2, z_values[0]],
            [l2, -w2, z_values[0]],
            [l2, w2, z_values[0]],
            [-l2, w2, z_values[0]],
            [-l2, -w2, z_values[1]],
            [l2, -w2, z_values[1]],
            [l2, w2, z_values[1]],
            [-l2, w2, z_values[1]],
        ],
        dtype=float,
    )
    rotated = local @ yaw_rotation_z(box.yaw).T
    return rotated + np.asarray(box.center_xyz, dtype=float)


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    transform = np.asarray(transform_4x4, dtype=float).reshape(4, 4)
    hom = np.column_stack([points, np.ones((points.shape[0],), dtype=float)])
    return (transform @ hom.T).T[:, :3]


def project_camera_points(points_camera: np.ndarray, camera_p_3x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera, dtype=float)
    p = np.asarray(camera_p_3x4, dtype=float).reshape(3, 4)
    hom = np.column_stack([points, np.ones((points.shape[0],), dtype=float)])
    uvw = (p @ hom.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv


def clip_xyxy(
    xyxy: Iterable[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(float(image_width - 1), x1))
    x2 = max(0.0, min(float(image_width - 1), x2))
    y1 = max(0.0, min(float(image_height - 1), y1))
    y2 = max(0.0, min(float(image_height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def project_lidar_box(
    box: Detection3D,
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    image_width: int,
    image_height: int,
    min_camera_depth: float = 1e-3,
) -> ProjectedDetection3D:
    corners_lidar = lidar_box_corners(box)
    corners_camera = transform_points(corners_lidar, camera_from_lidar_4x4)
    if np.all(corners_camera[:, 2] <= min_camera_depth):
        return ProjectedDetection3D(
            detection=box,
            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            valid=False,
            reason="behind_camera",
            corners_camera=tuple(map(tuple, corners_camera.tolist())),
        )

    # Use corners with positive camera depth. This keeps partially visible boxes
    # instead of requiring every corner to be inside the image.
    depth_mask = corners_camera[:, 2] > min_camera_depth
    uv = project_camera_points(corners_camera[depth_mask], camera_p_3x4)
    finite_mask = np.isfinite(uv).all(axis=1)
    uv = uv[finite_mask]
    if uv.shape[0] == 0:
        return ProjectedDetection3D(
            detection=box,
            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            valid=False,
            reason="no_projectable_corners",
            corners_camera=tuple(map(tuple, corners_camera.tolist())),
        )

    xyxy = (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))
    clipped = clip_xyxy(xyxy, image_width=image_width, image_height=image_height)
    if clipped is None:
        return ProjectedDetection3D(
            detection=box,
            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            valid=False,
            reason="outside_image",
            corners_camera=tuple(map(tuple, corners_camera.tolist())),
            corners_image=tuple(map(tuple, uv.tolist())),
        )

    return ProjectedDetection3D(
        detection=box,
        bbox_xyxy=clipped,
        valid=True,
        corners_camera=tuple(map(tuple, corners_camera.tolist())),
        corners_image=tuple(map(tuple, uv.tolist())),
    )

