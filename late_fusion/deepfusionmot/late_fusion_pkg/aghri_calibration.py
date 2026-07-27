"""AGHRI calibration loading with CameraInfo.p and camera-from-LiDAR TF semantics."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import zipfile

import numpy as np


def _load_json(path: Path, member: str | None = None) -> dict:
    if path.suffix.lower() == ".zip":
        if member is None:
            raise ValueError("ZIP calibration requires member path")
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as stream:
                return json.loads(stream.read().decode("utf-8"))
    if member and path.is_dir():
        target = path / member
        if not target.exists() and path.name == "calibration" and member.startswith("calibration/"):
            target = path / member.split("/", 1)[1]
    else:
        target = path
    return json.loads(target.read_text(encoding="utf-8"))


def camera_projection_from_intrinsics(
    intrinsics: dict,
    camera_name: str = "cam_zed_rgb",
    matrix_source: str = "p",
) -> tuple[np.ndarray, int, int, str]:
    camera = intrinsics[camera_name]
    frame_id = (camera.get("header") or {}).get("frame_id", "front_left_camera_optical_frame")
    width = int(camera.get("width", camera.get("image_width")))
    height = int(camera.get("height", camera.get("image_height")))
    if matrix_source == "p":
        values = camera.get("p") or (camera.get("projection_matrix") or {}).get("data")
        if values is None:
            raise KeyError("Camera projection matrix P not found")
        return np.asarray(values, dtype=float).reshape(3, 4), width, height, frame_id
    values = camera.get("k") or (camera.get("camera_matrix") or {}).get("data")
    if values is None:
        raise KeyError("Camera matrix K not found")
    k = np.asarray(values, dtype=float).reshape(3, 3)
    return np.column_stack([k, np.zeros((3, 1), dtype=float)]), width, height, frame_id


def _quat_to_rot(q: dict) -> np.ndarray:
    x = float(q["x"])
    y = float(q["y"])
    z = float(q["z"])
    w = float(q["w"])
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _edge_matrix(edge: dict) -> np.ndarray:
    transform = edge["transform"]
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_rot(transform["rotation"])
    matrix[:3, 3] = [
        float(transform["translation"]["x"]),
        float(transform["translation"]["y"]),
        float(transform["translation"]["z"]),
    ]
    return matrix


def _invert(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -(rot.T @ trans)
    return out


def transform_from_extrinsics(extrinsics: dict, source_frame: str, target_frame: str) -> tuple[np.ndarray, list[str]]:
    adjacency: dict[str, list[str]] = {}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for edge in extrinsics["transforms"]:
        parent = edge["header"]["frame_id"]
        child = edge["child_frame_id"]
        child_to_parent = _edge_matrix(edge)
        adjacency.setdefault(child, []).append(parent)
        adjacency.setdefault(parent, []).append(child)
        matrices[(child, parent)] = child_to_parent
        matrices[(parent, child)] = _invert(child_to_parent)

    queue = deque([[source_frame]])
    path = None
    while queue:
        candidate = queue.popleft()
        if candidate[-1] == target_frame:
            path = candidate
            break
        for nxt in adjacency.get(candidate[-1], []):
            if nxt not in candidate:
                queue.append(candidate + [nxt])
    if path is None:
        raise ValueError(f"No transform path from {source_frame} to {target_frame}")

    matrix = np.eye(4, dtype=float)
    for a, b in zip(path, path[1:]):
        matrix = matrices[(a, b)] @ matrix
    return matrix, path


def load_aghri_camera_lidar_calibration(
    calibration_path: Path,
    intrinsics_member: str = "calibration/intrinsics.json",
    extrinsics_member: str = "calibration/extrinsics.json",
    camera_name: str = "cam_zed_rgb",
    lidar_frame: str = "front_lidar_link",
    camera_frame: str = "front_left_camera_optical_frame",
    matrix_source: str = "p",
) -> dict:
    intrinsics = _load_json(calibration_path, intrinsics_member)
    extrinsics = _load_json(calibration_path, extrinsics_member)
    p, width, height, resolved_camera_frame = camera_projection_from_intrinsics(
        intrinsics, camera_name=camera_name, matrix_source=matrix_source
    )
    camera_from_lidar, path = transform_from_extrinsics(
        extrinsics, source_frame=lidar_frame, target_frame=camera_frame
    )
    base_from_lidar, base_path = transform_from_extrinsics(
        extrinsics, source_frame=lidar_frame, target_frame="base_link"
    )
    return {
        "camera_p": p,
        "camera_from_lidar": camera_from_lidar,
        "base_from_lidar": base_from_lidar,
        "image_width": width,
        "image_height": height,
        "camera_frame": resolved_camera_frame or camera_frame,
        "lidar_frame": lidar_frame,
        "transform_path": path,
        "base_transform_path": base_path,
        "matrix_source": matrix_source,
    }
