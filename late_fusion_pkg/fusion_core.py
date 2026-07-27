"""ROS-independent decision-level late fusion."""

from __future__ import annotations

from itertools import permutations
import math
import time
from typing import Sequence

import numpy as np

from late_fusion_pkg.detection_types import Detection2D, Detection3D, FusionResult, Match
from late_fusion_pkg.projection import project_lidar_box


CONFIDENCE_POLICIES = {"lidar_score", "minimum", "geometric_mean"}
OUTPUT_POLICY = "matched_plus_unmatched_lidar"


def iou_2d(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def fused_confidence(camera_score: float, lidar_score: float, policy: str) -> float:
    if policy not in CONFIDENCE_POLICIES:
        raise ValueError(f"Unsupported confidence policy: {policy}")
    if policy == "lidar_score":
        return float(lidar_score)
    if policy == "minimum":
        return float(min(camera_score, lidar_score))
    return float(math.sqrt(max(camera_score, 0.0) * max(lidar_score, 0.0)))


def _linear_assignment_maximize(matrix: np.ndarray) -> list[tuple[int, int]]:
    if matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-matrix)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        # Deterministic exact fallback for small unit-test cases.
        rows, cols = matrix.shape
        if min(rows, cols) > 8:
            # Greedy fallback keeps runtime bounded when SciPy is absent.
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            ordered = sorted(
                ((float(matrix[r, c]), r, c) for r in range(rows) for c in range(cols)),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            out: list[tuple[int, int]] = []
            for _score, row, col in ordered:
                if row not in used_rows and col not in used_cols:
                    out.append((row, col))
                    used_rows.add(row)
                    used_cols.add(col)
            return out

        if rows <= cols:
            best_score = -1.0
            best: list[tuple[int, int]] = []
            for cols_perm in permutations(range(cols), rows):
                pairs = list(enumerate(cols_perm))
                score = sum(float(matrix[r, c]) for r, c in pairs)
                if score > best_score:
                    best_score = score
                    best = [(int(r), int(c)) for r, c in pairs]
            return best
        best_score = -1.0
        best = []
        for rows_perm in permutations(range(rows), cols):
            pairs = [(r, c) for c, r in enumerate(rows_perm)]
            score = sum(float(matrix[r, c]) for r, c in pairs)
            if score > best_score:
                best_score = score
                best = [(int(r), int(c)) for r, c in pairs]
        return best


def fuse_detections(
    camera_detections: Sequence[Detection2D],
    lidar_detections: Sequence[Detection3D],
    camera_p_3x4: np.ndarray,
    camera_from_lidar_4x4: np.ndarray,
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.1,
    confidence_policy: str = "lidar_score",
) -> FusionResult:
    """Project LiDAR boxes, associate with 2D boxes, and preserve empty frames."""
    start = time.perf_counter()
    projected = tuple(
        project_lidar_box(
            det,
            camera_p_3x4=camera_p_3x4,
            camera_from_lidar_4x4=camera_from_lidar_4x4,
            image_width=image_width,
            image_height=image_height,
        )
        for det in lidar_detections
    )
    valid_lidar_indices = [idx for idx, proj in enumerate(projected) if proj.valid]

    iou_matrix = np.zeros((len(camera_detections), len(valid_lidar_indices)), dtype=float)
    for cam_idx, cam in enumerate(camera_detections):
        for col_idx, lidar_idx in enumerate(valid_lidar_indices):
            iou_matrix[cam_idx, col_idx] = iou_2d(cam.bbox_xyxy, projected[lidar_idx].bbox_xyxy)

    candidate_pairs = _linear_assignment_maximize(iou_matrix)
    matched_camera: set[int] = set()
    matched_lidar: set[int] = set()
    matches: list[Match] = []
    matched_3d: list[Detection3D] = []
    for cam_idx, valid_col_idx in candidate_pairs:
        if cam_idx >= len(camera_detections) or valid_col_idx >= len(valid_lidar_indices):
            continue
        iou = float(iou_matrix[cam_idx, valid_col_idx])
        if iou < float(iou_threshold):
            continue
        lidar_idx = valid_lidar_indices[valid_col_idx]
        cam = camera_detections[cam_idx]
        lidar = lidar_detections[lidar_idx]
        score = fused_confidence(cam.score, lidar.score, confidence_policy)
        fused = Detection3D(
            center_xyz=lidar.center_xyz,
            size_lwh=lidar.size_lwh,
            yaw=lidar.yaw,
            score=score,
            label_id=lidar.label_id,
            label_name=lidar.label_name,
            detection_id=lidar.detection_id,
            sample_id=lidar.sample_id,
            timestamp=lidar.timestamp,
            frame_id=lidar.frame_id,
            box_origin=lidar.box_origin,
            metadata={**lidar.metadata, "matched_camera_id": cam.detection_id, "fusion_iou": iou},
        )
        matches.append(Match(camera_index=cam_idx, lidar_index=lidar_idx, iou=iou, fused_score=score))
        matched_camera.add(cam_idx)
        matched_lidar.add(lidar_idx)
        matched_3d.append(fused)

    unmatched_camera = tuple(det for idx, det in enumerate(camera_detections) if idx not in matched_camera)
    unmatched_lidar = tuple(det for idx, det in enumerate(lidar_detections) if idx not in matched_lidar)
    final_3d = tuple(matched_3d) + unmatched_lidar

    _elapsed = time.perf_counter() - start
    return FusionResult(
        matches=tuple(matches),
        matched_3d=tuple(matched_3d),
        unmatched_lidar=unmatched_lidar,
        unmatched_camera=unmatched_camera,
        final_3d=final_3d,
        projected_lidar=projected,
        iou_matrix=tuple(tuple(float(v) for v in row) for row in iou_matrix.tolist()),
        confidence_policy=confidence_policy,
        output_policy=OUTPUT_POLICY,
    )

