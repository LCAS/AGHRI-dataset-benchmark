"""Geometric association helpers for the native AGHRI tracker."""

from __future__ import annotations

from itertools import permutations
import math
from typing import Sequence

import numpy as np

from late_fusion_pkg.detection_types import Detection2D, Detection3D
from late_fusion_pkg.fusion_core import iou_2d
from late_fusion_pkg.tracking_types import Track2D, Track3D


def linear_assignment_maximize(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Maximise a score matrix without making SciPy a hard dependency."""

    if matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-matrix)
        return [(int(row), int(col)) for row, col in zip(rows, cols)]
    except Exception:
        rows, cols = matrix.shape
        if min(rows, cols) > 8:
            ordered = sorted(
                ((float(matrix[row, col]), row, col) for row in range(rows) for col in range(cols)),
                key=lambda item: (-item[0], item[1], item[2]),
            )
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            out: list[tuple[int, int]] = []
            for score, row, col in ordered:
                if score <= -math.inf:
                    continue
                if row not in used_rows and col not in used_cols:
                    out.append((row, col))
                    used_rows.add(row)
                    used_cols.add(col)
            return out
        if rows <= cols:
            best_score = -math.inf
            best: list[tuple[int, int]] = []
            for cols_perm in permutations(range(cols), rows):
                pairs = list(enumerate(cols_perm))
                score = sum(float(matrix[row, col]) for row, col in pairs)
                if score > best_score:
                    best_score = score
                    best = [(int(row), int(col)) for row, col in pairs]
            return best
        best_score = -math.inf
        best = []
        for rows_perm in permutations(range(rows), cols):
            pairs = [(row, col) for col, row in enumerate(rows_perm)]
            score = sum(float(matrix[row, col]) for row, col in pairs)
            if score > best_score:
                best_score = score
                best = [(int(row), int(col)) for row, col in pairs]
        return best


def _axis_aligned_bev(det: Detection3D) -> tuple[float, float, float, float]:
    x, y = float(det.center_xyz[0]), float(det.center_xyz[1])
    length, width = float(det.size_lwh[0]), float(det.size_lwh[1])
    return (x - length / 2.0, y - width / 2.0, x + length / 2.0, y + width / 2.0)


def bev_iou_3d(a: Detection3D, b: Detection3D) -> float:
    """Stable BEV overlap proxy for AGHRI pedestrian boxes."""

    return iou_2d(_axis_aligned_bev(a), _axis_aligned_bev(b))


def non_max_suppress_3d(detections: Sequence[Detection3D], threshold: float) -> tuple[Detection3D, ...]:
    """Suppress duplicate LiDAR boxes with a deterministic BEV IoU NMS."""

    if threshold <= 0.0 or len(detections) <= 1:
        return tuple(detections)
    ordered = sorted(
        enumerate(detections),
        key=lambda item: (-float(item[1].score), item[0]),
    )
    kept: list[Detection3D] = []
    for _idx, detection in ordered:
        if all(bev_iou_3d(detection, previous) < threshold for previous in kept):
            kept.append(detection)
    return tuple(kept)


def center_distance_score(a: Detection3D, b: Detection3D, max_distance: float = 2.0) -> float:
    distance = float(np.linalg.norm(np.asarray(a.center_xyz, dtype=float) - np.asarray(b.center_xyz, dtype=float)))
    if distance >= max_distance:
        return 0.0
    return 1.0 - distance / max_distance


def score_detection3d_to_track(det: Detection3D, track: Track3D, metric: str, max_distance: float) -> float:
    if metric == "center_distance":
        return center_distance_score(det, track.detection, max_distance=max_distance)
    if metric == "hybrid":
        return max(bev_iou_3d(det, track.detection), center_distance_score(det, track.detection, max_distance=max_distance) * 0.5)
    return bev_iou_3d(det, track.detection)


def associate_3d(
    detections: Sequence[Detection3D],
    tracks: Sequence[Track3D],
    *,
    threshold: float,
    metric: str = "hybrid",
    max_center_distance: float = 2.0,
) -> tuple[list[tuple[int, int]], list[int], list[int], np.ndarray]:
    matrix = np.zeros((len(detections), len(tracks)), dtype=float)
    for det_idx, det in enumerate(detections):
        for track_idx, track in enumerate(tracks):
            matrix[det_idx, track_idx] = score_detection3d_to_track(det, track, metric, max_center_distance)
    candidate_pairs = linear_assignment_maximize(matrix)
    matches: list[tuple[int, int]] = []
    matched_dets: set[int] = set()
    matched_tracks: set[int] = set()
    for det_idx, track_idx in candidate_pairs:
        if matrix[det_idx, track_idx] >= threshold:
            matches.append((det_idx, track_idx))
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)
    return (
        matches,
        [idx for idx in range(len(detections)) if idx not in matched_dets],
        [idx for idx in range(len(tracks)) if idx not in matched_tracks],
        matrix,
    )


def associate_2d(
    detections: Sequence[Detection2D],
    tracks: Sequence[Track2D],
    *,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int], np.ndarray]:
    matrix = np.zeros((len(detections), len(tracks)), dtype=float)
    for det_idx, det in enumerate(detections):
        for track_idx, track in enumerate(tracks):
            matrix[det_idx, track_idx] = iou_2d(det.bbox_xyxy, track.bbox_xyxy)
    candidate_pairs = linear_assignment_maximize(matrix)
    matches: list[tuple[int, int]] = []
    matched_dets: set[int] = set()
    matched_tracks: set[int] = set()
    for det_idx, track_idx in candidate_pairs:
        if matrix[det_idx, track_idx] >= threshold:
            matches.append((det_idx, track_idx))
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)
    return (
        matches,
        [idx for idx in range(len(detections)) if idx not in matched_dets],
        [idx for idx in range(len(tracks)) if idx not in matched_tracks],
        matrix,
    )


def associate_2d_tracks_to_3d_tracks(
    tracks_2d: Sequence[Track2D],
    tracks_3d: Sequence[Track3D],
    *,
    threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int], np.ndarray]:
    matrix = np.zeros((len(tracks_2d), len(tracks_3d)), dtype=float)
    for idx2d, track2d in enumerate(tracks_2d):
        for idx3d, track3d in enumerate(tracks_3d):
            if track3d.projected_bbox_xyxy is None:
                continue
            matrix[idx2d, idx3d] = iou_2d(track2d.bbox_xyxy, track3d.projected_bbox_xyxy)
    candidate_pairs = linear_assignment_maximize(matrix)
    matches: list[tuple[int, int]] = []
    matched_2d: set[int] = set()
    matched_3d: set[int] = set()
    for idx2d, idx3d in candidate_pairs:
        if matrix[idx2d, idx3d] >= threshold:
            matches.append((idx2d, idx3d))
            matched_2d.add(idx2d)
            matched_3d.add(idx3d)
    return (
        matches,
        [idx for idx in range(len(tracks_2d)) if idx not in matched_2d],
        [idx for idx in range(len(tracks_3d)) if idx not in matched_3d],
        matrix,
    )
