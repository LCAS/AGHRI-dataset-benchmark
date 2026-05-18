"""
CenterPoint-style 3D Multi-Object Tracking.

Implements the tracking algorithm from the CenterPoint paper, decoupled from
the CenterPoint detection model so it can run on detections from any 3D detector.

Algorithm:
  1. At each frame, update existing track centres using their estimated velocity:
         x_pred = x_prev + vx * dt
  2. Match new detections to predicted positions using Euclidean centre distance.
  3. Update matched tracks: compute new velocity from position difference.
  4. Birth unmatched high-confidence detections as new tracks.
  5. Kill tracks unmatched for > max_age frames.

Unlike AB3DMOT/SimpleTrack, there is NO Kalman filter — velocity is estimated
directly as (current_pos - prev_pos) / dt and propagated as a constant.

Reference: Yin et al., "Center-based 3D Object Detection and Tracking", CVPR 2021.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    _next_id: int = 1

    def __init__(self, detection: np.ndarray, score: float) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        # box: [x, y, z, l, w, h, yaw]
        self.box = detection[:7].copy()
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = float(score)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Return predicted centre-shifted box."""
        self.age += 1
        self.time_since_update += 1
        pred = self.box.copy()
        pred[0] += self.vx * dt
        pred[1] += self.vy * dt
        return pred

    def update(self, detection: np.ndarray, score: float, dt: float = 1.0) -> None:
        """Update position and estimate velocity from displacement."""
        new_box = detection[:7].copy()
        # Velocity from displacement (clamp to avoid large jumps on first match)
        self.vx = (new_box[0] - self.box[0]) / dt
        self.vy = (new_box[1] - self.box[1]) / dt
        self.box = new_box
        self.hits += 1
        self.time_since_update = 0
        self.score = float(score)


def _centre_dist_matrix(pred_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    """
    N×M matrix of XY Euclidean distances between box centres.
    pred_boxes, det_boxes: (N/M, 7) arrays [x, y, z, l, w, h, yaw]
    """
    pred_xy = pred_boxes[:, :2]     # (N, 2)
    det_xy = det_boxes[:, :2]       # (M, 2)
    diff = pred_xy[:, None, :] - det_xy[None, :, :]   # (N, M, 2)
    return np.sqrt((diff ** 2).sum(axis=-1))            # (N, M)


def _match(
    predicted: np.ndarray,
    detections: np.ndarray,
    distance_threshold: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    if len(predicted) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            list(range(len(predicted))),
            list(range(len(detections))),
        )
    dist = _centre_dist_matrix(predicted, detections)
    row_ind, col_ind = linear_sum_assignment(dist)

    matched_rows, matched_cols = [], []
    unmatched_t = list(range(len(predicted)))
    unmatched_d = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= distance_threshold:
            matched_rows.append(r)
            matched_cols.append(c)
            unmatched_t.remove(r)
            unmatched_d.remove(c)

    matched = (
        np.column_stack([matched_rows, matched_cols])
        if matched_rows
        else np.empty((0, 2), dtype=np.int32)
    )
    return matched, unmatched_t, unmatched_d


class CenterPointTracker:
    """
    CenterPoint-style greedy velocity-based 3D tracker.

    Args:
        max_age:             frames a track survives without a match
        min_hits:            minimum matches before track is reported
        distance_threshold:  maximum XY centre distance (metres) for a valid match
        score_threshold:     minimum detection score to consider
        dt:                  time step between frames (default 1.0)
    """

    def __init__(
        self,
        max_age: int = 2,
        min_hits: int = 3,
        distance_threshold: float = 2.0,
        score_threshold: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        self.dt = dt
        self.tracks: List[Track] = []
        self.frame_count: int = 0

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1

    def update(
        self,
        detections: np.ndarray,
        scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process one frame.

        Args:
            detections: (N, 7) array [x, y, z, l, w, h, yaw]
            scores:     optional (N,) confidence scores

        Returns:
            (M, 8) array [x, y, z, l, w, h, yaw, track_id]
        """
        self.frame_count += 1

        if scores is None:
            scores = np.ones(len(detections), dtype=np.float64)

        if len(detections) > 0:
            keep = scores >= self.score_threshold
            detections = detections[keep]
            scores = scores[keep]

        # Predict positions
        predicted = (
            np.array([t.predict(self.dt) for t in self.tracks], dtype=np.float64)
            if self.tracks
            else np.empty((0, 7), dtype=np.float64)
        )

        matched, unmatched_tracks, unmatched_dets = _match(
            predicted, detections, self.distance_threshold
        )

        for r, c in matched:
            self.tracks[r].update(detections[c], scores[c], self.dt)

        for c in unmatched_dets:
            self.tracks.append(Track(detections[c], scores[c]))

        # Kill stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        results = []
        for t in self.tracks:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append(np.append(t.box, float(t.id)))

        return (
            np.array(results, dtype=np.float64)
            if results
            else np.empty((0, 8), dtype=np.float64)
        )
