"""
AB3DMOT: Baseline 3D Multi-Object Tracking.

Algorithm:
  1. Predict each track's next position via Kalman filter.
  2. Compute BEV IoU between predictions and new detections.
  3. Solve linear assignment with Hungarian algorithm.
  4. Update matched tracks; birth unmatched detections; age unmatched tracks.
  5. Delete tracks older than max_age; only output tracks with >= min_hits confirmed matches.

Reference: Weng et al., "3D Multi-Object Tracking: A Baseline and New Evaluation Metrics", IROS 2020.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "mot3d"))
from bev_iou import iou_matrix
from kalman3d import KalmanFilter3D


class Track:
    _next_id: int = 1

    def __init__(self, detection: np.ndarray, score: float) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter3D(detection)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.score = float(score)

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, detection: np.ndarray, score: float) -> None:
        self.kf.update(detection)
        self.hits += 1
        self.time_since_update = 0
        self.score = float(score)

    @property
    def box(self) -> np.ndarray:
        return self.kf.state


def _associate(
    predicted: np.ndarray,
    detections: np.ndarray,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match predictions to detections via Hungarian algorithm on BEV IoU.

    Returns:
        matched:          (M, 2) pairs of [track_idx, det_idx]
        unmatched_tracks: (K,) indices into predicted
        unmatched_dets:   (L,) indices into detections
    """
    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.arange(len(predicted), dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    if len(predicted) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.arange(len(detections), dtype=np.int32),
        )

    iou = iou_matrix(predicted, detections)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)

    matched_rows, matched_cols = [], []
    unmatched_tracks = list(range(len(predicted)))
    unmatched_dets = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
            matched_rows.append(r)
            matched_cols.append(c)
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)

    matched = np.column_stack([matched_rows, matched_cols]) if matched_rows else np.empty((0, 2), dtype=np.int32)
    return matched, np.array(unmatched_tracks, dtype=np.int32), np.array(unmatched_dets, dtype=np.int32)


class AB3DMOT:
    """
    AB3DMOT 3D multi-object tracker.

    Args:
        max_age:        frames a track can go unmatched before deletion
        min_hits:       minimum matched frames before a track is reported
        iou_threshold:  minimum BEV IoU for a valid match
        score_threshold: minimum detection confidence to consider
    """

    def __init__(
        self,
        max_age: int = 2,
        min_hits: int = 3,
        iou_threshold: float = 0.1,
        score_threshold: float = 0.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
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
        Process one frame of detections.

        Args:
            detections: (N, 7) array [x, y, z, l, w, h, yaw]
            scores:     optional (N,) confidence scores

        Returns:
            (M, 8) array [x, y, z, l, w, h, yaw, track_id] for active tracks
        """
        self.frame_count += 1

        if scores is None:
            scores = np.ones(len(detections), dtype=np.float64)

        # Filter low-confidence detections
        if len(detections) > 0:
            keep = scores >= self.score_threshold
            detections = detections[keep]
            scores = scores[keep]

        # Predict step
        predicted = (
            np.array([t.predict() for t in self.tracks], dtype=np.float64)
            if self.tracks
            else np.empty((0, 7), dtype=np.float64)
        )

        # Associate predictions → detections
        matched, unmatched_tracks, unmatched_dets = _associate(
            predicted, detections, self.iou_threshold
        )

        # Update matched tracks
        for r, c in matched:
            self.tracks[r].update(detections[c], scores[c])

        # Birth: create tracks for unmatched detections
        for c in unmatched_dets:
            self.tracks.append(Track(detections[c], scores[c]))

        # Death: remove stale tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Collect output: tracks with enough confirmed hits
        results = []
        for t in self.tracks:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                box = t.box
                results.append(np.append(box, float(t.id)))

        return np.array(results, dtype=np.float64) if results else np.empty((0, 8), dtype=np.float64)
