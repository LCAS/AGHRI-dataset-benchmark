"""
SimpleTrack: Simplified 3D Multi-Object Tracking with two-stage matching.

Key differences from AB3DMOT:
  1. Two-stage matching — high-confidence detections matched first; remaining
     low-confidence detections matched to unconfirmed / recently-lost tracks.
  2. Velocity-compensated Kalman filter using the same KF as AB3DMOT.
  3. Track states: Tentative → Confirmed → Lost (more robust birth/death logic).

Reference: Pang et al., "SimpleTrack: Understanding and Rethinking 3D Multi-object
Tracking", ECCV 2022.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "mot3d"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "ab3dmot" / "src"))
from bev_iou import iou_matrix
from kalman3d import KalmanFilter3D


class TrackState(Enum):
    Tentative = 0
    Confirmed = 1
    Lost = 2


class Track:
    _next_id: int = 1

    def __init__(self, detection: np.ndarray, score: float) -> None:
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter3D(detection)
        self.state = TrackState.Tentative
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


def _hungarian_match(
    predicted: np.ndarray,
    detections: np.ndarray,
    iou_threshold: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    if len(predicted) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=np.int32),
            list(range(len(predicted))),
            list(range(len(detections))),
        )
    iou = iou_matrix(predicted, detections)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou)

    matched_rows, matched_cols = [], []
    unmatched_t = list(range(len(predicted)))
    unmatched_d = list(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if iou[r, c] >= iou_threshold:
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


class SimpleTrack:
    """
    SimpleTrack 3D multi-object tracker.

    Two-stage matching:
      Stage 1: Match high-score detections to Confirmed tracks.
      Stage 2: Match remaining detections to Tentative / recently Lost tracks.

    Args:
        max_age:            frames a track survives without a match
        min_hits:           hits before Tentative → Confirmed
        iou_threshold_high: IoU threshold for stage-1 (high-score) matching
        iou_threshold_low:  IoU threshold for stage-2 (low-score) matching
        score_threshold:    minimum detection score to consider at all
        high_score_thresh:  split threshold between high/low-confidence detections
    """

    def __init__(
        self,
        max_age: int = 2,
        min_hits: int = 3,
        iou_threshold_high: float = 0.1,
        iou_threshold_low: float = 0.05,
        score_threshold: float = 0.0,
        high_score_thresh: float = 0.5,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.score_threshold = score_threshold
        self.high_score_thresh = high_score_thresh
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
            (M, 8) array [x, y, z, l, w, h, yaw, track_id] for active tracks
        """
        self.frame_count += 1

        if scores is None:
            scores = np.ones(len(detections), dtype=np.float64)

        # Discard very-low-confidence detections
        if len(detections) > 0:
            keep = scores >= self.score_threshold
            detections = detections[keep]
            scores = scores[keep]

        # Predict all tracks
        predicted = (
            np.array([t.predict() for t in self.tracks], dtype=np.float64)
            if self.tracks
            else np.empty((0, 7), dtype=np.float64)
        )

        # Split detections into high / low confidence
        if len(detections) > 0:
            high_mask = scores >= self.high_score_thresh
            high_dets = detections[high_mask]
            high_scores = scores[high_mask]
            low_dets = detections[~high_mask]
            low_scores = scores[~high_mask]
            high_idx = np.where(high_mask)[0]
            low_idx = np.where(~high_mask)[0]
        else:
            high_dets = low_dets = np.empty((0, 7), dtype=np.float64)
            high_scores = low_scores = np.empty(0, dtype=np.float64)
            high_idx = low_idx = np.empty(0, dtype=np.int32)

        # --- Stage 1: match high-score dets to Confirmed tracks ---
        confirmed_idx = [i for i, t in enumerate(self.tracks) if t.state == TrackState.Confirmed]
        confirmed_pred = predicted[confirmed_idx] if confirmed_idx else np.empty((0, 7))

        matched1, unmatched_conf, unmatched_high = _hungarian_match(
            confirmed_pred, high_dets, self.iou_threshold_high
        )
        for pair in matched1:
            self.tracks[confirmed_idx[pair[0]]].update(high_dets[pair[1]], high_scores[pair[1]])

        # --- Stage 2: match remaining dets + low-score dets to remaining tracks ---
        remaining_track_idx = [
            i for i in range(len(self.tracks))
            if not (i in confirmed_idx and i not in [confirmed_idx[r] for r in matched1[:, 0]
                                                     if len(matched1)])
            and self.tracks[i].time_since_update <= 1
        ]
        # Rebuild — tracks not updated in stage 1 are candidates
        stage1_updated = {confirmed_idx[pair[0]] for pair in matched1} if len(matched1) else set()
        stage2_track_idx = [i for i in range(len(self.tracks)) if i not in stage1_updated]
        stage2_pred = predicted[stage2_track_idx] if stage2_track_idx else np.empty((0, 7))
        stage2_dets = (
            np.concatenate([high_dets[unmatched_high], low_dets], axis=0)
            if len(unmatched_high) + len(low_dets) > 0
            else np.empty((0, 7), dtype=np.float64)
        )
        stage2_scores = (
            np.concatenate([high_scores[unmatched_high], low_scores])
            if len(unmatched_high) + len(low_scores) > 0
            else np.empty(0, dtype=np.float64)
        )

        matched2, unmatched_s2_tracks, unmatched_s2_dets = _hungarian_match(
            stage2_pred, stage2_dets, self.iou_threshold_low
        )
        for pair in matched2:
            self.tracks[stage2_track_idx[pair[0]]].update(stage2_dets[pair[1]], stage2_scores[pair[1]])

        # Birth: new tracks for unmatched high-confidence detections only
        for di in unmatched_s2_dets:
            if di < len(unmatched_high):
                orig_det_idx = unmatched_high[di]
                self.tracks.append(Track(high_dets[orig_det_idx], high_scores[orig_det_idx]))
            # Low-confidence unmatched dets are discarded (no new track births)

        # State transitions
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                t.state = TrackState.Confirmed
            elif t.time_since_update > self.max_age:
                t.state = TrackState.Lost

        # Death
        self.tracks = [t for t in self.tracks if t.state != TrackState.Lost]

        # Output Confirmed + Tentative-but-matched tracks
        results = []
        for t in self.tracks:
            if t.state == TrackState.Confirmed or (
                t.state == TrackState.Tentative and t.time_since_update == 0
            ):
                results.append(np.append(t.box, float(t.id)))

        return (
            np.array(results, dtype=np.float64)
            if results
            else np.empty((0, 8), dtype=np.float64)
        )
