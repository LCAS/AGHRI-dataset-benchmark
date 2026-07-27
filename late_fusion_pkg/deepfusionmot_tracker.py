"""Native AGHRI DeepFusionMOT-style tracker core."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np

from late_fusion_pkg.detection_types import Detection2D, Detection3D, FusionResult
from late_fusion_pkg.deepfusionmot_association import associate_2d, associate_2d_tracks_to_3d_tracks, associate_3d
from late_fusion_pkg.deepfusionmot_motion import (
    ConstantVelocityKalman2D,
    ConstantVelocityKalman3D,
    safe_dt,
    valid_detection2d,
    valid_detection3d,
)
from late_fusion_pkg.tracking_types import (
    Track2D,
    Track3D,
    TrackState,
    TrackedDetection2D,
    TrackedDetection3D,
    TrackingDiagnostics,
    TrackingFrameResult,
    UpdateSource,
)


@dataclass(frozen=True)
class DeepFusionMOTTrackerConfig:
    min_hits: int = 2
    max_age_frames: int = 5
    max_age_seconds: float = 0.75
    max_prediction_dt: float = 1.0
    association_3d_threshold: float = 0.05
    association_2d_threshold: float = 0.20
    support_2d_to_3d_threshold: float = 0.10
    association_3d_metric: str = "hybrid"
    max_center_distance_m: float = 2.0
    lidar_only_initiation_score_threshold: float = 0.0
    ego_motion_mode: str = "none"
    id_policy: str = "sequential"
    lifecycle_mode: str = "time"
    output_mode: str = "all"

    def __post_init__(self) -> None:
        if self.ego_motion_mode not in {"none", "aghri_odom"}:
            raise ValueError("ego_motion_mode must be 'none' or 'aghri_odom'")
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if self.max_age_frames < 0:
            raise ValueError("max_age_frames must be >= 0")
        if self.max_age_seconds < 0.0:
            raise ValueError("max_age_seconds must be >= 0")
        if self.lidar_only_initiation_score_threshold < 0.0:
            raise ValueError("lidar_only_initiation_score_threshold must be >= 0")
        if self.id_policy not in {"sequential", "deepfusionmot_even_odd"}:
            raise ValueError("id_policy must be 'sequential' or 'deepfusionmot_even_odd'")
        if self.lifecycle_mode not in {"time", "deepfusionmot_frame"}:
            raise ValueError("lifecycle_mode must be 'time' or 'deepfusionmot_frame'")
        if self.output_mode not in {"all", "deepfusionmot"}:
            raise ValueError("output_mode must be 'all' or 'deepfusionmot'")


class DeepFusionMOTTracker:
    """Stateful tracker that consumes existing per-frame AGHRI fusion outputs."""

    def __init__(self, config: DeepFusionMOTTrackerConfig | None = None):
        self.config = config or DeepFusionMOTTrackerConfig()
        self.tracks_3d: list[Track3D] = []
        self.tracks_2d: list[Track2D] = []
        self.recording_id: str | None = None
        self.last_timestamp: float | None = None
        self._next_track_id = 1
        self._next_track_id_3d = 0
        self._next_track_id_2d = 1
        self.frame_count = 0
        self.total_created_3d = 0
        self.total_deleted_3d = 0
        self.total_invalid_rejected = 0

    def reset(self, recording_id: str | None = None) -> None:
        self.tracks_3d = []
        self.tracks_2d = []
        self.recording_id = recording_id
        self.last_timestamp = None
        self._next_track_id = 1
        self._next_track_id_3d = 0
        self._next_track_id_2d = 1
        self.frame_count = 0

    def _allocate_id(self, track_kind: str) -> int:
        if self.config.id_policy == "deepfusionmot_even_odd":
            if track_kind == "3d":
                value = self._next_track_id_3d
                self._next_track_id_3d += 2
                return value
            if track_kind == "2d":
                value = self._next_track_id_2d
                self._next_track_id_2d += 2
                return value
            raise ValueError(f"Unknown track kind: {track_kind}")
        value = self._next_track_id
        self._next_track_id += 1
        return value

    def _confirm_if_ready(self, track: Track2D | Track3D) -> None:
        if track.state != TrackState.DELETED and track.hits >= self.config.min_hits:
            track.state = TrackState.CONFIRMED

    def _is_expired(self, track: Track2D | Track3D) -> bool:
        if self.config.lifecycle_mode == "deepfusionmot_frame":
            return track.time_since_update > self.config.max_age_frames
        return (
            track.missed_frames > self.config.max_age_frames
            or track.time_since_update > self.config.max_age_seconds
        )

    def _projected_bbox_by_detection_id(self, fusion_result: FusionResult) -> dict[str, tuple[float, float, float, float]]:
        out: dict[str, tuple[float, float, float, float]] = {}
        for projected in fusion_result.projected_lidar:
            if projected.valid and projected.detection.detection_id:
                out[projected.detection.detection_id] = projected.bbox_xyxy
        return out

    def _prediction_bbox_by_track_id(self) -> dict[int, tuple[float, float, float, float] | None]:
        return {track.track_id: track.projected_bbox_xyxy for track in self.tracks_3d}

    def _initiate_3d(
        self,
        detection: Detection3D,
        recording_id: str,
        timestamp: float | None,
        source: UpdateSource,
        projected_bbox: tuple[float, float, float, float] | None,
    ) -> Track3D:
        track = Track3D(
            track_id=self._allocate_id("3d"),
            recording_id=recording_id,
            detection=detection,
            timestamp=timestamp,
            state=TrackState.CONFIRMED if self.config.min_hits <= 1 else TrackState.TENTATIVE,
            update_source=source,
            projected_bbox_xyxy=projected_bbox,
            motion=ConstantVelocityKalman3D(detection),
        )
        self.tracks_3d.append(track)
        self.total_created_3d += 1
        return track

    def _initiate_2d(self, detection: Detection2D, recording_id: str, timestamp: float | None) -> Track2D:
        track = Track2D(
            track_id=self._allocate_id("2d"),
            recording_id=recording_id,
            bbox_xyxy=detection.bbox_xyxy,
            score=detection.score,
            timestamp=timestamp,
            state=TrackState.CONFIRMED if self.config.min_hits <= 1 else TrackState.TENTATIVE,
            detection=detection,
            motion=ConstantVelocityKalman2D(detection),
        )
        self.tracks_2d.append(track)
        return track

    def _predict_tracks(self, timestamp: float | None, dt: float, ego_motion_transform: np.ndarray | None = None) -> None:
        for track in self.tracks_3d:
            if track.state == TrackState.DELETED:
                continue
            track.age += 1
            track.detection = track.motion.predict(dt, track.detection, timestamp)
            if self.config.ego_motion_mode == "aghri_odom" and ego_motion_transform is not None:
                track.detection = track.motion.apply_ego_motion(ego_motion_transform, track.detection, timestamp)
            track.update_source = UpdateSource.PREDICTION_ONLY
        for track in self.tracks_2d:
            if track.state == TrackState.DELETED:
                continue
            track.age += 1
            track.bbox_xyxy = track.motion.predict(dt)
            track.update_source = UpdateSource.PREDICTION_ONLY

    def _update_3d(
        self,
        track: Track3D,
        detection: Detection3D,
        timestamp: float | None,
        source: UpdateSource,
        projected_bbox: tuple[float, float, float, float] | None,
    ) -> None:
        track.detection = track.motion.update(detection)
        track.timestamp = timestamp
        track.hits += 1
        track.missed_frames = 0
        track.time_since_update = 0.0
        track.update_source = source
        track.projected_bbox_xyxy = projected_bbox
        if track.state == TrackState.REACTIVATE:
            track.state = TrackState.CONFIRMED
        self._confirm_if_ready(track)

    def _update_2d(self, track: Track2D, detection: Detection2D, timestamp: float | None) -> None:
        track.bbox_xyxy = track.motion.update(detection)
        track.score = detection.score
        track.timestamp = timestamp
        track.detection = detection
        track.hits += 1
        track.missed_frames = 0
        track.time_since_update = 0.0
        track.update_source = UpdateSource.CAMERA_ONLY_SUPPORT
        if track.state == TrackState.REACTIVATE:
            track.state = TrackState.CONFIRMED
        self._confirm_if_ready(track)

    def _mark_missed(self, tracks: Sequence[Track2D | Track3D], dt: float) -> None:
        for track in tracks:
            if track.state == TrackState.DELETED:
                continue
            track.missed_frames += 1
            if self.config.lifecycle_mode == "deepfusionmot_frame":
                track.time_since_update += 1.0
            else:
                track.time_since_update += max(0.0, float(dt))
            track.update_source = UpdateSource.PREDICTION_ONLY
            if self.config.lifecycle_mode == "deepfusionmot_frame":
                if track.state == TrackState.TENTATIVE:
                    track.state = TrackState.DELETED
                elif track.state == TrackState.REACTIVATE and self._is_expired(track):
                    track.state = TrackState.DELETED
                elif track.state == TrackState.CONFIRMED:
                    track.state = TrackState.REACTIVATE
                elif track.state != TrackState.REACTIVATE and track.missed_frames >= 1:
                    track.state = TrackState.DELETED
            elif self._is_expired(track):
                track.state = TrackState.DELETED
            elif track.state == TrackState.CONFIRMED:
                track.state = TrackState.REACTIVATE

    def _support_3d_with_2d(self, track: Track3D, timestamp: float | None) -> None:
        track.timestamp = timestamp
        track.hits += 1
        track.missed_frames = 0
        track.time_since_update = 0.0
        track.update_source = UpdateSource.CAMERA_ONLY_SUPPORT
        self._confirm_if_ready(track)

    def update(
        self,
        *,
        recording_id: str,
        frame_index: int,
        timestamp: float | None,
        camera_detections: Sequence[Detection2D],
        lidar_detections: Sequence[Detection3D],
        fusion_result: FusionResult,
        ego_motion_transform: np.ndarray | None = None,
    ) -> TrackingFrameResult:
        frame_start = time.perf_counter()
        reset = False
        if self.recording_id != recording_id:
            self.reset(recording_id)
            reset = True
        self.frame_count += 1
        dt = safe_dt(timestamp, self.last_timestamp, max_dt=self.config.max_prediction_dt)
        self.last_timestamp = timestamp

        valid_camera = [det for det in camera_detections if valid_detection2d(det)]
        valid_matched = [det for det in fusion_result.matched_3d if valid_detection3d(det)]
        valid_lidar_only = [det for det in fusion_result.unmatched_lidar if valid_detection3d(det)]
        invalid_rejected = (
            len(camera_detections)
            + len(fusion_result.matched_3d)
            + len(fusion_result.unmatched_lidar)
            - len(valid_camera)
            - len(valid_matched)
            - len(valid_lidar_only)
        )
        self.total_invalid_rejected += invalid_rejected
        projected_by_id = self._projected_bbox_by_detection_id(fusion_result)

        self._predict_tracks(timestamp, dt, ego_motion_transform=ego_motion_transform)
        prediction_projection = self._prediction_bbox_by_track_id()

        association_start = time.perf_counter()
        active_3d = [track for track in self.tracks_3d if track.state != TrackState.DELETED]
        matched_fused, unmatched_fused, unmatched_3d, _ = associate_3d(
            valid_matched,
            active_3d,
            threshold=self.config.association_3d_threshold,
            metric=self.config.association_3d_metric,
            max_center_distance=self.config.max_center_distance_m,
        )
        matched_track_ids: set[int] = set()
        for det_idx, active_idx in matched_fused:
            det = valid_matched[det_idx]
            track = active_3d[active_idx]
            self._update_3d(
                track,
                det,
                timestamp,
                UpdateSource.MATCHED_CAMERA_LIDAR,
                projected_by_id.get(det.detection_id) or prediction_projection.get(track.track_id),
            )
            matched_track_ids.add(track.track_id)

        remaining_3d = [track for idx, track in enumerate(active_3d) if idx in unmatched_3d and track.state != TrackState.DELETED]
        matched_lidar, unmatched_lidar, _unmatched_remaining, _ = associate_3d(
            valid_lidar_only,
            remaining_3d,
            threshold=self.config.association_3d_threshold,
            metric=self.config.association_3d_metric,
            max_center_distance=self.config.max_center_distance_m,
        )
        for det_idx, remaining_idx in matched_lidar:
            det = valid_lidar_only[det_idx]
            track = remaining_3d[remaining_idx]
            self._update_3d(
                track,
                det,
                timestamp,
                UpdateSource.LIDAR_ONLY,
                projected_by_id.get(det.detection_id) or prediction_projection.get(track.track_id),
            )
            matched_track_ids.add(track.track_id)

        new_3d_tracks = 0
        low_score_lidar_only_initiations_suppressed = 0
        for det_idx in unmatched_fused:
            det = valid_matched[det_idx]
            track = self._initiate_3d(
                det,
                recording_id,
                timestamp,
                UpdateSource.MATCHED_CAMERA_LIDAR,
                projected_by_id.get(det.detection_id),
            )
            new_3d_tracks += 1
            matched_track_ids.add(track.track_id)
        for det_idx in unmatched_lidar:
            det = valid_lidar_only[det_idx]
            if det.score < self.config.lidar_only_initiation_score_threshold:
                low_score_lidar_only_initiations_suppressed += 1
                continue
            track = self._initiate_3d(
                det,
                recording_id,
                timestamp,
                UpdateSource.LIDAR_ONLY,
                projected_by_id.get(det.detection_id),
            )
            new_3d_tracks += 1
            matched_track_ids.add(track.track_id)

        active_2d = [track for track in self.tracks_2d if track.state != TrackState.DELETED]
        matched_2d, unmatched_camera, unmatched_2d_tracks, _ = associate_2d(
            list(fusion_result.unmatched_camera),
            active_2d,
            threshold=self.config.association_2d_threshold,
        )
        matched_2d_track_ids: set[int] = set()
        for det_idx, active_idx in matched_2d:
            det = fusion_result.unmatched_camera[det_idx]
            track = active_2d[active_idx]
            if valid_detection2d(det):
                self._update_2d(track, det, timestamp)
                matched_2d_track_ids.add(track.track_id)
        for det_idx in unmatched_camera:
            det = fusion_result.unmatched_camera[det_idx]
            if valid_detection2d(det):
                track = self._initiate_2d(det, recording_id, timestamp)
                matched_2d_track_ids.add(track.track_id)

        supported_3d_ids: set[int] = set()
        eligible_2d = [track for track in self.tracks_2d if track.state != TrackState.DELETED and track.track_id in matched_2d_track_ids]
        eligible_3d = [
            track
            for track in self.tracks_3d
            if track.state != TrackState.DELETED and track.track_id not in matched_track_ids and track.projected_bbox_xyxy is not None
        ]
        support_matches, _unmatched_support_2d, _unmatched_support_3d, _ = associate_2d_tracks_to_3d_tracks(
            eligible_2d,
            eligible_3d,
            threshold=self.config.support_2d_to_3d_threshold,
        )
        for idx2d, idx3d in support_matches:
            track = eligible_3d[idx3d]
            self._support_3d_with_2d(track, timestamp)
            supported_3d_ids.add(track.track_id)

        association_time_ms = (time.perf_counter() - association_start) * 1000.0

        missed_3d = [
            track
            for track in self.tracks_3d
            if track.state != TrackState.DELETED
            and track.track_id not in matched_track_ids
            and track.track_id not in supported_3d_ids
        ]
        self._mark_missed(missed_3d, dt)
        missed_2d = [
            track
            for track in self.tracks_2d
            if track.state != TrackState.DELETED and track.track_id not in matched_2d_track_ids
        ]
        self._mark_missed(missed_2d, dt)

        deleted_now = [track for track in self.tracks_3d if track.state == TrackState.DELETED]
        self.total_deleted_3d += len(deleted_now)
        self.tracks_3d = [track for track in self.tracks_3d if track.state != TrackState.DELETED]
        self.tracks_2d = [track for track in self.tracks_2d if track.state != TrackState.DELETED]
        output_3d_tracks = [
            track
            for track in self.tracks_3d
            if self.config.output_mode != "deepfusionmot"
            or track.state == TrackState.CONFIRMED
            or self.frame_count <= self.config.min_hits
        ]

        tracked_3d = tuple(
            TrackedDetection3D(
                recording_id=recording_id,
                frame_index=frame_index,
                timestamp=timestamp,
                track_id=track.track_id,
                track_state=track.state.value,
                update_source=track.update_source.value,
                detection=track.detection,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
                projected_bbox_xyxy=track.projected_bbox_xyxy,
            )
            for track in output_3d_tracks
        )
        tracked_2d = tuple(
            TrackedDetection2D(
                recording_id=recording_id,
                frame_index=frame_index,
                timestamp=timestamp,
                track_id=track.track_id,
                track_state=track.state.value,
                update_source=track.update_source.value,
                bbox_xyxy=track.bbox_xyxy,
                score=track.score,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
            )
            for track in self.tracks_2d
        )
        diagnostics = TrackingDiagnostics(
            recording_id=recording_id,
            frame_index=frame_index,
            timestamp=timestamp,
            camera_detection_count=len(valid_camera),
            lidar_detection_count=len(valid_matched) + len(valid_lidar_only),
            matched_detection_count=len(valid_matched),
            camera_only_count=len(fusion_result.unmatched_camera),
            lidar_only_count=len(valid_lidar_only),
            active_2d_tracks=len(self.tracks_2d),
            active_3d_tracks=len(self.tracks_3d),
            new_tracks=new_3d_tracks,
            confirmed_tracks=sum(1 for track in self.tracks_3d if track.state == TrackState.CONFIRMED),
            predicted_only_tracks=sum(1 for track in self.tracks_3d if track.update_source == UpdateSource.PREDICTION_ONLY),
            deleted_tracks=len(deleted_now),
            invalid_detections_rejected=invalid_rejected,
            association_time_ms=association_time_ms,
            tracker_update_time_ms=(time.perf_counter() - frame_start) * 1000.0,
            dt=dt,
            reset=reset,
            extra={
                "low_score_lidar_only_initiations_suppressed": low_score_lidar_only_initiations_suppressed,
                "lidar_only_initiation_score_threshold": self.config.lidar_only_initiation_score_threshold,
            },
        )
        return TrackingFrameResult(
            recording_id=recording_id,
            frame_index=frame_index,
            timestamp=timestamp,
            tracked_3d=tracked_3d,
            tracked_2d=tracked_2d,
            diagnostics=diagnostics,
        )
