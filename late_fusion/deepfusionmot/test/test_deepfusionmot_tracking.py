import math

import numpy as np

from late_fusion_pkg.aghri_ego_motion import AghriOdomEgoMotion
from late_fusion_pkg.detection_types import Detection2D, Detection3D, FusionResult
from late_fusion_pkg.deepfusionmot_association import non_max_suppress_3d
from late_fusion_pkg.deepfusionmot_motion import wrap_angle
from late_fusion_pkg.deepfusionmot_tracker import DeepFusionMOTTracker, DeepFusionMOTTrackerConfig


def det3d(x=5.0, y=0.0, z=0.0, yaw=0.0, score=0.9):
    return Detection3D(
        center_xyz=(x, y, z),
        size_lwh=(1.0, 0.8, 1.8),
        yaw=yaw,
        score=score,
        label_id=0,
        label_name="person",
        detection_id=f"d{x}_{y}_{yaw}",
        sample_id="rec/1",
        timestamp=1.0,
    )


def det2d(x1=10.0, y1=10.0, x2=40.0, y2=80.0, score=0.8):
    return Detection2D(
        bbox_xyxy=(x1, y1, x2, y2),
        score=score,
        label_id=0,
        label_name="person",
        detection_id=f"c{x1}_{y1}",
        sample_id="rec/1",
        timestamp=1.0,
    )


def fusion(matched=(), lidar_only=(), camera_only=()):
    final = tuple(matched) + tuple(lidar_only)
    return FusionResult(
        matches=(),
        matched_3d=tuple(matched),
        unmatched_lidar=tuple(lidar_only),
        unmatched_camera=tuple(camera_only),
        final_3d=final,
        projected_lidar=(),
        iou_matrix=(),
        confidence_policy="lidar_score",
        output_policy="matched_plus_unmatched_lidar",
    )


def tracker(min_hits=2, max_age_frames=2, max_age_seconds=0.5):
    return DeepFusionMOTTracker(
        DeepFusionMOTTrackerConfig(
            min_hits=min_hits,
            max_age_frames=max_age_frames,
            max_age_seconds=max_age_seconds,
            association_3d_threshold=0.01,
            association_3d_metric="hybrid",
            max_center_distance_m=2.0,
        )
    )


def deepfusionmot_strict_tracker():
    return DeepFusionMOTTracker(
        DeepFusionMOTTrackerConfig(
            min_hits=3,
            max_age_frames=25,
            max_age_seconds=0.75,
            association_3d_threshold=0.01,
            association_2d_threshold=0.50,
            support_2d_to_3d_threshold=0.50,
            association_3d_metric="bev_iou",
            id_policy="deepfusionmot_even_odd",
            lifecycle_mode="deepfusionmot_frame",
            output_mode="deepfusionmot",
        )
    )


def test_empty_frame_with_no_detections():
    trk = tracker()
    result = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[],
        fusion_result=fusion(),
    )
    assert result.tracked_3d == ()
    assert result.diagnostics.active_3d_tracks == 0


def test_new_3d_track_creation_and_confirmation_after_min_hits():
    trk = tracker(min_hits=2)
    first = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[det3d()],
        fusion_result=fusion(matched=[det3d()]),
    )
    assert len(first.tracked_3d) == 1
    assert first.tracked_3d[0].track_state == "tentative"
    second_det = det3d(x=5.05)
    second = trk.update(
        recording_id="rec",
        frame_index=1,
        timestamp=1.1,
        camera_detections=[],
        lidar_detections=[second_det],
        fusion_result=fusion(matched=[second_det]),
    )
    assert second.tracked_3d[0].track_id == first.tracked_3d[0].track_id
    assert second.tracked_3d[0].track_state == "confirmed"


def test_same_detection_across_frames_keeps_same_id():
    trk = tracker(min_hits=1)
    ids = []
    for index, x in enumerate((5.0, 5.02, 5.04)):
        item = det3d(x=x)
        result = trk.update(
            recording_id="rec",
            frame_index=index,
            timestamp=1.0 + index * 0.1,
            camera_detections=[],
            lidar_detections=[item],
            fusion_result=fusion(lidar_only=[item]),
        )
        ids.append(result.tracked_3d[0].track_id)
    assert ids == [1, 1, 1]


def test_short_missed_detection_gap_preserves_track():
    trk = tracker(min_hits=1, max_age_frames=2, max_age_seconds=0.5)
    item = det3d()
    trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[item], fusion_result=fusion(lidar_only=[item]))
    missed = trk.update(recording_id="rec", frame_index=1, timestamp=1.1, camera_detections=[], lidar_detections=[], fusion_result=fusion())
    assert len(missed.tracked_3d) == 1
    assert missed.tracked_3d[0].update_source == "prediction_only"
    recovered_det = det3d(x=5.01)
    recovered = trk.update(recording_id="rec", frame_index=2, timestamp=1.2, camera_detections=[], lidar_detections=[recovered_det], fusion_result=fusion(lidar_only=[recovered_det]))
    assert recovered.tracked_3d[0].track_id == 1
    assert recovered.tracked_3d[0].update_source == "lidar_only"


def test_long_missed_detection_gap_deletes_track():
    trk = tracker(min_hits=1, max_age_frames=2, max_age_seconds=0.15)
    item = det3d()
    trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[item], fusion_result=fusion(lidar_only=[item]))
    deleted = trk.update(recording_id="rec", frame_index=1, timestamp=1.3, camera_detections=[], lidar_detections=[], fusion_result=fusion())
    assert deleted.tracked_3d == ()
    assert deleted.diagnostics.deleted_tracks == 1


def test_new_recording_resets_ids():
    trk = tracker(min_hits=1)
    item = det3d()
    first = trk.update(recording_id="rec_a", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[item], fusion_result=fusion(lidar_only=[item]))
    second = trk.update(recording_id="rec_b", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[item], fusion_result=fusion(lidar_only=[item]))
    assert first.tracked_3d[0].track_id == 1
    assert second.tracked_3d[0].track_id == 1
    assert second.diagnostics.reset is True


def test_yaw_wrapping_around_pi_boundary():
    trk = tracker(min_hits=1)
    first_det = det3d(yaw=math.pi - 0.02)
    trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[first_det], fusion_result=fusion(lidar_only=[first_det]))
    second_det = det3d(yaw=-math.pi + 0.02)
    result = trk.update(recording_id="rec", frame_index=1, timestamp=1.1, camera_detections=[], lidar_detections=[second_det], fusion_result=fusion(lidar_only=[second_det]))
    assert -math.pi <= result.tracked_3d[0].detection.yaw < math.pi
    assert abs(wrap_angle(result.tracked_3d[0].detection.yaw - math.pi)) < 0.2


def test_invalid_or_nan_boxes_are_rejected_safely():
    trk = tracker(min_hits=1)
    invalid = det3d()
    invalid = Detection3D(
        center_xyz=(np.nan, 0.0, 0.0),
        size_lwh=invalid.size_lwh,
        yaw=0.0,
        score=0.9,
        label_id=0,
    )
    result = trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[invalid], fusion_result=fusion(lidar_only=[invalid]))
    assert result.tracked_3d == ()
    assert result.diagnostics.invalid_detections_rejected == 1


def test_duplicate_timestamps_do_not_crash_tracker():
    trk = tracker(min_hits=1)
    first = det3d()
    trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[first], fusion_result=fusion(lidar_only=[first]))
    second = det3d(x=5.01)
    result = trk.update(recording_id="rec", frame_index=1, timestamp=1.0, camera_detections=[], lidar_detections=[second], fusion_result=fusion(lidar_only=[second]))
    assert result.diagnostics.dt == 0.0
    assert result.tracked_3d[0].track_id == 1


def test_large_timestamp_gaps_are_handled_deterministically():
    trk = tracker(min_hits=1, max_age_frames=10, max_age_seconds=100.0)
    first = det3d()
    trk.update(recording_id="rec", frame_index=0, timestamp=1.0, camera_detections=[], lidar_detections=[first], fusion_result=fusion(lidar_only=[first]))
    second = det3d(x=5.01)
    result = trk.update(recording_id="rec", frame_index=1, timestamp=10.0, camera_detections=[], lidar_detections=[second], fusion_result=fusion(lidar_only=[second]))
    assert result.diagnostics.dt == trk.config.max_prediction_dt
    assert result.tracked_3d[0].track_id == 1


def test_camera_only_track_creation():
    trk = tracker(min_hits=1)
    cam = det2d()
    result = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[cam],
        lidar_detections=[],
        fusion_result=fusion(camera_only=[cam]),
    )
    assert len(result.tracked_2d) == 1
    assert result.tracked_2d[0].track_state == "confirmed"


def test_deepfusionmot_strict_uses_even_3d_and_odd_2d_ids():
    trk = deepfusionmot_strict_tracker()
    lidar = det3d()
    cam = det2d()
    result = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[cam],
        lidar_detections=[lidar],
        fusion_result=fusion(lidar_only=[lidar], camera_only=[cam]),
    )
    assert result.tracked_3d[0].track_id == 0
    assert result.tracked_2d[0].track_id == 1


def test_deepfusionmot_strict_deletes_unmatched_tentative_tracks():
    trk = deepfusionmot_strict_tracker()
    lidar = det3d()
    first = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[lidar],
        fusion_result=fusion(lidar_only=[lidar]),
    )
    assert first.tracked_3d[0].track_state == "tentative"
    missed = trk.update(
        recording_id="rec",
        frame_index=1,
        timestamp=1.1,
        camera_detections=[],
        lidar_detections=[],
        fusion_result=fusion(),
    )
    assert missed.tracked_3d == ()
    assert missed.diagnostics.active_3d_tracks == 0
    assert missed.diagnostics.deleted_tracks == 1


def test_deepfusionmot_strict_outputs_only_confirmed_or_early_frame_tracks():
    trk = deepfusionmot_strict_tracker()
    for index in range(3):
        trk.update(
            recording_id="rec",
            frame_index=index,
            timestamp=1.0 + index * 0.1,
            camera_detections=[],
            lidar_detections=[],
            fusion_result=fusion(),
        )
    lidar = det3d()
    result = trk.update(
        recording_id="rec",
        frame_index=3,
        timestamp=1.3,
        camera_detections=[],
        lidar_detections=[lidar],
        fusion_result=fusion(lidar_only=[lidar]),
    )
    assert result.diagnostics.active_3d_tracks == 1
    assert result.tracked_3d == ()


def test_lidar_nms_keeps_highest_scoring_duplicate_box():
    low = det3d(x=5.0, score=0.20)
    high = det3d(x=5.05, score=0.90)

    kept = non_max_suppress_3d([low, high], threshold=0.10)

    assert kept == (high,)


def test_low_score_lidar_only_detection_does_not_start_new_track():
    cfg = DeepFusionMOTTrackerConfig(
        min_hits=1,
        association_3d_threshold=0.01,
        lidar_only_initiation_score_threshold=0.20,
    )
    trk = DeepFusionMOTTracker(cfg)
    weak_lidar = det3d(score=0.10)

    result = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[weak_lidar],
        fusion_result=fusion(lidar_only=[weak_lidar]),
    )

    assert result.tracked_3d == ()
    assert result.diagnostics.active_3d_tracks == 0
    assert result.diagnostics.extra["low_score_lidar_only_initiations_suppressed"] == 1


def test_low_score_lidar_detection_can_start_track_when_camera_supported():
    cfg = DeepFusionMOTTrackerConfig(
        min_hits=1,
        association_3d_threshold=0.01,
        lidar_only_initiation_score_threshold=0.20,
    )
    trk = DeepFusionMOTTracker(cfg)
    weak_supported = det3d(score=0.10)

    result = trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[det2d()],
        lidar_detections=[weak_supported],
        fusion_result=fusion(matched=[weak_supported]),
    )

    assert len(result.tracked_3d) == 1
    assert result.tracked_3d[0].update_source == "matched_camera_lidar"


def test_low_score_lidar_only_detection_can_update_existing_track():
    cfg = DeepFusionMOTTrackerConfig(
        min_hits=1,
        association_3d_threshold=0.01,
        association_3d_metric="center_distance",
        max_center_distance_m=2.0,
        lidar_only_initiation_score_threshold=0.20,
    )
    trk = DeepFusionMOTTracker(cfg)
    strong = det3d(x=5.0, score=0.90)
    trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[strong],
        fusion_result=fusion(lidar_only=[strong]),
    )
    weak_followup = det3d(x=5.05, score=0.10)

    result = trk.update(
        recording_id="rec",
        frame_index=1,
        timestamp=1.1,
        camera_detections=[],
        lidar_detections=[weak_followup],
        fusion_result=fusion(lidar_only=[weak_followup]),
    )

    assert len(result.tracked_3d) == 1
    assert result.tracked_3d[0].track_id == 1
    assert result.tracked_3d[0].update_source == "lidar_only"
    assert result.diagnostics.extra["low_score_lidar_only_initiations_suppressed"] == 0


def test_aghri_odom_relative_lidar_transform_from_jsonl(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "odom_global.jsonl").write_text(
        "\n".join(
            [
                '{"t": 1.0, "p": {"x": 0.0, "y": 0.0, "z": 0.0}, "q": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}',
                '{"t": 2.0, "p": {"x": 1.0, "y": 0.0, "z": 0.0}, "q": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ego = AghriOdomEgoMotion.from_recording_dir(tmp_path, np.eye(4), max_time_delta_sec=0.1)
    current_from_previous = ego.relative_lidar_transform(1.0, 2.0)
    assert current_from_previous is not None
    point = current_from_previous @ np.asarray([5.0, 0.0, 0.0, 1.0])
    assert np.allclose(point[:3], [4.0, 0.0, 0.0])


def test_tracker_applies_aghri_odom_transform_before_association():
    cfg = DeepFusionMOTTrackerConfig(
        min_hits=1,
        max_age_frames=2,
        max_age_seconds=10.0,
        association_3d_threshold=0.01,
        association_3d_metric="center_distance",
        max_center_distance_m=1.0,
        ego_motion_mode="aghri_odom",
    )
    trk = DeepFusionMOTTracker(cfg)
    first = det3d(x=5.0)
    trk.update(
        recording_id="rec",
        frame_index=0,
        timestamp=1.0,
        camera_detections=[],
        lidar_detections=[first],
        fusion_result=fusion(lidar_only=[first]),
    )
    current_from_previous = np.eye(4)
    current_from_previous[0, 3] = -1.0
    second = det3d(x=4.0)
    result = trk.update(
        recording_id="rec",
        frame_index=1,
        timestamp=2.0,
        camera_detections=[],
        lidar_detections=[second],
        fusion_result=fusion(lidar_only=[second]),
        ego_motion_transform=current_from_previous,
    )
    assert len(result.tracked_3d) == 1
    assert result.tracked_3d[0].track_id == 1
