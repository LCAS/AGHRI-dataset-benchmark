import socket
from pathlib import Path

import numpy as np
import pytest

from late_fusion_pkg.live_conversions import decode_points, encode_points, finite_xyzi
from late_fusion_pkg.live_detector_profiles import (
    resolve_live_camera_profile,
    resolve_live_lidar_profile,
)
from late_fusion_pkg.live_worker_protocol import recv_json, send_json
from late_fusion_pkg.online_launch_modes import (
    assert_online_launch_mode,
    normalize_inference_mode,
    process_kinds_for_mode,
)


REPO_ROOT = Path("/home/prabuddhi/Desktop/late_fusion/deepfusionmot")


def test_online_mode_helpers_separate_live_and_cached():
    assert normalize_inference_mode("live") == "live"
    assert normalize_inference_mode("cached") == "cached"
    assert "live_yolo_detector" in process_kinds_for_mode("live")
    assert "cached_yolo_publisher" in process_kinds_for_mode("cached")
    assert assert_online_launch_mode("live") == "live"
    with pytest.raises(RuntimeError, match="fully online"):
        assert_online_launch_mode("cached")


def test_aghri_online_launch_starts_live_detectors_not_cached_publishers():
    launch_text = (REPO_ROOT / "launch/aghri_online_late_fusion.launch.py").read_text()
    helper_text = (REPO_ROOT / "late_fusion_pkg/online_ros_launch.py").read_text()
    setup_text = (REPO_ROOT / "setup.py").read_text()

    assert 'DeclareLaunchArgument("inference_mode", default_value="live")' in launch_text
    assert "assert_online_launch_mode" in helper_text
    assert "live_yolo_detector_node" in helper_text
    assert 'DeclareLaunchArgument("yolo_python", default_value="/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/bin/python")' in launch_text
    assert '"late_fusion_pkg.live_yolo_detector_node"' in helper_text
    assert "live_lidar_detector_node" in helper_text
    assert "online_performance_monitor" in setup_text
    assert "aghri_deepfusionmot_tracker_node" in setup_text
    assert "cached_yolo_publisher" not in helper_text
    assert "cached_second_publisher" not in helper_text
    assert "camera_source_label:=camera_live" in helper_text
    assert "lidar_source_label:=lidar_live" in helper_text


def test_online_launch_exposes_lidar_family_and_visualizer_controls():
    launch_text = (REPO_ROOT / "launch/aghri_online_late_fusion.launch.py").read_text()
    helper_text = (REPO_ROOT / "late_fusion_pkg/online_ros_launch.py").read_text()
    config_text = (REPO_ROOT / "config/aghri_visualization.yaml").read_text()

    assert 'DeclareLaunchArgument("lidar_detector", default_value="second")' in launch_text
    assert 'DeclareLaunchArgument("show_original_lidar_markers", default_value="true")' in launch_text
    assert 'DeclareLaunchArgument("show_unmatched_lidar_markers", default_value="true")' in launch_text
    assert 'DeclareLaunchArgument("show_fused_projection_image", default_value="true")' in launch_text
    assert 'DeclareLaunchArgument("lidar_score_threshold", default_value="0.20")' in launch_text
    assert "lidar_detector:={lidar.detector_family}" in helper_text
    assert "score_threshold_override:={_value(context, 'lidar_score_threshold')}" in helper_text
    assert "publish_unmatched_lidar_markers:={show_unmatched_lidar_markers}" in helper_text
    assert "draw_fused_projection:={show_fused_projection_image}" in helper_text
    visualizer_index = helper_text.index('package_lib + "/aghri_late_fusion_visualizer"')
    marker_index = helper_text.index("publish_original_lidar_markers:={show_original_lidar_markers}")
    assert visualizer_index < marker_index
    assert "marker_lifetime_sec: 1.00" in config_text


def test_online_launch_exposes_deepfusionmot_tracking_stage():
    launch_text = (REPO_ROOT / "launch/aghri_online_late_fusion.launch.py").read_text()
    helper_text = (REPO_ROOT / "late_fusion_pkg/online_ros_launch.py").read_text()
    visualizer_text = (REPO_ROOT / "late_fusion_pkg/aghri_visualization_node.py").read_text()
    rviz_text = (REPO_ROOT / "rviz/aghri_generic_late_fusion.rviz").read_text()

    assert 'DeclareLaunchArgument("enable_tracking", default_value="false")' in launch_text
    assert 'DeclareLaunchArgument("tracking_association_3d_threshold", default_value="0.03")' in launch_text
    assert 'DeclareLaunchArgument("tracking_association_2d_threshold", default_value="0.40")' in launch_text
    assert 'DeclareLaunchArgument("tracking_support_2d_to_3d_threshold", default_value="0.40")' in launch_text
    assert "aghri_deepfusionmot_tracker_node" in helper_text
    assert "matched_3d_topic:={_value(context, 'matched_3d_topic')}" in helper_text
    assert "tracked_3d_topic:={tracked_3d_topic}" in helper_text
    assert "tracking_diagnostics_topic:={tracking_diag}" in helper_text
    assert "publish_tracked_markers:={show_tracked_markers}" in helper_text
    assert "draw_tracked_projection:={show_tracked_projection_image}" in helper_text
    assert "tracked_output_cb" in visualizer_text
    assert "/visualization/tracked_3d_markers" in rviz_text
    assert "/visualization/tracked_projection_image" in rviz_text


def test_live_lidar_worker_uses_local_aghri_base_configs():
    node_text = (REPO_ROOT / "late_fusion_pkg/live_lidar_detector_node.py").read_text()
    second_base = (
        "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
        "3d-detection/benchmarks/mmdetection3d/configs/models/second_aghri.py"
    )
    pointpillars_base = (
        "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
        "3d-detection/benchmarks/mmdetection3d/configs/models/pointpillars_aghri.py"
    )

    assert second_base in node_text
    assert pointpillars_base in node_text
    assert "worker_ready" in node_text
    assert Path(second_base).exists()
    assert Path(pointpillars_base).exists()
    assert "/home/prabuddhi/cluster/aghri_second_finetune/code/mmdetection3d/configs/models/second_aghri.py" not in node_text
    assert "/home/prabuddhi/cluster/aghri_pointpillars_finetune/code/mmdetection3d/configs/models/pointpillars_aghri.py" not in node_text


def test_aghri_online_fps_matrix_runner_covers_all_detector_combinations():
    runner_text = (REPO_ROOT / "tools/run_aghri_online_fps_matrix.py").read_text()
    expected = {
        '("S1", "generic", "second", "generic")',
        '("S2", "finetuned", "second", "generic")',
        '("S3", "generic", "second", "finetuned")',
        '("S4", "finetuned", "second", "finetuned")',
        '("P1", "generic", "pointpillars", "generic")',
        '("P2", "finetuned", "pointpillars", "generic")',
        '("P3", "generic", "pointpillars", "finetuned")',
        '("P4", "finetuned", "pointpillars", "finetuned")',
    }

    for item in expected:
        assert item in runner_text
    assert "inference_mode:=live" in runner_text
    assert "play_bag:=false" in runner_text
    assert "enable_visualisation:=false" in runner_text
    assert "launch_rviz:=false" in runner_text
    assert "cached_yolo_publisher" not in runner_text
    assert "cached_second_publisher" not in runner_text
    assert "PYTHONNOUSERSITE" not in runner_text


def test_live_camera_profile_does_not_require_cache_paths():
    profile = resolve_live_camera_profile("aghri", "finetuned", validate_paths=False)
    assert profile.camera_model == "finetuned"
    assert "best.pt" in str(profile.checkpoint_path)
    assert profile.checkpoint_sha256 == "bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110"


def test_live_lidar_profile_keeps_finetuned_second_center_origin():
    profile = resolve_live_lidar_profile("aghri", "second", "finetuned", validate_paths=False)
    assert profile.detector_family == "second"
    assert profile.lidar_model == "finetuned"
    assert profile.box_origin == "center"
    assert profile.selected_class_name == "person"


def test_live_lidar_profile_keeps_finetuned_pointpillars_center_origin():
    profile = resolve_live_lidar_profile("aghri", "pointpillars", "finetuned", validate_paths=False)
    assert profile.detector_family == "pointpillars"
    assert profile.lidar_model == "finetuned"
    assert profile.box_origin == "center"
    assert profile.selected_class_name == "person"


def test_live_lidar_profile_supports_kitti_finetuned_pointpillars():
    profile = resolve_live_lidar_profile("kitti_tracking", "pointpillars", "finetuned", validate_paths=False)
    assert profile.detector_family == "pointpillars"
    assert profile.lidar_model == "finetuned"
    assert profile.selected_class_name == "person"
    assert profile.box_origin == "bottom_center"


def test_live_lidar_profile_allows_generic_kitti_pointpillars():
    profile = resolve_live_lidar_profile("kitti_tracking", "pointpillars", "generic", validate_paths=False)
    assert profile.detector_family == "pointpillars"
    assert profile.lidar_model == "generic"
    assert profile.box_origin == "bottom_center"
    assert profile.score_threshold == 0.1


def test_point_encoding_round_trip_filters_nonfinite_rows():
    points = np.array(
        [
            [1.0, 2.0, 3.0, 0.4],
            [np.nan, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 0.7],
        ],
        dtype=np.float32,
    )
    finite = finite_xyzi(points)
    encoded = encode_points(points)
    decoded = decode_points(encoded, finite.shape[0])
    assert decoded.dtype == np.float32
    assert np.allclose(decoded, finite)


def test_worker_protocol_length_prefixed_json_socketpair():
    left, right = socket.socketpair()
    try:
        payload = {"command": "ping", "nested": {"ok": True}, "values": [1, 2, 3]}
        send_json(left, payload)
        assert recv_json(right) == payload
    finally:
        left.close()
        right.close()
