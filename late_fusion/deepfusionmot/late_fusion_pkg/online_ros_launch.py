"""Shared launch helpers for fully online ROS 2 late-fusion workflows."""

from __future__ import annotations

import os


def _value(context, name: str) -> str:
    from launch.substitutions import LaunchConfiguration

    return str(LaunchConfiguration(name).perform(context))


def _package_lib_dir(package_name: str) -> str:
    from ament_index_python.packages import get_package_prefix

    return os.path.join(get_package_prefix(package_name), "lib", package_name)


def online_runtime_actions(
    context,
    dataset_profile: str,
    default_lidar_frame: str,
    default_camera_frame: str,
    late_fusion_node_name: str,
    visualizer_node_name: str,
    tracker_node_name: str,
    default_recording_label: str,
):
    from launch.actions import EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler, TimerAction
    from launch.conditions import IfCondition
    from launch.event_handlers import OnProcessExit
    from launch.events import Shutdown

    from late_fusion_pkg.live_detector_profiles import (
        live_startup_summary,
        resolve_live_camera_profile,
        resolve_live_lidar_profile,
    )
    from late_fusion_pkg.online_launch_modes import assert_online_launch_mode

    assert_online_launch_mode(_value(context, "inference_mode"))

    repo_root = "/home/prabuddhi/Desktop/late_fusion/deepfusionmot"
    package_lib = _package_lib_dir("late_fusion_pkg")
    camera = resolve_live_camera_profile(dataset_profile, _value(context, "camera_model"), validate_paths=True)
    lidar = resolve_live_lidar_profile(
        dataset_profile,
        _value(context, "lidar_detector"),
        _value(context, "lidar_model"),
        validate_paths=True,
        allow_cross_domain=_value(context, "allow_cross_domain").lower() == "true",
    )
    use_sim_time = _value(context, "use_sim_time")
    bag_path = _value(context, "bag_path")
    playback_rate = _value(context, "playback_rate")
    play_bag = _value(context, "play_bag")
    bag_play = ExecuteProcess(
        condition=IfCondition(play_bag),
        cmd=["ros2", "bag", "play", bag_path, "-r", playback_rate, "--clock", "--qos-profile-overrides-path", _value(context, "qos_overrides")],
        output="screen",
    )
    camera_diag = _value(context, "camera_diagnostics_topic")
    lidar_diag = _value(context, "lidar_diagnostics_topic")
    fusion_diag = _value(context, "fusion_diagnostics_topic")
    visualization_diag = _value(context, "visualization_diagnostics_topic")
    camera_detection_topic = _value(context, "camera_detection_topic")
    lidar_detection_topic = _value(context, "lidar_detection_topic")
    show_original_lidar_markers = _value(context, "show_original_lidar_markers")
    show_matched_markers = _value(context, "show_matched_markers")
    show_fused_markers = _value(context, "show_fused_markers")
    show_tracked_markers = _value(context, "show_tracked_markers")
    show_unmatched_lidar_markers = _value(context, "show_unmatched_lidar_markers")
    show_camera_detection_image = _value(context, "show_camera_detection_image")
    show_fused_projection_image = _value(context, "show_fused_projection_image")
    show_tracked_projection_image = _value(context, "show_tracked_projection_image")
    show_original_lidar_projection = _value(context, "show_original_lidar_projection")
    tracking_diag = _value(context, "tracking_diagnostics_topic")
    tracked_3d_topic = _value(context, "tracked_3d_topic")
    tracked_2d_topic = _value(context, "tracked_2d_topic")

    return [
        *(LogInfo(msg=line) for line in live_startup_summary(camera, lidar)),
        LogInfo(msg=f"Online recording label: {default_recording_label}"),
        ExecuteProcess(
            cmd=[
                _value(context, "yolo_python"),
                "-m",
                "late_fusion_pkg.live_yolo_detector_node",
                "--ros-args",
                "-r",
                "__node:=online_yolo_detector",
                "-p",
                f"dataset_profile:={dataset_profile}",
                "-p",
                f"camera_model:={camera.camera_model}",
                "-p",
                f"checkpoint_path:={camera.checkpoint_path}",
                "-p",
                f"checkpoint_sha256:={camera.checkpoint_sha256}",
                "-p",
                f"input_topic:={_value(context, 'image_topic')}",
                "-p",
                f"output_topic:={camera_detection_topic}",
                "-p",
                f"diagnostics_topic:={camera_diag}",
                "-p",
                f"device:={_value(context, 'detector_device')}",
                "-p",
                f"imgsz:={_value(context, 'yolo_imgsz')}",
                "-p",
                f"conf:={_value(context, 'yolo_conf')}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            additional_env={
                "PYTHONPATH": repo_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "PYTHONNOUSERSITE": "",
            },
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                package_lib + "/live_lidar_detector_node",
                "--ros-args",
                "-r",
                "__node:=online_lidar_detector",
                "-p",
                f"dataset_profile:={dataset_profile}",
                "-p",
                f"lidar_detector:={lidar.detector_family}",
                "-p",
                f"lidar_model:={lidar.lidar_model}",
                "-p",
                f"allow_cross_domain:={_value(context, 'allow_cross_domain')}",
                "-p",
                f"input_topic:={_value(context, 'pointcloud_topic')}",
                "-p",
                f"output_topic:={lidar_detection_topic}",
                "-p",
                f"diagnostics_topic:={lidar_diag}",
                "-p",
                f"frame_id:={default_lidar_frame}",
                "-p",
                f"device:={_value(context, 'mmdet3d_device')}",
                "-p",
                f"worker_python:={_value(context, 'worker_python')}",
                "-p",
                f"worker_timeout_sec:={_value(context, 'worker_timeout_sec')}",
                "-p",
                f"score_threshold_override:={_value(context, 'lidar_score_threshold')}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                package_lib + "/aghri_late_fusion_node",
                "--ros-args",
                "-r",
                f"__node:={late_fusion_node_name}",
                "--params-file",
                _value(context, "fusion_config"),
                "-p",
                f"camera_detections_topic:={camera_detection_topic}",
                "-p",
                f"lidar_detections_topic:={lidar_detection_topic}",
                "-p",
                f"camera_info_topic:={_value(context, 'camera_info_topic')}",
                "-p",
                f"fused_topic:={_value(context, 'fused_3d_topic')}",
                "-p",
                f"matched_topic:={_value(context, 'matched_3d_topic')}",
                "-p",
                f"unmatched_lidar_topic:={_value(context, 'unmatched_lidar_topic')}",
                "-p",
                f"unmatched_camera_topic:={_value(context, 'unmatched_camera_topic')}",
                "-p",
                f"diagnostics_topic:={fusion_diag}",
                "-p",
                f"lidar_frame:={default_lidar_frame}",
                "-p",
                f"camera_frame:={default_camera_frame}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            condition=IfCondition(_value(context, "enable_tracking")),
            cmd=[
                package_lib + "/aghri_deepfusionmot_tracker_node",
                "--ros-args",
                "-r",
                f"__node:={tracker_node_name}",
                "-p",
                f"matched_3d_topic:={_value(context, 'matched_3d_topic')}",
                "-p",
                f"unmatched_lidar_topic:={_value(context, 'unmatched_lidar_topic')}",
                "-p",
                f"unmatched_camera_topic:={_value(context, 'unmatched_camera_topic')}",
                "-p",
                f"tracked_3d_topic:={tracked_3d_topic}",
                "-p",
                f"tracked_2d_topic:={tracked_2d_topic}",
                "-p",
                f"diagnostics_topic:={tracking_diag}",
                "-p",
                f"recording_id:={default_recording_label}",
                "-p",
                f"min_hits:={_value(context, 'tracking_min_hits')}",
                "-p",
                f"max_age_frames:={_value(context, 'tracking_max_age_frames')}",
                "-p",
                f"association_3d_threshold:={_value(context, 'tracking_association_3d_threshold')}",
                "-p",
                f"association_2d_threshold:={_value(context, 'tracking_association_2d_threshold')}",
                "-p",
                f"support_2d_to_3d_threshold:={_value(context, 'tracking_support_2d_to_3d_threshold')}",
                "-p",
                f"ego_motion_mode:={_value(context, 'tracking_ego_motion_mode')}",
                "-p",
                f"fixed_frame:={_value(context, 'tracking_fixed_frame')}",
                "-p",
                f"lidar_frame:={default_lidar_frame}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                package_lib + "/cache_diagnostics_collector",
                "--ros-args",
                "-r",
                "__node:=online_diagnostics_collector",
                "-p",
                f"output_jsonl:={_value(context, 'diagnostics_output')}",
                "-p",
                f"camera_stats_topic:={camera_diag}",
                "-p",
                f"lidar_stats_topic:={lidar_diag}",
                "-p",
                f"fusion_diagnostics_topic:={fusion_diag}",
                "-p",
                f"tracking_diagnostics_topic:={tracking_diag}",
                "-p",
                f"visualization_diagnostics_topic:={visualization_diag}",
                "-p",
                "camera_source_label:=camera_live",
                "-p",
                "lidar_source_label:=lidar_live",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            condition=IfCondition(_value(context, "enable_visualisation")),
            cmd=[
                package_lib + "/aghri_late_fusion_visualizer",
                "--ros-args",
                "-r",
                f"__node:={visualizer_node_name}",
                "--params-file",
                _value(context, "visualization_config"),
                "-p",
                f"camera_image_topic:={_value(context, 'image_topic')}",
                "-p",
                f"camera_info_topic:={_value(context, 'camera_info_topic')}",
                "-p",
                f"camera_detections_topic:={camera_detection_topic}",
                "-p",
                f"lidar_detections_topic:={lidar_detection_topic}",
                "-p",
                f"matched_3d_topic:={_value(context, 'matched_3d_topic')}",
                "-p",
                f"fused_3d_topic:={_value(context, 'fused_3d_topic')}",
                "-p",
                f"unmatched_lidar_topic:={_value(context, 'unmatched_lidar_topic')}",
                "-p",
                f"unmatched_camera_topic:={_value(context, 'unmatched_camera_topic')}",
                "-p",
                f"tracked_3d_topic:={tracked_3d_topic}",
                "-p",
                f"diagnostics_topic:={visualization_diag}",
                "-p",
                f"lidar_frame:={default_lidar_frame}",
                "-p",
                f"camera_frame:={default_camera_frame}",
                "-p",
                f"publish_original_lidar_markers:={show_original_lidar_markers}",
                "-p",
                f"publish_matched_markers:={show_matched_markers}",
                "-p",
                f"publish_fused_markers:={show_fused_markers}",
                "-p",
                f"publish_tracked_markers:={show_tracked_markers}",
                "-p",
                f"publish_unmatched_lidar_markers:={show_unmatched_lidar_markers}",
                "-p",
                f"draw_camera_detections:={show_camera_detection_image}",
                "-p",
                f"draw_fused_projection:={show_fused_projection_image}",
                "-p",
                f"draw_tracked_projection:={show_tracked_projection_image}",
                "-p",
                f"draw_original_lidar_projection:={show_original_lidar_projection}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(condition=IfCondition(_value(context, "launch_rviz")), cmd=["rviz2", "-d", _value(context, "rviz_config")], output="screen"),
        TimerAction(period=float(_value(context, "bag_start_delay_sec")), actions=[bag_play]),
        RegisterEventHandler(OnProcessExit(target_action=bag_play, on_exit=[EmitEvent(event=Shutdown(reason="online late-fusion bag playback complete"))])),
    ]
