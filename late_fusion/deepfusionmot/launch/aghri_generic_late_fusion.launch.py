"""Launch cached AGHRI generic detector replay without importing launch_ros.

This launch file intentionally uses ExecuteProcess + installed executables instead of
``launch_ros.actions.Node`` because the local ROS Humble installation can have
``launch_ros`` newer than ``launch``. In that state, importing launch_ros fails
before a launch description can load.
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


def _value(context, name: str) -> str:
    return str(LaunchConfiguration(name).perform(context))


def _launch_runtime_actions(context):
    from ament_index_python.packages import get_package_prefix

    from late_fusion_pkg.aghri_detector_profiles import resolve_detector_selection, selection_summary_lines

    package_lib = os.path.join(get_package_prefix("late_fusion_pkg"), "lib", "late_fusion_pkg")
    selection = resolve_detector_selection(
        camera_model=_value(context, "camera_model"),
        lidar_detector=_value(context, "lidar_detector"),
        lidar_model=_value(context, "lidar_model"),
        detector_profile=_value(context, "detector_profile"),
        yolo_cache_override=_value(context, "YOLO_cache"),
        second_cache_override=_value(context, "SECOND_cache"),
    )
    bag_path = _value(context, "bag_path")
    recording_name = _value(context, "recording_name")
    fusion_config = _value(context, "fusion_config")
    visualization_config = _value(context, "visualization_config")
    rviz_config = _value(context, "rviz_config")
    launch_rviz = _value(context, "launch_rviz")
    enable_visualisation = _value(context, "enable_visualisation")
    diagnostics_output = _value(context, "diagnostics_output")
    show_original_lidar_markers = _value(context, "show_original_lidar_markers")
    show_matched_markers = _value(context, "show_matched_markers")
    show_fused_markers = _value(context, "show_fused_markers")
    show_unmatched_lidar_markers = _value(context, "show_unmatched_lidar_markers")
    show_camera_detection_image = _value(context, "show_camera_detection_image")
    show_fused_projection_image = _value(context, "show_fused_projection_image")
    use_sim_time = _value(context, "use_sim_time")
    playback_rate = _value(context, "playback_rate")
    play_bag = _value(context, "play_bag")
    qos_overrides = _value(context, "qos_overrides")

    bag_play_action = ExecuteProcess(
        condition=IfCondition(play_bag),
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "-r",
            playback_rate,
            "--clock",
            "--qos-profile-overrides-path",
            qos_overrides,
        ],
        output="screen",
    )

    return [
        *(LogInfo(msg=line) for line in selection_summary_lines(selection)),
        ExecuteProcess(
            cmd=[
                package_lib + "/cached_yolo_publisher",
                "--ros-args",
                "-r",
                "__node:=cached_yolo11s_replay",
                "-p",
                f"cache_csv:={selection.camera.cache_csv}",
                "-p",
                f"recording_name:={recording_name}",
                "-p",
                f"cache_label:=camera_{selection.camera.model}",
                "-p",
                f"checkpoint_sha256:={selection.camera.checkpoint_sha256}",
                "-p",
                f"checkpoint_path:={selection.camera.checkpoint_path}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            cmd=[
                package_lib + "/cached_second_publisher",
                "--ros-args",
                "-r",
                "__node:=cached_second_replay",
                "-p",
                f"cache_csv:={selection.lidar.cache_csv}",
                "-p",
                f"recording_name:={recording_name}",
                "-p",
                f"detector_family:={selection.lidar.detector_family}",
                "-p",
                f"cache_label:=lidar_{selection.lidar.model}",
                "-p",
                f"cache_z_origin:={selection.lidar.box_origin}",
                "-p",
                f"checkpoint_sha256:={selection.lidar.checkpoint_sha256}",
                "-p",
                f"checkpoint_path:={selection.lidar.checkpoint_path}",
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
                "__node:=aghri_late_fusion",
                "--params-file",
                fusion_config,
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
                "__node:=cache_diagnostics_collector",
                "-p",
                f"output_jsonl:={diagnostics_output}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        ExecuteProcess(
            condition=IfCondition(enable_visualisation),
            cmd=[
                package_lib + "/aghri_late_fusion_visualizer",
                "--ros-args",
                "-r",
                "__node:=aghri_late_fusion_visualizer",
                "--params-file",
                visualization_config,
                "-p",
                f"publish_original_lidar_markers:={show_original_lidar_markers}",
                "-p",
                f"publish_matched_markers:={show_matched_markers}",
                "-p",
                f"publish_fused_markers:={show_fused_markers}",
                "-p",
                f"publish_unmatched_lidar_markers:={show_unmatched_lidar_markers}",
                "-p",
                f"draw_camera_detections:={show_camera_detection_image}",
                "-p",
                f"draw_fused_projection:={show_fused_projection_image}",
                "-p",
                f"use_sim_time:={use_sim_time}",
            ],
            output="screen",
        ),
        TimerAction(
            period=8.0,
            actions=[bag_play_action],
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=bag_play_action,
                on_exit=[EmitEvent(event=Shutdown(reason="rosbag playback complete"))],
            )
        ),
        ExecuteProcess(
            condition=IfCondition(launch_rviz),
            cmd=[
                "rviz2",
                "-d",
                rviz_config,
            ],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_path", default_value="/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test/footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"),
            DeclareLaunchArgument("recording_name", default_value="footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"),
            DeclareLaunchArgument("camera_model", default_value="finetuned"),
            DeclareLaunchArgument("lidar_detector", default_value="second"),
            DeclareLaunchArgument("lidar_model", default_value="finetuned"),
            DeclareLaunchArgument("detector_profile", default_value="custom"),
            DeclareLaunchArgument("YOLO_cache", default_value=""),
            DeclareLaunchArgument("SECOND_cache", default_value=""),
            DeclareLaunchArgument("fusion_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/config/aghri_generic_late_fusion.yaml"),
            DeclareLaunchArgument("visualization_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/config/aghri_visualization.yaml"),
            DeclareLaunchArgument("rviz_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/rviz/aghri_generic_late_fusion.rviz"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("enable_visualisation", default_value="false"),
            DeclareLaunchArgument("show_original_lidar_markers", default_value="false"),
            DeclareLaunchArgument("show_matched_markers", default_value="true"),
            DeclareLaunchArgument("show_fused_markers", default_value="false"),
            DeclareLaunchArgument("show_unmatched_lidar_markers", default_value="true"),
            DeclareLaunchArgument("show_camera_detection_image", default_value="true"),
            DeclareLaunchArgument("show_fused_projection_image", default_value="true"),
            DeclareLaunchArgument("diagnostics_output", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/results/aghri_generic_baseline/rosbag_evaluation/live_diagnostics.jsonl"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument("play_bag", default_value="true"),
            DeclareLaunchArgument("qos_overrides", default_value="/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/config/aghri_rosbag2_qos_overrides.yaml"),
            OpaqueFunction(function=_launch_runtime_actions),
        ]
    )
