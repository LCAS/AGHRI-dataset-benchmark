"""Launch fully online AGHRI ROS 2 late fusion.

This launch file starts live YOLO and live MMDetection3D detector nodes. It is
separate from cached replay launches and rejects inference_mode:=cached.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction


def _runtime_actions(context):
    from late_fusion_pkg.online_ros_launch import online_runtime_actions

    return online_runtime_actions(
        context,
        dataset_profile="aghri",
        default_lidar_frame="front_lidar_link",
        default_camera_frame="front_left_camera_optical_frame",
        late_fusion_node_name="aghri_late_fusion",
        visualizer_node_name="aghri_late_fusion_visualizer",
        tracker_node_name="aghri_deepfusionmot_tracker",
        default_recording_label="aghri_online",
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("inference_mode", default_value="live"),
            DeclareLaunchArgument("bag_path", default_value="/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test/footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"),
            DeclareLaunchArgument("camera_model", default_value="finetuned"),
            DeclareLaunchArgument("lidar_detector", default_value="second"),
            DeclareLaunchArgument("lidar_model", default_value="finetuned"),
            DeclareLaunchArgument("allow_cross_domain", default_value="false"),
            DeclareLaunchArgument("image_topic", default_value="/dataset/cam_zed_rgb/image"),
            DeclareLaunchArgument("camera_info_topic", default_value="/dataset/cam_zed_rgb/camera_info"),
            DeclareLaunchArgument("pointcloud_topic", default_value="/dataset/lidar/points"),
            DeclareLaunchArgument("camera_detection_topic", default_value="/detector/camera/detections"),
            DeclareLaunchArgument("lidar_detection_topic", default_value="/detector/lidar/detections"),
            DeclareLaunchArgument("fused_3d_topic", default_value="/late_fusion/fused_3d"),
            DeclareLaunchArgument("matched_3d_topic", default_value="/late_fusion/matched_3d"),
            DeclareLaunchArgument("unmatched_lidar_topic", default_value="/late_fusion/unmatched_lidar"),
            DeclareLaunchArgument("unmatched_camera_topic", default_value="/late_fusion/unmatched_camera"),
            DeclareLaunchArgument("tracked_3d_topic", default_value="/deepfusionmot/tracked_3d"),
            DeclareLaunchArgument("tracked_2d_topic", default_value="/deepfusionmot/tracked_2d"),
            DeclareLaunchArgument("camera_diagnostics_topic", default_value="/detector/camera/live_stats"),
            DeclareLaunchArgument("lidar_diagnostics_topic", default_value="/detector/lidar/live_stats"),
            DeclareLaunchArgument("fusion_diagnostics_topic", default_value="/late_fusion/diagnostics"),
            DeclareLaunchArgument("tracking_diagnostics_topic", default_value="/deepfusionmot/diagnostics"),
            DeclareLaunchArgument("visualization_diagnostics_topic", default_value="/visualization/diagnostics"),
            DeclareLaunchArgument("fusion_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/config/aghri_generic_late_fusion.yaml"),
            DeclareLaunchArgument("visualization_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/config/aghri_visualization.yaml"),
            DeclareLaunchArgument("rviz_config", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/rviz/aghri_generic_late_fusion.rviz"),
            DeclareLaunchArgument("diagnostics_output", default_value="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/results/online_ros2_late_fusion/aghri/live_diagnostics.jsonl"),
            DeclareLaunchArgument("detector_device", default_value="auto"),
            DeclareLaunchArgument("mmdet3d_device", default_value="cuda:0"),
            DeclareLaunchArgument("yolo_python", default_value="/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/bin/python"),
            DeclareLaunchArgument("worker_python", default_value="/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/bin/python"),
            DeclareLaunchArgument("worker_timeout_sec", default_value="600.0"),
            DeclareLaunchArgument("yolo_imgsz", default_value="640"),
            DeclareLaunchArgument("yolo_conf", default_value="0.10"),
            DeclareLaunchArgument("lidar_score_threshold", default_value="0.20"),
            DeclareLaunchArgument("enable_tracking", default_value="false"),
            DeclareLaunchArgument("tracking_min_hits", default_value="3"),
            DeclareLaunchArgument("tracking_max_age_frames", default_value="10"),
            DeclareLaunchArgument("tracking_association_3d_threshold", default_value="0.03"),
            DeclareLaunchArgument("tracking_association_2d_threshold", default_value="0.40"),
            DeclareLaunchArgument("tracking_support_2d_to_3d_threshold", default_value="0.40"),
            DeclareLaunchArgument("tracking_ego_motion_mode", default_value="tf"),
            DeclareLaunchArgument("tracking_fixed_frame", default_value="map"),
            DeclareLaunchArgument("enable_visualisation", default_value="true"),
            DeclareLaunchArgument("show_original_lidar_markers", default_value="true"),
            DeclareLaunchArgument("show_matched_markers", default_value="true"),
            DeclareLaunchArgument("show_fused_markers", default_value="false"),
            DeclareLaunchArgument("show_tracked_markers", default_value="true"),
            DeclareLaunchArgument("show_unmatched_lidar_markers", default_value="true"),
            DeclareLaunchArgument("show_camera_detection_image", default_value="true"),
            DeclareLaunchArgument("show_fused_projection_image", default_value="true"),
            DeclareLaunchArgument("show_tracked_projection_image", default_value="true"),
            DeclareLaunchArgument("show_original_lidar_projection", default_value="false"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("playback_rate", default_value="1.0"),
            DeclareLaunchArgument("play_bag", default_value="true"),
            DeclareLaunchArgument("bag_start_delay_sec", default_value="12.0"),
            DeclareLaunchArgument("qos_overrides", default_value="/home/prabuddhi/Desktop/agri-human-dataset-benchmark/early_fusion/camera_lidar_humble/config/aghri_rosbag2_qos_overrides.yaml"),
            OpaqueFunction(function=_runtime_actions),
        ]
    )
