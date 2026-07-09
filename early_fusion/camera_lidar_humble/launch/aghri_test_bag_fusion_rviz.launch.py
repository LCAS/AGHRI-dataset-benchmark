import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    package_name = "camera_lidar_fusion"
    package_share = get_package_share_directory(package_name)
    package_prefix = get_package_prefix(package_name)
    fusion_executable = os.path.join(
        package_prefix,
        "lib",
        package_name,
        "lidar_fusion",
    )
    default_params = os.path.join(
        package_share,
        "config",
        "aghri_zed_livox_yolo11s_finetuned.yaml",
    )
    default_rviz = os.path.join(
        package_share,
        "rviz",
        "aghri_zed_livox_fusion.rviz",
    )
    default_qos_overrides = os.path.join(
        package_share,
        "config",
        "aghri_rosbag2_qos_overrides.yaml",
    )

    bag_path = LaunchConfiguration("bag_path")
    params_file = LaunchConfiguration("params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    playback_rate = LaunchConfiguration("playback_rate")
    qos_overrides = LaunchConfiguration("qos_overrides")
    loop = LaunchConfiguration("loop")
    start_offset = LaunchConfiguration("start_offset")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autoplay = LaunchConfiguration("autoplay")
    launch_rviz = LaunchConfiguration("launch_rviz")

    autoplay_without_loop = IfCondition(
        PythonExpression(["'", autoplay, "' == 'true' and '", loop, "' != 'true'"])
    )
    autoplay_with_loop = IfCondition(
        PythonExpression(["'", autoplay, "' == 'true' and '", loop, "' == 'true'"])
    )

    bag_play_base = [
        "ros2",
        "bag",
        "play",
        bag_path,
        "--clock",
        "--rate",
        playback_rate,
        "--start-offset",
        start_offset,
        "--qos-profile-overrides-path",
        qos_overrides,
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value="",
            description="Path to one converted AGHRI ROS 2 bag",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Fusion parameter YAML",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz,
            description="RViz2 configuration",
        ),
        DeclareLaunchArgument(
            "playback_rate",
            default_value="1.0",
            description="ros2 bag play rate",
        ),
        DeclareLaunchArgument(
            "qos_overrides",
            default_value=default_qos_overrides,
            description="ros2 bag play QoS overrides; makes /tf_static transient_local",
        ),
        DeclareLaunchArgument(
            "loop",
            default_value="false",
            description="Loop bag playback",
        ),
        DeclareLaunchArgument(
            "start_offset",
            default_value="0.0",
            description="Playback start offset in seconds",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock from ros2 bag play",
        ),
        DeclareLaunchArgument(
            "autoplay",
            default_value="true",
            description="Start ros2 bag play automatically",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Start RViz2 with the AGHRI fusion display",
        ),
        ExecuteProcess(
            cmd=[
                fusion_executable,
                "--ros-args",
                "--params-file",
                params_file,
                "-p",
                ["use_sim_time:=", use_sim_time],
            ],
            name="lidar_fusion",
            output="screen",
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config, "--ros-args", "-p", ["use_sim_time:=", use_sim_time]],
            name="rviz2",
            condition=IfCondition(launch_rviz),
            output="screen",
        ),
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=bag_play_base,
                    name="aghri_bag_play",
                    condition=autoplay_without_loop,
                    output="screen",
                ),
                ExecuteProcess(
                    cmd=bag_play_base + ["--loop"],
                    name="aghri_bag_play_loop",
                    condition=autoplay_with_loop,
                    output="screen",
                ),
            ],
        ),
    ])
