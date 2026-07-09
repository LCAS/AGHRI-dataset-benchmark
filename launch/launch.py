import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
            get_package_share_directory("camera_lidar_fusion"),
            'config',
            'config.yaml'
            )

    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=config_file,
            description='Path to the camera_lidar_fusion parameter file',
        ),
        Node(
            package="camera_lidar_fusion",
            executable='lidar_fusion',
            name='lidar_fusion',
            parameters=[params_file]
            )
        ])
