import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():
    config_file = os.path.join(
            get_package_share_directory("late_fusion_pkg"),
            'config',
            'late_fusion_pkg.config.yaml'
            )

    return LaunchDescription([
        Node(
            package="late_fusion_pkg",
            executable='late_fusion_node',
            name='late_fusion_node',
            parameters=[config_file]
            )
        ])
