from glob import glob
from os.path import isfile
from setuptools import find_packages, setup

package_name = 'late_fusion_pkg'


def files(pattern):
    return [path for path in glob(pattern) if isfile(path)]

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        include=[
            'late_fusion_pkg',
            'late_fusion_pkg.*',
            'late_fusion_scripts',
            'late_fusion_scripts.*'],
        exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/aghri_second_finetune', files('config/aghri_second_finetune/*')),
        ('share/' + package_name + '/config/aghri_pointpillars_finetune', files('config/aghri_pointpillars_finetune/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ernstmv',
    maintainer_email='ernestoroque777@gmail.com',
    description='Implementation of a late fusion detector using 2d image-based and 3d lidar-based detections',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'late_fusion_node = late_fusion_pkg.late_fusion_node:main',
            'aghri_yolo_detector_node = late_fusion_pkg.aghri_yolo_detector_node:main',
            'aghri_second_detector_node = late_fusion_pkg.aghri_second_detector_node:main',
            'aghri_late_fusion_node = late_fusion_pkg.aghri_late_fusion_node:main',
            'cached_yolo_publisher = late_fusion_pkg.cached_yolo_publisher:main',
            'cached_second_publisher = late_fusion_pkg.cached_second_publisher:main',
            'live_yolo_detector_node = late_fusion_pkg.live_yolo_detector_node:main',
            'live_lidar_detector_node = late_fusion_pkg.live_lidar_detector_node:main',
            'mmdet3d_inference_worker = late_fusion_pkg.mmdet3d_inference_worker:main',
            'online_performance_monitor = late_fusion_pkg.online_performance_monitor:main',
            'cache_diagnostics_collector = late_fusion_pkg.cache_diagnostics_collector:main',
            'aghri_deepfusionmot_tracker_node = late_fusion_pkg.aghri_deepfusionmot_tracker_node:main',
            'aghri_late_fusion_visualizer = late_fusion_pkg.aghri_visualization_node:main',
        ],
    },
)
