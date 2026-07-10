# Copyright 2025 LCAS GROUP, University of Lincoln
#
# Licensed under the GLWTS  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://github.com/me-shaon/GLWTPL/tree/master
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sensor Fusion Node (Camera and LiDAR)

This ROS 2 node subscribes to synchronized camera images and LiDAR point clouds,
fuses them using a YOLO-based early fusion approach, and publishes the
detection results and annotated images.

Author: LCAS GROUP, University of Lincoln
License: GLWTS
"""

from collections import deque
import json
from time import time
import zipfile

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber, TimeSynchronizer
import tf2_ros

from cv_bridge import CvBridge
import numpy as np
from sensor_msgs_py import point_cloud2 as pc2

from scripts.fusser import Fusser


POINTCLOUD_MODES = {'organised_rings', 'unorganised_xyz'}
SYNC_MODES = {'exact', 'approximate'}
CALIBRATION_MODES = {'kitti_topic', 'aghri_json', 'ros_camera_info_tf'}
CAMERA_MATRIX_SOURCES = {'p', 'k'}


def validate_sync_config(sync_mode: str, queue_size: int, slop_sec: float):
    """Validate synchronisation settings and return normalised values."""
    if sync_mode not in SYNC_MODES:
        raise ValueError(
            f"Unsupported sync_mode '{sync_mode}'. "
            f"Expected one of: {sorted(SYNC_MODES)}"
        )
    queue_size = int(queue_size)
    slop_sec = float(slop_sec)
    if queue_size <= 0:
        raise ValueError('sync_queue_size must be positive')
    if slop_sec < 0.0:
        raise ValueError('sync_slop_sec must be non-negative')
    return sync_mode, queue_size, slop_sec


def pointcloud2_to_xyz_array(
    lidar_msg: PointCloud2,
    mode: str,
    total_rings: int = 64,
    rings_to_use: int = 64,
) -> tuple[np.ndarray, int, int]:
    """Convert PointCloud2 XYZ data using the selected cloud layout mode."""
    if mode not in POINTCLOUD_MODES:
        raise ValueError(
            f"Unsupported pointcloud_mode '{mode}'. "
            f"Expected one of: {sorted(POINTCLOUD_MODES)}"
        )

    points = pc2.read_points_numpy(
        lidar_msg,
        field_names=('x', 'y', 'z'),
        skip_nans=False,
    )
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    raw_count = int(points.shape[0])

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    valid_count = int(points.shape[0])

    if mode == 'unorganised_xyz':
        return points, raw_count, valid_count

    if total_rings <= 0 or rings_to_use <= 0:
        raise ValueError('total_rings and rings_to_use must be positive')
    if total_rings % rings_to_use != 0:
        raise ValueError('total_rings must be divisible by rings_to_use')

    points_per_ring = points.shape[0] // total_rings
    if points_per_ring == 0:
        return np.empty((0, 3), dtype=np.float32), raw_count, valid_count

    ring_step = total_rings // rings_to_use
    usable_count = points_per_ring * total_rings
    points = points[:usable_count]
    points = points.reshape(total_rings, points_per_ring, 3)
    points = points[::ring_step]

    return points.reshape(rings_to_use * points_per_ring, 3), raw_count, valid_count


def load_json_from_zip(zip_path: str, member: str) -> dict:
    """Load a JSON member from a ZIP archive."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(member) as file:
                return json.loads(file.read().decode('utf-8'))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Calibration ZIP not found: {zip_path}") from exc
    except KeyError as exc:
        raise KeyError(f"Calibration ZIP member not found: {member}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Calibration member is not valid JSON: {member}") from exc


def _as_matrix(values, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    try:
        return arr.reshape(shape)
    except ValueError as exc:
        raise ValueError(f"Malformed {name}; expected shape {shape}") from exc


def projection_from_intrinsics(
    intrinsics: dict,
    camera_sensor_name: str,
    camera_matrix_source: str,
) -> tuple[np.ndarray, str | None]:
    """Extract a 3x4 projection matrix from an AGHRI intrinsics dictionary."""
    if camera_matrix_source not in CAMERA_MATRIX_SOURCES:
        raise ValueError(
            f"Unsupported camera_matrix_source '{camera_matrix_source}'. "
            f"Expected one of: {sorted(CAMERA_MATRIX_SOURCES)}"
        )
    if camera_sensor_name not in intrinsics:
        raise KeyError(f"Camera sensor '{camera_sensor_name}' not found")

    camera = intrinsics[camera_sensor_name]
    frame_id = (camera.get('header') or {}).get('frame_id')

    if camera_matrix_source == 'p':
        if 'p' in camera:
            return _as_matrix(camera['p'], (3, 4), 'camera projection p'), frame_id
        projection_matrix = camera.get('projection_matrix')
        if isinstance(projection_matrix, dict) and 'data' in projection_matrix:
            return _as_matrix(
                projection_matrix['data'],
                (3, 4),
                'camera projection_matrix.data',
            ), frame_id
        raise KeyError(f"Camera '{camera_sensor_name}' has no projection matrix")

    if 'k' in camera:
        k = _as_matrix(camera['k'], (3, 3), 'camera matrix k')
    else:
        camera_matrix = camera.get('camera_matrix')
        if not isinstance(camera_matrix, dict) or 'data' not in camera_matrix:
            raise KeyError(f"Camera '{camera_sensor_name}' has no camera matrix")
        k = _as_matrix(camera_matrix['data'], (3, 3), 'camera_matrix.data')

    return np.column_stack([k, np.zeros((3, 1))]), frame_id


def _quat_to_rotation_matrix(rotation: dict) -> np.ndarray:
    try:
        x = float(rotation['x'])
        y = float(rotation['y'])
        z = float(rotation['z'])
        w = float(rotation['w'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Malformed quaternion in extrinsics') from exc

    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError('Malformed quaternion in extrinsics: zero norm')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def _transform_to_matrix(edge: dict) -> np.ndarray:
    transform = edge.get('transform')
    if not isinstance(transform, dict):
        raise ValueError('Malformed transform entry')

    translation = transform.get('translation')
    rotation = transform.get('rotation')
    if not isinstance(translation, dict) or not isinstance(rotation, dict):
        raise ValueError('Malformed transform translation or rotation')

    try:
        t = np.array([
            float(translation['x']),
            float(translation['y']),
            float(translation['z']),
        ], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Malformed translation in extrinsics') from exc

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_rotation_matrix(rotation)
    matrix[:3, 3] = t
    return matrix


def _transform_stamped_to_matrix(transform_stamped) -> np.ndarray:
    transform = transform_stamped.transform
    rotation = {
        'x': transform.rotation.x,
        'y': transform.rotation.y,
        'z': transform.rotation.z,
        'w': transform.rotation.w,
    }
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_rotation_matrix(rotation)
    matrix[:3, 3] = [
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
    ]
    return matrix


def projection_from_camera_info(
    camera_info: CameraInfo,
    camera_matrix_source: str,
) -> np.ndarray:
    """Extract a 3x4 projection matrix from sensor_msgs/CameraInfo."""
    if camera_matrix_source not in CAMERA_MATRIX_SOURCES:
        raise ValueError(
            f"Unsupported camera_matrix_source '{camera_matrix_source}'. "
            f"Expected one of: {sorted(CAMERA_MATRIX_SOURCES)}"
        )
    if camera_matrix_source == 'p':
        return _as_matrix(camera_info.p, (3, 4), 'CameraInfo.p')
    k = _as_matrix(camera_info.k, (3, 3), 'CameraInfo.k')
    return np.column_stack([k, np.zeros((3, 1))])


def _invert_transform(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=float)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def _build_transform_graph(extrinsics: dict):
    transforms = extrinsics.get('transforms')
    if not isinstance(transforms, list):
        raise ValueError("Extrinsics JSON must contain a 'transforms' list")

    adjacency: dict[str, list[str]] = {}
    transform_map: dict[tuple[str, str], np.ndarray] = {}
    for edge in transforms:
        header = edge.get('header') or {}
        parent = header.get('frame_id')
        child = edge.get('child_frame_id')
        if not parent or not child:
            raise ValueError('Transform entry is missing parent or child frame')

        matrix = _transform_to_matrix(edge)
        adjacency.setdefault(parent, []).append(child)
        adjacency.setdefault(child, []).append(parent)

        # ROS TransformStamped stores the child frame pose in the parent frame.
        # The matrix therefore maps child-frame points into parent-frame points.
        transform_map[(child, parent)] = matrix
        transform_map[(parent, child)] = _invert_transform(matrix)

    return adjacency, transform_map


def _find_unique_shortest_path(adjacency, start: str, goal: str) -> list[str]:
    if start not in adjacency:
        raise KeyError(f"Frame '{start}' not found in calibration graph")
    if goal not in adjacency:
        raise KeyError(f"Frame '{goal}' not found in calibration graph")

    queue = deque([[start]])
    shortest_paths = []
    shortest_len = None
    while queue:
        path = queue.popleft()
        if shortest_len is not None and len(path) > shortest_len:
            continue
        node = path[-1]
        if node == goal:
            shortest_len = len(path)
            shortest_paths.append(path)
            continue
        for neighbor in adjacency.get(node, []):
            if neighbor not in path:
                queue.append(path + [neighbor])

    if not shortest_paths:
        raise ValueError(f"No transform path from '{start}' to '{goal}'")
    if len(shortest_paths) > 1:
        raise ValueError(f"Ambiguous transform path from '{start}' to '{goal}'")

    return shortest_paths[0]


def transform_between_frames(extrinsics: dict, source_frame: str, target_frame: str):
    """Return target-from-source transform and the frame path used."""
    adjacency, transform_map = _build_transform_graph(extrinsics)
    path = _find_unique_shortest_path(adjacency, source_frame, target_frame)

    transform = np.eye(4, dtype=float)
    for source, target in zip(path[:-1], path[1:]):
        transform = transform_map[(source, target)] @ transform

    return transform, path


def load_aghri_projection(
    calibration_zip_path: str,
    intrinsics_member: str,
    extrinsics_member: str,
    camera_sensor_name: str,
    lidar_frame_name: str,
    camera_optical_frame_name: str,
    camera_matrix_source: str = 'p',
) -> tuple[np.ndarray, np.ndarray, list[str], str | None]:
    """Load AGHRI calibration and return image and camera transforms."""
    intrinsics = load_json_from_zip(calibration_zip_path, intrinsics_member)
    extrinsics = load_json_from_zip(calibration_zip_path, extrinsics_member)
    camera_projection, inferred_frame = projection_from_intrinsics(
        intrinsics,
        camera_sensor_name,
        camera_matrix_source,
    )

    if inferred_frame and inferred_frame != camera_optical_frame_name:
        raise ValueError(
            "Camera optical frame does not match intrinsics: "
            f"{camera_optical_frame_name} != {inferred_frame}"
        )

    camera_from_lidar, path = transform_between_frames(
        extrinsics,
        lidar_frame_name,
        camera_optical_frame_name,
    )
    projection_lidar_to_image = camera_projection @ camera_from_lidar

    return projection_lidar_to_image, camera_from_lidar, path, inferred_frame


class SensorFusionNode(Node):
    """Sensor Fusion Node combining 2D and 3D sensor data."""

    def __init__(self):
        """
        Initialize the SensorFusionNode.

        Sets up subscribers, publishers, parameters, and internal tools.
        """
        super().__init__('lidar_fusion')

        # Declare and get parameters
        self.declare_parameter('image_sub_topic', '/camera/image_raw')
        self.declare_parameter('calib_sub_topic', '/camera/calibration')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('lidar_sub_topic', '/lidar/points')

        self.image_sub_topic = self.get_parameter('image_sub_topic').value
        self.calib_sub_topic = self.get_parameter('calib_sub_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.lidar_sub_topic = self.get_parameter('lidar_sub_topic').value

        # Publishers
        self.declare_parameter('result_pub_topic', '/camera_lidar_fusion/result')
        self.declare_parameter('bboxes_pub_topic', '/camera_lidar_fusion/pred_bboxes')

        result_pub_topic = self.get_parameter('result_pub_topic').value
        bboxes_pub_topic = self.get_parameter('bboxes_pub_topic').value

        self.final_publisher = self.create_publisher(Image, result_pub_topic, 1)
        self.bboxes_publisher = self.create_publisher(String, bboxes_pub_topic, 1)

        # Early fusion parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('reduction_factor', 0.9)
        self.declare_parameter('yolo_classes', [0, 2])
        self.declare_parameter('yolo_threshold', 0.8)
        self.declare_parameter('draw_points', True)
        self.declare_parameter('draw_bboxes', True)
        self.declare_parameter('write_distance', True)
        self.declare_parameter('device', 'auto')
        self.declare_parameter('point_depth_filter_mode', 'legacy_lidar_x')
        self.declare_parameter('minimum_forward_distance', 2.0)
        self.declare_parameter('output_coordinate_mode', 'legacy_kitti')
        self.declare_parameter('coordinate_label_mode', 'xyz')
        self.declare_parameter('association_method', 'legacy_box')
        self.declare_parameter('centre_statistic', 'mean')
        self.declare_parameter('depth_cluster_gap_m', 0.35)
        self.declare_parameter('depth_cluster_min_points', 5)
        self.declare_parameter('box_bottom_trim_ratio', 0.0)
        self.declare_parameter('hybrid_min_cluster_points', 5)
        self.declare_parameter('hybrid_min_support_ratio', 0.35)
        self.declare_parameter('hybrid_max_depth_spread_m', 0.35)
        self.declare_parameter('hybrid_min_cluster_separation_m', 0.20)
        self.declare_parameter('hybrid_fallback_on_single_cluster', True)
        self.declare_parameter('hybrid_fallback_on_sparse_candidates', True)

        # Node tools
        self.bridge = CvBridge()
        self.fusser = None
        self._cloud_log_counter = 0
        self._camera_info_msg = None
        self._ros_calibration_warned = False
        self._tf_wait_warned = False
        self._tf_buffer = None
        self._tf_listener = None

        # LiDAR ring settings
        self.declare_parameter('total_rings', 64)
        self.declare_parameter('rings_to_use', 64)
        self.total_rings = self.get_parameter('total_rings').value
        self.rings_to_use = self.get_parameter('rings_to_use').value

        # Runtime compatibility settings
        self.declare_parameter('pointcloud_mode', 'organised_rings')
        self.declare_parameter('sync_mode', 'exact')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop_sec', 0.1)
        self.declare_parameter('calibration_mode', 'kitti_topic')
        self.declare_parameter('calibration_zip_path', '')
        self.declare_parameter('intrinsics_member', 'calibration/intrinsics.json')
        self.declare_parameter('extrinsics_member', 'calibration/extrinsics.json')
        self.declare_parameter('camera_sensor_name', 'cam_zed_rgb')
        self.declare_parameter('lidar_frame_name', 'front_lidar_link')
        self.declare_parameter(
            'camera_optical_frame_name',
            'front_left_camera_optical_frame',
        )
        self.declare_parameter('camera_matrix_source', 'p')

        self.pointcloud_mode = self.get_parameter('pointcloud_mode').value
        self.sync_mode = self.get_parameter('sync_mode').value
        sync_config = validate_sync_config(
            self.sync_mode,
            self.get_parameter('sync_queue_size').value,
            self.get_parameter('sync_slop_sec').value,
        )
        _, self.sync_queue_size, self.sync_slop_sec = sync_config
        self.calibration_mode = self.get_parameter('calibration_mode').value

        if self.pointcloud_mode not in POINTCLOUD_MODES:
            raise ValueError(f"Unsupported pointcloud_mode: {self.pointcloud_mode}")
        if self.calibration_mode not in CALIBRATION_MODES:
            raise ValueError(f"Unsupported calibration_mode: {self.calibration_mode}")

        self._setup_calibration()
        self._setup_synchronizer()

        self.get_logger().info("SensorFusionNode initialized and listening...")
        self.get_logger().info(
            f"pointcloud_mode={self.pointcloud_mode}, "
            f"output_coordinate_mode="
            f"{self.get_parameter('output_coordinate_mode').value}"
        )

    def _setup_synchronizer(self) -> None:
        """Create the selected image/cloud synchronizer."""
        subscribers = [
            Subscriber(self, Image, self.image_sub_topic),
            Subscriber(self, PointCloud2, self.lidar_sub_topic),
        ]

        if self.sync_mode == 'exact':
            self.synchronizer = TimeSynchronizer(
                subscribers,
                queue_size=int(self.sync_queue_size),
            )
        elif self.sync_mode == 'approximate':
            self.synchronizer = ApproximateTimeSynchronizer(
                subscribers,
                queue_size=int(self.sync_queue_size),
                slop=float(self.sync_slop_sec),
            )
        else:
            raise ValueError(f"Unsupported sync_mode: {self.sync_mode}")

        self.synchronizer.registerCallback(self._main_pipeline)
        self.get_logger().info(
            f"sync_mode={self.sync_mode}, queue_size={self.sync_queue_size}, "
            f"slop_sec={self.sync_slop_sec}"
        )

    def _setup_calibration(self) -> None:
        """Initialise the selected calibration mode."""
        if self.calibration_mode == 'kitti_topic':
            self.calib_subscription = self.create_subscription(
                String,
                self.calib_sub_topic,
                self._calib_callback,
                1,
            )
            self.get_logger().info(
                f"calibration_mode=kitti_topic, topic={self.calib_sub_topic}"
            )
            return

        if self.calibration_mode == 'ros_camera_info_tf':
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self.camera_info_subscription = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_callback,
                10,
            )
            self.get_logger().info(
                "calibration_mode=ros_camera_info_tf, "
                f"camera_info_topic={self.camera_info_topic}, "
                f"target={self.get_parameter('camera_optical_frame_name').value}, "
                f"source={self.get_parameter('lidar_frame_name').value}"
            )
            return

        projection, camera_from_lidar, path, inferred_frame = load_aghri_projection(
            self.get_parameter('calibration_zip_path').value,
            self.get_parameter('intrinsics_member').value,
            self.get_parameter('extrinsics_member').value,
            self.get_parameter('camera_sensor_name').value,
            self.get_parameter('lidar_frame_name').value,
            self.get_parameter('camera_optical_frame_name').value,
            self.get_parameter('camera_matrix_source').value,
        )
        self.fusser = self._make_fusser(
            projection_matrix=projection,
            lidar_to_camera_matrix=camera_from_lidar,
        )
        self.get_logger().info(
            "calibration_mode=aghri_json, transform_path="
            f"{' -> '.join(path)}, camera_frame={inferred_frame}"
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Store CameraInfo and initialise ROS-native calibration when TF is ready."""
        self._camera_info_msg = msg
        self._try_setup_ros_calibration()

    def _try_setup_ros_calibration(self) -> bool:
        if self.calibration_mode != 'ros_camera_info_tf':
            return self.fusser is not None
        if self.fusser is not None:
            return True
        if self._camera_info_msg is None:
            if not self._ros_calibration_warned:
                self.get_logger().warn(
                    f"Waiting for CameraInfo on {self.camera_info_topic}"
                )
                self._ros_calibration_warned = True
            return False
        camera_frame = self.get_parameter('camera_optical_frame_name').value
        lidar_frame = self.get_parameter('lidar_frame_name').value
        if self._camera_info_msg.header.frame_id != camera_frame:
            raise ValueError(
                "CameraInfo frame mismatch: "
                f"{self._camera_info_msg.header.frame_id} != {camera_frame}"
            )
        if self._tf_buffer is None:
            raise RuntimeError("TF buffer was not initialised")
        try:
            transform = self._tf_buffer.lookup_transform(
                camera_frame,
                lidar_frame,
                Time(),
            )
        except Exception as exc:  # noqa: BLE001 - keep ROS distro exception-compatible.
            if not self._tf_wait_warned:
                self.get_logger().warn(
                    f"Waiting for TF {lidar_frame} -> {camera_frame}: {exc}"
                )
                self._tf_wait_warned = True
            return False

        camera_projection = projection_from_camera_info(
            self._camera_info_msg,
            self.get_parameter('camera_matrix_source').value,
        )
        camera_from_lidar = _transform_stamped_to_matrix(transform)
        projection = camera_projection @ camera_from_lidar
        self.fusser = self._make_fusser(
            projection_matrix=projection,
            lidar_to_camera_matrix=camera_from_lidar,
        )

        json_zip = self.get_parameter('calibration_zip_path').value
        if json_zip:
            try:
                json_projection, json_camera_from_lidar, _, _ = load_aghri_projection(
                    json_zip,
                    self.get_parameter('intrinsics_member').value,
                    self.get_parameter('extrinsics_member').value,
                    self.get_parameter('camera_sensor_name').value,
                    lidar_frame,
                    camera_frame,
                    self.get_parameter('camera_matrix_source').value,
                )
                projection_diff = float(np.max(np.abs(projection - json_projection)))
                transform_diff = float(np.max(np.abs(camera_from_lidar - json_camera_from_lidar)))
                self.get_logger().info(
                    "ROS-native calibration matched aghri_json fallback: "
                    f"max_projection_abs_diff={projection_diff:.6g}, "
                    f"max_transform_abs_diff={transform_diff:.6g}"
                )
            except Exception as exc:  # noqa: BLE001 - comparison is diagnostic only.
                self.get_logger().warn(
                    f"Could not compare ROS-native calibration with aghri_json: {exc}"
                )
        return True

    def _calib_callback(self, msg: String) -> None:
        """
        Parse calibration data from a String message and initialize the Fusser object.

        Args:
            msg (String): The calibration information.
        """
        if not self.fusser:
            data = msg.data if hasattr(msg, 'data') else str(msg)
            for line in data.splitlines():
                if line.startswith('P') or line.startswith('std_msgs'):
                    if line.startswith('std_msgs'):
                        values = line.split("'")[1].split(':')[1].split()
                    else:
                        values = line.split(':', maxsplit=1)[1].split()
                    P = np.array([float(val) for val in values])
                elif line.startswith('R_rect'):
                    values = [val for val in line.split()]
                    R0 = np.array([float(val) for val in values[1:]])
                elif line.startswith('Tr_velo_cam') or line.startswith('Tr_velo_to_cam'):
                    values = [val for val in line.split()]
                    V2C = np.array([float(val) for val in values[1:]])

            P, R0, V2C = P.reshape(3, 4), R0.reshape(3, 3), V2C.reshape(3, 4)

            self.fusser = self._make_fusser(P=P, R0=R0, V2C=V2C)

    def _make_fusser(self, P=None, R0=None, V2C=None, **kwargs) -> Fusser:
        """Create Fusser with shared runtime parameters."""
        fusser = Fusser(
            P, R0, V2C,
            self.get_parameter('YOLO_model').value,
            self.get_parameter('reduction_factor').value,
            self.get_parameter('yolo_classes').value,
            self.get_parameter('yolo_threshold').value,
            self.get_parameter('draw_points').value,
            self.get_parameter('draw_bboxes').value,
            self.get_parameter('write_distance').value,
            device=self.get_parameter('device').value,
            point_depth_filter_mode=self.get_parameter(
                'point_depth_filter_mode',
            ).value,
            minimum_forward_distance=self.get_parameter(
                'minimum_forward_distance',
            ).value,
            output_coordinate_mode=self.get_parameter(
                'output_coordinate_mode',
            ).value,
            coordinate_label_mode=self.get_parameter(
                'coordinate_label_mode',
            ).value,
            association_method=self.get_parameter('association_method').value,
            centre_statistic=self.get_parameter('centre_statistic').value,
            depth_cluster_gap_m=self.get_parameter('depth_cluster_gap_m').value,
            depth_cluster_min_points=self.get_parameter(
                'depth_cluster_min_points',
            ).value,
            box_bottom_trim_ratio=self.get_parameter(
                'box_bottom_trim_ratio',
            ).value,
            hybrid_min_cluster_points=self.get_parameter(
                'hybrid_min_cluster_points',
            ).value,
            hybrid_min_support_ratio=self.get_parameter(
                'hybrid_min_support_ratio',
            ).value,
            hybrid_max_depth_spread_m=self.get_parameter(
                'hybrid_max_depth_spread_m',
            ).value,
            hybrid_min_cluster_separation_m=self.get_parameter(
                'hybrid_min_cluster_separation_m',
            ).value,
            hybrid_fallback_on_single_cluster=self.get_parameter(
                'hybrid_fallback_on_single_cluster',
            ).value,
            hybrid_fallback_on_sparse_candidates=self.get_parameter(
                'hybrid_fallback_on_sparse_candidates',
            ).value,
            **kwargs,
        )
        self.get_logger().info(f"device={fusser.device}")
        return fusser

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to a numpy array.

        Args:
            img_msg (Image): Image message.

        Returns:
            np.ndarray: OpenCV image.
        """
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        """
        Convert a numpy array to a sensor_msgs/Image message.

        Args:
            arr (np.ndarray): OpenCV image.

        Returns:
            Image: Image message.
        """
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

    def _lidarmsg2np(self, lidar_msg: PointCloud2) -> np.ndarray:
        """
        Convert a sensor_msgs/PointCloud2 to a filtered numpy array.

        Args:
            lidar_msg (PointCloud2): PointCloud2 message.

        Returns:
            np.ndarray: Filtered point cloud array.
        """
        points, raw_count, valid_count = pointcloud2_to_xyz_array(
            lidar_msg,
            self.pointcloud_mode,
            self.total_rings,
            self.rings_to_use,
        )
        self._cloud_log_counter += 1
        if self._cloud_log_counter == 1 or self._cloud_log_counter % 30 == 0:
            self.get_logger().debug(
                f"PointCloud2 mode={self.pointcloud_mode}, "
                f"raw={raw_count}, valid_xyz={valid_count}, "
                f"fusion_points={len(points)}"
            )
        return points

    def _np2String(self, arr: np.ndarray) -> String:
        """
        Convert a numpy array to a std_msgs/String message.

        Args:
            arr (np.ndarray): Array to convert.

        Returns:
            String: String message.
        """
        arr_string = np.array2string(arr)
        msg = String()
        msg.data = arr_string
        return msg

    def _main_pipeline(self, image_msg: Image, lidar_msg: PointCloud2) -> None:
        """
        Main processing pipeline for synchronized camera and LiDAR messages.

        Args:
            image_msg (Image): Incoming camera image.
            lidar_msg (PointCloud2): Incoming LiDAR point cloud.
        """
        if self.calibration_mode == 'ros_camera_info_tf' and not self._try_setup_ros_calibration():
            return

        if self.fusser:
            img = self._imgmsg2np(image_msg)
            points = self._lidarmsg2np(lidar_msg)

            start = time()
            predictions, final_img = self.fusser.pipeline(img, points)
            spend = time() - start

            final_img_msg = self._np2imgmsg(final_img)
            predictions_msg = self._np2String(predictions)

            self.final_publisher.publish(final_img_msg)
            self.bboxes_publisher.publish(predictions_msg)

            self.get_logger().info(f'{len(predictions)} objects detected in {spend:.4f} seconds')


def main(args=None) -> None:
    """
    ROS 2 main entrypoint.
    """
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
