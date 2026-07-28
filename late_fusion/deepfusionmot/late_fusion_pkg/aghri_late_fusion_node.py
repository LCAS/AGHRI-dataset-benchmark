"""ROS 2 AGHRI late-fusion node using CameraInfo/TF and standard messages."""

from __future__ import annotations

import math
import time

import numpy as np


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from sensor_msgs.msg import CameraInfo
    from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
    from std_msgs.msg import String
    import tf2_ros

    from late_fusion_pkg.fusion_core import fuse_detections
    from late_fusion_pkg.ros_adapters import detections2d_from_msg, detections3d_from_msg, detection3d_to_msg, detection2d_to_msg
    from vision_msgs.msg import Detection2D, BoundingBox2D

    class AghriLateFusion(Node):
        def __init__(self):
            super().__init__("aghri_late_fusion")
            self.declare_parameter("camera_detections_topic", "/detector/camera/detections")
            self.declare_parameter("lidar_detections_topic", "/detector/lidar/detections")
            self.declare_parameter("camera_info_topic", "/dataset/cam_zed_rgb/camera_info")
            self.declare_parameter("fused_topic", "/late_fusion/fused_3d")
            self.declare_parameter("matched_topic", "/late_fusion/matched_3d")
            self.declare_parameter("unmatched_lidar_topic", "/late_fusion/unmatched_lidar")
            self.declare_parameter("unmatched_camera_topic", "/late_fusion/unmatched_camera")
            self.declare_parameter("diagnostics_topic", "/late_fusion/diagnostics")
            self.declare_parameter("sync_queue_size", 30)
            self.declare_parameter("sync_slop_sec", 0.10)
            self.declare_parameter("iou_threshold", 0.10)
            self.declare_parameter("confidence_policy", "lidar_score")
            self.declare_parameter("lidar_frame", "front_lidar_link")
            self.declare_parameter("camera_frame", "front_left_camera_optical_frame")
            self.declare_parameter("image_width", 672)
            self.declare_parameter("image_height", 376)
            self.declare_parameter("use_source_time_tf", False)

            self.camera_p = None
            self.image_width = int(self.get_parameter("image_width").value)
            self.image_height = int(self.get_parameter("image_height").value)
            self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self.camera_info_cb, 10)
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            self.pub_fused = self.create_publisher(Detection3DArray, self.get_parameter("fused_topic").value, 10)
            self.pub_matched = self.create_publisher(Detection3DArray, self.get_parameter("matched_topic").value, 10)
            self.pub_unmatched_lidar = self.create_publisher(Detection3DArray, self.get_parameter("unmatched_lidar_topic").value, 10)
            self.pub_unmatched_camera = self.create_publisher(Detection2DArray, self.get_parameter("unmatched_camera_topic").value, 10)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)

            sub_cam = Subscriber(self, Detection2DArray, self.get_parameter("camera_detections_topic").value)
            sub_lidar = Subscriber(self, Detection3DArray, self.get_parameter("lidar_detections_topic").value)
            self.sync = ApproximateTimeSynchronizer(
                [sub_cam, sub_lidar],
                queue_size=int(self.get_parameter("sync_queue_size").value),
                slop=float(self.get_parameter("sync_slop_sec").value),
            )
            self.sync.registerCallback(self.callback)

        def camera_info_cb(self, msg: CameraInfo) -> None:
            self.camera_p = np.asarray(msg.p, dtype=float).reshape(3, 4)
            self.image_width = int(msg.width)
            self.image_height = int(msg.height)

        def camera_from_lidar(self, stamp=None):
            target = str(self.get_parameter("camera_frame").value)
            source = str(self.get_parameter("lidar_frame").value)
            lookup_time = (
                rclpy.time.Time.from_msg(stamp)
                if stamp is not None and bool(self.get_parameter("use_source_time_tf").value)
                else rclpy.time.Time()
            )
            tf = self.tf_buffer.lookup_transform(target, source, lookup_time)
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = _quat_to_matrix(tf.transform.rotation)
            matrix[:3, 3] = [
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
                float(tf.transform.translation.z),
            ]
            return matrix

        def _publish_3d(self, publisher, header, detections):
            out = Detection3DArray()
            out.header = header
            for det in detections:
                out.detections.append(detection3d_to_msg(det, Detection3D, BoundingBox3D, ObjectHypothesisWithPose))
            publisher.publish(out)

        def callback(self, camera_msg: Detection2DArray, lidar_msg: Detection3DArray) -> None:
            start = time.perf_counter()
            if self.camera_p is None:
                return
            try:
                transform = self.camera_from_lidar(lidar_msg.header.stamp)
            except Exception as exc:
                self.pub_diag.publish(String(data=f"tf_unavailable: {exc}"))
                return
            cam = detections2d_from_msg(camera_msg)
            lidar = detections3d_from_msg(lidar_msg)
            result = fuse_detections(
                cam,
                lidar,
                self.camera_p,
                transform,
                self.image_width,
                self.image_height,
                iou_threshold=float(self.get_parameter("iou_threshold").value),
                confidence_policy=str(self.get_parameter("confidence_policy").value),
            )
            self._publish_3d(self.pub_fused, lidar_msg.header, result.final_3d)
            self._publish_3d(self.pub_matched, lidar_msg.header, result.matched_3d)
            self._publish_3d(self.pub_unmatched_lidar, lidar_msg.header, result.unmatched_lidar)
            unmatched_cam_msg = Detection2DArray()
            unmatched_cam_msg.header = camera_msg.header
            for det in result.unmatched_camera:
                unmatched_cam_msg.detections.append(detection2d_to_msg(det, Detection2D, BoundingBox2D, ObjectHypothesisWithPose))
            self.pub_unmatched_camera.publish(unmatched_cam_msg)
            cam_t = camera_msg.header.stamp.sec + camera_msg.header.stamp.nanosec / 1e9
            lidar_t = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec / 1e9
            total = time.perf_counter() - start
            self.pub_diag.publish(
                String(
                    data=(
                        f"camera_stamp={cam_t:.9f},lidar_stamp={lidar_t:.9f},"
                        f"delta={abs(cam_t-lidar_t):.9f},camera={len(cam)},lidar={len(lidar)},"
                        f"matches={len(result.matches)},unmatched_camera={len(result.unmatched_camera)},"
                        f"unmatched_lidar={len(result.unmatched_lidar)},total_callback_time={total:.6f}"
                    )
                )
            )

    rclpy.init(args=args)
    node = AghriLateFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
