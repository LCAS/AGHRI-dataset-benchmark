"""ROS 2 visualisation node for AGHRI cached late-fusion replay."""

from __future__ import annotations

import json
import math
from collections import deque

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
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Point
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from vision_msgs.msg import Detection2DArray, Detection3DArray
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros

    from late_fusion_pkg.ros_adapters import detections2d_from_msg, detections3d_from_msg
    from late_fusion_pkg.visualization_utils import (
        color_rgba,
        detections3d_to_marker_array,
        draw_detection2d_array,
        draw_projected_wireframes,
        VisualBoxSmoother,
    )

    class AghriVisualization(Node):
        def __init__(self):
            super().__init__("aghri_late_fusion_visualizer")
            self.declare_parameter("camera_image_topic", "/dataset/cam_zed_rgb/image")
            self.declare_parameter("camera_info_topic", "/dataset/cam_zed_rgb/camera_info")
            self.declare_parameter("camera_detections_topic", "/detector/camera/detections")
            self.declare_parameter("lidar_detections_topic", "/detector/lidar/detections")
            self.declare_parameter("matched_3d_topic", "/late_fusion/matched_3d")
            self.declare_parameter("fused_3d_topic", "/late_fusion/fused_3d")
            self.declare_parameter("tracked_3d_topic", "/deepfusionmot/tracked_3d")
            self.declare_parameter("unmatched_camera_topic", "/late_fusion/unmatched_camera")
            self.declare_parameter("unmatched_lidar_topic", "/late_fusion/unmatched_lidar")
            self.declare_parameter("camera_detections_image_topic", "/visualization/camera_detections_image")
            self.declare_parameter("fused_projection_image_topic", "/visualization/fused_projection_image")
            self.declare_parameter("tracked_projection_image_topic", "/visualization/tracked_projection_image")
            self.declare_parameter("lidar_projection_image_topic", "/visualization/lidar_projection_image")
            self.declare_parameter("lidar_markers_topic", "/visualization/lidar_detections_markers")
            self.declare_parameter("matched_markers_topic", "/visualization/matched_3d_markers")
            self.declare_parameter("fused_markers_topic", "/visualization/fused_3d_markers")
            self.declare_parameter("tracked_markers_topic", "/visualization/tracked_3d_markers")
            self.declare_parameter("unmatched_lidar_markers_topic", "/visualization/unmatched_lidar_markers")
            self.declare_parameter("diagnostics_topic", "/visualization/diagnostics")
            self.declare_parameter("lidar_frame", "front_lidar_link")
            self.declare_parameter("camera_frame", "front_left_camera_optical_frame")
            self.declare_parameter("draw_camera_detections", True)
            self.declare_parameter("draw_fused_projection", True)
            self.declare_parameter("draw_tracked_projection", False)
            self.declare_parameter("draw_original_lidar_projection", False)
            self.declare_parameter("draw_unmatched_camera", False)
            self.declare_parameter("publish_original_lidar_markers", False)
            self.declare_parameter("publish_matched_markers", True)
            self.declare_parameter("publish_fused_markers", False)
            self.declare_parameter("publish_tracked_markers", False)
            self.declare_parameter("publish_unmatched_lidar_markers", False)
            self.declare_parameter("draw_labels", True)
            self.declare_parameter("draw_confidence", True)
            self.declare_parameter("line_width_2d", 2)
            self.declare_parameter("marker_line_width", 0.04)
            self.declare_parameter("marker_lifetime_sec", 0.25)
            self.declare_parameter("sync_queue_size", 30)
            self.declare_parameter("sync_slop_sec", 0.10)
            self.declare_parameter("draw_projected_envelope", True)
            self.declare_parameter("enable_visual_smoothing", False)
            self.declare_parameter("visual_smoothing_alpha", 0.35)
            self.declare_parameter("visual_smoothing_max_gap_sec", 0.20)
            self.declare_parameter("visual_smoothing_max_distance_m", 1.5)

            self.bridge = CvBridge()
            self.camera_p = None
            self.image_cache = deque(maxlen=int(self.get_parameter("sync_queue_size").value))
            smoothing_args = (
                float(self.get_parameter("visual_smoothing_alpha").value),
                float(self.get_parameter("visual_smoothing_max_gap_sec").value),
                float(self.get_parameter("visual_smoothing_max_distance_m").value),
            )
            self.smoothers = {
                name: VisualBoxSmoother(*smoothing_args)
                for name in ("matched", "lidar", "fused", "tracked", "unmatched")
            }
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self.camera_info_cb, 10)
            if bool(self.get_parameter("draw_fused_projection").value) or bool(self.get_parameter("draw_tracked_projection").value):
                self.create_subscription(Image, self.get_parameter("camera_image_topic").value, self.image_cache_cb, 30)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)

            self.pub_camera_img = self.create_publisher(Image, self.get_parameter("camera_detections_image_topic").value, 10)
            self.pub_fused_projection = self.create_publisher(Image, self.get_parameter("fused_projection_image_topic").value, 10)
            self.pub_tracked_projection = self.create_publisher(Image, self.get_parameter("tracked_projection_image_topic").value, 10)
            self.pub_lidar_projection = self.create_publisher(Image, self.get_parameter("lidar_projection_image_topic").value, 10)
            self.pub_lidar_markers = self.create_publisher(MarkerArray, self.get_parameter("lidar_markers_topic").value, 10)
            self.pub_matched_markers = self.create_publisher(MarkerArray, self.get_parameter("matched_markers_topic").value, 10)
            self.pub_fused_markers = self.create_publisher(MarkerArray, self.get_parameter("fused_markers_topic").value, 10)
            self.pub_tracked_markers = self.create_publisher(MarkerArray, self.get_parameter("tracked_markers_topic").value, 10)
            self.pub_unmatched_lidar_markers = self.create_publisher(MarkerArray, self.get_parameter("unmatched_lidar_markers_topic").value, 10)

            queue = int(self.get_parameter("sync_queue_size").value)
            slop = float(self.get_parameter("sync_slop_sec").value)
            if bool(self.get_parameter("draw_camera_detections").value):
                self.cam_sync = ApproximateTimeSynchronizer(
                    [
                        Subscriber(self, Image, self.get_parameter("camera_image_topic").value),
                        Subscriber(self, Detection2DArray, self.get_parameter("camera_detections_topic").value),
                    ],
                    queue_size=queue,
                    slop=slop,
                )
                self.cam_sync.registerCallback(self.camera_image_cb)
            if bool(self.get_parameter("draw_original_lidar_projection").value):
                self.lidar_projection_sync = ApproximateTimeSynchronizer(
                    [
                        Subscriber(self, Image, self.get_parameter("camera_image_topic").value),
                        Subscriber(self, Detection3DArray, self.get_parameter("lidar_detections_topic").value),
                    ],
                    queue_size=queue,
                    slop=slop,
                )
                self.lidar_projection_sync.registerCallback(self.lidar_projection_cb)

            if bool(self.get_parameter("publish_original_lidar_markers").value):
                self.create_subscription(Detection3DArray, self.get_parameter("lidar_detections_topic").value, self.lidar_marker_cb, 10)
            if bool(self.get_parameter("publish_matched_markers").value) or bool(self.get_parameter("draw_fused_projection").value):
                self.create_subscription(Detection3DArray, self.get_parameter("matched_3d_topic").value, self.matched_output_cb, 30)
            if bool(self.get_parameter("publish_fused_markers").value):
                self.create_subscription(Detection3DArray, self.get_parameter("fused_3d_topic").value, self.fused_marker_cb, 10)
            if bool(self.get_parameter("publish_tracked_markers").value) or bool(self.get_parameter("draw_tracked_projection").value):
                self.create_subscription(Detection3DArray, self.get_parameter("tracked_3d_topic").value, self.tracked_output_cb, 30)
            if bool(self.get_parameter("publish_unmatched_lidar_markers").value):
                self.create_subscription(Detection3DArray, self.get_parameter("unmatched_lidar_topic").value, self.unmatched_lidar_marker_cb, 10)

        def camera_info_cb(self, msg: CameraInfo) -> None:
            self.camera_p = np.asarray(msg.p, dtype=float).reshape(3, 4)

        @staticmethod
        def _stamp_seconds(stamp) -> float:
            return float(stamp.sec) + float(stamp.nanosec) / 1e9

        def image_cache_cb(self, msg: Image) -> None:
            self.image_cache.append(msg)

        def _nearest_image(self, stamp):
            if not self.image_cache:
                return None, None
            target = self._stamp_seconds(stamp)
            image = min(self.image_cache, key=lambda item: abs(self._stamp_seconds(item.header.stamp) - target))
            delta = abs(self._stamp_seconds(image.header.stamp) - target)
            if delta > float(self.get_parameter("sync_slop_sec").value):
                return None, delta
            return image, delta

        def camera_from_lidar(self, source_stamp):
            lookup_time = rclpy.time.Time.from_msg(source_stamp)
            tf = self.tf_buffer.lookup_transform(
                str(self.get_parameter("camera_frame").value),
                str(self.get_parameter("lidar_frame").value),
                lookup_time,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = _quat_to_matrix(tf.transform.rotation)
            matrix[:3, 3] = [
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
                float(tf.transform.translation.z),
            ]
            return matrix

        def _image_to_bgr(self, msg: Image):
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if image is None:
                raise RuntimeError("cv_bridge returned no image")
            return image

        def _publish_image(self, publisher, header, image_bgr) -> None:
            msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
            msg.header = header
            publisher.publish(msg)

        def camera_image_cb(self, image_msg: Image, detections_msg: Detection2DArray) -> None:
            image = self._image_to_bgr(image_msg)
            detections = detections2d_from_msg(detections_msg)
            annotated = draw_detection2d_array(
                image,
                detections,
                (255, 80, 20),
                draw_labels=bool(self.get_parameter("draw_labels").value),
                line_width=int(self.get_parameter("line_width_2d").value),
            )
            self._publish_image(self.pub_camera_img, image_msg.header, annotated)
            self._publish_diag("camera_image", image_msg, detections_msg, {"detections": len(detections)})

        def _visual_detections(self, detections, stamp, stream):
            if not bool(self.get_parameter("enable_visual_smoothing").value):
                return detections
            return self.smoothers[stream].update(detections, self._stamp_seconds(stamp))

        def _project_image_cb(
            self,
            image_msg: Image,
            detections_msg: Detection3DArray,
            publisher,
            color_bgr,
            label: str,
            detections=None,
            label_prefix: str | None = None,
        ) -> None:
            delta = abs(self._stamp_seconds(image_msg.header.stamp) - self._stamp_seconds(detections_msg.header.stamp))
            if self.camera_p is None:
                self._projection_diag(label, detections_msg, delta, camera_info_failures=1)
                return
            try:
                transform = self.camera_from_lidar(detections_msg.header.stamp)
            except Exception as exc:
                self._projection_diag(label, detections_msg, delta, tf_failures=1, error=str(exc))
                return
            image = self._image_to_bgr(image_msg)
            if detections is None:
                detections = detections3d_from_msg(detections_msg)
            annotated, stats = draw_projected_wireframes(
                image,
                detections,
                self.camera_p,
                transform,
                color_bgr,
                draw_labels=bool(self.get_parameter("draw_labels").value),
                draw_envelope=bool(self.get_parameter("draw_projected_envelope").value),
                timestamp_delta_sec=delta,
                line_width=int(self.get_parameter("line_width_2d").value),
                label_prefix=label_prefix or label,
            )
            self._publish_image(publisher, image_msg.header, annotated)
            self._projection_diag(label, detections_msg, delta, stats=stats)

        def fused_projection_cb(self, image_msg: Image, detections_msg: Detection3DArray) -> None:
            self._project_image_cb(image_msg, detections_msg, self.pub_fused_projection, (0, 230, 0), "fused_projection")

        def lidar_projection_cb(self, image_msg: Image, detections_msg: Detection3DArray) -> None:
            self._project_image_cb(image_msg, detections_msg, self.pub_lidar_projection, (0, 220, 255), "lidar_projection")

        def _publish_markers(
            self, publisher, msg: Detection3DArray, box_ns: str, label_ns: str,
            color_name: str, prefix: str, detections=None,
        ) -> None:
            if detections is None:
                detections = detections3d_from_msg(msg)
            arr = detections3d_to_marker_array(
                detections,
                msg.header,
                MarkerArray,
                Marker,
                Point,
                box_namespace=box_ns,
                label_namespace=label_ns,
                rgba=color_rgba(color_name),
                line_width=float(self.get_parameter("marker_line_width").value),
                lifetime_sec=float(self.get_parameter("marker_lifetime_sec").value),
                draw_labels=bool(self.get_parameter("draw_labels").value),
                label_prefix=prefix,
            )
            publisher.publish(arr)
            self.pub_diag.publish(String(data=json.dumps({"source": box_ns, "markers": len(arr.markers), "detections": len(detections)})))

        def lidar_marker_cb(self, msg: Detection3DArray) -> None:
            detections = self._visual_detections(detections3d_from_msg(msg), msg.header.stamp, "lidar")
            self._publish_markers(self.pub_lidar_markers, msg, "lidar_original_boxes", "lidar_original_labels", "yellow", "LiDAR person", detections)

        def matched_output_cb(self, msg: Detection3DArray) -> None:
            detections = self._visual_detections(detections3d_from_msg(msg), msg.header.stamp, "matched")
            if bool(self.get_parameter("publish_matched_markers").value):
                self._publish_markers(self.pub_matched_markers, msg, "matched_boxes", "matched_labels", "green", "matched", detections)
            if not bool(self.get_parameter("draw_fused_projection").value):
                return
            image_msg, delta = self._nearest_image(msg.header.stamp)
            if image_msg is None:
                self._projection_diag("fused_projection", msg, delta, no_synchronised_image=1)
                return
            self._project_image_cb(image_msg, msg, self.pub_fused_projection, (0, 230, 0), "fused_projection", detections, "matched")

        def fused_marker_cb(self, msg: Detection3DArray) -> None:
            detections = self._visual_detections(detections3d_from_msg(msg), msg.header.stamp, "fused")
            self._publish_markers(self.pub_fused_markers, msg, "fused_boxes", "fused_labels", "cyan", "fused", detections)

        def tracked_output_cb(self, msg: Detection3DArray) -> None:
            detections = self._visual_detections(detections3d_from_msg(msg), msg.header.stamp, "tracked")
            if bool(self.get_parameter("publish_tracked_markers").value):
                self._publish_markers(self.pub_tracked_markers, msg, "tracked_boxes", "tracked_labels", "magenta", "track", detections)
            if not bool(self.get_parameter("draw_tracked_projection").value):
                return
            image_msg, delta = self._nearest_image(msg.header.stamp)
            if image_msg is None:
                self._projection_diag("track", msg, delta, no_synchronised_image=1)
                return
            self._project_image_cb(image_msg, msg, self.pub_tracked_projection, (255, 0, 255), "track", detections, "track")

        def unmatched_lidar_marker_cb(self, msg: Detection3DArray) -> None:
            detections = self._visual_detections(detections3d_from_msg(msg), msg.header.stamp, "unmatched")
            self._publish_markers(self.pub_unmatched_lidar_markers, msg, "unmatched_lidar_boxes", "unmatched_lidar_labels", "red", "unmatched", detections)

        def _projection_diag(
            self, source, detections_msg, delta, *, stats=None, tf_failures=0,
            camera_info_failures=0, no_synchronised_image=0, error=None,
        ) -> None:
            stats = stats or {}
            payload = {
                "source": source,
                "detection_stamp": self._stamp_seconds(detections_msg.header.stamp),
                "matched_box_count": len(detections_msg.detections),
                "boxes_successfully_projected": int(stats.get("drawn", 0)),
                "boxes_rejected_behind_camera": int(stats.get("behind_camera", 0)),
                "boxes_rejected_outside_image": int(stats.get("outside_image", 0)),
                "boxes_rejected_invalid_geometry": int(stats.get("invalid_geometry", 0)),
                "tf_failures": int(tf_failures),
                "camera_info_failures": int(camera_info_failures),
                "no_synchronised_image": int(no_synchronised_image),
                "image_detection_timestamp_delta": None if delta is None else float(delta),
                "visual_smoothing_enabled": bool(self.get_parameter("enable_visual_smoothing").value),
            }
            if error:
                payload["error"] = error
            self.pub_diag.publish(String(data=json.dumps(payload, sort_keys=True)))

        def _publish_diag(self, source: str, image_msg, detections_msg, extra: dict) -> None:
            image_t = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9
            det_t = detections_msg.header.stamp.sec + detections_msg.header.stamp.nanosec / 1e9
            payload = {"source": source, "image_stamp": image_t, "detection_stamp": det_t, "delta": abs(image_t - det_t)}
            payload.update(extra)
            self.pub_diag.publish(String(data=json.dumps(payload, sort_keys=True)))

    rclpy.init(args=args)
    node = AghriVisualization()
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
