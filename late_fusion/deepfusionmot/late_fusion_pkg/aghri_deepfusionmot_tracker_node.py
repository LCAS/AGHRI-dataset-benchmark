"""ROS 2 online DeepFusionMOT tracker for AGHRI live late-fusion outputs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace

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
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from std_msgs.msg import String
    from vision_msgs.msg import BoundingBox2D, BoundingBox3D, Detection2D, Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
    import tf2_ros

    from late_fusion_pkg.deepfusionmot_tracker import DeepFusionMOTTracker, DeepFusionMOTTrackerConfig
    from late_fusion_pkg.detection_types import Detection2D as TypedDetection2D, FusionResult
    from late_fusion_pkg.ros_adapters import detection2d_to_msg, detection3d_to_msg, detections2d_from_msg, detections3d_from_msg

    class AghriDeepFusionMOTTrackerNode(Node):
        def __init__(self):
            super().__init__("aghri_deepfusionmot_tracker")
            self.declare_parameter("matched_3d_topic", "/late_fusion/matched_3d")
            self.declare_parameter("unmatched_lidar_topic", "/late_fusion/unmatched_lidar")
            self.declare_parameter("unmatched_camera_topic", "/late_fusion/unmatched_camera")
            self.declare_parameter("tracked_3d_topic", "/deepfusionmot/tracked_3d")
            self.declare_parameter("tracked_2d_topic", "/deepfusionmot/tracked_2d")
            self.declare_parameter("diagnostics_topic", "/deepfusionmot/diagnostics")
            self.declare_parameter("recording_id", "aghri_online")
            self.declare_parameter("sync_queue_size", 30)
            self.declare_parameter("sync_slop_sec", 0.10)
            self.declare_parameter("min_hits", 3)
            self.declare_parameter("max_age_seconds", 0.75)
            self.declare_parameter("max_age_frames", 10)
            self.declare_parameter("association_3d_threshold", 0.03)
            self.declare_parameter("association_2d_threshold", 0.40)
            self.declare_parameter("support_2d_to_3d_threshold", 0.40)
            self.declare_parameter("association_3d_metric", "bev_iou")
            self.declare_parameter("max_center_distance_m", 2.0)
            self.declare_parameter("lidar_only_initiation_score_threshold", 0.0)
            self.declare_parameter("id_policy", "deepfusionmot_even_odd")
            self.declare_parameter("lifecycle_mode", "deepfusionmot_frame")
            self.declare_parameter("output_mode", "deepfusionmot")
            self.declare_parameter("ego_motion_mode", "tf")
            self.declare_parameter("fixed_frame", "map")
            self.declare_parameter("lidar_frame", "front_lidar_link")
            self.declare_parameter("tf_lookup_timeout_sec", 0.05)

            tracker_ego_mode = "aghri_odom" if str(self.get_parameter("ego_motion_mode").value) == "tf" else "none"
            self.tracker = DeepFusionMOTTracker(
                DeepFusionMOTTrackerConfig(
                    min_hits=int(self.get_parameter("min_hits").value),
                    max_age_frames=int(self.get_parameter("max_age_frames").value),
                    max_age_seconds=float(self.get_parameter("max_age_seconds").value),
                    association_3d_threshold=float(self.get_parameter("association_3d_threshold").value),
                    association_2d_threshold=float(self.get_parameter("association_2d_threshold").value),
                    support_2d_to_3d_threshold=float(self.get_parameter("support_2d_to_3d_threshold").value),
                    association_3d_metric=str(self.get_parameter("association_3d_metric").value),
                    max_center_distance_m=float(self.get_parameter("max_center_distance_m").value),
                    lidar_only_initiation_score_threshold=float(self.get_parameter("lidar_only_initiation_score_threshold").value),
                    ego_motion_mode=tracker_ego_mode,
                    id_policy=str(self.get_parameter("id_policy").value),
                    lifecycle_mode=str(self.get_parameter("lifecycle_mode").value),
                    output_mode=str(self.get_parameter("output_mode").value),
                )
            )
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.previous_map_from_lidar: np.ndarray | None = None
            self.frame_index = 0
            self.tf_failures = 0
            self.pub_tracked_3d = self.create_publisher(Detection3DArray, self.get_parameter("tracked_3d_topic").value, 10)
            self.pub_tracked_2d = self.create_publisher(Detection2DArray, self.get_parameter("tracked_2d_topic").value, 10)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)

            queue = int(self.get_parameter("sync_queue_size").value)
            slop = float(self.get_parameter("sync_slop_sec").value)
            self.sync = ApproximateTimeSynchronizer(
                [
                    Subscriber(self, Detection3DArray, self.get_parameter("matched_3d_topic").value),
                    Subscriber(self, Detection3DArray, self.get_parameter("unmatched_lidar_topic").value),
                    Subscriber(self, Detection2DArray, self.get_parameter("unmatched_camera_topic").value),
                ],
                queue_size=queue,
                slop=slop,
            )
            self.sync.registerCallback(self.callback)
            self.get_logger().info("online DeepFusionMOT tracker ready")

        @staticmethod
        def _stamp_seconds(stamp) -> float:
            return float(stamp.sec) + float(stamp.nanosec) / 1e9

        def _map_from_lidar(self, stamp) -> np.ndarray | None:
            if str(self.get_parameter("ego_motion_mode").value) != "tf":
                return None
            try:
                tf = self.tf_buffer.lookup_transform(
                    str(self.get_parameter("fixed_frame").value),
                    str(self.get_parameter("lidar_frame").value),
                    rclpy.time.Time.from_msg(stamp),
                    timeout=rclpy.duration.Duration(seconds=float(self.get_parameter("tf_lookup_timeout_sec").value)),
                )
            except Exception as exc:
                self.tf_failures += 1
                self.get_logger().debug(f"DeepFusionMOT TF ego-motion unavailable: {exc}")
                return None
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = _quat_to_matrix(tf.transform.rotation)
            matrix[:3, 3] = [
                float(tf.transform.translation.x),
                float(tf.transform.translation.y),
                float(tf.transform.translation.z),
            ]
            return matrix

        def _ego_motion_transform(self, stamp) -> np.ndarray | None:
            current = self._map_from_lidar(stamp)
            if current is None:
                self.previous_map_from_lidar = None
                return None
            previous = self.previous_map_from_lidar
            self.previous_map_from_lidar = current
            if previous is None:
                return None
            return np.linalg.inv(current) @ previous

        def _publish_3d(self, header, tracks) -> None:
            out = Detection3DArray()
            out.header = header
            for track in tracks:
                det = replace(
                    track.detection,
                    detection_id=str(track.track_id),
                    metadata={**track.detection.metadata, "track_state": track.track_state, "update_source": track.update_source},
                )
                out.detections.append(detection3d_to_msg(det, Detection3D, BoundingBox3D, ObjectHypothesisWithPose))
            self.pub_tracked_3d.publish(out)

        def _publish_2d(self, header, tracks) -> None:
            out = Detection2DArray()
            out.header = header
            for track in tracks:
                det = TypedDetection2D(
                    bbox_xyxy=track.bbox_xyxy,
                    score=float(track.score),
                    label_id=0,
                    label_name="person",
                    detection_id=str(track.track_id),
                    timestamp=track.timestamp,
                    metadata={"track_state": track.track_state, "update_source": track.update_source},
                )
                out.detections.append(detection2d_to_msg(det, Detection2D, BoundingBox2D, ObjectHypothesisWithPose))
            self.pub_tracked_2d.publish(out)

        def callback(self, matched_msg: Detection3DArray, unmatched_lidar_msg: Detection3DArray, unmatched_camera_msg: Detection2DArray) -> None:
            start = time.perf_counter()
            matched = tuple(detections3d_from_msg(matched_msg))
            unmatched_lidar = tuple(detections3d_from_msg(unmatched_lidar_msg))
            unmatched_camera = tuple(detections2d_from_msg(unmatched_camera_msg))
            timestamp = self._stamp_seconds(matched_msg.header.stamp)
            fusion_result = FusionResult(
                matches=(),
                matched_3d=matched,
                unmatched_lidar=unmatched_lidar,
                unmatched_camera=unmatched_camera,
                final_3d=matched + unmatched_lidar,
                projected_lidar=(),
                iou_matrix=(),
                confidence_policy="lidar_score",
                output_policy="matched_plus_unmatched_lidar",
            )
            tracking = self.tracker.update(
                recording_id=str(self.get_parameter("recording_id").value),
                frame_index=self.frame_index,
                timestamp=timestamp,
                camera_detections=unmatched_camera,
                lidar_detections=matched + unmatched_lidar,
                fusion_result=fusion_result,
                ego_motion_transform=self._ego_motion_transform(matched_msg.header.stamp),
            )
            self._publish_3d(matched_msg.header, tracking.tracked_3d)
            self._publish_2d(unmatched_camera_msg.header, tracking.tracked_2d)
            elapsed = time.perf_counter() - start
            diag = {
                "frame_index": self.frame_index,
                "stamp": timestamp,
                "matched_3d": len(matched),
                "unmatched_lidar": len(unmatched_lidar),
                "unmatched_camera": len(unmatched_camera),
                "tracked_3d": len(tracking.tracked_3d),
                "tracked_2d": len(tracking.tracked_2d),
                "active_3d_tracks": tracking.diagnostics.active_3d_tracks,
                "confirmed_tracks": tracking.diagnostics.confirmed_tracks,
                "new_tracks": tracking.diagnostics.new_tracks,
                "deleted_tracks": tracking.diagnostics.deleted_tracks,
                "ego_motion_mode": str(self.get_parameter("ego_motion_mode").value),
                "tf_failures": self.tf_failures,
                "tracker_update_time_sec": elapsed,
            }
            self.pub_diag.publish(String(data=json.dumps(diag)))
            self.frame_index += 1

    rclpy.init(args=args)
    node = AghriDeepFusionMOTTrackerNode()
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
