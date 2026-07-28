"""ROS 2 publisher that replays cached generic YOLO detections by image stamp."""

from __future__ import annotations

import json


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

    from late_fusion_pkg.cache_replay import load_yolo_cache
    from late_fusion_pkg.ros_adapters import detection2d_to_msg

    class CachedYoloPublisher(Node):
        def __init__(self):
            super().__init__("cached_yolo_publisher")
            self.declare_parameter("cache_csv", "results/aghri_generic_baseline/test_full/yolo_detections.csv")
            self.declare_parameter("recording_name", "")
            self.declare_parameter("input_topic", "/dataset/cam_zed_rgb/image")
            self.declare_parameter("output_topic", "/detector/camera/detections")
            self.declare_parameter("diagnostics_topic", "/detector/camera/cache_stats")
            self.declare_parameter("timestamp_tolerance_sec", 0.02)
            self.declare_parameter("cache_label", "camera_generic")
            self.declare_parameter("checkpoint_sha256", "unknown")
            self.declare_parameter("checkpoint_path", "unknown")
            self.recording_name = str(self.get_parameter("recording_name").value)
            self.cache_label = str(self.get_parameter("cache_label").value)
            self.checkpoint_sha256 = str(self.get_parameter("checkpoint_sha256").value)
            self.checkpoint_path = str(self.get_parameter("checkpoint_path").value)
            self.cache = load_yolo_cache(
                str(self.get_parameter("cache_csv").value),
                timestamp_tolerance_sec=float(self.get_parameter("timestamp_tolerance_sec").value),
            )
            self.hits = 0
            self.misses = 0
            self.messages = 0
            self.pub = self.create_publisher(Detection2DArray, self.get_parameter("output_topic").value, 10)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)
            self.create_subscription(Image, self.get_parameter("input_topic").value, self.callback, 10)
            self.get_logger().info(
                f"loaded cached YOLO replay label={self.cache_label} "
                f"checkpoint_sha256={self.checkpoint_sha256} "
                f"checkpoint_path={self.checkpoint_path}: {self.cache.validation}"
            )

        def callback(self, msg: Image) -> None:
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
            detections = self.cache.lookup(self.recording_name, stamp)
            self.messages += 1
            if self.cache.contains_timestamp(self.recording_name, stamp):
                self.hits += 1
            else:
                self.misses += 1
            out = Detection2DArray()
            out.header = msg.header
            for det in detections:
                out.detections.append(detection2d_to_msg(det, Detection2D, BoundingBox2D, ObjectHypothesisWithPose))
            self.pub.publish(out)
            self.pub_diag.publish(
                String(
                    data=json.dumps(
                        {
                            "mode": "cached_generic_detector_ros_replay",
                            "cache_label": self.cache_label,
                            "checkpoint_sha256": self.checkpoint_sha256,
                            "cache_csv": self.cache.validation.path,
                            "recording": self.recording_name,
                            "stamp": stamp,
                            "published_detections": len(out.detections),
                            "messages": self.messages,
                            "hits": self.hits,
                            "misses": self.misses,
                        },
                        sort_keys=True,
                    )
                )
            )

    rclpy.init(args=args)
    node = CachedYoloPublisher()
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
