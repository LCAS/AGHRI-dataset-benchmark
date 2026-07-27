"""Collect cached replay and late-fusion diagnostics during ROS playback."""

from __future__ import annotations

import json
from pathlib import Path


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class CacheDiagnosticsCollector(Node):
        def __init__(self):
            super().__init__("cache_diagnostics_collector")
            self.declare_parameter("output_jsonl", "results/aghri_generic_baseline/rosbag_evaluation/live_diagnostics.jsonl")
            self.declare_parameter("camera_stats_topic", "/detector/camera/cache_stats")
            self.declare_parameter("lidar_stats_topic", "/detector/lidar/cache_stats")
            self.declare_parameter("fusion_diagnostics_topic", "/late_fusion/diagnostics")
            self.declare_parameter("tracking_diagnostics_topic", "/deepfusionmot/diagnostics")
            self.declare_parameter("visualization_diagnostics_topic", "/visualization/diagnostics")
            self.declare_parameter("camera_source_label", "camera_cache")
            self.declare_parameter("lidar_source_label", "lidar_cache")
            self.output_path = Path(str(self.get_parameter("output_jsonl").value)).expanduser()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.output_path.open("a", encoding="utf-8")
            camera_label = str(self.get_parameter("camera_source_label").value)
            lidar_label = str(self.get_parameter("lidar_source_label").value)
            self.create_subscription(String, self.get_parameter("camera_stats_topic").value, lambda msg: self._write(camera_label, msg), 10)
            self.create_subscription(String, self.get_parameter("lidar_stats_topic").value, lambda msg: self._write(lidar_label, msg), 10)
            self.create_subscription(String, self.get_parameter("fusion_diagnostics_topic").value, lambda msg: self._write("fusion", msg), 10)
            self.create_subscription(String, self.get_parameter("tracking_diagnostics_topic").value, lambda msg: self._write("tracking", msg), 10)
            self.create_subscription(String, self.get_parameter("visualization_diagnostics_topic").value, lambda msg: self._write("visualization", msg), 10)
            self.get_logger().info(f"writing cached replay diagnostics to {self.output_path}")

        def _write(self, source: str, msg: String) -> None:
            payload = {
                "collector_time_sec": self.get_clock().now().nanoseconds / 1e9,
                "source": source,
                "data": msg.data,
            }
            self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self.stream.flush()

        def destroy_node(self):
            try:
                self.stream.close()
            finally:
                super().destroy_node()

    rclpy.init(args=args)
    node = CacheDiagnosticsCollector()
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
