"""Wall-clock performance monitor for fully online ROS 2 late fusion."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _stamp_to_key(stamp) -> str:
    return f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"


def _stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _parse_json_or_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        parsed = {}
        for item in str(text).split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
        return parsed or {"message": text}


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import String
    from vision_msgs.msg import Detection2DArray, Detection3DArray

    class OnlinePerformanceMonitor(Node):
        def __init__(self):
            super().__init__("online_performance_monitor")
            self.declare_parameter("output_jsonl", "results/aghri_online_fps_benchmark/raw_runs/live_monitor.jsonl")
            self.declare_parameter("image_topic", "/dataset/cam_zed_rgb/image")
            self.declare_parameter("pointcloud_topic", "/dataset/lidar/points")
            self.declare_parameter("camera_detection_topic", "/detector/camera/detections")
            self.declare_parameter("lidar_detection_topic", "/detector/lidar/detections")
            self.declare_parameter("fused_topic", "/late_fusion/fused_3d")
            self.declare_parameter("matched_topic", "/late_fusion/matched_3d")
            self.declare_parameter("tracked_3d_topic", "/deepfusionmot/tracked_3d")
            self.declare_parameter("camera_diagnostics_topic", "/detector/camera/live_stats")
            self.declare_parameter("lidar_diagnostics_topic", "/detector/lidar/live_stats")
            self.declare_parameter("fusion_diagnostics_topic", "/late_fusion/diagnostics")
            self.declare_parameter("tracking_diagnostics_topic", "/deepfusionmot/diagnostics")
            self.declare_parameter("visualization_diagnostics_topic", "/visualization/diagnostics")

            self.output_path = Path(str(self.get_parameter("output_jsonl").value)).expanduser()
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.output_path.open("w", encoding="utf-8")
            self.counts: dict[str, int] = {}
            self.image_receipts: dict[str, int] = {}
            self.lidar_receipts: dict[str, int] = {}
            self.max_receipts = 5000

            self.create_subscription(Image, self.get_parameter("image_topic").value, self.image_cb, 50)
            self.create_subscription(PointCloud2, self.get_parameter("pointcloud_topic").value, self.lidar_cb, 50)
            self.create_subscription(Detection2DArray, self.get_parameter("camera_detection_topic").value, self.camera_detection_cb, 50)
            self.create_subscription(Detection3DArray, self.get_parameter("lidar_detection_topic").value, self.lidar_detection_cb, 50)
            self.create_subscription(Detection3DArray, self.get_parameter("fused_topic").value, self.fused_cb, 50)
            self.create_subscription(Detection3DArray, self.get_parameter("matched_topic").value, self.matched_cb, 50)
            self.create_subscription(Detection3DArray, self.get_parameter("tracked_3d_topic").value, self.tracked_cb, 50)
            self.create_subscription(String, self.get_parameter("camera_diagnostics_topic").value, lambda msg: self.diag_cb("camera_live", msg), 50)
            self.create_subscription(String, self.get_parameter("lidar_diagnostics_topic").value, lambda msg: self.diag_cb("lidar_live", msg), 50)
            self.create_subscription(String, self.get_parameter("fusion_diagnostics_topic").value, lambda msg: self.diag_cb("fusion", msg), 50)
            self.create_subscription(String, self.get_parameter("tracking_diagnostics_topic").value, lambda msg: self.diag_cb("tracking", msg), 50)
            self.create_subscription(String, self.get_parameter("visualization_diagnostics_topic").value, lambda msg: self.diag_cb("visualization", msg), 50)
            self.write({"event": "monitor_start", "topics": self._topics()})
            self.get_logger().info(f"writing online performance monitor events to {self.output_path}")

        def _topics(self) -> dict[str, str]:
            names = (
                "image_topic",
                "pointcloud_topic",
                "camera_detection_topic",
                "lidar_detection_topic",
                "fused_topic",
                "matched_topic",
                "tracked_3d_topic",
                "camera_diagnostics_topic",
                "lidar_diagnostics_topic",
                "fusion_diagnostics_topic",
                "tracking_diagnostics_topic",
                "visualization_diagnostics_topic",
            )
            return {name: str(self.get_parameter(name).value) for name in names}

        def now_ns(self) -> int:
            return time.perf_counter_ns()

        def increment(self, name: str) -> int:
            self.counts[name] = self.counts.get(name, 0) + 1
            return self.counts[name]

        def write(self, payload: dict) -> None:
            payload = {
                "monitor_time_ns": self.now_ns(),
                "ros_time_sec": self.get_clock().now().nanoseconds / 1e9,
                **payload,
            }
            self.stream.write(json.dumps(payload, sort_keys=True) + "\n")
            self.stream.flush()

        def _remember(self, table: dict[str, int], key: str, wall_ns: int) -> None:
            table[key] = wall_ns
            if len(table) > self.max_receipts:
                for old_key in list(table)[: len(table) - self.max_receipts]:
                    table.pop(old_key, None)

        def image_cb(self, msg: Image) -> None:
            wall_ns = self.now_ns()
            key = _stamp_to_key(msg.header.stamp)
            self._remember(self.image_receipts, key, wall_ns)
            self.write({
                "event": "input_image",
                "count": self.increment("input_image"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": key,
                "frame_id": msg.header.frame_id,
            })

        def lidar_cb(self, msg: PointCloud2) -> None:
            wall_ns = self.now_ns()
            key = _stamp_to_key(msg.header.stamp)
            self._remember(self.lidar_receipts, key, wall_ns)
            self.write({
                "event": "input_pointcloud",
                "count": self.increment("input_pointcloud"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": key,
                "frame_id": msg.header.frame_id,
            })

        def camera_detection_cb(self, msg: Detection2DArray) -> None:
            self.write({
                "event": "camera_detections",
                "count": self.increment("camera_detections"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": _stamp_to_key(msg.header.stamp),
                "detections": len(msg.detections),
            })

        def lidar_detection_cb(self, msg: Detection3DArray) -> None:
            self.write({
                "event": "lidar_detections",
                "count": self.increment("lidar_detections"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": _stamp_to_key(msg.header.stamp),
                "detections": len(msg.detections),
            })

        def fused_cb(self, msg: Detection3DArray) -> None:
            wall_ns = self.now_ns()
            key = _stamp_to_key(msg.header.stamp)
            receipt_ns = self.lidar_receipts.get(key)
            latency_ms = None
            if receipt_ns is not None:
                latency_ms = (wall_ns - receipt_ns) / 1_000_000.0
            self.write({
                "event": "fused_output",
                "count": self.increment("fused_output"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": key,
                "detections": len(msg.detections),
                "sensor_receipt_wall_ns": receipt_ns,
                "end_to_end_latency_ms": latency_ms,
            })

        def matched_cb(self, msg: Detection3DArray) -> None:
            self.write({
                "event": "matched_output",
                "count": self.increment("matched_output"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": _stamp_to_key(msg.header.stamp),
                "detections": len(msg.detections),
            })

        def tracked_cb(self, msg: Detection3DArray) -> None:
            wall_ns = self.now_ns()
            key = _stamp_to_key(msg.header.stamp)
            receipt_ns = self.lidar_receipts.get(key)
            latency_ms = None
            if receipt_ns is not None:
                latency_ms = (wall_ns - receipt_ns) / 1_000_000.0
            self.write({
                "event": "tracked_3d_output",
                "count": self.increment("tracked_3d_output"),
                "stamp": _stamp_to_float(msg.header.stamp),
                "stamp_key": key,
                "detections": len(msg.detections),
                "sensor_receipt_wall_ns": receipt_ns,
                "end_to_end_latency_ms": latency_ms,
            })

        def diag_cb(self, source: str, msg: String) -> None:
            self.write({
                "event": "diagnostic",
                "source": source,
                "count": self.increment(f"diagnostic_{source}"),
                "data": _parse_json_or_text(msg.data),
                "raw": msg.data,
            })

        def destroy_node(self):
            try:
                self.write({"event": "monitor_stop", "counts": self.counts})
                self.stream.close()
            finally:
                super().destroy_node()

    rclpy.init(args=args)
    node = OnlinePerformanceMonitor()
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
