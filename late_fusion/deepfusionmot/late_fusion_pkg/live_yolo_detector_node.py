"""Online ROS 2 YOLO detector node for raw camera images."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np


def _image_to_numpy(msg):
    """Convert a ROS Image to an ndarray without requiring cache files."""

    try:
        from cv_bridge import CvBridge

        bridge = CvBridge()
        return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception:
        pass

    dtype = np.uint8
    channels = 1
    encoding = str(msg.encoding).lower()
    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding without cv_bridge: {msg.encoding}")

    row_bytes = int(msg.step)
    width_bytes = int(msg.width) * channels
    raw = np.frombuffer(msg.data, dtype=dtype)
    rows = raw.reshape(int(msg.height), row_bytes)
    packed = rows[:, :width_bytes]
    image = packed.reshape(int(msg.height), int(msg.width), channels)
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, [2, 1, 0]].copy()
    if encoding == "bgra8":
        return image[:, :, :3].copy()
    if channels == 1:
        return image.reshape(int(msg.height), int(msg.width))
    return image.copy()


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose

    from late_fusion_pkg.detection_types import Detection2D as TypedDetection2D
    from late_fusion_pkg.live_conversions import latency_ms, stamp_to_float
    from late_fusion_pkg.live_detector_profiles import resolve_live_camera_profile
    from late_fusion_pkg.ros_adapters import detection2d_to_msg

    class LiveYoloDetector(Node):
        def __init__(self):
            super().__init__("live_yolo_detector")
            self.declare_parameter("dataset_profile", "aghri")
            self.declare_parameter("camera_model", "generic")
            self.declare_parameter("checkpoint_path", "")
            self.declare_parameter("checkpoint_sha256", "")
            self.declare_parameter("input_topic", "/dataset/cam_zed_rgb/image")
            self.declare_parameter("output_topic", "/detector/camera/detections")
            self.declare_parameter("diagnostics_topic", "/detector/camera/live_stats")
            self.declare_parameter("device", "auto")
            self.declare_parameter("imgsz", 640)
            self.declare_parameter("conf", 0.10)
            self.declare_parameter("class_id", 0)
            self.declare_parameter("queue_depth", 1)

            dataset_profile = str(self.get_parameter("dataset_profile").value)
            camera_model = str(self.get_parameter("camera_model").value)
            profile = resolve_live_camera_profile(dataset_profile, camera_model, validate_paths=True)
            checkpoint_override = str(self.get_parameter("checkpoint_path").value).strip()
            sha_override = str(self.get_parameter("checkpoint_sha256").value).strip()
            self.checkpoint_path = Path(checkpoint_override) if checkpoint_override else profile.checkpoint_path
            self.checkpoint_sha256 = sha_override or profile.checkpoint_sha256
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Live YOLO checkpoint does not exist: {self.checkpoint_path}")

            from late_fusion_pkg.checkpoints import sha256_file

            actual_sha = sha256_file(self.checkpoint_path)
            if self.checkpoint_sha256 not in ("", "unknown") and actual_sha != self.checkpoint_sha256:
                raise ValueError(
                    f"Live YOLO checkpoint SHA256 mismatch for {self.checkpoint_path}: "
                    f"expected {self.checkpoint_sha256}, got {actual_sha}"
                )
            self.checkpoint_sha256 = actual_sha

            try:
                from ultralytics import YOLO
                import torch
            except Exception as exc:
                raise RuntimeError(
                    "Live YOLO mode requires ultralytics and torch in the ROS Python environment. "
                    "No detector cache is used in live mode."
                ) from exc

            requested_device = str(self.get_parameter("device").value)
            self.device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else (
                "cpu" if requested_device == "auto" else requested_device
            )
            load_start = time.perf_counter()
            self.model = YOLO(str(self.checkpoint_path)).to(self.device)
            self.model_load_ms = latency_ms(load_start)
            self.imgsz = int(self.get_parameter("imgsz").value)
            self.conf = float(self.get_parameter("conf").value)
            self.class_id = int(self.get_parameter("class_id").value)
            self.profile = profile
            self.lock = threading.Lock()
            self.received = 0
            self.processed = 0
            self.dropped = 0
            self.failures = 0
            self.pub = self.create_publisher(Detection2DArray, self.get_parameter("output_topic").value, 10)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)
            self.create_subscription(
                Image,
                self.get_parameter("input_topic").value,
                self.callback,
                int(self.get_parameter("queue_depth").value),
            )
            self.get_logger().info(
                "LIVE YOLO inference enabled; cached camera replay disabled; "
                f"model={profile.label}; checkpoint={self.checkpoint_path}; "
                f"sha256={self.checkpoint_sha256}; device={self.device}; load_ms={self.model_load_ms:.1f}"
            )

        def _publish_diag(self, msg, detections: int, start_time: float, error: str = "") -> None:
            payload = {
                "mode": "live_yolo_inference",
                "cache_used": False,
                "dataset_profile": self.profile.dataset_profile,
                "camera_model": self.profile.camera_model,
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "stamp": stamp_to_float(msg.header.stamp),
                "frame_id": msg.header.frame_id,
                "published_detections": int(detections),
                "received": self.received,
                "processed": self.processed,
                "dropped": self.dropped,
                "failures": self.failures,
                "latency_ms": latency_ms(start_time),
                "device": self.device,
                "error": error,
            }
            self.pub_diag.publish(String(data=json.dumps(payload, sort_keys=True)))

        def callback(self, msg: Image) -> None:
            self.received += 1
            if not self.lock.acquire(blocking=False):
                self.dropped += 1
                return
            start_time = time.perf_counter()
            try:
                out = Detection2DArray()
                out.header = msg.header
                image = _image_to_numpy(msg)
                result = self.model(
                    image,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    classes=[self.class_id],
                    verbose=False,
                )[0]
                boxes = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else []
                for index, box in enumerate(boxes):
                    x1, y1, x2, y2, score, class_id = box[:6]
                    det = TypedDetection2D(
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        score=float(score),
                        label_id=int(class_id),
                        label_name=str(result.names.get(int(class_id), int(class_id))),
                        detection_id=f"live_yolo_{self.processed}_{index}",
                        timestamp=stamp_to_float(msg.header.stamp),
                    )
                    out.detections.append(detection2d_to_msg(det, Detection2D, BoundingBox2D, ObjectHypothesisWithPose))
                self.pub.publish(out)
                self.processed += 1
                self._publish_diag(msg, len(out.detections), start_time)
            except Exception as exc:
                self.failures += 1
                self.get_logger().error(f"Live YOLO inference failed: {exc}")
                self._publish_diag(msg, 0, start_time, error=str(exc))
            finally:
                self.lock.release()

    rclpy.init(args=args)
    node = LiveYoloDetector()
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
