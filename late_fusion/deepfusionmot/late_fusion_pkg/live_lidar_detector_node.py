"""Online ROS 2 LiDAR detector node backed by a persistent MMDetection3D worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def _reserve_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from vision_msgs.msg import BoundingBox3D, Detection3D, Detection3DArray, ObjectHypothesisWithPose

    from late_fusion_pkg.detection_types import Detection3D as TypedDetection3D
    from late_fusion_pkg.live_conversions import encode_points, latency_ms, pointcloud2_to_xyzi, stamp_to_float
    from late_fusion_pkg.live_detector_profiles import resolve_live_lidar_profile
    from late_fusion_pkg.live_worker_protocol import recv_json, send_json
    from late_fusion_pkg.ros_adapters import detection3d_to_msg

    class LiveLidarDetector(Node):
        def __init__(self):
            super().__init__("live_lidar_detector")
            self.declare_parameter("dataset_profile", "aghri")
            self.declare_parameter("lidar_detector", "pointpillars")
            self.declare_parameter("lidar_model", "generic")
            self.declare_parameter("allow_cross_domain", False)
            self.declare_parameter("input_topic", "/dataset/lidar/points")
            self.declare_parameter("output_topic", "/detector/lidar/detections")
            self.declare_parameter("diagnostics_topic", "/detector/lidar/live_stats")
            self.declare_parameter("frame_id", "front_lidar_link")
            self.declare_parameter("device", "cuda:0")
            self.declare_parameter("worker_python", "/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/bin/python")
            self.declare_parameter("worker_host", "127.0.0.1")
            self.declare_parameter("worker_timeout_sec", 600.0)
            self.declare_parameter("score_threshold_override", -1.0)
            self.declare_parameter("queue_depth", 1)

            self.profile = resolve_live_lidar_profile(
                dataset_profile=str(self.get_parameter("dataset_profile").value),
                lidar_detector=str(self.get_parameter("lidar_detector").value),
                lidar_model=str(self.get_parameter("lidar_model").value),
                validate_paths=True,
                allow_cross_domain=bool(self.get_parameter("allow_cross_domain").value),
            )
            self.frame_id = str(self.get_parameter("frame_id").value)
            self.worker_host = str(self.get_parameter("worker_host").value)
            self.worker_port = _reserve_local_port(self.worker_host)
            self.worker_timeout_sec = float(self.get_parameter("worker_timeout_sec").value)
            score_threshold_override = float(self.get_parameter("score_threshold_override").value)
            self.score_threshold = (
                score_threshold_override
                if score_threshold_override >= 0.0
                else float(self.profile.score_threshold)
            )
            self.lock = threading.Lock()
            self.received = 0
            self.processed = 0
            self.dropped = 0
            self.failures = 0
            self.worker = None
            self.sock = None

            self._start_worker()
            self.get_logger().info("worker_ready")
            self.pub = self.create_publisher(Detection3DArray, self.get_parameter("output_topic").value, 10)
            self.pub_diag = self.create_publisher(String, self.get_parameter("diagnostics_topic").value, 10)
            self.create_subscription(
                PointCloud2,
                self.get_parameter("input_topic").value,
                self.callback,
                int(self.get_parameter("queue_depth").value),
            )
            self.get_logger().info(
                "LIVE LiDAR inference enabled; cached LiDAR replay disabled; "
                f"detector={self.profile.label}; config={self.profile.config_path}; "
                f"checkpoint={self.profile.checkpoint_path}; sha256={self.profile.checkpoint_sha256}; "
                f"box_origin={self.profile.box_origin}"
            )

        def _start_worker(self) -> None:
            worker_python = str(self.get_parameter("worker_python").value)
            if not Path(worker_python).exists():
                raise FileNotFoundError(f"MMDetection3D worker Python does not exist: {worker_python}")
            repo_root = "/home/prabuddhi/Desktop/late_fusion/deepfusionmot"
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONNOUSERSITE"] = "1"
            env.setdefault(
                "AGHRI_SECOND_BASE_CONFIG",
                "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d/configs/models/second_aghri.py",
            )
            env.setdefault(
                "AGHRI_SECOND_DATA_ROOT",
                "/home/prabuddhi/cluster/aghri_second_finetune/data/aghri_mmdet3d",
            )
            env.setdefault(
                "AGHRI_SECOND_GENERIC_CHECKPOINT",
                "/home/prabuddhi/Desktop/late_fusion/deepfusionmot/checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
            )
            env.setdefault("AGHRI_SECOND_WORK_DIR", "/tmp/aghri_second_online_ros")
            env.setdefault(
                "AGHRI_POINTPILLARS_BASE_CONFIG",
                "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d/configs/models/pointpillars_aghri.py",
            )
            env.setdefault(
                "AGHRI_POINTPILLARS_DATA_ROOT",
                "/home/prabuddhi/cluster/aghri_second_finetune/data/aghri_mmdet3d",
            )
            env.setdefault(
                "AGHRI_POINTPILLARS_GENERIC_CHECKPOINT",
                "/home/prabuddhi/Desktop/late_fusion/deepfusionmot/checkpoints/kitti_pretrained/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth",
            )
            env.setdefault("AGHRI_POINTPILLARS_WORK_DIR", "/tmp/aghri_pointpillars_online_ros")
            cmd = [
                worker_python,
                "-m",
                "late_fusion_pkg.mmdet3d_inference_worker",
                "--host",
                self.worker_host,
                "--port",
                str(self.worker_port),
                "--dataset-profile",
                self.profile.dataset_profile,
                "--lidar-detector",
                self.profile.detector_family,
                "--lidar-model",
                self.profile.lidar_model,
                "--config",
                str(self.profile.config_path),
                "--checkpoint",
                str(self.profile.checkpoint_path),
                "--checkpoint-sha256",
                self.profile.checkpoint_sha256,
                "--device",
                str(self.get_parameter("device").value),
                "--selected-class-id",
                str(self.profile.selected_class_id),
                "--selected-class-name",
                self.profile.selected_class_name,
                "--box-origin",
                self.profile.box_origin,
                "--score-threshold",
                str(self.score_threshold),
            ]
            self.worker = subprocess.Popen(cmd, env=env)
            deadline = time.time() + self.worker_timeout_sec
            last_error = None
            while time.time() < deadline:
                if self.worker.poll() is not None:
                    raise RuntimeError(f"MMDetection3D worker exited before accepting connections: rc={self.worker.returncode}")
                try:
                    sock = socket.create_connection((self.worker_host, self.worker_port), timeout=2.0)
                    sock.settimeout(self.worker_timeout_sec)
                    self.sock = sock
                    send_json(sock, {"command": "ping"})
                    response = recv_json(sock)
                    if response.get("status") != "ok":
                        raise RuntimeError(f"MMDetection3D worker ping failed: {response}")
                    return
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.5)
            raise TimeoutError(f"Timed out waiting for MMDetection3D worker: {last_error}")

        def _publish_diag(self, msg, detections: int, start_time: float, raw_points: int, valid_points: int, error: str = "") -> None:
            payload = {
                "mode": "live_lidar_inference",
                "cache_used": False,
                "dataset_profile": self.profile.dataset_profile,
                "detector_family": self.profile.detector_family,
                "lidar_model": self.profile.lidar_model,
                "config_path": str(self.profile.config_path),
                "checkpoint_path": str(self.profile.checkpoint_path),
                "checkpoint_sha256": self.profile.checkpoint_sha256,
                "box_origin": self.profile.box_origin,
                "score_threshold": self.score_threshold,
                "stamp": stamp_to_float(msg.header.stamp),
                "frame_id": self.frame_id,
                "raw_points": raw_points,
                "valid_points": valid_points,
                "published_detections": int(detections),
                "received": self.received,
                "processed": self.processed,
                "dropped": self.dropped,
                "failures": self.failures,
                "latency_ms": latency_ms(start_time),
                "error": error,
            }
            self.pub_diag.publish(String(data=json.dumps(payload, sort_keys=True)))

        def callback(self, msg: PointCloud2) -> None:
            self.received += 1
            if not self.lock.acquire(blocking=False):
                self.dropped += 1
                return
            start_time = time.perf_counter()
            raw_points = 0
            valid_points = 0
            try:
                points, raw_points, valid_points = pointcloud2_to_xyzi(msg)
                request = {
                    "command": "infer",
                    "stamp": stamp_to_float(msg.header.stamp),
                    "frame_id": msg.header.frame_id,
                    "point_count": int(points.shape[0]),
                    "points_b64": encode_points(points),
                }
                send_json(self.sock, request)
                response = recv_json(self.sock)
                if response.get("status") != "ok":
                    raise RuntimeError(str(response.get("error", response)))
                out = Detection3DArray()
                out.header = msg.header
                out.header.frame_id = self.frame_id
                for index, det in enumerate(response.get("detections", [])):
                    typed = TypedDetection3D(
                        center_xyz=(float(det["x"]), float(det["y"]), float(det["z"])),
                        size_lwh=(float(det["length"]), float(det["width"]), float(det["height"])),
                        yaw=float(det["yaw"]),
                        score=float(det["score"]),
                        label_id=int(det["label_id"]),
                        label_name=str(det.get("label_name", self.profile.selected_class_name)),
                        detection_id=str(det.get("id", f"live_lidar_{self.processed}_{index}")),
                        timestamp=stamp_to_float(msg.header.stamp),
                        frame_id=self.frame_id,
                        box_origin=self.profile.box_origin,
                    )
                    out.detections.append(detection3d_to_msg(typed, Detection3D, BoundingBox3D, ObjectHypothesisWithPose))
                self.pub.publish(out)
                self.processed += 1
                self._publish_diag(msg, len(out.detections), start_time, raw_points, valid_points)
            except Exception as exc:
                self.failures += 1
                self.get_logger().error(f"Live LiDAR inference failed: {exc}")
                self._publish_diag(msg, 0, start_time, raw_points, valid_points, error=str(exc))
            finally:
                self.lock.release()

        def destroy_node(self):
            try:
                if self.sock is not None:
                    try:
                        send_json(self.sock, {"command": "shutdown"})
                        recv_json(self.sock)
                    except Exception:
                        pass
                    self.sock.close()
            finally:
                if self.worker is not None and self.worker.poll() is None:
                    self.worker.terminate()
                    try:
                        self.worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.worker.kill()
                super().destroy_node()

    rclpy.init(args=args)
    node = LiveLidarDetector()
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
