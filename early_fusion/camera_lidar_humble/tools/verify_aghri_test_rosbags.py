#!/usr/bin/env python3
"""Verify AGHRI test ROS 2 bags produced by agri-human-dataset-tools."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from convert_aghri_test_to_rosbags import (
    CALIBRATION_CAMERA_FRAME,
    CALIBRATION_LIDAR_FRAME,
    CAMERA_INFO_TOPIC,
    CAMERA_INFO_TOPICS,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_EXTRINSICS_MEMBER,
    DATASET_TOOLS_IMAGE_FRAME,
    DATASET_TOOLS_LIDAR_FRAME,
    DEFAULT_DATASET_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    IMAGE_TOPIC,
    LIDAR_TOPIC,
    SYNC_SLOP_SEC,
    TF_CHILD_FRAME,
    TF_PARENT_FRAME,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    load_manifest_recordings,
)

CAMERA_IMAGE_TOPICS = {
    "cam_zed_rgb": IMAGE_TOPIC,
    "cam_fish_front": "/dataset/cam_fish_front/image",
    "cam_fish_left": "/dataset/cam_fish_left/image",
    "cam_fish_right": "/dataset/cam_fish_right/image",
}
EXPECTED_CAMERA_FROM_LIDAR = np.array([
    [0.0, -1.0, 0.0, -0.220],
    [0.0, 0.0, -1.0, 0.295],
    [1.0, 0.0, 0.0, 0.075],
    [0.0, 0.0, 0.0, 1.000],
], dtype=float)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def load_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read(member).decode("utf-8"))


def quat_to_matrix(rotation: dict[str, float]) -> np.ndarray:
    x = float(rotation["x"])
    y = float(rotation["y"])
    z = float(rotation["z"])
    w = float(rotation["w"])
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_msg_to_matrix(transform) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quat_to_matrix({
        "x": transform.rotation.x,
        "y": transform.rotation.y,
        "z": transform.rotation.z,
        "w": transform.rotation.w,
    })
    matrix[:3, 3] = [
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
    ]
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=float)
    inv[:3, :3] = matrix[:3, :3].T
    inv[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inv


def compose_json_transform(extrinsics: dict[str, Any], source: str, target: str) -> tuple[np.ndarray, list[str]]:
    adjacency: dict[str, list[str]] = {}
    transform_map: dict[tuple[str, str], np.ndarray] = {}
    for edge in extrinsics.get("transforms", []):
        parent = (edge.get("header") or {}).get("frame_id")
        child = edge.get("child_frame_id")
        if not parent or not child:
            continue
        transform = edge["transform"]
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = quat_to_matrix(transform["rotation"])
        matrix[:3, 3] = [
            float(transform["translation"]["x"]),
            float(transform["translation"]["y"]),
            float(transform["translation"]["z"]),
        ]
        adjacency.setdefault(parent, []).append(child)
        adjacency.setdefault(child, []).append(parent)
        transform_map[(child, parent)] = matrix
        transform_map[(parent, child)] = invert_transform(matrix)

    queue = [[source]]
    found = []
    shortest = None
    while queue:
        path = queue.pop(0)
        if shortest is not None and len(path) > shortest:
            continue
        if path[-1] == target:
            shortest = len(path)
            found.append(path)
            continue
        for neighbor in adjacency.get(path[-1], []):
            if neighbor not in path:
                queue.append(path + [neighbor])
    if len(found) != 1:
        raise RuntimeError(f"Expected one transform path from {source} to {target}, found {len(found)}")

    result = np.eye(4, dtype=float)
    for frm, to in zip(found[0][:-1], found[0][1:]):
        result = transform_map[(frm, to)] @ result
    return result, found[0]


def nearest_delta(value: float, candidates: list[float]) -> float | None:
    if not candidates:
        return None
    idx = bisect_left(candidates, value)
    options = []
    if idx < len(candidates):
        options.append(abs(candidates[idx] - value))
    if idx > 0:
        options.append(abs(candidates[idx - 1] - value))
    return min(options) if options else None


def sync_summary(image_stamps: list[float], lidar_stamps: list[float], slop_sec: float) -> dict[str, Any]:
    deltas = [nearest_delta(stamp, image_stamps) for stamp in lidar_stamps]
    finite_deltas = [delta for delta in deltas if delta is not None]
    compatible = [delta for delta in finite_deltas if delta <= slop_sec]
    return {
        "sync_slop_sec": slop_sec,
        "compatible_lidar_message_count": len(compatible),
        "unmatched_lidar_message_count": len(lidar_stamps) - len(compatible),
        "mean_nearest_image_delta_sec": float(np.mean(finite_deltas)) if finite_deltas else None,
        "max_compatible_image_delta_sec": max(compatible) if compatible else None,
        "max_nearest_image_delta_sec": max(finite_deltas) if finite_deltas else None,
    }


def verify_bag(
    bag_path: Path,
    recording: dict[str, Any],
    source_manifest_hash: str,
    sync_slop_sec: float,
    calibration_path: Path,
    extrinsics_member: str,
) -> dict[str, Any]:
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2
        from sensor_msgs_py import point_cloud2
        from tf2_msgs.msg import TFMessage
        import tf2_ros
        from rclpy.time import Time
    except Exception as exc:  # noqa: BLE001 - clear ROS setup failure.
        raise RuntimeError("ROS 2 bag verification requires ROS Humble sourced") from exc

    row: dict[str, Any] = {
        "recording": recording["recording"],
        "bag_path": str(bag_path),
        "source_manifest_sha256": source_manifest_hash,
        "verification_status": "failed",
        "failure_reason": None,
    }
    try:
        metadata = bag_path / "metadata.yaml"
        if not metadata.exists():
            raise RuntimeError("metadata.yaml missing")
        if not list(bag_path.glob("*.db3")):
            raise RuntimeError("sqlite3 database missing")

        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        if topic_types.get(IMAGE_TOPIC) != "sensor_msgs/msg/Image":
            raise RuntimeError(f"Missing or wrong image topic type: {topic_types.get(IMAGE_TOPIC)}")
        if topic_types.get(CAMERA_INFO_TOPIC) != "sensor_msgs/msg/CameraInfo":
            raise RuntimeError(f"Missing or wrong CameraInfo topic type: {topic_types.get(CAMERA_INFO_TOPIC)}")
        for camera, info_topic in CAMERA_INFO_TOPICS.items():
            if topic_types.get(info_topic) != "sensor_msgs/msg/CameraInfo":
                raise RuntimeError(f"Missing or wrong {camera} CameraInfo topic type: {topic_types.get(info_topic)}")
            image_topic = CAMERA_IMAGE_TOPICS[camera]
            if topic_types.get(image_topic) != "sensor_msgs/msg/Image":
                raise RuntimeError(f"Missing or wrong {camera} image topic type: {topic_types.get(image_topic)}")
        if topic_types.get(LIDAR_TOPIC) != "sensor_msgs/msg/PointCloud2":
            raise RuntimeError(f"Missing or wrong lidar topic type: {topic_types.get(LIDAR_TOPIC)}")
        if topic_types.get(TF_TOPIC) != "tf2_msgs/msg/TFMessage":
            raise RuntimeError(f"Missing or wrong /tf topic type: {topic_types.get(TF_TOPIC)}")
        if topic_types.get(TF_STATIC_TOPIC) != "tf2_msgs/msg/TFMessage":
            raise RuntimeError(f"Missing or wrong /tf_static topic type: {topic_types.get(TF_STATIC_TOPIC)}")

        image_stamps: list[float] = []
        camera_info_stamps: list[float] = []
        image_stamps_by_camera: dict[str, list[float]] = {camera: [] for camera in CAMERA_INFO_TOPICS}
        camera_info_stamps_by_camera: dict[str, list[float]] = {camera: [] for camera in CAMERA_INFO_TOPICS}
        camera_info_counts_by_camera: dict[str, int] = {camera: 0 for camera in CAMERA_INFO_TOPICS}
        image_counts_by_camera: dict[str, int] = {camera: 0 for camera in CAMERA_INFO_TOPICS}
        camera_info_frames_by_camera: dict[str, set[str]] = {camera: set() for camera in CAMERA_INFO_TOPICS}
        lidar_stamps: list[float] = []
        image_frame_ids: set[str] = set()
        camera_info_frame_ids: set[str] = set()
        lidar_frame_ids: set[str] = set()
        image_count = 0
        camera_info_count = 0
        lidar_count = 0
        tf_count = 0
        tf_static_count = 0
        tf_frames: list[tuple[str, str]] = []
        tf_static_frames: list[tuple[str, str]] = []
        finite_cloud_checks = 0
        last_camera_info = None
        tf_buffer = tf2_ros.Buffer()

        while reader.has_next():
            topic, data, _bag_stamp = reader.read_next()
            if topic in set(CAMERA_IMAGE_TOPICS.values()):
                msg = deserialize_message(data, Image)
                stamp = stamp_to_float(msg.header.stamp)
                if stamp <= 0.0:
                    raise RuntimeError("Zero image header timestamp")
                if msg.height <= 0 or msg.width <= 0 or not msg.data:
                    raise RuntimeError("Invalid image payload")
                for camera, image_topic in CAMERA_IMAGE_TOPICS.items():
                    if topic == image_topic:
                        image_counts_by_camera[camera] += 1
                        image_stamps_by_camera[camera].append(stamp)
                        if camera == "cam_zed_rgb":
                            image_count += 1
                            image_frame_ids.add(msg.header.frame_id)
                            image_stamps.append(stamp)
                        break
            elif topic in set(CAMERA_INFO_TOPICS.values()):
                msg = deserialize_message(data, CameraInfo)
                stamp = stamp_to_float(msg.header.stamp)
                if msg.width <= 0 or msg.height <= 0:
                    raise RuntimeError("Invalid CameraInfo dimensions")
                if len(msg.k) != 9 or len(msg.r) != 9 or len(msg.p) != 12:
                    raise RuntimeError("Invalid CameraInfo matrix dimensions")
                for camera, info_topic in CAMERA_INFO_TOPICS.items():
                    if topic == info_topic:
                        camera_info_counts_by_camera[camera] += 1
                        camera_info_stamps_by_camera[camera].append(stamp)
                        camera_info_frames_by_camera[camera].add(msg.header.frame_id)
                        if camera == "cam_zed_rgb":
                            camera_info_count += 1
                            last_camera_info = msg
                            camera_info_frame_ids.add(msg.header.frame_id)
                            camera_info_stamps.append(stamp)
                        break
            elif topic == LIDAR_TOPIC:
                msg = deserialize_message(data, PointCloud2)
                lidar_count += 1
                lidar_frame_ids.add(msg.header.frame_id)
                stamp = stamp_to_float(msg.header.stamp)
                if stamp <= 0.0:
                    raise RuntimeError("Zero point-cloud header timestamp")
                pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
                if pts.size == 0:
                    raise RuntimeError("Empty finite point cloud")
                finite_cloud_checks += 1
                lidar_stamps.append(stamp)
            elif topic == TF_STATIC_TOPIC:
                msg = deserialize_message(data, TFMessage)
                tf_static_count += 1
                for transform in msg.transforms:
                    tf_static_frames.append((transform.header.frame_id, transform.child_frame_id))
                    tf_buffer.set_transform_static(transform, "verify_aghri_test_rosbags")
            elif topic == TF_TOPIC:
                msg = deserialize_message(data, TFMessage)
                tf_count += 1
                for transform in msg.transforms:
                    tf_frames.append((transform.header.frame_id, transform.child_frame_id))

        if image_count <= 0:
            raise RuntimeError("No ZED RGB image messages")
        if camera_info_count <= 0:
            raise RuntimeError("No ZED CameraInfo messages")
        if lidar_count <= 0:
            raise RuntimeError("No LiDAR point-cloud messages")
        if tf_count <= 0:
            raise RuntimeError("No /tf messages")
        if tf_static_count <= 0:
            raise RuntimeError("No /tf_static messages")
        if (TF_PARENT_FRAME, TF_CHILD_FRAME) not in tf_frames:
            raise RuntimeError(f"/tf missing {TF_PARENT_FRAME} -> {TF_CHILD_FRAME}")
        if recording["rgb_image_count"] and image_count != recording["rgb_image_count"]:
            raise RuntimeError(f"Image count {image_count} != manifest count {recording['rgb_image_count']}")
        if recording["lidar_point_cloud_count"] and lidar_count != recording["lidar_point_cloud_count"]:
            raise RuntimeError(
                f"Point-cloud count {lidar_count} != manifest count {recording['lidar_point_cloud_count']}"
            )
        if image_frame_ids != {DATASET_TOOLS_IMAGE_FRAME}:
            raise RuntimeError(f"Unexpected image frame IDs: {sorted(image_frame_ids)}")
        if camera_info_frame_ids != {DATASET_TOOLS_IMAGE_FRAME}:
            raise RuntimeError(f"Unexpected CameraInfo frame IDs: {sorted(camera_info_frame_ids)}")
        if lidar_frame_ids != {DATASET_TOOLS_LIDAR_FRAME}:
            raise RuntimeError(f"Unexpected lidar frame IDs: {sorted(lidar_frame_ids)}")
        if image_stamps != camera_info_stamps:
            raise RuntimeError("Image and CameraInfo timestamps do not match one-for-one")
        for camera in CAMERA_INFO_TOPICS:
            if image_counts_by_camera[camera] <= 0:
                raise RuntimeError(f"No image messages for {camera}")
            if camera_info_counts_by_camera[camera] <= 0:
                raise RuntimeError(f"No CameraInfo messages for {camera}")
            if image_stamps_by_camera[camera] != camera_info_stamps_by_camera[camera]:
                raise RuntimeError(f"Image and CameraInfo timestamps do not match for {camera}")
        if image_stamps != sorted(image_stamps):
            raise RuntimeError("Image timestamps are not monotonic")
        if camera_info_stamps != sorted(camera_info_stamps):
            raise RuntimeError("CameraInfo timestamps are not monotonic")
        if lidar_stamps != sorted(lidar_stamps):
            raise RuntimeError("Point-cloud timestamps are not monotonic")

        tf_transform = tf_buffer.lookup_transform(
            CALIBRATION_CAMERA_FRAME,
            CALIBRATION_LIDAR_FRAME,
            Time(),
        )
        tf_camera_from_lidar = transform_msg_to_matrix(tf_transform.transform)
        extrinsics = load_json_from_zip(calibration_path, extrinsics_member)
        json_camera_from_lidar, tf_path = compose_json_transform(
            extrinsics,
            CALIBRATION_LIDAR_FRAME,
            CALIBRATION_CAMERA_FRAME,
        )
        tf_json_max_abs_diff = float(np.max(np.abs(tf_camera_from_lidar - json_camera_from_lidar)))
        tf_expected_max_abs_diff = float(np.max(np.abs(tf_camera_from_lidar - EXPECTED_CAMERA_FROM_LIDAR)))
        if tf_json_max_abs_diff > 1e-9:
            raise RuntimeError(f"tf2 transform does not match JSON transform: {tf_json_max_abs_diff}")
        if tf_expected_max_abs_diff > 1e-6:
            raise RuntimeError(f"tf2 transform does not match expected matrix: {tf_expected_max_abs_diff}")

        sync = sync_summary(image_stamps, lidar_stamps, sync_slop_sec)
        expected_sync = recording["expected_synchronised_frame_count"]
        if expected_sync and sync["compatible_lidar_message_count"] < expected_sync:
            raise RuntimeError(
                "Approximate synchronization coverage is too low: "
                f"{sync['compatible_lidar_message_count']} < {expected_sync}"
            )

        all_stamps = image_stamps + lidar_stamps
        row.update({
            "verification_status": "passed",
            "failure_reason": None,
            "image_count": image_count,
            "camera_info_count": camera_info_count,
            "camera_info_counts_by_camera": camera_info_counts_by_camera,
            "image_counts_by_camera": image_counts_by_camera,
            "camera_info_frames_by_camera": {
                camera: sorted(frames)
                for camera, frames in camera_info_frames_by_camera.items()
            },
            "point_cloud_count": lidar_count,
            "tf_count": tf_count,
            "tf_static_count": tf_static_count,
            "finite_cloud_checks": finite_cloud_checks,
            "first_timestamp": min(all_stamps),
            "last_timestamp": max(all_stamps),
            "duration_sec": max(all_stamps) - min(all_stamps),
            "image_frame_ids": sorted(image_frame_ids),
            "camera_info_frame_ids": sorted(camera_info_frame_ids),
            "lidar_frame_ids": sorted(lidar_frame_ids),
            "calibration_camera_frame": CALIBRATION_CAMERA_FRAME,
            "calibration_lidar_frame": CALIBRATION_LIDAR_FRAME,
            "tf_static_frame_count": len(tf_static_frames),
            "tf_frames": sorted(set(tf_frames)),
            "tf_path": tf_path,
            "tf2_camera_from_lidar_matrix": tf_camera_from_lidar.tolist(),
            "tf_json_max_abs_diff": tf_json_max_abs_diff,
            "tf_expected_max_abs_diff": tf_expected_max_abs_diff,
            "camera_info": {
                "width": int(last_camera_info.width) if last_camera_info else None,
                "height": int(last_camera_info.height) if last_camera_info else None,
                "distortion_model": last_camera_info.distortion_model if last_camera_info else None,
                "d": list(last_camera_info.d) if last_camera_info else None,
                "k": list(last_camera_info.k) if last_camera_info else None,
                "r": list(last_camera_info.r) if last_camera_info else None,
                "p": list(last_camera_info.p) if last_camera_info else None,
            },
            "expected_synchronised_frame_count": expected_sync,
            "compatible_lidar_message_count": sync["compatible_lidar_message_count"],
            "unmatched_lidar_message_count": sync["unmatched_lidar_message_count"],
            "mean_nearest_image_delta_sec": sync["mean_nearest_image_delta_sec"],
            "max_compatible_image_delta_sec": sync["max_compatible_image_delta_sec"],
            "max_nearest_image_delta_sec": sync["max_nearest_image_delta_sec"],
            "sync_slop_sec": sync_slop_sec,
            "topics": topic_types,
        })
    except Exception as exc:  # noqa: BLE001 - keep per-bag failure explicit.
        row["failure_reason"] = str(exc)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "recording",
        "bag_path",
        "image_count",
        "camera_info_count",
        "tf_count",
        "point_cloud_count",
        "tf_static_count",
        "duration_sec",
        "first_timestamp",
        "last_timestamp",
        "image_frame_ids",
        "camera_info_frame_ids",
        "lidar_frame_ids",
        "tf_json_max_abs_diff",
        "tf_expected_max_abs_diff",
        "expected_synchronised_frame_count",
        "compatible_lidar_message_count",
        "unmatched_lidar_message_count",
        "mean_nearest_image_delta_sec",
        "max_compatible_image_delta_sec",
        "max_nearest_image_delta_sec",
        "sync_slop_sec",
        "source_manifest_sha256",
        "verification_status",
        "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bag-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--extrinsics-member", type=str, default=DEFAULT_EXTRINSICS_MEMBER)
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--sync-slop-sec", type=float, default=SYNC_SLOP_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recordings = load_manifest_recordings(args.test_manifest, args.dataset_root)
    names = sorted(recordings) if args.all_recordings else args.recording
    if not names:
        raise SystemExit("Provide --recording NAME or --all-recordings")

    missing = [name for name in names if name not in recordings]
    if missing:
        raise SystemExit(f"Requested recordings are not in the official test manifest: {missing}")

    source_manifest_hash = sha256_file(args.test_manifest)
    rows = [
        verify_bag(
            args.bag_root / name,
            recordings[name],
            source_manifest_hash,
            args.sync_slop_sec,
            args.calibration_path,
            args.extrinsics_member,
        )
        for name in names
    ]
    write_csv(args.bag_root / "bag_manifest.csv", rows)
    write_json(args.bag_root / "bag_manifest.json", rows)
    write_json(args.bag_root / "verification_summary.json", {
        "bag_producer": "agri-human-dataset-tools/ros2bag/check_and_make_rosbag2.py",
        "source_manifest": str(args.test_manifest),
        "source_manifest_sha256": source_manifest_hash,
        "recording_count": len(rows),
        "passed_count": sum(1 for row in rows if row["verification_status"] == "passed"),
        "failed_count": sum(1 for row in rows if row["verification_status"] != "passed"),
        "image_topic": IMAGE_TOPIC,
        "camera_info_topics": CAMERA_INFO_TOPICS,
        "lidar_topic": LIDAR_TOPIC,
        "tf_topic": TF_TOPIC,
        "tf_static_topic": TF_STATIC_TOPIC,
        "dataset_tools_image_frame": DATASET_TOOLS_IMAGE_FRAME,
        "dataset_tools_lidar_frame": DATASET_TOOLS_LIDAR_FRAME,
        "calibration_camera_frame": CALIBRATION_CAMERA_FRAME,
        "calibration_lidar_frame": CALIBRATION_LIDAR_FRAME,
        "sync_slop_sec": args.sync_slop_sec,
        "rows": rows,
    })
    failed = [row for row in rows if row["verification_status"] != "passed"]
    print(json.dumps({"rows": rows}, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} bag verification(s) failed")


if __name__ == "__main__":
    main()
