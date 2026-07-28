#!/usr/bin/env python3
"""Add explicit empty YOLO cache rows for AGHRI ROS bag image timestamps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_TOPIC = "/dataset/cam_zed_rgb/image"


def stamp_key(value: float) -> str:
    return f"{value:.9f}"


def read_bag_image_stamps(bag_path: Path) -> list[float]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    stamps = []
    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        if topic != IMAGE_TOPIC:
            continue
        msg = deserialize_message(data, Image)
        stamps.append(float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9)
    return stamps


def update_cache(cache_csv: Path, metadata_json: Path, recording_name: str, stamps: list[float]) -> int:
    with cache_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    if not fields:
        raise ValueError(f"Cannot read CSV header: {cache_csv}")
    existing = {
        stamp_key(float(row["timestamp"]))
        for row in rows
        if row.get("sample_id", "").split("/", 1)[0] == recording_name and row.get("timestamp")
    }
    checkpoint_sha = ""
    for row in rows:
        if row.get("checkpoint_sha256"):
            checkpoint_sha = row["checkpoint_sha256"]
        if checkpoint_sha:
            break
    added = []
    for stamp in stamps:
        key = stamp_key(stamp)
        if key in existing:
            continue
        stamp_token = key.replace(".", "_")
        added.append(
            {
                "sample_id": f"{recording_name}/ros_image_{stamp_token}",
                "timestamp": key,
                "image_path": "",
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "confidence": "",
                "class_id": "",
                "class_name": "__EMPTY__",
                "checkpoint_sha256": checkpoint_sha,
                "inference_time_sec": "0.0",
                "empty_frame": "True",
            }
        )
    if not added:
        return 0
    with cache_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(added)
    if metadata_json.exists():
        metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
        by_recording = metadata.setdefault("ros_empty_frame_rows_added", {})
        by_recording[recording_name] = int(by_recording.get(recording_name, 0)) + len(added)
        metadata["empty_frame_policy"] = "Explicit empty rows added for camera image timestamps present in ROS bags but absent from paired offline fusion detections."
        metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return len(added)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording-name", required=True)
    parser.add_argument("--bag-path", type=Path, required=True)
    parser.add_argument("--cache-csv", type=Path, action="append", required=True)
    args = parser.parse_args()

    stamps = read_bag_image_stamps(args.bag_path)
    result = {}
    for cache_csv in args.cache_csv:
        added = update_cache(cache_csv, cache_csv.with_name("metadata.json"), args.recording_name, stamps)
        result[str(cache_csv)] = {"bag_image_messages": len(stamps), "empty_rows_added": added}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
