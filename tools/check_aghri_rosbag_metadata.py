#!/usr/bin/env python3
"""Validate converted AGHRI test rosbag metadata without playing bags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


REQUIRED_TOPICS = {
    "/dataset/cam_zed_rgb/image": "sensor_msgs/msg/Image",
    "/dataset/cam_zed_rgb/camera_info": "sensor_msgs/msg/CameraInfo",
    "/dataset/lidar/points": "sensor_msgs/msg/PointCloud2",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
}


def read_manifest(manifest: Path) -> list[dict]:
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def inspect_bag(bag_path: Path) -> dict:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        return {"bag_path": str(bag_path), "ok": False, "missing_metadata": True}
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = data["rosbag2_bagfile_information"]
    topics = {}
    for item in info.get("topics_with_message_count", []):
        meta = item["topic_metadata"]
        topics[meta["name"]] = {"type": meta["type"], "message_count": int(item["message_count"])}
    missing = []
    wrong_type = []
    empty = []
    for topic, expected_type in REQUIRED_TOPICS.items():
        if topic not in topics:
            missing.append(topic)
            continue
        if topics[topic]["type"] != expected_type:
            wrong_type.append({"topic": topic, "expected": expected_type, "actual": topics[topic]["type"]})
        if topics[topic]["message_count"] <= 0:
            empty.append(topic)
    db_files = [bag_path / item for item in info.get("relative_file_paths", [])]
    missing_db = [str(path) for path in db_files if not path.exists()]
    return {
        "bag_path": str(bag_path),
        "ok": not (missing or wrong_type or empty or missing_db),
        "duration_sec": info.get("duration", {}).get("nanoseconds", 0) / 1e9,
        "message_count": int(info.get("message_count", 0)),
        "required_topic_counts": {topic: topics.get(topic, {}).get("message_count", 0) for topic in REQUIRED_TOPICS},
        "missing_topics": missing,
        "wrong_type_topics": wrong_type,
        "empty_topics": empty,
        "missing_db_files": missing_db,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-manifest", type=Path, default=Path("/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test/bag_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/aghri_generic_baseline/rosbag_metadata_smoke.json"))
    args = parser.parse_args()
    rows = read_manifest(args.bag_manifest)
    reports = [inspect_bag(Path(row["bag_path"])) for row in rows]
    summary = {
        "bag_manifest": str(args.bag_manifest),
        "bags": len(reports),
        "passed": sum(1 for item in reports if item["ok"]),
        "failed": sum(1 for item in reports if not item["ok"]),
        "required_topics": REQUIRED_TOPICS,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("bags", "passed", "failed")}, indent=2))


if __name__ == "__main__":
    main()
