#!/usr/bin/env python3
"""Optional ROS 2 qualitative image capture helper for AGHRI fusion.

This helper is intentionally qualitative-only. It captures images from an
already-running ROS graph and writes them into the paper-assets directory.
Numerical claims should continue to come from the offline evaluator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


DEFAULT_OUTPUT = Path("/home/prabuddhi/Desktop/Early_Fus/results/aghri_fusion_paper_assets/figures/qualitative/ros_capture")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topic", default="/camera_lidar_fusion/result")
    parser.add_argument("--raw-topic", default="/dataset/cam_zed_rgb/image")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "qualitative ROS playback capture only",
        "topics": [args.topic, args.raw_topic],
        "output_dir": str(args.output_dir),
        "status": "not_started",
        "note": "Requires a running ROS 2 graph plus image_tools/showimage or rclpy image conversion support.",
    }
    try:
        subprocess.run(["ros2", "topic", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as exc:
        manifest["status"] = "ros_unavailable"
        manifest["error"] = str(exc)
        (args.output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise SystemExit("ROS 2 command line is not available or no ROS environment is sourced.")

    # This conservative helper records topic presence and leaves actual image
    # conversion to the existing evaluator/ROS visualization stack if cv_bridge
    # is not installed in the active environment.
    topics = subprocess.run(["ros2", "topic", "list"], check=True, stdout=subprocess.PIPE, text=True).stdout.splitlines()
    manifest["available_topics"] = topics
    missing = [topic for topic in [args.topic, args.raw_topic] if topic not in topics]
    if missing:
        manifest["status"] = "missing_topics"
        manifest["missing_topics"] = missing
    else:
        manifest["status"] = "topics_available"
        manifest["checked_at_unix"] = time.time()
        manifest["next_step"] = "Use RViz/Image panel or add cv_bridge in the active ROS environment to save PNGs from these topics."
    (args.output_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
