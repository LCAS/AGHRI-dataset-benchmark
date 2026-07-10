#!/usr/bin/env python3
"""Run the canonical AGHRI dataset-tools ROS 2 bag converter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_DATASET_ROOT = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
DEFAULT_MANIFEST = Path(
    "/home/prabuddhi/Desktop/Early_Fus/results/aghri_generic_vs_finetuned/manifest/test_manifest.csv"
)
DEFAULT_DATASET_TOOLS_ROOT = Path("/home/prabuddhi/Desktop/agri-human-dataset-tools")
DEFAULT_OUTPUT_ROOT = Path("/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test")
DEFAULT_PYTHON = Path("/usr/bin/python3")
DEFAULT_CALIBRATION_PATH = Path("/media/prabuddhi/Backup2/Updated Dataset_PW/calibration.zip")
DEFAULT_INTRINSICS_MEMBER = "calibration/intrinsics.json"
DEFAULT_EXTRINSICS_MEMBER = "calibration/extrinsics.json"

IMAGE_TOPIC = "/dataset/cam_zed_rgb/image"
CAMERA_INFO_TOPIC = "/dataset/cam_zed_rgb/camera_info"
FISH_FRONT_CAMERA_INFO_TOPIC = "/dataset/cam_fish_front/camera_info"
FISH_LEFT_CAMERA_INFO_TOPIC = "/dataset/cam_fish_left/camera_info"
FISH_RIGHT_CAMERA_INFO_TOPIC = "/dataset/cam_fish_right/camera_info"
CAMERA_INFO_TOPICS = {
    "cam_zed_rgb": CAMERA_INFO_TOPIC,
    "cam_fish_front": FISH_FRONT_CAMERA_INFO_TOPIC,
    "cam_fish_left": FISH_LEFT_CAMERA_INFO_TOPIC,
    "cam_fish_right": FISH_RIGHT_CAMERA_INFO_TOPIC,
}
LIDAR_TOPIC = "/dataset/lidar/points"
CALIBRATION_CAMERA_FRAME = "front_left_camera_optical_frame"
CALIBRATION_LIDAR_FRAME = "front_lidar_link"
DATASET_TOOLS_IMAGE_FRAME = CALIBRATION_CAMERA_FRAME
DATASET_TOOLS_LIDAR_FRAME = CALIBRATION_LIDAR_FRAME
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TF_PARENT_FRAME = "map"
TF_CHILD_FRAME = "base_link"
SYNC_SLOP_SEC = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_manifest_recordings(manifest: Path, dataset_root: Path) -> dict[str, dict[str, Any]]:
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    recordings: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split") != "test" or not truthy(row.get("supported")):
            continue
        if truthy(row.get("train_val_leakage")):
            raise RuntimeError(f"Train/validation leakage in manifest: {row['recording_name']}")

        recording_path = Path(row["recording_path"])
        try:
            recording_path.relative_to(dataset_root)
        except ValueError as exc:
            raise RuntimeError(f"Recording outside dataset root: {recording_path}") from exc

        name = row["recording_name"]
        if name in recordings:
            raise RuntimeError(f"Duplicate recording in manifest: {name}")
        recordings[name] = {
            "recording": name,
            "recording_path": recording_path,
            "rgb_image_count": int(row.get("rgb_image_count") or 0),
            "lidar_point_cloud_count": int(row.get("lidar_point_cloud_count") or 0),
            "expected_synchronised_frame_count": int(row.get("expected_synchronised_frame_count") or 0),
        }

    if len(recordings) != 6:
        raise RuntimeError(f"Expected six official test recordings, found {len(recordings)}")
    return recordings


def dataset_tools_script(tools_root: Path) -> Path:
    script = tools_root / "ros2bag/check_and_make_rosbag2.py"
    if not script.exists():
        raise FileNotFoundError(f"Dataset-tools converter not found: {script}")
    return script


def bag_dir(output_root: Path, recording: str) -> Path:
    return output_root / recording


def converter_command(
    python: Path,
    tools_root: Path,
    recording_path: Path,
    output_path: Path,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    intrinsics_member: str = DEFAULT_INTRINSICS_MEMBER,
    extrinsics_member: str = DEFAULT_EXTRINSICS_MEMBER,
    make_rosbag: bool = True,
) -> list[str]:
    command = [
        str(python),
        str(dataset_tools_script(tools_root)),
        "--bag-dir",
        str(recording_path),
    ]
    if make_rosbag:
        command.extend([
            "--make-rosbag",
            "--rosbag-out",
            str(output_path),
            "--write-tf",
            "--tf-parent",
            TF_PARENT_FRAME,
            "--tf-child",
            TF_CHILD_FRAME,
            "--tf-xyzrpy",
            "0,0,0,0,0,0",
            "--write-calibration",
            "--calibration-path",
            str(calibration_path),
            "--intrinsics-member",
            intrinsics_member,
            "--extrinsics-member",
            extrinsics_member,
            "--camera-name",
            "cam_zed_rgb",
            "--camera-info-topic",
            CAMERA_INFO_TOPIC,
            "--camera-info-cameras",
            ",".join(CAMERA_INFO_TOPICS),
        ])
    return command


def write_summary(
    output_root: Path,
    manifest: Path,
    selected: list[str],
    rows: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "converter": "agri-human-dataset-tools/ros2bag/check_and_make_rosbag2.py",
        "calibration_mode": "dataset_tools_camera_info_tf_static",
        "camera_info_topics": CAMERA_INFO_TOPICS,
        "image_topic": IMAGE_TOPIC,
        "lidar_topic": LIDAR_TOPIC,
        "tf_topic": TF_TOPIC,
        "tf_static_topic": TF_STATIC_TOPIC,
        "tf_parent_frame": TF_PARENT_FRAME,
        "tf_child_frame": TF_CHILD_FRAME,
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "selected_recordings": selected,
        "rows": rows,
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-tools-root", type=Path, default=DEFAULT_DATASET_TOOLS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--intrinsics-member", type=str, default=DEFAULT_INTRINSICS_MEMBER)
    parser.add_argument("--extrinsics-member", type=str, default=DEFAULT_EXTRINSICS_MEMBER)
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--all-recordings", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run dataset-tools checks without writing rosbag2 output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recordings = load_manifest_recordings(args.test_manifest, args.dataset_root)
    selected_names = sorted(recordings) if args.all_recordings else args.recording
    if not selected_names:
        raise SystemExit("Provide --recording NAME or --all-recordings")

    missing = [name for name in selected_names if name not in recordings]
    if missing:
        raise SystemExit(f"Requested recordings are not in the official test manifest: {missing}")

    rows: list[dict[str, Any]] = []
    for name in selected_names:
        recording = recordings[name]
        out_dir = bag_dir(args.output_root, name)
        make_rosbag = not args.check_only
        command = converter_command(
            args.python,
            args.dataset_tools_root,
            recording["recording_path"],
            out_dir,
            calibration_path=args.calibration_path,
            intrinsics_member=args.intrinsics_member,
            extrinsics_member=args.extrinsics_member,
            make_rosbag=make_rosbag,
        )
        row = {
            **recording,
            "bag_path": str(out_dir),
            "command": command,
            "status": "dry_run" if args.dry_run else "pending",
        }

        if args.dry_run:
            rows.append(row)
            continue

        if make_rosbag and out_dir.exists():
            if not args.overwrite:
                raise SystemExit(f"Bag already exists; pass --overwrite to replace it: {out_dir}")
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(command, check=True)
        row["status"] = "checked" if args.check_only else "converted"
        rows.append(row)

    write_summary(args.output_root, args.test_manifest, selected_names, rows)
    print(json.dumps({
        "output_root": str(args.output_root),
        "selected_recordings": selected_names,
        "rows": rows,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
