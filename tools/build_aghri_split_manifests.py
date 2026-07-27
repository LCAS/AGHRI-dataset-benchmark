#!/usr/bin/env python3
"""Build AGHRI late-fusion manifests for train/val/test split scene lists."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from late_fusion_pkg.aghri_manifest import DEFAULT_SLOP_SEC, TimestampedFile, nearest_neighbor, timestamp_from_name


FIELDS = [
    "split",
    "recording_name",
    "sample_id",
    "lidar_frame_id",
    "lidar_timestamp",
    "lidar_point_cloud_path",
    "zed_rgb_frame_id",
    "zed_rgb_timestamp",
    "image_path",
    "camera_lidar_delta_sec",
    "pair_status",
    "gt_2d_reference",
    "gt_3d_reference",
    "calibration_source",
    "camera_frame",
    "lidar_frame",
]


def find_recording(dataset_root: Path, recording_name: str) -> Path | None:
    for part in sorted(dataset_root.glob("dataset_part*")):
        candidate = part / recording_name
        if candidate.exists():
            return candidate
    return None


def scan_modality(recording_path: Path, rel: str, frame_id: str) -> list[TimestampedFile]:
    folder = recording_path / rel
    items = []
    if not folder.exists():
        return items
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        ts = timestamp_from_name(path)
        if ts is not None:
            items.append(TimestampedFile(ts, path, frame_id))
    return items


def build_split_manifest(
    dataset_root: Path,
    split: str,
    output_csv: Path,
    max_delta_sec: float,
    calibration_source: Path,
) -> dict:
    split_path = dataset_root / "split_lists" / f"{split}.txt"
    recordings = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    unmatched = 0
    missing_recordings = 0
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for recording_name in recordings:
            recording_path = find_recording(dataset_root, recording_name)
            if recording_path is None:
                missing_recordings += 1
                unmatched += 1
                written += 1
                writer.writerow(
                    {
                        "split": split,
                        "recording_name": recording_name,
                        "sample_id": "",
                        "lidar_frame_id": "front_lidar_link",
                        "lidar_timestamp": "",
                        "lidar_point_cloud_path": "",
                        "zed_rgb_frame_id": "front_left_camera_optical_frame",
                        "zed_rgb_timestamp": "",
                        "image_path": "",
                        "camera_lidar_delta_sec": "",
                        "pair_status": "recording_path_missing",
                        "gt_2d_reference": "",
                        "gt_3d_reference": "",
                        "calibration_source": str(calibration_source),
                        "camera_frame": "front_left_camera_optical_frame",
                        "lidar_frame": "front_lidar_link",
                    }
                )
                continue
            lidar_items = scan_modality(recording_path, "sensor_data/lidar", "front_lidar_link")
            image_items = scan_modality(recording_path, "sensor_data/cam_zed_rgb", "front_left_camera_optical_frame")
            for lidar, image, delta in nearest_neighbor(lidar_items, image_items, max_delta_sec):
                status = "matched" if image is not None else "unmatched_lidar"
                if image is None:
                    unmatched += 1
                writer.writerow(
                    {
                        "split": split,
                        "recording_name": recording_name,
                        "sample_id": f"{recording_name}/{lidar.path.stem}",
                        "lidar_frame_id": lidar.frame_id,
                        "lidar_timestamp": f"{lidar.timestamp:.9f}",
                        "lidar_point_cloud_path": str(lidar.path),
                        "zed_rgb_frame_id": image.frame_id if image else "front_left_camera_optical_frame",
                        "zed_rgb_timestamp": f"{image.timestamp:.9f}" if image else "",
                        "image_path": str(image.path) if image else "",
                        "camera_lidar_delta_sec": f"{delta:.9f}" if delta is not None else "",
                        "pair_status": status,
                        "gt_2d_reference": str(recording_path / "annotations/cam_zed_rgb_ann.json"),
                        "gt_3d_reference": str(recording_path / "annotations/lidar_ann.json"),
                        "calibration_source": str(calibration_source),
                        "camera_frame": "front_left_camera_optical_frame",
                        "lidar_frame": "front_lidar_link",
                    }
                )
                written += 1
    return {
        "split": split,
        "output_csv": str(output_csv),
        "recordings": len(recordings),
        "rows": written,
        "unmatched_or_missing": unmatched,
        "missing_recordings": missing_recordings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_manifests"))
    parser.add_argument("--max-delta-sec", type=float, default=DEFAULT_SLOP_SEC)
    parser.add_argument("--calibration-source", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW/calibration.zip"))
    args = parser.parse_args()
    summaries = [
        build_split_manifest(
            args.dataset_root,
            split,
            args.output_dir / f"aghri_late_fusion_{split}_manifest.csv",
            args.max_delta_sec,
            args.calibration_source,
        )
        for split in ("train", "val", "test")
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
