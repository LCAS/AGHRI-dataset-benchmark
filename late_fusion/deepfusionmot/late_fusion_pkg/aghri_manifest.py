"""AGHRI manifest helpers for deterministic offline late-fusion pairing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


DEFAULT_SLOP_SEC = 0.10


@dataclass(frozen=True)
class TimestampedFile:
    timestamp: float
    path: Path
    frame_id: str


def timestamp_from_name(path: Path) -> float | None:
    stem = path.stem
    if "_" not in stem:
        return None
    sec, frac = stem.split("_", 1)
    try:
        return float(Decimal(f"{sec}.{frac}"))
    except InvalidOperation:
        return None


def nearest_neighbor(
    anchors: Iterable[TimestampedFile],
    candidates: Iterable[TimestampedFile],
    max_delta_sec: float = DEFAULT_SLOP_SEC,
) -> list[tuple[TimestampedFile, TimestampedFile | None, float | None]]:
    ordered_candidates = sorted(candidates, key=lambda item: item.timestamp)
    pairs = []
    for anchor in sorted(anchors, key=lambda item: item.timestamp):
        best = None
        best_delta = None
        for candidate in ordered_candidates:
            delta = abs(candidate.timestamp - anchor.timestamp)
            if best_delta is None or delta < best_delta:
                best = candidate
                best_delta = delta
            elif candidate.timestamp > anchor.timestamp and best_delta is not None and delta > best_delta:
                break
        if best_delta is None or best_delta > max_delta_sec:
            pairs.append((anchor, None, best_delta))
        else:
            pairs.append((anchor, best, best_delta))
    return pairs


def read_official_test_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _scan_modality(recording_path: Path, rel: str, frame_id: str) -> list[TimestampedFile]:
    folder = recording_path / rel
    if not folder.exists():
        return []
    items = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        ts = timestamp_from_name(path)
        if ts is None:
            continue
        items.append(TimestampedFile(timestamp=ts, path=path, frame_id=frame_id))
    return items


def build_late_fusion_manifest(
    official_manifest: Path,
    output_csv: Path,
    max_delta_sec: float = DEFAULT_SLOP_SEC,
    calibration_source: str = "",
) -> dict[str, int | str]:
    rows = read_official_test_manifest(official_manifest)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    written = 0
    unmatched = 0
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for manifest_row in rows:
            recording_path = Path(manifest_row["recording_path"])
            recording_name = manifest_row["recording_name"]
            split = manifest_row.get("split", "test")
            lidar_items = _scan_modality(recording_path, "sensor_data/lidar", "front_lidar_link")
            image_items = _scan_modality(recording_path, "sensor_data/cam_zed_rgb", "front_left_camera_optical_frame")
            if not recording_path.exists():
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
                        "calibration_source": calibration_source,
                        "camera_frame": "front_left_camera_optical_frame",
                        "lidar_frame": "front_lidar_link",
                    }
                )
                written += 1
                unmatched += 1
                continue
            for lidar, image, delta in nearest_neighbor(lidar_items, image_items, max_delta_sec):
                status = "matched" if image is not None else "unmatched_lidar"
                if image is None:
                    unmatched += 1
                sample_id = f"{recording_name}/{lidar.path.stem}"
                writer.writerow(
                    {
                        "split": split,
                        "recording_name": recording_name,
                        "sample_id": sample_id,
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
                        "calibration_source": calibration_source,
                        "camera_frame": "front_left_camera_optical_frame",
                        "lidar_frame": "front_lidar_link",
                    }
                )
                written += 1
    return {"output_csv": str(output_csv), "rows": written, "unmatched_or_missing": unmatched}

