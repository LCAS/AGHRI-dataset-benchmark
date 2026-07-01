#!/usr/bin/env python3
"""Shared AGHRI frame-manifest and annotation-alignment helpers."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def sorted_sensor_images(camera_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def timestamp_from_filename(filename: str) -> float | None:
    stem = Path(filename).stem
    if "_" not in stem:
        return None
    seconds, fractional = stem.split("_", 1)
    if not seconds.isdigit() or not fractional.isdigit():
        return None
    return float(f"{seconds}.{fractional}")


def load_annotations(annotation_path: Path) -> list[dict[str, Any]]:
    annotations = read_json(annotation_path)
    if not isinstance(annotations, list):
        raise ValueError(f"Annotation JSON is not a list: {annotation_path}")
    return annotations


def build_frame_manifest(
    *,
    split: str,
    dataset_part: str,
    scene: str,
    camera: str,
    fps: float,
    camera_dir: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    images = sorted_sensor_images(camera_dir)
    image_names = [path.name for path in images]

    annotations = load_annotations(annotation_path) if annotation_path.exists() else []
    file_to_annotation_records: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(annotations):
        filename = record.get("File")
        if isinstance(filename, str):
            file_to_annotation_records[filename].append(index)

    duplicate_annotation_files = sorted(
        filename
        for filename, indices in file_to_annotation_records.items()
        if len(indices) > 1
    )
    if duplicate_annotation_files:
        raise ValueError(
            "Duplicate annotation File values cannot map uniquely: "
            + ", ".join(duplicate_annotation_files[:10])
        )

    file_to_video_frame = {name: index for index, name in enumerate(image_names)}
    missing_annotation_files = sorted(
        filename
        for filename in file_to_annotation_records
        if filename not in file_to_video_frame
    )
    if missing_annotation_files:
        raise ValueError(
            "Annotation File values missing from sensor images: "
            + ", ".join(missing_annotation_files[:10])
        )

    annotation_record_to_video_frame = {
        str(record_index): file_to_video_frame[filename]
        for filename, indices in file_to_annotation_records.items()
        for record_index in indices
    }

    mapped_indices = [
        annotation_record_to_video_frame[str(index)]
        for index in range(len(annotations))
        if str(index) in annotation_record_to_video_frame
    ]
    if mapped_indices != sorted(mapped_indices) or len(mapped_indices) != len(set(mapped_indices)):
        raise ValueError("Mapped annotation video-frame indices are not strictly increasing.")

    frames = []
    for video_frame_index, filename in enumerate(image_names):
        annotation_indices = file_to_annotation_records.get(filename, [])
        frames.append(
            {
                "video_frame_index": video_frame_index,
                "file": filename,
                "timestamp_from_filename": timestamp_from_filename(filename),
                "has_annotation": bool(annotation_indices),
                "annotation_record_index": annotation_indices[0] if annotation_indices else None,
            }
        )

    return {
        "split": split,
        "dataset_part": dataset_part,
        "scene": scene,
        "camera": camera,
        "fps": float(fps),
        "total_sensor_images": len(image_names),
        "total_annotation_records": len(annotations),
        "video_frame_count": len(image_names),
        "frames": frames,
        "file_to_video_frame": file_to_video_frame,
        "annotation_record_to_video_frame": annotation_record_to_video_frame,
    }


def load_frame_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Frame manifest is not a JSON object: {path}")

    frames = manifest.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Frame manifest has no frames list: {path}")

    filenames = []
    file_to_video_frame: dict[str, int] = {}
    annotation_record_to_video_frame: dict[str, int] = {}

    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Manifest frame {expected_index} is not an object.")
        video_frame_index = frame.get("video_frame_index")
        filename = frame.get("file")
        if video_frame_index != expected_index:
            raise ValueError(
                f"Manifest frame index mismatch at {expected_index}: {video_frame_index}"
            )
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Manifest frame {expected_index} has invalid file.")
        if filename in file_to_video_frame:
            raise ValueError(f"Duplicate manifest filename: {filename}")
        filenames.append(filename)
        file_to_video_frame[filename] = expected_index
        record_index = frame.get("annotation_record_index")
        if record_index is not None:
            annotation_record_to_video_frame[str(record_index)] = expected_index

    if manifest.get("video_frame_count") not in (None, len(frames)):
        raise ValueError(
            f"Manifest video_frame_count={manifest.get('video_frame_count')} "
            f"but frames={len(frames)}"
        )

    manifest["file_to_video_frame"] = manifest.get("file_to_video_frame") or file_to_video_frame
    manifest["annotation_record_to_video_frame"] = (
        manifest.get("annotation_record_to_video_frame") or annotation_record_to_video_frame
    )
    return manifest


def validate_manifest_against_sources(
    *,
    manifest: dict[str, Any],
    camera_dir: Path,
    annotations: list[dict[str, Any]],
    expected_fps: float,
    video_frame_count: int | None = None,
) -> dict[str, Any]:
    image_names = [path.name for path in sorted_sensor_images(camera_dir)]
    frame_names = [frame["file"] for frame in manifest["frames"]]
    if frame_names != image_names:
        raise ValueError("Manifest filenames do not match sorted source image filenames.")

    manifest_fps = float(manifest.get("fps", 0.0))
    if not math.isclose(manifest_fps, float(expected_fps), abs_tol=0.05):
        raise ValueError(f"Manifest FPS {manifest_fps} does not match expected {expected_fps}.")

    if video_frame_count is not None and video_frame_count != len(frame_names):
        raise ValueError(
            f"Video frame count {video_frame_count} does not match manifest frames {len(frame_names)}."
        )

    file_to_video_frame = manifest["file_to_video_frame"]
    mapped_indices = []
    missing_files = []
    for record_index, record in enumerate(annotations):
        filename = record.get("File")
        if filename not in file_to_video_frame:
            missing_files.append(filename)
            continue
        mapped_indices.append(file_to_video_frame[filename])

    if missing_files:
        raise ValueError(
            "Annotation File values missing from manifest: "
            + ", ".join(str(value) for value in missing_files[:10])
        )
    if mapped_indices != sorted(mapped_indices) or len(mapped_indices) != len(set(mapped_indices)):
        raise ValueError("Annotation File mappings are not strictly increasing and unique.")

    return {
        "source_image_count": len(image_names),
        "annotation_count": len(annotations),
        "unannotated_frame_count": len(image_names) - len(annotations),
        "annotation_mapping_verified": True,
    }


def collapse_duplicate_boxes(labels: list[dict[str, Any]], target_class: str, tolerance: float = 1e-6):
    target_boxes = []
    other_boxes = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        bbox = label.get("BoundingBoxes")
        label_class = str(label.get("Class", "")).strip()
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            box = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if box[2] <= 0 or box[3] <= 0:
            continue
        if label_class == target_class:
            target_boxes.append(box)
        else:
            other_boxes.append({"class": label_class, "bbox_xywh": box})

    unique_target_boxes = []
    for box in target_boxes:
        if not any(all(abs(a - b) <= tolerance for a, b in zip(box, existing)) for existing in unique_target_boxes):
            unique_target_boxes.append(box)

    if len(unique_target_boxes) > 1:
        return "ambiguous_duplicate_target", None, other_boxes
    if len(unique_target_boxes) == 1:
        return "visible", unique_target_boxes[0], other_boxes
    return "absent", None, other_boxes
