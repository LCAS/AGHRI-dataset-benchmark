#!/usr/bin/env python3
"""Build split-safe AGHRI ReID manifests for optional AGHRI fine-tuning.

The ReID split is identity-based:
  train: person_07, person_08, person_10
  val: person_03, person_04
  reserved_test: person_01, person_02, person_05, person_06, person_09

Original dataset split provenance is retained, but original test scenes are
never used for crop export.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2


CAMERAS = ("cam_fish_front", "cam_fish_left", "cam_fish_right", "cam_zed_rgb")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
RESERVED = {"person_01", "person_02", "person_05", "person_06", "person_09"}
TRAIN = {"person_07", "person_08", "person_10"}
VAL = {"person_03", "person_04"}


FIELDS = [
    "sample_id",
    "split",
    "original_split",
    "dataset_part",
    "scene_name",
    "camera",
    "annotation_file",
    "annotation_record_index",
    "image_path",
    "video_frame",
    "local_class",
    "global_person_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "image_width",
    "image_height",
    "bbox_area",
    "exclusion_reason",
]


def read_split(dataset_root: Path, split: str) -> list[str]:
    path = dataset_root / "split_lists" / f"{split}.txt"
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def build_scene_index(dataset_root: Path) -> dict[str, tuple[str, Path]]:
    index = {}
    for part_dir in sorted(dataset_root.glob("dataset_part*")):
        if not part_dir.is_dir():
            continue
        for scene_dir in sorted(part_dir.iterdir()):
            if scene_dir.is_dir():
                index[scene_dir.name] = (part_dir.name, scene_dir)
    return index


def sorted_sensor_images(camera_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name,
    )


def load_video_manifest(video_root: Path, original_split: str, dataset_part: str, scene: str, camera: str) -> dict[str, int] | None:
    path = video_root / original_split / dataset_part / scene / f"{camera}_frame_manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "file_to_video_frame" in data and isinstance(data["file_to_video_frame"], dict):
        return {str(k): int(v) for k, v in data["file_to_video_frame"].items()}
    frames = data.get("frames", [])
    return {str(frame["file"]): int(frame["video_frame_index"]) for frame in frames}


def local_frame_manifest(camera_dir: Path) -> dict[str, int]:
    return {path.name: idx for idx, path in enumerate(sorted_sensor_images(camera_dir))}


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        height, width = img.shape[:2]
        return width, height
    except Exception:
        return None


def read_identity_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
        key = (row["scene_name"], f"{int(row['local_class']):02d}")
        rows[key] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_box(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/media/prabuddhi/Backup2/Updated Dataset_PW"))
    parser.add_argument("--video-root", type=Path, default=Path("/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos"))
    parser.add_argument("--stage-root", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive"))
    parser.add_argument("--identity-map", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive/identity_audit/aghri_identity_map_confirmed.csv"))
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-box-width", type=float, default=5.0)
    parser.add_argument("--min-box-height", type=float, default=10.0)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    identity_map = read_identity_map(args.identity_map)
    scene_index = build_scene_index(args.dataset_root)
    split_scenes = {split: read_split(args.dataset_root, split) for split in ("train", "val", "test")}

    accepted_pre_stride: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    sample_counter = 0
    boxes_by_identity_all = Counter()
    crops_by_identity_camera_valid = Counter()
    manifest_source_counts = Counter()

    for original_split, scenes in split_scenes.items():
        for scene in scenes:
            if scene not in scene_index:
                continue
            dataset_part, scene_dir = scene_index[scene]
            for camera in CAMERAS:
                ann_file = scene_dir / "annotations" / f"{camera}_ann.json"
                camera_dir = scene_dir / "sensor_data" / camera
                if not ann_file.exists() or not camera_dir.exists():
                    continue
                annotations = json.loads(ann_file.read_text(encoding="utf-8"))
                file_to_frame = load_video_manifest(args.video_root, original_split, dataset_part, scene, camera)
                if file_to_frame is None:
                    file_to_frame = local_frame_manifest(camera_dir)
                    manifest_source_counts["sensor_image_order"] += 1
                else:
                    manifest_source_counts["video_frame_manifest"] += 1

                for record_index, record in enumerate(annotations):
                    file_name = str(record.get("File", ""))
                    image_path = camera_dir / file_name
                    video_frame = file_to_frame.get(file_name, -1)
                    for label in record.get("Labels", []) or []:
                        local_class = str(label.get("Class", "")).strip()
                        if not local_class:
                            continue
                        local_class = f"{int(local_class):02d}"
                        mapping = identity_map.get((scene, local_class))
                        global_id = mapping["manual_name_or_id"] if mapping else f"person_{int(local_class):02d}"
                        reid_split = mapping["reid_split"] if mapping else "unknown"
                        approved = str(mapping["approved"]) == "1" if mapping else False
                        boxes_by_identity_all[global_id] += 1
                        base = {
                            "sample_id": "",
                            "split": reid_split,
                            "original_split": original_split,
                            "dataset_part": dataset_part,
                            "scene_name": scene,
                            "camera": camera,
                            "annotation_file": str(ann_file),
                            "annotation_record_index": record_index,
                            "image_path": str(image_path),
                            "video_frame": video_frame,
                            "local_class": local_class,
                            "global_person_id": global_id,
                            "bbox_x": "",
                            "bbox_y": "",
                            "bbox_w": "",
                            "bbox_h": "",
                            "image_width": "",
                            "image_height": "",
                            "bbox_area": "",
                            "exclusion_reason": "",
                        }
                        if original_split == "test":
                            base["exclusion_reason"] = "original_test_scene_excluded"
                            excluded.append(base)
                            continue
                        if reid_split == "reserved_test" or global_id in RESERVED:
                            base["exclusion_reason"] = "reserved_test_identity_excluded"
                            excluded.append(base)
                            continue
                        if not approved or reid_split not in {"train", "val"}:
                            base["exclusion_reason"] = "unapproved_identity"
                            excluded.append(base)
                            continue
                        if not image_path.exists():
                            base["exclusion_reason"] = "missing_image_file"
                            excluded.append(base)
                            continue
                        size = image_size(image_path)
                        if size is None:
                            base["exclusion_reason"] = "unreadable_image_file"
                            excluded.append(base)
                            continue
                        image_width, image_height = size
                        bbox = label.get("BoundingBoxes")
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            base["exclusion_reason"] = "invalid_bbox_format"
                            excluded.append(base)
                            continue
                        x, y, w, h = [float(v) for v in bbox]
                        x1 = max(0.0, min(float(image_width), x))
                        y1 = max(0.0, min(float(image_height), y))
                        x2 = max(0.0, min(float(image_width), x + w))
                        y2 = max(0.0, min(float(image_height), y + h))
                        clipped_w = x2 - x1
                        clipped_h = y2 - y1
                        area = clipped_w * clipped_h
                        base.update(
                            {
                                "bbox_x": format_box(x1),
                                "bbox_y": format_box(y1),
                                "bbox_w": format_box(clipped_w),
                                "bbox_h": format_box(clipped_h),
                                "image_width": image_width,
                                "image_height": image_height,
                                "bbox_area": format_box(area),
                            }
                        )
                        if clipped_w < args.min_box_width or clipped_h < args.min_box_height or area < args.min_box_area:
                            base["exclusion_reason"] = "invalid_or_extremely_small_box"
                            excluded.append(base)
                            continue
                        sample_counter += 1
                        base["sample_id"] = f"aghri_{sample_counter:08d}"
                        accepted_pre_stride.append(base)
                        crops_by_identity_camera_valid[(global_id, camera)] += 1

    sampled: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted_pre_stride:
        groups[(row["split"], row["global_person_id"], row["scene_name"], row["camera"])].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r["video_frame"]))
        for idx, row in enumerate(rows):
            if idx % args.frame_stride == 0:
                sampled.append(row)
            else:
                skipped = row.copy()
                skipped["exclusion_reason"] = f"temporal_stride_skip_{args.frame_stride}"
                excluded.append(skipped)

    train_rows = [row for row in sampled if row["split"] == "train"]
    val_rows = [row for row in sampled if row["split"] == "val"]

    assert all(row["original_split"] != "test" for row in train_rows + val_rows)
    assert all(row["global_person_id"] not in RESERVED for row in train_rows + val_rows)
    assert all(row["global_person_id"] in TRAIN for row in train_rows)
    assert all(row["global_person_id"] in VAL for row in val_rows)
    assert all(Path(row["image_path"]).exists() for row in train_rows + val_rows)
    assert all(float(row["bbox_w"]) > 0 and float(row["bbox_h"]) > 0 for row in train_rows + val_rows)

    manifest_dir = args.stage_root / "manifests"
    write_csv(manifest_dir / "all_samples_manifest.csv", accepted_pre_stride)
    write_csv(manifest_dir / "train_manifest.csv", train_rows)
    write_csv(manifest_dir / "val_manifest.csv", val_rows)
    write_csv(manifest_dir / "excluded_samples.csv", excluded)

    by_reid_split = Counter(row["split"] for row in accepted_pre_stride)
    by_reid_split_sampled = Counter(row["split"] for row in sampled)
    crops_per_identity_sampled = Counter(row["global_person_id"] for row in sampled)
    crops_per_identity_camera_sampled = Counter((row["global_person_id"], row["camera"]) for row in sampled)
    boxes = [float(row["bbox_area"]) for row in accepted_pre_stride]
    boxes_sorted = sorted(boxes)
    median_area = boxes_sorted[len(boxes_sorted) // 2] if boxes_sorted else 0.0

    min_required = 8
    few_samples = {
        identity: crops_per_identity_sampled.get(identity, 0)
        for identity in sorted(TRAIN | VAL)
        if crops_per_identity_sampled.get(identity, 0) < min_required
    }
    pk_ok = all(crops_per_identity_sampled.get(identity, 0) >= 4 for identity in TRAIN)
    retrieval_ok = all(crops_per_identity_sampled.get(identity, 0) >= 2 for identity in VAL)

    report = f"""# STAGE 1 PRE-EXPORT CHECKS

Frame stride: {args.frame_stride}

## Scene/Class Rows By ReID Split

These rows come from `aghri_identity_map_confirmed.csv`.

| ReID split | rows |
|---|---:|
| train | {sum(1 for row in identity_map.values() if row['reid_split'] == 'train')} |
| val | {sum(1 for row in identity_map.values() if row['reid_split'] == 'val')} |
| reserved_test | {sum(1 for row in identity_map.values() if row['reid_split'] == 'reserved_test')} |

## Annotated Boxes By Identity

{dict(sorted(boxes_by_identity_all.items()))}

## Valid Boxes Before Temporal Sampling

{dict(sorted(by_reid_split.items()))}

## Crops After Temporal Sampling

{dict(sorted(by_reid_split_sampled.items()))}

Per identity after temporal sampling:

{dict(sorted(crops_per_identity_sampled.items()))}

Per identity/camera after temporal sampling:

{dict((f"{identity}:{camera}", count) for (identity, camera), count in sorted(crops_per_identity_camera_sampled.items()))}

## Box Area

- min: {min(boxes) if boxes else 0:.1f}
- median: {median_area:.1f}
- max: {max(boxes) if boxes else 0:.1f}

## Manifest Sources

{dict(sorted(manifest_source_counts.items()))}

Training PK sampling status: {"PASS" if pk_ok else "FAIL"}.

Validation retrieval image status: {"PASS" if retrieval_ok else "FAIL"}.

Identities below {min_required} sampled crops:

{few_samples}

## Leakage Assertions

- No original final-test scene rows in train/val manifests: PASS
- No reserved-test identity in train/val manifests: PASS
- No unapproved identity in train/val manifests: PASS
- No missing image files in train/val manifests: PASS
- No invalid bounding boxes in train/val manifests: PASS
- No final-test scene images accessed for crop export: PASS by construction; original `test.txt` scenes are excluded before image-size reads/crop export.

## Decision

Phase B manifest generation passed leakage assertions. Crop export may proceed
for `train_manifest.csv` and `val_manifest.csv`.
"""
    (args.stage_root / "reports" / "PRE_EXPORT_CHECKS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
