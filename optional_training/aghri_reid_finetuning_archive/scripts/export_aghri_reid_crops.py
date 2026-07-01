#!/usr/bin/env python3
"""Export AGHRI ReID crops from validated optional fine-tuning manifests."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CSV_FIELDS = [
    "sample_id",
    "split",
    "original_split",
    "dataset_part",
    "scene_name",
    "camera",
    "annotation_file",
    "annotation_record_index",
    "source_image_path",
    "crop_path",
    "video_frame",
    "local_class",
    "global_person_id",
    "integer_identity_label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "crop_width",
    "crop_height",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def identity_labels(rows: list[dict[str, str]]) -> dict[str, int]:
    return {identity: idx for idx, identity in enumerate(sorted({row["global_person_id"] for row in rows}))}


def crop_one(row: dict[str, str], out_root: Path, label_map: dict[str, int]) -> dict[str, Any]:
    image_path = Path(row["image_path"])
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    x = int(round(float(row["bbox_x"])))
    y = int(round(float(row["bbox_y"])))
    bw = int(round(float(row["bbox_w"])))
    bh = int(round(float(row["bbox_h"])))
    x1 = max(0, min(w, x))
    y1 = max(0, min(h, y))
    x2 = max(0, min(w, x1 + max(1, bw)))
    y2 = max(0, min(h, y1 + max(1, bh)))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop for sample {row['sample_id']}")

    identity = row["global_person_id"]
    crop_dir = out_root / row["split"] / identity
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{row['sample_id']}_{row['camera']}_f{int(row['video_frame']):06d}.jpg"
    ok = cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Could not write crop: {crop_path}")

    return {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "original_split": row["original_split"],
        "dataset_part": row["dataset_part"],
        "scene_name": row["scene_name"],
        "camera": row["camera"],
        "annotation_file": row["annotation_file"],
        "annotation_record_index": row["annotation_record_index"],
        "source_image_path": str(image_path),
        "crop_path": str(crop_path),
        "video_frame": row["video_frame"],
        "local_class": row["local_class"],
        "global_person_id": identity,
        "integer_identity_label": label_map[identity],
        "bbox_x": row["bbox_x"],
        "bbox_y": row["bbox_y"],
        "bbox_w": row["bbox_w"],
        "bbox_h": row["bbox_h"],
        "crop_width": crop.shape[1],
        "crop_height": crop.shape[0],
    }


def write_reid_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['crop_path']} {row['integer_identity_label']}\n")


def make_contact_sheet(identity: str, rows: list[dict[str, Any]], output: Path, max_images: int = 24) -> None:
    selected = rows[:max_images]
    thumbs = []
    for row in selected:
        img = cv2.imread(str(row["crop_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        thumb = np.full((160, 100, 3), 255, dtype=np.uint8)
        h, w = img.shape[:2]
        scale = min(100 / max(w, 1), 140 / max(h, 1))
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (100 - nw) // 2
        y = (140 - nh) // 2
        thumb[y:y + nh, x:x + nw] = resized
        cv2.putText(thumb, row["camera"].replace("cam_", ""), (4, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 6
    rows_n = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows_n * 160 + 30, cols * 100, 3), 245, dtype=np.uint8)
    cv2.putText(sheet, identity, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        y = 30 + r * 160
        x = c * 100
        sheet[y:y + 160, x:x + 100] = thumb
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    stage_root = args.stage_root
    output_root = args.output_root or stage_root
    train_manifest = read_rows(stage_root / "manifests" / "train_manifest.csv")
    val_manifest = read_rows(stage_root / "manifests" / "val_manifest.csv")

    train_labels = identity_labels(train_manifest)
    val_labels = identity_labels(val_manifest)
    exported = []
    for row in train_manifest:
        exported.append(crop_one(row, output_root / "crops", train_labels))
    for row in val_manifest:
        exported.append(crop_one(row, output_root / "crops", val_labels))

    train_rows = [row for row in exported if row["split"] == "train"]
    val_rows = [row for row in exported if row["split"] == "val"]
    write_csv(output_root / "manifests" / "crop_export_manifest.csv", exported)
    write_reid_txt(output_root / "manifests" / "train_reid.txt", train_rows)
    write_reid_txt(output_root / "manifests" / "val_reid.txt", val_rows)

    by_identity = defaultdict(list)
    for row in exported:
        by_identity[row["global_person_id"]].append(row)
    for identity, rows in by_identity.items():
        rows.sort(key=lambda row: (row["camera"], int(row["video_frame"])))
        make_contact_sheet(identity, rows, output_root / "identity_audit" / "contact_sheets" / f"{identity}.jpg")

    crop_counts = Counter(row["global_person_id"] for row in exported)
    camera_counts = Counter(row["camera"] for row in exported)
    scene_counts = Counter(row["scene_name"] for row in exported)
    widths = [int(row["crop_width"]) for row in exported]
    heights = [int(row["crop_height"]) for row in exported]
    areas = [w * h for w, h in zip(widths, heights)]
    reserved = {"person_01", "person_02", "person_05", "person_06", "person_09"}
    leakage = [row for row in exported if row["global_person_id"] in reserved or row["original_split"] == "test"]
    report = f"""# DATASET EXPORT REPORT

## Summary

- Train identities: {len(train_labels)} ({', '.join(sorted(train_labels))})
- Validation identities: {len(val_labels)} ({', '.join(sorted(val_labels))})
- Training crops: {len(train_rows)}
- Validation crops: {len(val_rows)}
- Total exported crops: {len(exported)}

## Crops Per Identity

{dict(sorted(crop_counts.items()))}

## Crops Per Camera

{dict(sorted(camera_counts.items()))}

## Crops Per Scene

{dict(sorted(scene_counts.items()))}

## Box/Crop Size

- min area: {min(areas) if areas else 0}
- median area: {sorted(areas)[len(areas)//2] if areas else 0}
- max area: {max(areas) if areas else 0}
- min width: {min(widths) if widths else 0}
- min height: {min(heights) if heights else 0}

## Leakage Checks

- Reserved identities exported: {sum(1 for row in exported if row['global_person_id'] in reserved)}
- Original final-test scene crops exported: {sum(1 for row in exported if row['original_split'] == 'test')}
- Leakage rows: {len(leakage)}

## Notes

Crops are annotation-box crops saved at natural cropped resolution. Resizing and
normalization are left to the training pipeline.
"""
    (output_root / "reports" / "DATASET_EXPORT_REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "DATASET_EXPORT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
