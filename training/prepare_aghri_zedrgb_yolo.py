#!/usr/bin/env python3
"""Prepare an AGHRI ZED RGB person dataset in Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")
TRAIN_VAL_SPLITS = ("train", "val")
PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class YoloBox:
    """A validated YOLO-format box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_line(self) -> str:
        """Return the normalized label row."""
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(frozen=True)
class ConvertedBox:
    """Box-conversion result with rejection context."""

    box: Optional[YoloBox]
    reason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/media/prabuddhi/Backup2/Updated Dataset_PW",
        help="Read-only AGHRI dataset root.",
    )
    parser.add_argument(
        "--split-lists-dir",
        default="/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists",
        help="Directory containing official train.txt, val.txt, and test.txt.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Place images as symlinks or copies.",
    )
    parser.add_argument(
        "--audit-output-root",
        default=(
            "/home/prabuddhi/Desktop/Early_Fus/training_outputs/"
            "aghri_yolo11s_finetune/dataset_audit"
        ),
        help="Directory for deterministic train/val visual-audit images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate an existing output dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect inputs and build in-memory manifests without writing outputs.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing prepared dataset without generating it.",
    )
    return parser.parse_args()


def read_split_file(path: Path) -> List[str]:
    """Read one official split-list file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing split list: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate recording in split list: {path}")
    return names


def load_official_splits(split_lists_dir: Path) -> Dict[str, List[str]]:
    """Load official recording-level train, validation, and test splits."""
    splits = {
        split: read_split_file(split_lists_dir / f"{split}.txt")
        for split in SPLITS
    }
    validate_split_sets(splits)
    return splits


def validate_split_sets(splits: Dict[str, Sequence[str]]) -> None:
    """Fail if any recording belongs to more than one official split."""
    split_sets = {split: set(values) for split, values in splits.items()}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(
                    f"Official split overlap between {left} and {right}: "
                    f"{sorted(overlap)}"
                )
    leakage = (split_sets["train"] | split_sets["val"]) & split_sets["test"]
    if leakage:
        raise ValueError(f"Test recordings leaked into train/val: {sorted(leakage)}")


def find_recordings(dataset_root: Path) -> Dict[str, Path]:
    """Index AGHRI recording directories by recording name."""
    recordings: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for part in sorted(dataset_root.glob("dataset_part*")):
        if not part.is_dir():
            continue
        for recording_dir in sorted(part.glob("*_label")):
            if not recording_dir.is_dir():
                continue
            if recording_dir.name in recordings:
                duplicates.setdefault(recording_dir.name, [recordings[recording_dir.name]])
                duplicates[recording_dir.name].append(recording_dir)
            recordings[recording_dir.name] = recording_dir
    if duplicates:
        details = {
            name: [str(path) for path in paths]
            for name, paths in sorted(duplicates.items())
        }
        raise ValueError(f"Duplicate recording directories found: {details}")
    return recordings


def require_recordings(
    splits: Dict[str, Sequence[str]],
    recordings: Dict[str, Path],
) -> None:
    """Fail if any train/val/test recording listed in splits is missing."""
    missing = {
        split: [name for name in names if name not in recordings]
        for split, names in splits.items()
    }
    missing = {split: names for split, names in missing.items() if names}
    if missing:
        raise FileNotFoundError(f"Missing official split recordings: {missing}")


def read_image_size(image_path: Path) -> Tuple[int, int]:
    """Return image size as (width, height)."""
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size
    except Exception as pil_error:
        try:
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError("cv2.imread returned None")
            height, width = image.shape[:2]
            return width, height
        except Exception as cv_error:
            raise RuntimeError(
                f"Cannot read image size for {image_path}: "
                f"PIL={pil_error}; cv2={cv_error}"
            ) from cv_error


def load_annotation_index(annotation_path: Path) -> Dict[str, dict]:
    """Load ZED RGB annotations keyed by filename and stem."""
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list annotations in {annotation_path}")
    index: Dict[str, dict] = {}
    for frame in data:
        if not isinstance(frame, dict):
            continue
        filename = frame.get("File") or frame.get("file") or frame.get("filename")
        if not filename:
            continue
        for key in {str(filename).lower(), Path(str(filename)).stem.lower()}:
            index[key] = frame
    return index


def labels_for_image(annotation_index: Dict[str, dict], filename: str) -> dict:
    """Return an annotation frame for an image filename, or an empty frame."""
    key = filename.lower()
    stem = Path(filename).stem.lower()
    return annotation_index.get(key) or annotation_index.get(stem) or {
        "File": filename,
        "Timestamp": timestamp_from_filename(filename),
        "Labels": [],
    }


def timestamp_from_filename(filename: str) -> str:
    """Derive a stable timestamp string from AGHRI image filenames."""
    return Path(filename).stem.replace("_", ".", 1)


def convert_xywh_box_to_yolo(
    bbox: Sequence[Any],
    image_width: int,
    image_height: int,
) -> ConvertedBox:
    """Convert source pixel x,y,width,height into clipped normalized YOLO."""
    if len(bbox) < 4:
        return ConvertedBox(None, "bbox_has_fewer_than_4_values")
    try:
        x, y, width, height = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return ConvertedBox(None, "bbox_contains_non_numeric_value")
    if width <= 0 or height <= 0:
        return ConvertedBox(None, "bbox_non_positive_source_extent")
    if image_width <= 0 or image_height <= 0:
        return ConvertedBox(None, "image_non_positive_extent")

    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return ConvertedBox(None, "bbox_outside_image_after_clipping")

    yolo = YoloBox(
        class_id=PERSON_CLASS_ID,
        x_center=((x1 + x2) / 2.0) / float(image_width),
        y_center=((y1 + y2) / 2.0) / float(image_height),
        width=clipped_width / float(image_width),
        height=clipped_height / float(image_height),
    )
    values = (yolo.x_center, yolo.y_center, yolo.width, yolo.height)
    if not all(0.0 <= value <= 1.0 for value in values):
        return ConvertedBox(None, "normalized_coordinate_out_of_bounds")
    if yolo.width <= 0.0 or yolo.height <= 0.0:
        return ConvertedBox(None, "normalized_non_positive_extent")
    return ConvertedBox(yolo)


def convert_frame_labels(
    labels: Sequence[dict],
    image_width: int,
    image_height: int,
) -> Tuple[List[YoloBox], List[dict]]:
    """Convert one frame's source labels into YOLO person labels."""
    boxes: List[YoloBox] = []
    rejected: List[dict] = []
    for index, obj in enumerate(labels):
        if not isinstance(obj, dict):
            rejected.append({"index": index, "reason": "label_is_not_mapping"})
            continue
        bbox = obj.get("BoundingBoxes") or obj.get("bbox") or obj.get("box")
        if bbox is None:
            rejected.append({"index": index, "reason": "missing_bbox"})
            continue
        converted = convert_xywh_box_to_yolo(bbox, image_width, image_height)
        if converted.box is None:
            rejected.append(
                {
                    "index": index,
                    "reason": converted.reason,
                    "source_class": obj.get("Class") or obj.get("class"),
                    "bbox": bbox,
                }
            )
        else:
            boxes.append(converted.box)
    return boxes, rejected


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    """Create an output directory, optionally replacing an existing one."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to recreate it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def place_image(src: Path, dst: Path, mode: str) -> None:
    """Place an image as a symlink or copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    os.symlink(src.resolve(), dst)


def write_data_yaml(output_root: Path) -> None:
    """Write an Ultralytics dataset YAML with absolute root path."""
    data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "person"},
    }
    try:
        import yaml

        text = yaml.safe_dump(data, sort_keys=False)
    except Exception:
        text = (
            f"path: {data['path']}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: person\n"
        )
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def read_dataset_yaml(path: Path) -> Dict[str, Any]:
    """Read the small Ultralytics dataset YAML, with a dependency-free fallback."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        data: Dict[str, Any] = {}
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            raw = lines[index]
            index += 1
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "names" and not value:
                names: Dict[int, str] = {}
                while index < len(lines) and lines[index].startswith("  "):
                    child = lines[index].strip()
                    index += 1
                    if ":" not in child:
                        continue
                    name_key, name_value = child.split(":", 1)
                    names[int(name_key.strip().strip("'\""))] = name_value.strip().strip("'\"")
                data[key] = names
            else:
                data[key] = value.strip("'\"")
        return data


def sha256_file(path: Path) -> str:
    """Hash a file with SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, rows: Sequence[dict]) -> None:
    """Write a deterministic CSV manifest."""
    fields = [
        "split",
        "recording",
        "source_image_path",
        "generated_image_path",
        "source_annotation_path",
        "image_timestamp",
        "image_width",
        "image_height",
        "num_person_boxes",
        "generated_label_path",
        "invalid_rejected_box_count",
        "preparation_status",
        "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_manifest(path: Path) -> List[dict]:
    """Read a CSV manifest as dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_zed_images(recording_dir: Path) -> List[Path]:
    """List ZED RGB images for one recording."""
    image_dir = recording_dir / "sensor_data" / "cam_zed_rgb"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing ZED RGB image directory: {image_dir}")
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_split_manifest(
    split: str,
    recording_names: Sequence[str],
    recordings: Dict[str, Path],
    output_root: Path,
    mode: str,
    dry_run: bool,
) -> Tuple[List[dict], List[dict]]:
    """Generate one train/val split and return manifest rows and rejections."""
    rows: List[dict] = []
    rejected_boxes: List[dict] = []
    identities = set()

    for recording in sorted(recording_names):
        recording_dir = recordings[recording]
        annotation_path = recording_dir / "annotations" / "cam_zed_rgb_ann.json"
        annotation_index = load_annotation_index(annotation_path)
        for source_image in list_zed_images(recording_dir):
            identity = f"{recording}/{source_image.name}"
            if identity in identities:
                raise ValueError(f"Duplicate output identity in {split}: {identity}")
            identities.add(identity)

            generated_image = output_root / "images" / split / recording / source_image.name
            generated_label = (
                output_root
                / "labels"
                / split
                / recording
                / f"{source_image.stem}.txt"
            )
            failure_reason = ""
            status = "ok"
            boxes: List[YoloBox] = []
            rejected: List[dict] = []
            try:
                width, height = read_image_size(source_image)
                annotation_frame = labels_for_image(annotation_index, source_image.name)
                labels = annotation_frame.get("Labels") or annotation_frame.get("labels") or []
                boxes, rejected = convert_frame_labels(labels, width, height)
                if not dry_run:
                    place_image(source_image, generated_image, mode)
                    generated_label.parent.mkdir(parents=True, exist_ok=True)
                    label_text = "\n".join(box.as_line() for box in boxes)
                    generated_label.write_text(
                        label_text + ("\n" if label_text else ""),
                        encoding="utf-8",
                    )
                for rejected_box in rejected:
                    rejected_boxes.append(
                        {
                            "split": split,
                            "recording": recording,
                            "source_image_path": str(source_image),
                            **rejected_box,
                        }
                    )
            except Exception as error:
                width, height = 0, 0
                status = "failed"
                failure_reason = str(error)

            rows.append(
                {
                    "split": split,
                    "recording": recording,
                    "source_image_path": str(source_image.resolve()),
                    "generated_image_path": str(generated_image.resolve()),
                    "source_annotation_path": str(annotation_path.resolve()),
                    "image_timestamp": timestamp_from_filename(source_image.name),
                    "image_width": width,
                    "image_height": height,
                    "num_person_boxes": len(boxes),
                    "generated_label_path": str(generated_label.resolve()),
                    "invalid_rejected_box_count": len(rejected),
                    "preparation_status": status,
                    "failure_reason": failure_reason,
                }
            )
    return rows, rejected_boxes


def write_excluded_test_manifest(
    path: Path,
    test_recordings: Sequence[str],
    recordings: Dict[str, Path],
) -> None:
    """Write a manifest proving official test recordings were excluded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "recording", "recording_path", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for recording in sorted(test_recordings):
            writer.writerow(
                {
                    "split": "test",
                    "recording": recording,
                    "recording_path": str(recordings[recording].resolve()),
                    "status": "excluded_from_training_and_validation",
                }
            )


def load_label_file(path: Path) -> List[YoloBox]:
    """Parse a YOLO label file."""
    boxes: List[YoloBox] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} expected 5 fields")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id != PERSON_CLASS_ID:
            raise ValueError(f"{path}:{line_number} class id is not 0")
        x_center, y_center, width, height = values
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number} normalized value out of range")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number} non-positive width/height")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return boxes


def verify_dataset(
    output_root: Path,
    split_lists_dir: Path,
    write_summary: bool = True,
) -> dict:
    """Verify a prepared AGHRI YOLO dataset."""
    splits = load_official_splits(split_lists_dir)
    manifests = {
        split: read_manifest(output_root / "manifests" / f"{split}.csv")
        for split in TRAIN_VAL_SPLITS
    }
    train_ids = {
        f"{row['recording']}/{Path(row['source_image_path']).name}"
        for row in manifests["train"]
    }
    val_ids = {
        f"{row['recording']}/{Path(row['source_image_path']).name}"
        for row in manifests["val"]
    }
    train_val_overlap = train_ids & val_ids
    test_recordings = set(splits["test"])
    leakage_rows = [
        row
        for split in TRAIN_VAL_SPLITS
        for row in manifests[split]
        if row["recording"] in test_recordings
    ]

    counts = {
        "train_recording_count": len(splits["train"]),
        "validation_recording_count": len(splits["val"]),
        "excluded_test_recording_count": len(splits["test"]),
        "train_image_count": len(manifests["train"]),
        "validation_image_count": len(manifests["val"]),
        "train_person_box_count": 0,
        "validation_person_box_count": 0,
        "train_negative_image_count": 0,
        "validation_negative_image_count": 0,
        "train_rejected_box_count": 0,
        "validation_rejected_box_count": 0,
        "missing_file_count": 0,
        "split_overlap_count": len(train_val_overlap),
        "test_leakage_count": len(leakage_rows),
        "duplicate_image_identity_count": 0,
    }

    seen_identities = set()
    duplicate_identities = set()
    for split in TRAIN_VAL_SPLITS:
        for row in manifests[split]:
            image_path = Path(row["generated_image_path"])
            label_path = Path(row["generated_label_path"])
            identity = f"{row['recording']}/{image_path.name}"
            if identity in seen_identities:
                duplicate_identities.add(identity)
            seen_identities.add(identity)
            if not image_path.exists():
                counts["missing_file_count"] += 1
            if not label_path.exists():
                counts["missing_file_count"] += 1
                continue
            labels = load_label_file(label_path)
            box_key = "train_person_box_count" if split == "train" else "validation_person_box_count"
            neg_key = (
                "train_negative_image_count"
                if split == "train"
                else "validation_negative_image_count"
            )
            rejected_key = (
                "train_rejected_box_count"
                if split == "train"
                else "validation_rejected_box_count"
            )
            counts[box_key] += len(labels)
            if not labels:
                counts[neg_key] += 1
            counts[rejected_key] += int(row.get("invalid_rejected_box_count") or 0)

    counts["duplicate_image_identity_count"] = len(duplicate_identities)

    data_yaml = output_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    data_cfg = read_dataset_yaml(data_yaml)
    dataset_path = Path(str(data_cfg.get("path", output_root)))
    for key in ("train", "val"):
        value = data_cfg.get(key)
        if not value:
            raise ValueError(f"data.yaml missing {key}")
        resolved = Path(value)
        if not resolved.is_absolute():
            resolved = dataset_path / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"data.yaml {key} path missing: {resolved}")
    names = data_cfg.get("names")
    if names not in ({0: "person"}, {"0": "person"}, ["person"]):
        raise ValueError(f"Unexpected data.yaml names: {names}")

    if any(
        counts[key] != 0
        for key in (
            "missing_file_count",
            "split_overlap_count",
            "test_leakage_count",
            "duplicate_image_identity_count",
        )
    ):
        raise ValueError(f"Dataset verification failed: {counts}")

    if write_summary:
        summary_path = output_root / "manifests" / "preparation_summary.json"
        existing = {}
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing.update(counts)
        existing["verification_status"] = "passed"
        summary_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return counts


def manifest_hash(path: Path) -> str:
    """Return a SHA256 hash for a manifest."""
    return sha256_file(path)


def write_summary(
    output_root: Path,
    split_lists_dir: Path,
    rows_by_split: Dict[str, Sequence[dict]],
    rejected_boxes: Sequence[dict],
    dry_run: bool,
) -> dict:
    """Write preparation summary JSON."""
    splits = load_official_splits(split_lists_dir)
    summary = {
        "dataset_root": None,
        "output_root": str(output_root.resolve()),
        "split_lists_dir": str(split_lists_dir.resolve()),
        "train_recording_count": len(splits["train"]),
        "validation_recording_count": len(splits["val"]),
        "excluded_test_recording_count": len(splits["test"]),
        "train_image_count": len(rows_by_split.get("train", [])),
        "validation_image_count": len(rows_by_split.get("val", [])),
        "train_person_box_count": sum(
            int(row["num_person_boxes"]) for row in rows_by_split.get("train", [])
        ),
        "validation_person_box_count": sum(
            int(row["num_person_boxes"]) for row in rows_by_split.get("val", [])
        ),
        "train_negative_image_count": sum(
            1 for row in rows_by_split.get("train", []) if int(row["num_person_boxes"]) == 0
        ),
        "validation_negative_image_count": sum(
            1 for row in rows_by_split.get("val", []) if int(row["num_person_boxes"]) == 0
        ),
        "train_rejected_box_count": sum(
            int(row["invalid_rejected_box_count"]) for row in rows_by_split.get("train", [])
        ),
        "validation_rejected_box_count": sum(
            int(row["invalid_rejected_box_count"]) for row in rows_by_split.get("val", [])
        ),
        "rejected_box_count": len(rejected_boxes),
        "missing_file_count": sum(
            1
            for rows in rows_by_split.values()
            for row in rows
            if row["preparation_status"] != "ok"
        ),
        "split_overlap_count": 0,
        "test_leakage_count": 0,
        "dry_run": dry_run,
        "box_source_convention": "pixel x,y,width,height",
        "box_clipping_rule": "clip source x1,y1,x2,y2 to image extent before normalization",
        "class_mapping": "all AGHRI ZED RGB annotated people are YOLO class 0 person",
    }
    if not dry_run:
        summary["train_manifest_sha256"] = manifest_hash(output_root / "manifests" / "train.csv")
        summary["val_manifest_sha256"] = manifest_hash(output_root / "manifests" / "val.csv")
        (output_root / "manifests" / "preparation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def draw_visual_audit(output_root: Path, audit_root: Path, max_per_split: int = 8) -> List[Path]:
    """Draw deterministic train/val samples with converted boxes."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return []

    created: List[Path] = []
    if audit_root.exists():
        shutil.rmtree(audit_root)
    audit_root.mkdir(parents=True, exist_ok=True)

    for split in TRAIN_VAL_SPLITS:
        rows = read_manifest(output_root / "manifests" / f"{split}.csv")
        non_empty = [row for row in rows if int(row["num_person_boxes"]) > 0]
        empty = [row for row in rows if int(row["num_person_boxes"]) == 0]
        chosen = (non_empty[: max_per_split - 1] + empty[:1])[:max_per_split]
        split_out = audit_root / split
        split_out.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(chosen):
            image_path = Path(row["generated_image_path"])
            label_path = Path(row["generated_label_path"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                width, height = image.size
                for box in load_label_file(label_path):
                    x1 = (box.x_center - box.width / 2.0) * width
                    y1 = (box.y_center - box.height / 2.0) * height
                    x2 = (box.x_center + box.width / 2.0) * width
                    y2 = (box.y_center + box.height / 2.0) * height
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                out_path = split_out / f"{index:02d}_{row['recording']}_{image_path.name}"
                image.save(out_path)
                created.append(out_path)
    return created


def prepare_dataset(
    dataset_root: Path,
    split_lists_dir: Path,
    output_root: Path,
    mode: str = "symlink",
    overwrite: bool = False,
    dry_run: bool = False,
    audit_output_root: Optional[Path] = None,
) -> dict:
    """Prepare train/val YOLO dataset and manifests."""
    splits = load_official_splits(split_lists_dir)
    recordings = find_recordings(dataset_root)
    require_recordings(splits, recordings)

    if not dry_run:
        ensure_empty_dir(output_root, overwrite)
        for split in TRAIN_VAL_SPLITS:
            (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    rows_by_split: Dict[str, List[dict]] = {}
    all_rejected: List[dict] = []
    for split in TRAIN_VAL_SPLITS:
        rows, rejected = build_split_manifest(
            split=split,
            recording_names=splits[split],
            recordings=recordings,
            output_root=output_root,
            mode=mode,
            dry_run=dry_run,
        )
        rows_by_split[split] = rows
        all_rejected.extend(rejected)
        if not dry_run:
            write_manifest(output_root / "manifests" / f"{split}.csv", rows)

    if not dry_run:
        write_excluded_test_manifest(
            output_root / "manifests" / "excluded_test.csv",
            splits["test"],
            recordings,
        )
        (output_root / "manifests" / "rejected_boxes.json").write_text(
            json.dumps(all_rejected, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_data_yaml(output_root)

    summary = write_summary(output_root, split_lists_dir, rows_by_split, all_rejected, dry_run)
    summary["dataset_root"] = str(dataset_root.resolve())
    if not dry_run:
        verify_dataset(output_root, split_lists_dir, write_summary=True)
        if audit_output_root is not None:
            audit_images = draw_visual_audit(output_root, audit_output_root)
            summary["visual_audit_image_count"] = len(audit_images)
            summary_path = output_root / "manifests" / "preparation_summary.json"
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            existing.update(
                {
                    "dataset_root": summary["dataset_root"],
                    "visual_audit_dir": str(audit_output_root.resolve()),
                    "visual_audit_image_count": len(audit_images),
                }
            )
            summary_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return summary


def main() -> None:
    """Command-line entrypoint."""
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    split_lists_dir = Path(args.split_lists_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if args.verify_only:
        summary = verify_dataset(output_root, split_lists_dir, write_summary=True)
    else:
        summary = prepare_dataset(
            dataset_root=dataset_root,
            split_lists_dir=split_lists_dir,
            output_root=output_root,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            audit_output_root=Path(args.audit_output_root).resolve(),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
