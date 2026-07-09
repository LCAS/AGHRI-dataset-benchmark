"""Focused tests for AGHRI ZED RGB YOLO training utilities."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "training"
sys.path.insert(0, str(TRAINING_DIR))

import prepare_aghri_zedrgb_yolo as prep
from write_checkpoint_manifest import sha256_file


def _write_image(path: Path, size=(100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def _write_recording(root: Path, recording: str, frames: dict) -> None:
    rec = root / "dataset_part1" / recording
    image_dir = rec / "sensor_data" / "cam_zed_rgb"
    ann_dir = rec / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    for filename, labels in frames.items():
        _write_image(image_dir / filename)
        annotations.append(
            {
                "Timestamp": float(Path(filename).stem.replace("_", ".", 1)),
                "File": filename,
                "Labels": labels,
            }
        )
    (ann_dir / "cam_zed_rgb_ann.json").write_text(json.dumps(annotations), encoding="utf-8")


def _write_splits(root: Path, train, val, test) -> Path:
    split_dir = root / "split_lists"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, names in {"train": train, "val": val, "test": test}.items():
        (split_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return split_dir


def test_split_list_parsing_and_overlap_rejection(tmp_path):
    split_dir = _write_splits(tmp_path, ["a_label"], ["b_label"], ["c_label"])
    splits = prep.load_official_splits(split_dir)
    assert splits["train"] == ["a_label"]
    assert splits["val"] == ["b_label"]
    assert splits["test"] == ["c_label"]

    _write_splits(tmp_path, ["a_label"], ["a_label"], ["c_label"])
    with pytest.raises(ValueError, match="overlap"):
        prep.load_official_splits(split_dir)


def test_box_conversion_clips_and_rejects_invalid_boxes():
    clipped = prep.convert_xywh_box_to_yolo([-10, 10, 30, 40], 100, 80)
    assert clipped.box is not None
    assert clipped.box.class_id == 0
    assert clipped.box.x_center == pytest.approx(0.10)
    assert clipped.box.width == pytest.approx(0.20)

    invalid = prep.convert_xywh_box_to_yolo([10, 10, 0, 40], 100, 80)
    assert invalid.box is None
    assert invalid.reason == "bbox_non_positive_source_extent"

    outside = prep.convert_xywh_box_to_yolo([120, 10, 5, 5], 100, 80)
    assert outside.box is None
    assert outside.reason == "bbox_outside_image_after_clipping"


def test_multiple_persons_and_empty_label_file_handling(tmp_path):
    labels = [
        {"Class": "01", "BoundingBoxes": [0, 0, 10, 10]},
        {"Class": "02", "BoundingBoxes": [20, 20, 20, 10]},
    ]
    boxes, rejected = prep.convert_frame_labels(labels, 100, 80)
    assert len(boxes) == 2
    assert rejected == []

    label_path = tmp_path / "empty.txt"
    label_path.write_text("", encoding="utf-8")
    assert prep.load_label_file(label_path) == []


def test_prepare_dry_run_and_full_dataset_are_deterministic(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_rec = "train_recording_label"
    val_rec = "val_recording_label"
    test_rec = "test_recording_label"
    frame = "1000_000000001.png"
    _write_recording(
        dataset_root,
        train_rec,
        {
            frame: [
                {"Class": "01", "BoundingBoxes": [5, 5, 20, 20]},
                {"Class": "02", "BoundingBoxes": [50, 10, 15, 25]},
            ],
            "1000_000000002.png": [],
        },
    )
    _write_recording(
        dataset_root,
        val_rec,
        {frame: [{"Class": "01", "BoundingBoxes": [-5, 0, 15, 15]}]},
    )
    _write_recording(dataset_root, test_rec, {frame: [{"Class": "01", "BoundingBoxes": [1, 1, 5, 5]}]})
    split_dir = _write_splits(tmp_path, [train_rec], [val_rec], [test_rec])
    output_root = tmp_path / "yolo"

    dry_summary = prep.prepare_dataset(
        dataset_root,
        split_dir,
        output_root,
        dry_run=True,
    )
    assert dry_summary["train_image_count"] == 2
    assert not output_root.exists()

    prep.prepare_dataset(
        dataset_root,
        split_dir,
        output_root,
        mode="symlink",
        overwrite=True,
        audit_output_root=tmp_path / "audit",
    )
    summary = prep.verify_dataset(output_root, split_dir)
    assert summary["test_leakage_count"] == 0
    assert summary["split_overlap_count"] == 0
    assert summary["train_image_count"] == 2
    assert summary["validation_image_count"] == 1
    assert summary["train_negative_image_count"] == 1
    assert (output_root / "images" / "train" / train_rec / frame).is_symlink()
    assert (output_root / "labels" / "train" / train_rec / f"{Path(frame).stem}.txt").is_file()

    with (output_root / "manifests" / "excluded_test.csv").open(newline="", encoding="utf-8") as handle:
        excluded = list(csv.DictReader(handle))
    assert excluded[0]["recording"] == test_rec

    first_hash = prep.sha256_file(output_root / "manifests" / "train.csv")
    prep.prepare_dataset(dataset_root, split_dir, output_root, overwrite=True)
    second_hash = prep.sha256_file(output_root / "manifests" / "train.csv")
    assert first_hash == second_hash


def test_dataset_yaml_generation(tmp_path):
    output_root = tmp_path / "dataset"
    output_root.mkdir()
    prep.write_data_yaml(output_root)
    data = yaml.safe_load((output_root / "data.yaml").read_text(encoding="utf-8"))
    assert Path(data["path"]) == output_root.resolve()
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"] == {0: "person"}


def test_checkpoint_manifest_hash_helper(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert sha256_file(checkpoint) == (
        "47320987f9a49d5b00119b960f247a95"
        "6773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_finetuned_fusion_config_only_changes_model_path():
    generic = yaml.safe_load((REPO_ROOT / "config" / "aghri_zed_livox.yaml").read_text())
    finetuned_path = REPO_ROOT / "config" / "aghri_zed_livox_yolo11s_finetuned.yaml"
    if not finetuned_path.exists():
        pytest.skip("fine-tuned fusion config is created after training")
    finetuned = yaml.safe_load(finetuned_path.read_text())
    generic_params = generic["lidar_fusion"]["ros__parameters"]
    finetuned_params = finetuned["lidar_fusion"]["ros__parameters"]
    changed = {
        key
        for key in set(generic_params) | set(finetuned_params)
        if generic_params.get(key) != finetuned_params.get(key)
    }
    assert changed == {"YOLO_model"}
    assert finetuned_params["association_method"] == "legacy_box"
    assert finetuned_params["centre_statistic"] == "mean"
    assert finetuned_params["box_bottom_trim_ratio"] == 0.0
