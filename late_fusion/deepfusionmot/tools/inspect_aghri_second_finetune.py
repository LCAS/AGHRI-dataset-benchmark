#!/usr/bin/env python3
"""Write preflight reports for AGHRI SECOND fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from aghri_second_finetune_common import (  # noqa: E402
    EXPECTED_GENERIC_SECOND_SHA256,
    discover_scenes,
    ensure_no_split_overlap,
    iter_scene_frames,
    normalize_box,
    read_pcd_header,
    read_split_scenes,
    sha256_file,
    summarize,
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def inspect_splits(raw_root: Path, splits_dir: Path) -> dict:
    split_scenes = read_split_scenes(splits_dir)
    scene_records = discover_scenes(raw_root)
    missing = {
        split: [scene for scene in scenes if scene not in scene_records]
        for split, scenes in split_scenes.items()
    }
    missing = {split: scenes for split, scenes in missing.items() if scenes}
    frame_counts = {}
    for split, scenes in split_scenes.items():
        count = 0
        for scene in scenes:
            if scene in scene_records:
                count += sum(1 for _ in iter_scene_frames(scene_records[scene]))
        frame_counts[split] = count
    return {
        "splits_dir": str(splits_dir),
        "raw_root": str(raw_root),
        "split_scene_counts": {split: len(scenes) for split, scenes in split_scenes.items()},
        "split_frame_counts": frame_counts,
        "missing_scene_count": sum(len(v) for v in missing.values()),
        "missing_scenes": missing,
        "overlaps": ensure_no_split_overlap(split_scenes),
        "ok": not missing and not ensure_no_split_overlap(split_scenes),
    }


def inspect_pointclouds(raw_root: Path, splits_dir: Path, max_samples_per_split: int) -> dict:
    split_scenes = read_split_scenes(splits_dir)
    scene_records = discover_scenes(raw_root)
    samples = []
    point_counts: Dict[str, List[float]] = defaultdict(list)
    field_layouts = Counter()
    data_modes = Counter()
    for split, scenes in split_scenes.items():
        sampled = 0
        for scene in scenes:
            if scene not in scene_records:
                continue
            for frame in iter_scene_frames(scene_records[scene]):
                if sampled >= max_samples_per_split:
                    break
                header, _header_bytes = read_pcd_header(frame["pcd_path"])
                fields = " ".join(header.get("FIELDS", []))
                field_layouts[fields] += 1
                data_modes[" ".join(header.get("DATA", []))] += 1
                point_count = int(header.get("POINTS", ["0"])[0])
                point_counts[split].append(float(point_count))
                samples.append(
                    {
                        "split": split,
                        "scene": scene,
                        "file": str(frame["pcd_path"]),
                        "fields": header.get("FIELDS", []),
                        "type": header.get("TYPE", []),
                        "size": header.get("SIZE", []),
                        "points": point_count,
                        "data": header.get("DATA", []),
                    }
                )
                sampled += 1
            if sampled >= max_samples_per_split:
                break
    return {
        "sample_count": len(samples),
        "samples": samples,
        "point_count_summary": {split: summarize(values) for split, values in point_counts.items()},
        "field_layouts": dict(field_layouts),
        "data_modes": dict(data_modes),
        "expected_mmdet3d_features": ["x", "y", "z", "intensity"],
        "conversion_note": "Raw AGHRI PCD uses x/y/z plus rgb in sampled files; rgb is decoded to normalized intensity.",
    }


def inspect_boxes(raw_root: Path, splits_dir: Path) -> dict:
    split_scenes = read_split_scenes(splits_dir)
    scene_records = discover_scenes(raw_root)
    rows = []
    per_split = {}
    for split, scenes in split_scenes.items():
        classes = Counter()
        dims_l: List[float] = []
        dims_w: List[float] = []
        dims_h: List[float] = []
        bottoms: List[float] = []
        yaws: List[float] = []
        frames = 0
        frames_with_boxes = 0
        for scene in scenes:
            if scene not in scene_records:
                continue
            for frame in iter_scene_frames(scene_records[scene]):
                frames += 1
                had_box = False
                for label in frame["labels"]:
                    try:
                        x, y, z, length, width, height, yaw = normalize_box(label.get("BoundingBoxes", []))
                    except ValueError:
                        continue
                    classes[str(label.get("Class", ""))] += 1
                    dims_l.append(length)
                    dims_w.append(width)
                    dims_h.append(height)
                    bottoms.append(z)
                    yaws.append(yaw)
                    had_box = True
                    rows.append(
                        {
                            "split": split,
                            "scene": scene,
                            "sample_idx": f"{scene}/{frame['pcd_stem']}",
                            "raw_class": str(label.get("Class", "")),
                            "x": x,
                            "y": y,
                            "z_bottom": z,
                            "length": length,
                            "width": width,
                            "height": height,
                            "yaw": yaw,
                        }
                    )
                if had_box:
                    frames_with_boxes += 1
        per_split[split] = {
            "frames": frames,
            "frames_with_boxes": frames_with_boxes,
            "instances": sum(classes.values()),
            "raw_class_counts": dict(classes),
            "length": summarize(dims_l),
            "width": summarize(dims_w),
            "height": summarize(dims_h),
            "z_bottom": summarize(bottoms),
            "yaw": summarize(yaws),
        }
    return {"per_split": per_split, "rows": rows}


def inspect_checkpoint(checkpoint: Path) -> dict:
    result = {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.exists(),
        "expected_sha256": EXPECTED_GENERIC_SECOND_SHA256,
        "sha256": None,
        "sha256_ok": False,
        "torch_load_ok": False,
        "state_dict_key_count": 0,
        "bbox_head_key_count": 0,
        "estimated_backbone_transfer_key_count": 0,
        "estimated_backbone_transfer_key_ratio": None,
        "notes": [],
    }
    if not checkpoint.exists():
        result["notes"].append("Checkpoint file was not found.")
        return result
    result["sha256"] = sha256_file(checkpoint)
    result["sha256_ok"] = result["sha256"] == EXPECTED_GENERIC_SECOND_SHA256
    try:
        import torch

        payload = torch.load(str(checkpoint), map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        keys = list(state_dict.keys())
        bbox_keys = [key for key in keys if "bbox_head" in key]
        transferable = [
            key
            for key in keys
            if "bbox_head" not in key
            and not key.endswith("num_batches_tracked")
        ]
        result.update(
            {
                "torch_load_ok": True,
                "state_dict_key_count": len(keys),
                "bbox_head_key_count": len(bbox_keys),
                "estimated_backbone_transfer_key_count": len(transferable),
                "estimated_backbone_transfer_key_ratio": len(transferable) / len(keys) if keys else None,
                "meta_keys": sorted(list(payload.get("meta", {}).keys())) if isinstance(payload, dict) else [],
            }
        )
        result["notes"].append(
            "This is a name-based estimate. The AGHRI config reuses SECOND VFE/middle/backbone/neck shapes "
            "and reinitializes the one-class bbox head."
        )
    except Exception as exc:  # pragma: no cover - depends on local torch install
        result["notes"].append(f"torch.load failed: {exc}")
    return result


def inspect_config(mmdet3d_root: Path) -> dict:
    model_config = mmdet3d_root / "configs" / "models" / "second_aghri.py"
    dataset_config = mmdet3d_root / "configs" / "datasets" / "aghri_lidar_detection.py"
    metric = mmdet3d_root / "aghri3d" / "metric.py"
    model_text = load_text(model_config)
    dataset_text = load_text(dataset_config)
    metric_text = load_text(metric)
    return {
        "mmdet3d_root": str(mmdet3d_root),
        "model_config": str(model_config),
        "dataset_config": str(dataset_config),
        "metric": str(metric),
        "point_cloud_range": "[-10.24, -10.24, -2.0, 25.6, 10.24, 3.0]" in model_text,
        "load_dim_4": "load_dim=4" in dataset_text or "load_dim': 4" in dataset_text,
        "use_dim_4": "use_dim=4" in dataset_text or "use_dim': 4" in dataset_text,
        "save_best_metric_present": "bev_ap_mean" in metric_text,
        "anchor_size_configured": "anchor_size = [0.46, 0.57, 1.50]" in model_text,
        "iou_thresholds": "[0.25, 0.5, 0.75]" in dataset_text,
    }


def write_markdown(path: Path, split: dict, pcd: dict, boxes: dict, ckpt: dict, cfg: dict) -> None:
    lines = [
        "# AGHRI SECOND Fine-tuning Inspection",
        "",
        "## Verdict",
        "",
        f"- Split integrity: {'PASS' if split['ok'] else 'CHECK REQUIRED'}",
        f"- Generic checkpoint SHA256: {'PASS' if ckpt['sha256_ok'] else 'FAIL'}",
        f"- Checkpoint torch load: {'PASS' if ckpt['torch_load_ok'] else 'NOT RUN/FAILED'}",
        f"- Sampled PCD layouts: {pcd['field_layouts']}",
        "",
        "## Dataset Splits",
        "",
        "| split | scenes | frames |",
        "|---|---:|---:|",
    ]
    for split_name in ("train", "val", "test"):
        lines.append(
            f"| {split_name} | {split['split_scene_counts'].get(split_name, 0)} | "
            f"{split['split_frame_counts'].get(split_name, 0)} |"
        )
    lines += [
        "",
        "## Box Audit",
        "",
        "| split | frames | boxed frames | instances | raw classes | mean LWH | mean bottom z |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for split_name, values in boxes["per_split"].items():
        mean_l = values["length"]["mean"]
        mean_w = values["width"]["mean"]
        mean_h = values["height"]["mean"]
        lwh = f"{mean_l:.3f}, {mean_w:.3f}, {mean_h:.3f}" if mean_l is not None else "n/a"
        z_mean = values["z_bottom"]["mean"]
        z_text = f"{z_mean:.3f}" if z_mean is not None else "n/a"
        lines.append(
            f"| {split_name} | {values['frames']} | {values['frames_with_boxes']} | "
            f"{values['instances']} | {values['raw_class_counts']} | {lwh} | "
            f"{z_text} |"
        )
    lines += [
        "",
        "## MMDetection3D Compatibility",
        "",
        f"- AGHRI SECOND config found: `{cfg['model_config']}`",
        f"- Dataset config uses 4 point features: {cfg['load_dim_4'] and cfg['use_dim_4']}",
        f"- AGHRI metric exposes BEV/3D AP thresholds 0.25/0.50/0.75: {cfg['iou_thresholds']}",
        f"- Estimated non-head transfer keys: {ckpt['estimated_backbone_transfer_key_count']} / {ckpt['state_dict_key_count']}",
        "",
        "## Notes",
        "",
        "- The fine-tune export must stage only train/val data. The held-out test split is audited here but excluded from the training root by default.",
        "- Raw AGHRI labels are numeric IDs in sampled files, so the fine-tune converter maps every valid LiDAR box to the single `person` class.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--mmdet3d-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-pcd-samples-per-split", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    mmdet3d_root = Path(args.mmdet3d_root).resolve()

    split = inspect_splits(raw_root, splits_dir)
    pcd = inspect_pointclouds(raw_root, splits_dir, args.max_pcd_samples_per_split)
    boxes = inspect_boxes(raw_root, splits_dir)
    ckpt = inspect_checkpoint(Path(args.checkpoint).resolve())
    cfg = inspect_config(mmdet3d_root)

    (out_dir / "split_verification.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (out_dir / "pointcloud_profile.json").write_text(json.dumps(pcd, indent=2), encoding="utf-8")
    (out_dir / "box_audit_summary.json").write_text(json.dumps(boxes["per_split"], indent=2), encoding="utf-8")
    (out_dir / "checkpoint_loading_audit.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    (out_dir / "mmdetection3d_config_audit.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with (out_dir / "box_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["split", "scene", "sample_idx", "raw_class", "x", "y", "z_bottom", "length", "width", "height", "yaw"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boxes["rows"])

    write_markdown(out_dir / "AGHRI_SECOND_FINETUNE_INSPECTION.md", split, pcd, boxes, ckpt, cfg)
    print(json.dumps({"out_dir": str(out_dir), "split_ok": split["ok"], "checkpoint_sha_ok": ckpt["sha256_ok"]}, indent=2))


if __name__ == "__main__":
    main()
