#!/usr/bin/env python3
"""Prepare a train/val AGHRI LiDAR export for SECOND fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from aghri_second_finetune_common import (  # noqa: E402
    discover_scenes,
    iter_scene_frames,
    normalize_box,
    pcd_dtype_fields,
    read_pcd_header,
    read_split_scenes,
)


CLASS_NAME = "person"


def parse_pcd(path: Path) -> np.ndarray:
    import numpy as np

    header, header_bytes = read_pcd_header(path)
    dtype = np.dtype(pcd_dtype_fields(header))
    count = int(header["POINTS"][0])
    with path.open("rb") as stream:
        stream.seek(header_bytes)
        return np.fromfile(stream, dtype=dtype, count=count)


def decode_intensity(points: np.ndarray) -> np.ndarray:
    import numpy as np

    if "intensity" in points.dtype.names:
        return points["intensity"].astype(np.float32)
    if "rgb" in points.dtype.names:
        rgb_float = points["rgb"].astype(np.float32)
        rgb_uint = rgb_float.view(np.uint32)
        red = ((rgb_uint >> 16) & 255).astype(np.float32)
        green = ((rgb_uint >> 8) & 255).astype(np.float32)
        blue = (rgb_uint & 255).astype(np.float32)
        return ((red + green + blue) / 3.0) / 255.0
    return np.zeros(points.shape[0], dtype=np.float32)


def convert_pcd_to_bin(source: Path, dest: Path, overwrite: bool) -> None:
    import numpy as np

    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    points = parse_pcd(source)
    converted = np.column_stack(
        [
            points["x"].astype(np.float32),
            points["y"].astype(np.float32),
            points["z"].astype(np.float32),
            decode_intensity(points),
        ]
    ).astype(np.float32)
    converted.tofile(dest)


def frame_to_info(frame: dict) -> dict:
    sample_idx = f"{frame['scene']}/{frame['pcd_stem']}"
    info = {
        "sample_idx": sample_idx,
        "scene_name": frame["scene"],
        "timestamp": frame["timestamp"],
        "lidar_points": {
            "lidar_path": f"points/{frame['scene']}/{frame['pcd_stem']}.bin",
            "num_pts_feats": 4,
        },
        "instances": [],
    }
    for raw_instance in frame["labels"]:
        box_payload = raw_instance.get("BoundingBoxes", [])
        try:
            box = normalize_box(box_payload)
        except ValueError:
            continue
        info["instances"].append(
            {
                "bbox_3d": box,
                "bbox_label_3d": 0,
                "bbox_label": 0,
                "bbox_3d_isvalid": True,
                "num_lidar_pts": -1,
                "raw_class": str(raw_instance.get("Class", "")),
            }
        )
    return info


def write_infos(path: Path, data_list: List[dict]) -> None:
    payload = {
        "metainfo": {
            "dataset": "aghri_lidar",
            "info_version": "aghri_second_finetune_scene_split_v1",
            "classes": [CLASS_NAME],
        },
        "data_list": data_list,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(payload, stream)


def prepare_dataset(args: argparse.Namespace) -> dict:
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    splits_dir = Path(args.splits_dir).resolve()
    active_splits = ["train", "val"] if not args.include_test else ["train", "val", "test"]

    split_scenes = read_split_scenes(splits_dir)
    scene_records = discover_scenes(raw_root)
    missing = {
        split: [scene for scene in split_scenes[split] if scene not in scene_records]
        for split in active_splits
    }
    missing = {split: scenes for split, scenes in missing.items() if scenes}
    if missing:
        raise FileNotFoundError(f"Split scenes missing under {raw_root}: {missing}")

    out_root.mkdir(parents=True, exist_ok=True)
    infos_dir = out_root / "infos"
    image_sets_dir = out_root / "ImageSets"
    image_sets_dir.mkdir(parents=True, exist_ok=True)

    split_to_infos: Dict[str, List[dict]] = {split: [] for split in active_splits}
    split_to_samples: Dict[str, List[str]] = {split: [] for split in active_splits}
    convert_jobs = []

    for split in active_splits:
        for scene_name in split_scenes[split]:
            for frame in iter_scene_frames(scene_records[scene_name]):
                if not frame["pcd_path"].exists():
                    raise FileNotFoundError(f"Missing point cloud: {frame['pcd_path']}")
                info = frame_to_info(frame)
                split_to_infos[split].append(info)
                split_to_samples[split].append(info["sample_idx"])
                dest = out_root / info["lidar_points"]["lidar_path"]
                convert_jobs.append((frame["pcd_path"], dest))

    if args.dry_run:
        return {
            "dry_run": True,
            "out_root": str(out_root),
            "active_splits": active_splits,
            "split_scenes": {split: split_scenes[split] for split in active_splits},
            "split_frames": {split: len(split_to_infos[split]) for split in active_splits},
            "total_conversion_jobs": len(convert_jobs),
        }

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(convert_pcd_to_bin, source, dest, args.overwrite_points)
                for source, dest in convert_jobs
            ]
            for future in futures:
                future.result()
    else:
        for source, dest in convert_jobs:
            convert_pcd_to_bin(source, dest, args.overwrite_points)

    for split in active_splits:
        write_infos(infos_dir / f"agri_person_infos_{split}.pkl", split_to_infos[split])
        write_infos(out_root / f"agri_person_infos_{split}.pkl", split_to_infos[split])
        (image_sets_dir / f"{split}.txt").write_text("\n".join(split_to_samples[split]) + "\n", encoding="utf-8")
        (image_sets_dir / f"{split}_scenes.txt").write_text("\n".join(split_scenes[split]) + "\n", encoding="utf-8")

    if not args.include_test:
        for stale in [
            infos_dir / "agri_person_infos_test.pkl",
            out_root / "agri_person_infos_test.pkl",
            image_sets_dir / "test.txt",
            image_sets_dir / "test_scenes.txt",
        ]:
            if stale.exists():
                stale.unlink()

    rows = []
    for split in active_splits:
        rows.append(
            {
                "split": split,
                "scenes": len(split_scenes[split]),
                "frames": len(split_to_infos[split]),
                "instances": sum(len(info["instances"]) for info in split_to_infos[split]),
            }
        )
    with (out_root / "aghri_second_finetune_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "scenes", "frames", "instances"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dry_run": False,
        "out_root": str(out_root),
        "active_splits": active_splits,
        "test_split_staged": bool(args.include_test),
        "class_name": CLASS_NAME,
        "split_scenes": {split: split_scenes[split] for split in active_splits},
        "split_frames": {split: len(split_to_infos[split]) for split in active_splits},
        "split_instances": {
            split: sum(len(info["instances"]) for info in split_to_infos[split])
            for split in active_splits
        },
    }
    (out_root / "dataset_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite-points", action="store_true")
    parser.add_argument("--include-test", action="store_true", help="Also export held-out test split. Off by default.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = prepare_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
