"""
Convert raw AGHRI LiDAR annotations to per-scene MOT3D CSV ground-truth files.

Reads from the original labelled_dataset annotation JSONs to preserve per-person
numeric identities (for example, '01' and '10'), which serve as stable track IDs.

Usage:
    python convert_gt_to_mot3d.py \\
        --raw-root D:/AOC/datasets/agri-human-sensing/labelled_dataset \\
        --splits-dir D:/AOC/datasets/agri-human-sensing/labelled_dataset/splits/default \\
        --out-dir D:/AOC/datasets/agri-human-sensing/mmdet3d_lidar_aghri/gt_mot3d \\
        --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _person_id_from_class(raw_class: object) -> int:
    """Convert a positive digit-only AGHRI identity into an integer track ID."""
    if not isinstance(raw_class, str):
        raise ValueError(
            f"Invalid AGHRI person identity {raw_class!r}; expected a string such as '01'."
        )
    identity = raw_class.strip()
    if not identity.isdigit() or int(identity) <= 0:
        raise ValueError(
            f"Invalid AGHRI person identity {raw_class!r}; expected a positive "
            "digit-only Class value such as '01' or '10'."
        )
    return int(identity)


def _normalize_box(raw_box) -> Optional[List[float]]:
    """Return [x, y, z, l, w, h, yaw] from annotation BoundingBoxes payload."""
    box = raw_box
    if (
        isinstance(box, list)
        and len(box) == 2
        and all(isinstance(item, list) and len(item) == 9 for item in box)
    ):
        box = box[0]
    if not isinstance(box, list) or len(box) != 9:
        return None
    x, y, z, dx, dy, dz, _rx, _ry, rz = [float(v) for v in box]
    return [x, y, z, dx, dy, dz, float(rz)]


def _load_scene_splits(splits_dir: Path, split: str) -> set:
    """Return the set of 'scene_name/pcd_stem' keys that belong to this split."""
    split_file = splits_dir / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    keys: set = set()
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lidar_rel = Path(line.split()[0])
            scene_name = lidar_rel.parts[0]
            pcd_stem = lidar_rel.stem
            keys.add(f"{scene_name}/{pcd_stem}")
    return keys


def convert_scene(
    scene_dir: Path,
    split_keys: set,
    out_dir: Path,
) -> Tuple[int, int]:
    """
    Convert one scene's lidar_ann.json to a MOT3D CSV.

    Returns (frame_count, annotation_count).
    """
    ann_path = scene_dir / "annotations" / "lidar_ann.json"
    if not ann_path.exists():
        return 0, 0

    ann_data = _load_json(ann_path)
    scene_name = scene_dir.name

    out_path = out_dir / f"{scene_name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["frame_id,track_id,x,y,z,l,w,h,yaw,score\n"]
    frame_id = 0
    total_anns = 0

    for frame_ann in ann_data:
        filename = str(frame_ann.get("File", ""))
        if not filename.endswith(".pcd"):
            continue
        pcd_stem = Path(filename).stem
        key = f"{scene_name}/{pcd_stem}"
        if key not in split_keys:
            continue

        frame_id += 1
        for label in frame_ann.get("Labels", []):
            track_id = _person_id_from_class(label.get("Class"))
            raw_box = label.get("BoundingBoxes", [])
            box = _normalize_box(raw_box)
            if box is None:
                continue
            x, y, z, l, w, h, yaw = box
            lines.append(
                f"{frame_id},{track_id},{x:.4f},{y:.4f},{z:.4f},"
                f"{l:.4f},{w:.4f},{h:.4f},{yaw:.4f},1.0000\n"
            )
            total_anns += 1

    if frame_id == 0:
        return 0, 0

    out_path.write_text("".join(lines), encoding="utf-8")
    return frame_id, total_anns


def convert_gt(
    raw_root: Path,
    splits_dir: Path,
    out_dir: Path,
    split: str = "val",
) -> None:
    split_keys = _load_scene_splits(splits_dir, split)
    scene_dirs = sorted(d for d in raw_root.iterdir() if d.is_dir() and d.name != "splits")

    total_frames = 0
    total_anns = 0
    total_scenes = 0

    for scene_dir in scene_dirs:
        frames, anns = convert_scene(scene_dir, split_keys, out_dir)
        if frames > 0:
            total_scenes += 1
            total_frames += frames
            total_anns += anns
            print(f"  {scene_dir.name}: {frames} frames, {anns} annotations")

    print(f"\nConverted {split} split: {total_scenes} scenes, {total_frames} frames, {total_anns} annotations")
    print(f"Output: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw AGHRI LiDAR annotations to per-scene MOT3D CSV GT files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--raw-root",
        default=r"D:\AOC\datasets\agri-human-sensing\labelled_dataset",
        help="Path to the raw labelled_dataset directory.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Directory with train.txt/val.txt/test.txt. Defaults to <raw-root>/splits/default.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"D:\AOC\datasets\agri-human-sensing\mmdet3d_lidar_aghri\gt_mot3d",
        help="Output directory for per-scene MOT3D CSV files.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else raw_root / "splits" / "default"
    convert_gt(raw_root, splits_dir, Path(args.out_dir), args.split)


if __name__ == "__main__":
    main()
