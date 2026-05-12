"""
Convert raw AGHRI lidar_ann.json annotations into the two formats the 3D tracking
suite needs:

  1. Per-frame detection JSONs  (tracker input — identity stripped, score=1.0)
     Output: <out_det_dir>/<scene_name>/<pcd_stem>.json
     Format: list of {"x","y","z","l","w","h","yaw","score","label"} dicts

  2. Per-scene GT MOT3D CSV     (evaluation ground truth — identity preserved)
     Output: <out_gt_dir>/<scene_name>.csv
     Format: frame_id,track_id,x,y,z,l,w,h,yaw,score

Frame IDs in both outputs follow the same sort order (timestamp → alphabetical PCD
stem), so tracker frame_id 1 corresponds to GT frame_id 1 in the same scene.

Usage (standalone):
    python common/mot3d/gt_as_detections.py \\
        --scene-dir "D:/AOC/datasets/.../labelled_dataset/footpath1_..." \\
        --out-det-dir  3d-tracking/reports/runs/_gt_detections \\
        --out-gt-dir   3d-tracking/reports/runs/_gt_mot3d
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple


def _person_id_from_class(class_name: str) -> int:
    """Stable integer track ID from a label like 'human3'."""
    digits = "".join(c for c in class_name if c.isdigit())
    if digits:
        return int(digits)
    digest = hashlib.sha1(class_name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def _parse_box(raw_box) -> Optional[List[float]]:
    """Return [x, y, z, l, w, h, yaw] or None if malformed."""
    box = raw_box
    if (
        isinstance(box, list)
        and len(box) == 2
        and all(isinstance(item, list) and len(item) == 9 for item in box)
    ):
        box = box[0]
    if not isinstance(box, list) or len(box) < 9:
        return None
    x, y, z, l, w, h = [float(v) for v in box[:6]]
    yaw = float(box[8])
    return [x, y, z, l, w, h, yaw]


def convert_scene_to_detections(scene_dir: Path, out_dir: Path) -> int:
    """
    Write per-frame detection JSON files from one scene's lidar_ann.json.

    Identity is stripped — all persons become {"score": 1.0, "label": 0}.
    Files are named after PCD stems so they match future model detection exports.

    Returns the number of frames written.
    """
    ann_path = scene_dir / "annotations" / "lidar_ann.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    frames = sorted(
        json.loads(ann_path.read_text(encoding="utf-8")),
        key=lambda f: f["Timestamp"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for frame in frames:
        fname = str(frame.get("File", ""))
        if not fname.endswith(".pcd"):
            continue
        pcd_stem = Path(fname).stem

        detections = []
        for label in frame.get("Labels", []):
            cls = str(label.get("Class", "")).lower()
            if not cls.startswith("human"):
                continue
            box = _parse_box(label.get("BoundingBoxes", []))
            if box is None:
                continue
            x, y, z, l, w, h, yaw = box
            detections.append({
                "x": x, "y": y, "z": z,
                "l": l, "w": w, "h": h,
                "yaw": yaw, "score": 1.0, "label": 0,
            })

        out_path = out_dir / f"{pcd_stem}.json"
        out_path.write_text(json.dumps(detections), encoding="utf-8")
        count += 1

    return count


def convert_scene_to_gt_mot3d(scene_dir: Path, out_csv: Path) -> Tuple[int, int]:
    """
    Write a GT MOT3D CSV from one scene's lidar_ann.json.

    Identity IS preserved — class names like 'human1' become track_id=1.
    frame_id is the 1-indexed position in the timestamp-sorted frame list.

    Returns (frame_count, annotation_count).
    """
    ann_path = scene_dir / "annotations" / "lidar_ann.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    frames = sorted(
        json.loads(ann_path.read_text(encoding="utf-8")),
        key=lambda f: f["Timestamp"],
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    lines = ["frame_id,track_id,x,y,z,l,w,h,yaw,score\n"]
    frame_id = 0
    total_anns = 0

    for frame in frames:
        fname = str(frame.get("File", ""))
        if not fname.endswith(".pcd"):
            continue
        frame_id += 1

        for label in frame.get("Labels", []):
            cls = str(label.get("Class", "")).lower()
            if not cls.startswith("human"):
                continue
            box = _parse_box(label.get("BoundingBoxes", []))
            if box is None:
                continue
            tid = _person_id_from_class(cls)
            x, y, z, l, w, h, yaw = box
            lines.append(
                f"{frame_id},{tid},{x:.4f},{y:.4f},{z:.4f},"
                f"{l:.4f},{w:.4f},{h:.4f},{yaw:.4f},1.0000\n"
            )
            total_anns += 1

    out_csv.write_text("".join(lines), encoding="utf-8")
    return frame_id, total_anns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw lidar_ann.json to detection JSONs and GT MOT3D CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scene-dir", required=True, type=Path,
                        help="Raw labelled scene directory (contains annotations/ and sensor_data/).")
    parser.add_argument("--out-det-dir", required=True, type=Path,
                        help="Output directory for per-frame detection JSONs (one subdir per scene).")
    parser.add_argument("--out-gt-dir", required=True, type=Path,
                        help="Output directory for GT MOT3D CSV files.")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    scene_name = scene_dir.name

    det_out = args.out_det_dir / scene_name
    gt_out = args.out_gt_dir / f"{scene_name}.csv"

    frames = convert_scene_to_detections(scene_dir, det_out)
    print(f"[det] {scene_name}: {frames} detection frames → {det_out}")

    frame_count, ann_count = convert_scene_to_gt_mot3d(scene_dir, gt_out)
    print(f"[gt]  {scene_name}: {frame_count} frames, {ann_count} annotations → {gt_out}")


if __name__ == "__main__":
    main()
