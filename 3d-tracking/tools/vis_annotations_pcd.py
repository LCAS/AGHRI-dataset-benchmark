#!/usr/bin/env python3
"""
Visualise 3D LiDAR annotations from a lidar_ann.json file over matching PCD frames.

This script is intended for quality control of annotation/detection inputs before
running a tracker. It opens a single Open3D window and updates the displayed point
cloud and 3D boxes when the user moves frame by frame.

Expected annotation format:
    [
      {
        "Timestamp": 1732099879.1882894,
        "File": "1732099879_188289505.pcd",
        "Labels": [
          {
            "Class": "01",
            "BoundingBoxes": [x, y, z, l, w, h, roll, pitch, yaw]
          }
        ]
      }
    ]

Fixed convention:
    BoundingBoxes = [x, y, z, l, w, h, roll, pitch, yaw]
    yaw is read from index 8 and interpreted as a rotation around the Z axis.

Viewer controls:
    - Right arrow / N : next frame
    - Left arrow / B  : previous frame
    - R               : reset camera view
    - Q / ESC         : exit

Open3D mouse controls:
    - Mouse wheel              : zoom in/out
    - Hold mouse wheel + drag  : pan the scene
    - Hold left click + drag   : rotate the scene

Notes:
    - PCD files are expected to be directly inside --pcd_dir.
    - Colours are assigned from the numeric person identity in `Class`.
    - The visualisation background is white and the point cloud is rendered in black.

Usage example:
    a) scenario 1
    python 3d-tracking/tools/vis_annotations_pcd.py `
        --json_path "D:/AOC/datasets/agri-human-sensing/labelled_dataset/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/annotations/lidar_ann.json" `
        --pcd_dir   "D:/AOC/datasets/agri-human-sensing/labelled_dataset/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/sensor_data/lidar"
    b) scenario 2
    python 3d-tracking/tools/vis_annotations_pcd.py `
        --json_path "D:/AOC/datasets/agri-human-sensing/labelled_dataset/in_straw_3pick_diff_st_10_24_2024_5_a_label/annotations/lidar_ann.json" `
        --pcd_dir   "D:/AOC/datasets/agri-human-sensing/labelled_dataset/in_straw_3pick_diff_st_10_24_2024_5_a_label/sensor_data/lidar"
    c) scenario 3
    python 3d-tracking/tools/vis_annotations_pcd.py `
        --json_path "D:/AOC/datasets/agri-human-sensing/labelled_dataset/out_vine_4swap+walk_st_ly_11_06_2024_2_label/annotations/lidar_ann.json" `
        --pcd_dir   "D:/AOC/datasets/agri-human-sensing/labelled_dataset/out_vine_4swap+walk_st_ly_11_06_2024_2_label/sensor_data/lidar"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the annotation viewer."""
    parser = argparse.ArgumentParser(
        description="Visualise 3D LiDAR annotations from lidar_ann.json over PCD frames."
    )
    parser.add_argument("--json_path", required=True, help="Path to lidar_ann.json.")
    parser.add_argument("--pcd_dir", required=True, help="Directory containing the PCD frames.")
    parser.add_argument("--start", type=int, default=0, help="Initial frame index.")
    parser.add_argument("--stop", type=int, default=-1, help="Exclusive final frame index (-1 = until the end).")
    parser.add_argument("--step", type=int, default=1, help="Frame step used when browsing the sequence.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size in the Open3D viewer.")
    parser.add_argument(
        "--only-classes",
        nargs="*",
        default=None,
        help="Optional identity filter. Example: --only-classes 01 02",
    )
    return parser.parse_args()


def load_frames(json_path: str) -> list[dict]:
    """Load and time-sort annotation frames from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    return sorted(frames, key=lambda x: x["Timestamp"])


def class_to_color(name: str) -> Tuple[float, float, float]:
    """
    Map annotation labels to stable high-contrast colours.

    The predefined mapping preserves colour consistency for released AGHRI person
    identities. Any other positive numeric identity receives a deterministic colour.
    """
    predefined = {
        "01": (1.0, 0.0, 0.0),
        "02": (0.0, 0.60, 0.0),
        "03": (0.0, 0.35, 1.0),
        "04": (1.0, 0.55, 0.0),
        "05": (0.65, 0.0, 0.85),
        "06": (0.0, 0.75, 0.75),
        "07": (0.85, 0.25, 0.25),
        "08": (0.25, 0.65, 0.95),
        "09": (0.70, 0.55, 0.10),
        "10": (0.45, 0.20, 0.75),
    }
    key = str(name).strip()
    if not key.isdigit() or int(key) <= 0:
        raise ValueError(
            f"Invalid AGHRI person identity {name!r}; expected a positive "
            "digit-only Class value such as '01' or '10'."
        )
    if key in predefined:
        return predefined[key]

    import colorsys

    seed = abs(hash(key)) % 10000
    h = (seed * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
    return (float(r), float(g), float(b))


def make_box_geometry(label: dict) -> tuple[o3d.geometry.LineSet, dict]:
    """Create an Open3D oriented 3D bounding box from a single annotation label."""
    bb = label["BoundingBoxes"]
    cls_name = label.get("Class", "unknown")
    if len(bb) < 9:
        raise ValueError(f"Invalid BoundingBoxes for {cls_name}: {bb}")

    center = np.array(bb[:3], dtype=float)
    extent = np.array(bb[3:6], dtype=float)
    roll = float(bb[6])
    pitch = float(bb[7])
    yaw = float(bb[8])

    # The current visualisation follows the project convention: only yaw is used
    # to orient the box around the vertical Z axis.
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, yaw))
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=rotation, extent=extent)
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color(class_to_color(cls_name))

    meta = {
        "class": cls_name,
        "center": center.tolist(),
        "extent": extent.tolist(),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }
    return lines, meta


def find_pcd_path(pcd_dir: Path, file_name: str) -> Path | None:
    """Return the PCD path expected from the frame's File field."""
    pcd_path = pcd_dir / file_name
    return pcd_path if pcd_path.exists() else None


def build_frame_geometries(
    frame: dict,
    args: argparse.Namespace,
    pcd_dir: Path,
) -> tuple[list[o3d.geometry.Geometry], list[dict], Path]:
    """Load the point cloud and build all geometries for a single frame."""
    pcd_path = find_pcd_path(pcd_dir, frame["File"])
    if pcd_path is None:
        raise FileNotFoundError(f"PCD file not found: {pcd_dir / frame['File']}")

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if pcd.is_empty():
        raise ValueError(f"Empty or unreadable PCD file: {pcd_path}")

    # Black points on a white background match the track visualisation style.
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    labels = frame.get("Labels", [])
    if args.only_classes is not None:
        selected_classes = set(args.only_classes)
        labels = [label for label in labels if label.get("Class") in selected_classes]

    geoms: list[o3d.geometry.Geometry] = [pcd]
    meta_rows: list[dict] = []

    for label in labels:
        box_geom, meta = make_box_geometry(label)
        geoms.append(box_geom)
        meta_rows.append(meta)

    return geoms, meta_rows, pcd_path


def main() -> None:
    args = parse_args()
    frames = load_frames(args.json_path)
    pcd_dir = Path(args.pcd_dir)

    if not pcd_dir.exists():
        raise FileNotFoundError(f"PCD directory does not exist: {pcd_dir}")
    if not pcd_dir.is_dir():
        raise NotADirectoryError(f"pcd_dir is not a directory: {pcd_dir}")

    stop = len(frames) if args.stop == -1 else min(args.stop, len(frames))
    indices = list(range(args.start, stop, args.step))
    if not indices:
        raise ValueError("No frames to visualise with the requested range.")

    print(f"Loaded frames: {len(frames)}")
    print("Assumed format: BoundingBoxes=[x,y,z,l,w,h,roll,pitch,yaw], yaw=bb[8]")
    print("Controls: right arrow/N next, left arrow/B previous, R reset, Q/ESC exit")

    state = {
        "frames": frames,
        "indices": indices,
        "pos": 0,
        "args": args,
        "pcd_dir": pcd_dir,
        "geoms": [],
        "first": True,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR Annotation Viewer", width=1400, height=900)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(args.point_size)
    render_opt.background_color = np.array([1.0, 1.0, 1.0])

    def show_current(reset_camera: bool = False):
        """Refresh the Open3D window with the current frame geometries."""
        frame_idx = state["indices"][state["pos"]]
        frame = state["frames"][frame_idx]

        for geom in state["geoms"]:
            vis.remove_geometry(geom, reset_bounding_box=False)
        state["geoms"] = []

        try:
            geoms, meta_rows, pcd_path = build_frame_geometries(
                frame=frame,
                args=state["args"],
                pcd_dir=state["pcd_dir"],
            )
        except Exception as exc:
            print(f"[WARN] frame {frame_idx}: {exc}")
            return

        for geom in geoms:
            vis.add_geometry(geom, reset_bounding_box=state["first"] or reset_camera)
        state["geoms"] = geoms

        print("\n" + "=" * 90)
        print(f"Frame idx          : {frame_idx}")
        print(f"Timestamp          : {frame['Timestamp']}")
        print(f"PCD                : {pcd_path}")
        print(f"Visible detections : {len(meta_rows)}")

        for idx, row in enumerate(meta_rows):
            center = np.round(row["center"], 3).tolist()
            extent = np.round(row["extent"], 3).tolist()
            print(
                f"  [{idx}] class={row['class']:<10} "
                f"center={center} extent={extent} "
                f"roll={row['roll']:.4f} pitch={row['pitch']:.4f} yaw={row['yaw']:.4f}"
            )

        # Keep the camera stable while browsing. Reset only on the first frame or
        # when the user explicitly requests it with R.
        if state["first"] or reset_camera:
            ctr = vis.get_view_control()
            try:
                bbox = geoms[0].get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
            except Exception:
                pass
            ctr.set_up([0, 0, 1])
            ctr.set_front([0, -1, 0])
            ctr.set_zoom(0.7)
            state["first"] = False

        vis.poll_events()
        vis.update_renderer()

    def next_cb(_):
        if state["pos"] < len(state["indices"]) - 1:
            state["pos"] += 1
            show_current(reset_camera=False)
        return False

    def prev_cb(_):
        if state["pos"] > 0:
            state["pos"] -= 1
            show_current(reset_camera=False)
        return False

    def reset_cb(_):
        show_current(reset_camera=True)
        return False

    def quit_cb(vis_obj):
        vis_obj.close()
        return False

    # Open3D key codes: 262 = right arrow, 263 = left arrow, 256 = ESC.
    vis.register_key_callback(ord("N"), next_cb)
    vis.register_key_callback(262, next_cb)
    vis.register_key_callback(ord("B"), prev_cb)
    vis.register_key_callback(263, prev_cb)
    vis.register_key_callback(ord("R"), reset_cb)
    vis.register_key_callback(ord("Q"), quit_cb)
    vis.register_key_callback(256, quit_cb)

    show_current(reset_camera=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
