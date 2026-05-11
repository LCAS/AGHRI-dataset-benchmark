#!/usr/bin/env python3
"""
3D track visualiser for PCD point-cloud sequences.

Reads per-scene MOT3D CSV files produced by the AB3DMOT / SimpleTrack /
CenterPoint runners and overlays the predicted tracks on the corresponding
raw PCD frames for qualitative inspection.

MOT3D CSV format (produced by the trackers in this project):
    frame_id,track_id,x,y,z,l,w,h,yaw,score

Frame IDs in the CSV are 1-indexed and match the sorted order of PCD files
in --pcd-dir, which is the same ordering the trackers use when processing
scene detections.

Expected inputs
---------------
1. --pcd-dir : raw scene PCD directory, e.g.
       D:/AOC/datasets/agri-human-sensing/labelled_dataset/<scene>/sensor_data/lidar/
2. --tracks  : per-scene MOT3D CSV, e.g.
       3d-tracking/reports/runs/ab3dmot/pointpillars_aghri/<scene>.csv

Controls
--------
- N or Right Arrow : next frame
- B or Left Arrow  : previous frame
- R                : reset camera view
- Q or ESC         : quit

Open3D mouse controls:
    - Mouse wheel              : zoom in/out
    - Hold mouse wheel + drag  : pan the scene
    - Hold left click + drag   : rotate the scene
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageFont, ImageDraw

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def color_from_id(tid: int) -> np.ndarray:
    """Deterministic RGB colour from a track ID via golden-ratio hue spacing."""
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return np.array([r, g, b], dtype=float)


def bbox_lineset(x, y, z, l, w, h, yaw, color):
    """Build a wireframe 3D bounding box as an Open3D LineSet."""
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0
    corners = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = (R @ corners.T).T + np.array([x, y, z])
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def create_text_point_cloud(text: str, height: float = 0.6, font_path: str = FONT_PATH) -> o3d.geometry.PointCloud:
    """Rasterise a text label as a vertical point cloud for 3D overlay."""
    try:
        font = ImageFont.truetype(font_path, 64)
    except OSError:
        font = ImageFont.load_default()

    W, H = 512, 512
    img = Image.new("L", (W, H), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        return o3d.geometry.PointCloud()
    x, y = (W - w) // 2, (H - h) // 2
    draw.text((x - bbox[0], y - bbox[1]), text, fill=255, font=font)

    img_np = np.array(img)
    ys, xs = np.where(img_np > 0)
    if len(xs) == 0:
        return o3d.geometry.PointCloud()

    zs = -ys
    pts = np.stack([xs, np.zeros_like(xs), zs], axis=1).astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    current_h = pts[:, 2].max() - pts[:, 2].min()
    if current_h > 0:
        pts *= (height / current_h)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def list_pcd_files(pcd_dir: Path, ext: str = ".pcd") -> list[str]:
    """Return PCD filenames sorted lexicographically (matches tracker frame ordering)."""
    return [p.name for p in sorted(pcd_dir.glob(f"*{ext}"))]


def load_mot3d_csv(tracks_csv: Path, pcd_names: list[str]) -> dict[str, list]:
    """
    Load a MOT3D CSV and index tracks by PCD filename.

    Frame IDs in the CSV are 1-indexed and correspond to the sorted order of
    PCD files in the scene directory — the same ordering used by the trackers.
    """
    by_file: dict[str, list] = {}
    with tracks_csv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("frame_id"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            pcd_idx = frame_id - 1  # 1-indexed → 0-indexed
            if pcd_idx < 0 or pcd_idx >= len(pcd_names):
                continue
            fname = pcd_names[pcd_idx]
            by_file.setdefault(fname, []).append({
                "track_id": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "l": float(parts[5]),
                "w": float(parts[6]),
                "h": float(parts[7]),
                "theta": float(parts[8]),   # yaw column
                "score": float(parts[9]) if len(parts) > 9 else 1.0,
            })
    return by_file


def main():
    parser = argparse.ArgumentParser(
        description="Visualise MOT3D CSV tracker output over raw PCD frames.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python tools/vis_tracks_pcd.py \\\n"
            "    --pcd-dir D:/AOC/datasets/agri-human-sensing/labelled_dataset"
            "/<scene>/sensor_data/lidar \\\n"
            "    --tracks  3d-tracking/reports/runs/ab3dmot/pointpillars_aghri/<scene>.csv"
        ),
    )
    parser.add_argument("--pcd-dir", required=True, type=str,
                        help="Directory containing raw .pcd frames for one scene.")
    parser.add_argument("--tracks", required=True, type=str,
                        help="Per-scene MOT3D CSV file (frame_id,track_id,x,y,z,l,w,h,yaw,score).")
    parser.add_argument("--ext", default=".pcd", type=str,
                        help="Point-cloud file extension (default: .pcd).")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Rendered point size (default: 2.0).")
    args = parser.parse_args()

    pcd_dir = Path(args.pcd_dir).expanduser().resolve()
    tracks_csv = Path(args.tracks).expanduser().resolve()

    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"pcd_dir not found: {pcd_dir}")
    if not tracks_csv.is_file():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv}")

    pcd_names = list_pcd_files(pcd_dir, ext=args.ext)
    if not pcd_names:
        raise RuntimeError(f"No {args.ext} files found in {pcd_dir}")

    tracks_by_file = load_mot3d_csv(tracks_csv, pcd_names)

    print(f"[INFO] {len(pcd_names)} point-cloud frames in {pcd_dir}")
    print(f"[INFO] {len(tracks_by_file)} frames with tracks in {tracks_csv.name}")
    print("[INFO] Controls: N/→ next, B/← prev, R reset camera, Q/ESC quit")

    state = {
        "idx": 0,
        "pcd_names": pcd_names,
        "pcd_dir": pcd_dir,
        "tracks_by_file": tracks_by_file,
        "pcd_geom": None,
        "bbox_geoms": [],
        "text_geoms": [],
        "first": True,
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("3D Tracks Viewer", width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = float(args.point_size)

    def load_frame(idx: int, recenter: bool = False):
        fname = state["pcd_names"][idx]
        pcd_path = state["pcd_dir"] / fname

        cloud = o3d.io.read_point_cloud(str(pcd_path))
        if cloud.is_empty():
            print(f"[WARN] Empty point cloud: {pcd_path}")
            return
        cloud.paint_uniform_color([0.0, 0.0, 0.0])

        if state["pcd_geom"] is None:
            state["pcd_geom"] = cloud
            vis.add_geometry(state["pcd_geom"], reset_bounding_box=True)
        else:
            vis.remove_geometry(state["pcd_geom"], reset_bounding_box=False)
            state["pcd_geom"] = cloud
            vis.add_geometry(state["pcd_geom"], reset_bounding_box=False)

        for g in state["bbox_geoms"] + state["text_geoms"]:
            vis.remove_geometry(g, reset_bounding_box=False)
        state["bbox_geoms"] = []
        state["text_geoms"] = []

        tracks = state["tracks_by_file"].get(fname, [])
        sensor_origin = np.array([0.0, 0.0, 0.0])

        for t in tracks:
            tid = int(t["track_id"])
            color = color_from_id(tid)

            ls = bbox_lineset(t["x"], t["y"], t["z"], t["l"], t["w"], t["h"], t["theta"], color)
            state["bbox_geoms"].append(ls)
            vis.add_geometry(ls, reset_bounding_box=False)

            center = np.array([t["x"], t["y"], t["z"]], dtype=float)
            text_pcd = create_text_point_cloud(str(tid), height=0.6)
            text_pcd.paint_uniform_color(color)

            dx, dy = center[0] - sensor_origin[0], center[1] - sensor_origin[1]
            angle = np.arctan2(dy, dx) - np.pi / 2.0
            c, s = np.cos(angle), np.sin(angle)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
            text_pcd.rotate(Rz, center=(0.0, 0.0, 0.0))
            text_pcd.translate([center[0], center[1], center[2] + t["h"] / 2.0 + 0.8])

            state["text_geoms"].append(text_pcd)
            vis.add_geometry(text_pcd, reset_bounding_box=False)

        if state["first"] or recenter:
            ctr = vis.get_view_control()
            ctr.set_lookat(cloud.get_center())
            ctr.set_up([0, 0, 1])
            ctr.set_front([0, -1, 0])
            ctr.set_zoom(0.7)
            state["first"] = False

        vis.update_renderer()
        frame_id = idx + 1
        score_info = ""
        if tracks:
            scores = [t["score"] for t in tracks]
            score_info = f" (scores {min(scores):.2f}–{max(scores):.2f})"
        print(f"[INFO] frame {frame_id}/{len(pcd_names)}: {fname}  tracks={len(tracks)}{score_info}")

    def next_cb(v):
        if state["idx"] < len(state["pcd_names"]) - 1:
            state["idx"] += 1
            load_frame(state["idx"])
        return False

    def prev_cb(v):
        if state["idx"] > 0:
            state["idx"] -= 1
            load_frame(state["idx"])
        return False

    def reset_cb(v):
        state["first"] = True
        load_frame(state["idx"], recenter=True)
        return False

    def quit_cb(v):
        v.close()
        return False

    vis.register_key_callback(ord("N"), next_cb)
    vis.register_key_callback(262, next_cb)   # right arrow
    vis.register_key_callback(ord("B"), prev_cb)
    vis.register_key_callback(263, prev_cb)   # left arrow
    vis.register_key_callback(ord("R"), reset_cb)
    vis.register_key_callback(ord("Q"), quit_cb)
    vis.register_key_callback(256, quit_cb)   # ESC

    load_frame(state["idx"], recenter=True)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
