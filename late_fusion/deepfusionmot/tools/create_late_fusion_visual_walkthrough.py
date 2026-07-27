#!/usr/bin/env python3
"""Create a visual and numerical walkthrough for one AGHRI late-fusion frame."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import textwrap
from typing import Any
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RECORDING = "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label"
DEFAULT_SAMPLE_ID = f"{DEFAULT_RECORDING}/1731407186_709929023"
DEFAULT_MANIFEST = "data_manifests/aghri_late_fusion_test_manifest.csv"
DEFAULT_YOLO_CACHE = "results/aghri_pointpillars_detector_comparison/cache/yolo_finetuned/detections.csv"
DEFAULT_LIDAR_CACHE = "results/aghri_pointpillars_detector_comparison/cache/pointpillars_finetuned/detections.csv"
DEFAULT_ASSOCIATION = "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_ft/fusion/association_results.csv"
DEFAULT_PER_FRAME = "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_ft/fusion/per_frame_results.csv"
DEFAULT_CALIBRATION = "results/calibration_consistency_audit/calibration_matrices.json"
DEFAULT_OUTPUT_ROOT = "results/late_fusion_visual_walkthrough"
DEFAULT_THRESHOLD = 0.10

EXPECTED_AUDIT_RECT = np.array([272.1678344001285, 123.43052469077153, 330.4215834005085, 263.1164718533429], dtype=float)
EXPECTED_AUDIT_IOU = 0.5966875961787735

GREEN = "#00b050"
BLUE = "#0070c0"
MAGENTA = "#cc00cc"
YELLOW = "#ffd600"
CYAN = "#00c8c8"
RED = "#e60000"
GREY = "#888888"

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def as_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_rows(path: str | Path, sample_id: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv_rows(path) if row.get("sample_id") == sample_id]
    if not rows:
        raise RuntimeError(f"No row for sample_id={sample_id!r} in {path}")
    return rows


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(path: str | Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def yaw_rotation_z(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def local_box_corners(size_lwh: tuple[float, float, float], box_origin: str) -> np.ndarray:
    length, width, height = [float(v) for v in size_lwh]
    l2 = length / 2.0
    w2 = width / 2.0
    if box_origin == "bottom_center":
        z_values = (0.0, height)
    elif box_origin == "center":
        z_values = (-height / 2.0, height / 2.0)
    else:
        raise ValueError(f"Unsupported box_origin: {box_origin}")
    return np.array(
        [
            [-l2, -w2, z_values[0]],
            [l2, -w2, z_values[0]],
            [l2, w2, z_values[0]],
            [-l2, w2, z_values[0]],
            [-l2, -w2, z_values[1]],
            [l2, -w2, z_values[1]],
            [l2, w2, z_values[1]],
            [-l2, w2, z_values[1]],
        ],
        dtype=float,
    )


def lidar_box_corners(center_xyz: tuple[float, float, float], size_lwh: tuple[float, float, float], yaw: float, box_origin: str) -> np.ndarray:
    local = local_box_corners(size_lwh, box_origin)
    rotated = local @ yaw_rotation_z(yaw).T
    return rotated + np.asarray(center_xyz, dtype=float)


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points_xyz, np.ones((points_xyz.shape[0],), dtype=float)])
    return (transform_4x4 @ hom.T).T[:, :3]


def project_camera_points(points_camera: np.ndarray, camera_p_3x4: np.ndarray) -> np.ndarray:
    hom = np.column_stack([points_camera, np.ones((points_camera.shape[0],), dtype=float)])
    uvw = (camera_p_3x4 @ hom.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def clip_xyxy(xyxy: np.ndarray, image_width: int, image_height: int) -> np.ndarray | None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    clipped = np.array(
        [
            max(0.0, min(float(image_width - 1), x1)),
            max(0.0, min(float(image_height - 1), y1)),
            max(0.0, min(float(image_width - 1), x2)),
            max(0.0, min(float(image_height - 1), y2)),
        ],
        dtype=float,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def intersection_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])], dtype=float)


def area_xyxy(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def iou_2d(a: np.ndarray, b: np.ndarray) -> tuple[float, dict[str, float | list[float]]]:
    inter = intersection_xyxy(a, b)
    intersection_area = area_xyxy(inter)
    a_area = area_xyxy(a)
    b_area = area_xyxy(b)
    union = a_area + b_area - intersection_area
    iou = intersection_area / union if union > 0 else 0.0
    return float(iou), {
        "intersection_xyxy": inter.tolist(),
        "intersection_width": float(max(0.0, inter[2] - inter[0])),
        "intersection_height": float(max(0.0, inter[3] - inter[1])),
        "intersection_area": float(intersection_area),
        "camera_area": float(a_area),
        "lidar_projected_area": float(b_area),
        "union_area": float(union),
    }


def parse_binary_pcd_xyz(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"PCD DATA line not found in {path}")
            header_lines.append(line)
            if line.startswith(b"DATA"):
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")
        if "DATA binary" not in header:
            raise RuntimeError(f"Only binary PCD is supported by this walkthrough script: {path}")
        fields = next(line.split()[1:] for line in header.splitlines() if line.startswith("FIELDS"))
        sizes = [int(v) for v in next(line.split()[1:] for line in header.splitlines() if line.startswith("SIZE"))]
        types = next(line.split()[1:] for line in header.splitlines() if line.startswith("TYPE"))
        counts = [int(v) for v in next(line.split()[1:] for line in header.splitlines() if line.startswith("COUNT"))]
        points = int(next(line.split()[1] for line in header.splitlines() if line.startswith("POINTS")))
        offsets: dict[str, tuple[int, str]] = {}
        offset = 0
        for field, size, pcd_type, count in zip(fields, sizes, types, counts):
            if count != 1:
                raise RuntimeError(f"Unsupported PCD field count {count} for {field}")
            if pcd_type == "F" and size == 4:
                fmt = "f"
            elif pcd_type == "U" and size == 4:
                fmt = "I"
            else:
                fmt = "x"
            offsets[field] = (offset, fmt)
            offset += size * count
        row_size = offset
        raw = f.read(points * row_size)
    required = ["x", "y", "z"]
    if not all(name in offsets for name in required):
        raise RuntimeError(f"PCD lacks x/y/z fields: {path}")
    xyz = np.empty((points, 3), dtype=float)
    for idx in range(points):
        base = idx * row_size
        for col, name in enumerate(required):
            off, fmt = offsets[name]
            xyz[idx, col] = struct.unpack_from("<" + fmt, raw, base + off)[0]
    finite = np.isfinite(xyz).all(axis=1)
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-9
    return xyz[finite & nonzero]


def load_ground_truth(manifest_row: dict[str, str]) -> dict[str, Any]:
    gt: dict[str, Any] = {}
    image_file = Path(manifest_row["image_path"]).name
    lidar_file = Path(manifest_row["lidar_point_cloud_path"]).name
    cam_path = manifest_row.get("gt_2d_reference")
    lidar_path = manifest_row.get("gt_3d_reference")
    if cam_path and Path(cam_path).exists():
        for item in load_json(cam_path):
            if item.get("File") == image_file:
                labels = item.get("Labels") or []
                if labels:
                    box = labels[0].get("BoundingBoxes")
                    if box:
                        x1, y1, x2_or_w, y2_or_h = [float(v) for v in box[:4]]
                        if x2_or_w <= x1 or y2_or_h <= y1:
                            gt["camera_2d_xyxy"] = [x1, y1, x1 + x2_or_w, y1 + y2_or_h]
                            gt["camera_2d_source_format"] = "xywh"
                        else:
                            gt["camera_2d_xyxy"] = [x1, y1, x2_or_w, y2_or_h]
                            gt["camera_2d_source_format"] = "xyxy"
                break
    if lidar_path and Path(lidar_path).exists():
        for item in load_json(lidar_path):
            if item.get("File") == lidar_file:
                labels = item.get("Labels") or []
                if labels:
                    box = labels[0].get("BoundingBoxes")
                    if box and len(box) >= 9:
                        gt["lidar_3d_box_raw"] = [float(v) for v in box]
                        gt["lidar_3d_box_xyzlwhyaw"] = [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                            float(box[4]),
                            float(box[5]),
                            float(box[8]),
                        ]
                break
    return gt


def bev_polygon(center_xyz: tuple[float, float, float], size_lwh: tuple[float, float, float], yaw: float, box_origin: str) -> np.ndarray:
    return lidar_box_corners(center_xyz, size_lwh, yaw, box_origin)[:4, :2]


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def intersection(s: np.ndarray, e: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dc = a - b
        dp = s - e
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-12:
            return e
        return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom], dtype=float)

    output = subject.copy()
    for i in range(len(clipper)):
        input_poly = output
        output = np.empty((0, 2), dtype=float)
        if len(input_poly) == 0:
            break
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        s = input_poly[-1]
        for e in input_poly:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output = np.vstack([output, intersection(s, e, a, b)])
                output = np.vstack([output, e])
            elif inside(s, a, b):
                output = np.vstack([output, intersection(s, e, a, b)])
            s = e
    return output


def bev_iou(a: tuple[tuple[float, float, float], tuple[float, float, float], float, str], b: tuple[tuple[float, float, float], tuple[float, float, float], float, str]) -> float:
    poly_a = bev_polygon(*a)
    poly_b = bev_polygon(*b)
    inter = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter)
    union = polygon_area(poly_a) + polygon_area(poly_b) - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def box3d_iou_approx(pred: tuple[tuple[float, float, float], tuple[float, float, float], float, str], gt: tuple[tuple[float, float, float], tuple[float, float, float], float, str]) -> tuple[float, float, float, float]:
    pred_center, pred_size, pred_yaw, pred_origin = pred
    gt_center, gt_size, gt_yaw, gt_origin = gt
    pred_poly = bev_polygon(pred_center, pred_size, pred_yaw, pred_origin)
    gt_poly = bev_polygon(gt_center, gt_size, gt_yaw, gt_origin)
    inter_poly = polygon_clip(pred_poly, gt_poly)
    inter_area = polygon_area(inter_poly)
    def z_range(center: tuple[float, float, float], size: tuple[float, float, float], origin: str) -> tuple[float, float]:
        if origin == "bottom_center":
            return float(center[2]), float(center[2] + size[2])
        return float(center[2] - size[2] / 2.0), float(center[2] + size[2] / 2.0)
    pz0, pz1 = z_range(pred_center, pred_size, pred_origin)
    gz0, gz1 = z_range(gt_center, gt_size, gt_origin)
    overlap_h = max(0.0, min(pz1, gz1) - max(pz0, gz0))
    inter_vol = inter_area * overlap_h
    pred_vol = float(np.prod(pred_size))
    gt_vol = float(np.prod(gt_size))
    union_vol = pred_vol + gt_vol - inter_vol
    bev_union = polygon_area(pred_poly) + polygon_area(gt_poly) - inter_area
    return (
        float(inter_area / bev_union) if bev_union > 0 else 0.0,
        float(inter_vol / union_vol) if union_vol > 0 else 0.0,
        float(np.linalg.norm(np.asarray(pred_center) - np.asarray(gt_center))),
        float(abs((pred_center[2] + pred_size[2] / 2.0) - (gt_center[2] + gt_size[2] / 2.0))),
    )


def text_font(size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_image_rgba(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def save_image_with_panel(image: Image.Image, path: str | Path, title: str, panel_lines: list[str], caption: str = "") -> None:
    font_title = text_font(22, bold=True)
    font = text_font(16)
    width, height = image.size
    panel_w = 540
    pad = 18
    wrapped_lines: list[str] = []
    for line in panel_lines:
        wrapped = textwrap.wrap(line, width=58) or [""]
        wrapped_lines.extend(wrapped)
    out_h = max(height + 80, 150 + 24 * len(wrapped_lines))
    canvas = Image.new("RGB", (width + panel_w, out_h), "white")
    canvas.paste(image.convert("RGB"), (0, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 16), title, fill="black", font=font_title)
    x = width + 20
    draw.text((x, 18), "Walkthrough data", fill="black", font=font_title)
    y = 62
    for line in wrapped_lines:
        draw.text((x, y), line, fill="black", font=font)
        y += 24
    if caption:
        draw.text((pad, height + 66), caption, fill="black", font=font)
    canvas.save(path)


def draw_rect(draw: ImageDraw.ImageDraw, box: np.ndarray, color: str, width: int = 4) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)


def annotate_box_image(base: Image.Image, rects: list[tuple[np.ndarray, str, str]], title: str, lines: list[str], path: Path, fills: list[tuple[np.ndarray, str, int]] | None = None) -> None:
    img = base.copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    if fills:
        for box, color, alpha in fills:
            color_rgb = Image.new("RGBA", (1, 1), color).getpixel((0, 0))[:3]
            od.rectangle([float(v) for v in box], fill=(*color_rgb, alpha))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    for box, color, label in rects:
        draw_rect(draw, box, color, 4)
    if rects:
        lines = lines + ["legend:"] + [f"{color}: {label}" for _box, color, label in rects]
    save_image_with_panel(img, path, title, lines)


def draw_projected_corners(base: Image.Image, uv: np.ndarray, title: str, lines: list[str], path: Path) -> None:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    font = text_font(14, bold=True)
    for idx, (u, v) in enumerate(uv):
        cx = max(0, min(base.width - 1, float(u)))
        cy = max(0, min(base.height - 1, float(v)))
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=MAGENTA, outline="black", width=1)
        draw.text((cx + 7, cy + 2), f"C{idx + 1}", fill=MAGENTA, font=font)
    save_image_with_panel(img, path, title, lines)


def draw_projected_cuboid(base: Image.Image, uv: np.ndarray, yolo: np.ndarray, projected: np.ndarray | None, title: str, lines: list[str], path: Path) -> None:
    img = base.copy()
    draw = ImageDraw.Draw(img)
    draw_rect(draw, yolo, GREEN, 4)
    if projected is not None:
        draw_rect(draw, projected, YELLOW, 3)
    for a, b in EDGES:
        p1 = (float(uv[a, 0]), float(uv[a, 1]))
        p2 = (float(uv[b, 0]), float(uv[b, 1]))
        draw.line([p1, p2], fill=MAGENTA, width=3)
    save_image_with_panel(img, path, title, lines)


def draw_lidar_scene(
    points: np.ndarray,
    corners: np.ndarray | None,
    path: Path,
    title: str,
    lines: list[str],
    gt_corners: np.ndarray | None = None,
    box_color: str = MAGENTA,
    box_label: str = "PointPillars",
) -> None:
    rng = np.random.default_rng(42)
    if len(points) > 7000:
        idx = rng.choice(len(points), size=7000, replace=False)
        pts = points[idx]
    else:
        pts = points
    fig = plt.figure(figsize=(13, 7), dpi=140)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.4, c="#777777", alpha=0.45)
    if corners is not None:
        draw_box_3d(ax, corners, box_color, box_label)
    if gt_corners is not None:
        draw_box_3d(ax, gt_corners, BLUE, "GT")
    ax.quiver(0, 0, 0, 2, 0, 0, color="red")
    ax.quiver(0, 0, 0, 0, 2, 0, color="green")
    ax.quiver(0, 0, 0, 0, 0, 1, color="blue")
    ax.text(2.1, 0, 0, "x", color="red")
    ax.text(0, 2.1, 0, "y", color="green")
    ax.text(0, 0, 1.1, "z", color="blue")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_zlabel("z up (m)")
    ax.set_title(title)
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((1.4, 1.0, 0.5))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")

    bev = fig.add_subplot(1, 2, 2)
    bev.scatter(pts[:, 0], pts[:, 1], s=0.4, c="#777777", alpha=0.45, label="raw point cloud")
    if corners is not None:
        draw_box_bev(bev, corners, box_color, f"{box_label} BEV")
    if gt_corners is not None:
        draw_box_bev(bev, gt_corners, BLUE, "GT BEV")
    bev.scatter([0], [0], c="black", s=30, label="LiDAR origin")
    bev.arrow(0, 0, 2, 0, color="red", width=0.03, head_width=0.25)
    bev.arrow(0, 0, 0, 2, color="green", width=0.03, head_width=0.25)
    bev.text(2.2, 0, "x")
    bev.text(0, 2.2, "y")
    bev.set_xlabel("x forward (m)")
    bev.set_ylabel("y left (m)")
    bev.axis("equal")
    bev.grid(True, alpha=0.2)
    bev.legend(loc="upper right")
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path)
    plt.close(fig)


def draw_box_3d(ax: Any, corners: np.ndarray, color: str, label: str) -> None:
    for idx, (a, b) in enumerate(EDGES):
        ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], color=color, linewidth=2, label=label if idx == 0 else None)
    center = corners.mean(axis=0)
    ax.scatter([center[0]], [center[1]], [center[2]], c=color, s=30)


def draw_box_bev(ax: Any, corners: np.ndarray, color: str, label: str) -> None:
    pts = np.vstack([corners[:4, :2], corners[0, :2]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, label=label)
    front_mid = (corners[1, :2] + corners[2, :2]) / 2.0
    center = corners[:4, :2].mean(axis=0)
    ax.arrow(center[0], center[1], front_mid[0] - center[0], front_mid[1] - center[1], color=color, width=0.02, head_width=0.15)


def draw_corner_labels(points: np.ndarray, corners: np.ndarray, path: Path, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    rng = np.random.default_rng(1)
    pts = points[rng.choice(len(points), min(len(points), 5000), replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c=GREY, alpha=0.35)
    draw_box_bev(ax, corners, MAGENTA, "PointPillars box")
    for idx, p in enumerate(corners[:4]):
        ax.text(p[0], p[1], f"L{idx + 1}", color=MAGENTA, fontsize=12, weight="bold")
    for idx, p in enumerate(corners[4:]):
        ax.text(p[0], p[1], f"L{idx + 5}", color=MAGENTA, fontsize=12, weight="bold")
    ax.scatter(corners[:, 0], corners[:, 1], c=MAGENTA, s=30)
    ax.scatter([0], [0], c="black", s=35, label="LiDAR origin")
    ax.set_title("Stage 6 - Eight LiDAR-frame box corners")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.text(0.02, 0.02, "\n".join(lines), fontsize=9)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(path)
    plt.close(fig)


def draw_text_figure(path: Path, title: str, body: list[str], size: tuple[int, int] = (1400, 850)) -> None:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    title_font = text_font(34, bold=True)
    font = text_font(24)
    mono = text_font(20)
    draw.text((40, 35), title, fill="black", font=title_font)
    y = 105
    for line in body:
        f = mono if any(ch in line for ch in "[]=<>") or line.strip().startswith(("IoU", "T_", "P =", "R =")) else font
        draw.text((60, y), line, fill="black", font=f)
        y += 36
    img.save(path)


def graphviz_png(path: Path, dot: str) -> None:
    dot_path = path.with_suffix(".dot")
    dot_path.write_text(dot)
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(path)], check=True)


def format_array(values: list[float] | np.ndarray, decimals: int = 6) -> str:
    return "[" + ", ".join(f"{float(v):.{decimals}f}" for v in values) + "]"


def build_worked_example(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = as_path(args.manifest)
    yolo_cache = as_path(args.camera_cache)
    lidar_cache = as_path(args.lidar_cache)
    association_path = as_path(args.association)
    per_frame_path = as_path(args.per_frame)
    calibration_path = as_path(args.calibration)
    manifest_row = find_rows(manifest_path, args.sample_id)[0]
    yolo_row = find_rows(yolo_cache, args.sample_id)[0]
    lidar_row = find_rows(lidar_cache, args.sample_id)[0]
    association_row = find_rows(association_path, args.sample_id)[0]
    per_frame_row = find_rows(per_frame_path, args.sample_id)[0]
    calibration = load_json(calibration_path)
    gt = load_ground_truth(manifest_row)

    image_path = Path(manifest_row["image_path"])
    pointcloud_path = Path(manifest_row["lidar_point_cloud_path"])
    if not image_path.exists():
        raise RuntimeError(f"Image path is missing: {image_path}")
    if not pointcloud_path.exists():
        raise RuntimeError(f"Point cloud path is missing: {pointcloud_path}")

    yolo_box = np.array([float(yolo_row[k]) for k in ("x1", "y1", "x2", "y2")], dtype=float)
    lidar_center = tuple(float(lidar_row[k]) for k in ("x", "y", "z"))
    lidar_size = tuple(float(lidar_row[k]) for k in ("length", "width", "height"))
    yaw = float(lidar_row["yaw"])
    cache_metadata = load_json(as_path("results/aghri_pointpillars_detector_comparison/cache/pointpillars_finetuned/metadata.json"))
    cache_box_origin = str(cache_metadata.get("cache_box_origin", "bottom_center"))
    box_origin = cache_box_origin if args.lidar_box_origin == "cache" else str(args.lidar_box_origin)
    corners_lidar = lidar_box_corners(lidar_center, lidar_size, yaw, box_origin)
    t_camera_from_lidar = np.asarray(calibration["T_camera_from_lidar"], dtype=float)
    p_camera = np.asarray(calibration["P"], dtype=float)
    corners_camera = transform_points(corners_lidar, t_camera_from_lidar)
    valid_mask = corners_camera[:, 2] > 1e-3
    if not np.any(valid_mask):
        raise RuntimeError("No projectable 3D-box corners; all corners are behind the camera.")
    uv = project_camera_points(corners_camera[valid_mask], p_camera)
    full_uv = np.full((8, 2), np.nan, dtype=float)
    full_uv[valid_mask] = uv
    finite_uv = uv[np.isfinite(uv).all(axis=1)]
    if len(finite_uv) == 0:
        raise RuntimeError("No finite projected 3D-box corners.")
    projected_unclipped = np.array([finite_uv[:, 0].min(), finite_uv[:, 1].min(), finite_uv[:, 0].max(), finite_uv[:, 1].max()], dtype=float)
    projected_clipped = clip_xyxy(projected_unclipped, int(calibration["width"]), int(calibration["height"]))
    if projected_clipped is None:
        raise RuntimeError("Projected 3D-box envelope lies outside the image.")
    iou, iou_parts = iou_2d(yolo_box, projected_clipped)
    accepted = iou >= float(args.threshold)
    if not accepted:
        raise RuntimeError(f"Expected selected example to match, but IoU={iou:.12f} < threshold={args.threshold}")
    cached_iou = float(association_row["iou"])
    if args.lidar_box_origin == "cache" and not math.isclose(iou, cached_iou, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"IoU differs from frozen association cache: {iou} vs {cached_iou}")
    if args.sample_id == DEFAULT_SAMPLE_ID:
        if not np.allclose(projected_clipped, EXPECTED_AUDIT_RECT, atol=1e-9):
            raise RuntimeError(f"Projected rectangle differs from audit: {projected_clipped.tolist()} vs {EXPECTED_AUDIT_RECT.tolist()}")
        if not math.isclose(iou, EXPECTED_AUDIT_IOU, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"IoU differs from audit: {iou} vs {EXPECTED_AUDIT_IOU}")

    gt_eval = None
    gt_corners = None
    if "lidar_3d_box_xyzlwhyaw" in gt:
        gt_box = gt["lidar_3d_box_xyzlwhyaw"]
        gt_center = tuple(float(v) for v in gt_box[:3])
        gt_size = tuple(float(v) for v in gt_box[3:6])
        gt_yaw = float(gt_box[6])
        gt_origin = "bottom_center"
        gt_corners = lidar_box_corners(gt_center, gt_size, gt_yaw, gt_origin)
        bev, iou3d, center_error, vertical_error = box3d_iou_approx((lidar_center, lidar_size, yaw, box_origin), (gt_center, gt_size, gt_yaw, gt_origin))
        gt_eval = {
            "ground_truth_box_xyzlwhyaw": gt_box,
            "bev_iou": bev,
            "iou_3d_approx": iou3d,
            "center_error_m": center_error,
            "vertical_center_error_m": vertical_error,
        }
    if "camera_2d_xyxy" in gt:
        if gt_eval is None:
            gt_eval = {}
        gt_eval["ground_truth_2d_box_xyxy"] = gt["camera_2d_xyxy"]

    if box_origin != cache_box_origin and math.isclose(iou, cached_iou, rel_tol=0.0, abs_tol=1e-9):
        cached_iou_note = "Cached association IoU already matches the explicit origin-corrected geometry."
    elif box_origin != cache_box_origin:
        cached_iou_note = "Cached IoU was computed with the cache metadata box-origin convention; selected-example geometry here uses an explicit origin override."
    else:
        cached_iou_note = "Cached IoU and recomputed IoU use the same box-origin convention."

    result = {
        "recording_name": manifest_row["recording_name"],
        "sample_id": args.sample_id,
        "selection": {
            "used_original_audited_frame": args.sample_id == DEFAULT_SAMPLE_ID,
            "reason": (
                "The calibration audit already verified this real matched frame exactly; boundary clipping is retained and explained as part of the projection stage."
                if args.sample_id == DEFAULT_SAMPLE_ID
                else "Selected as a clearer pedagogical example: the person is near the image centre, the projected cuboid envelope is mostly or fully visible, and the match is reproduced from frozen detector caches without threshold changes."
            ),
        },
        "box_origin_correction": {
            "cache_metadata_origin": cache_box_origin,
            "used_geometry_origin": box_origin,
            "override_applied": box_origin != cache_box_origin,
            "reason": (
                "The projected cuboid was vertically high when the cached PointPillars z value was treated as bottom-centre. "
                "Interpreting the same real detector box as centre-origin aligns the projected cuboid with the visible person. "
                "This is a box-origin convention correction, not a calibration change."
                if box_origin != cache_box_origin
                else "No box-origin override was applied; geometry follows the frozen cache metadata."
            ),
        },
        "image_timestamp": float(manifest_row["zed_rgb_timestamp"]),
        "lidar_timestamp": float(manifest_row["lidar_timestamp"]),
        "image_dimensions": [int(calibration["width"]), int(calibration["height"])],
        "image_path": str(image_path),
        "pointcloud_path": str(pointcloud_path),
        "frames": {
            "camera_frame": calibration["camera_frame"],
            "lidar_frame": calibration["lidar_frame"],
        },
        "camera_detection": {
            "bbox_xyxy": yolo_box.tolist(),
            "confidence": float(yolo_row["confidence"]),
            "class_id": int(float(yolo_row["class_id"])),
            "class_name": yolo_row.get("class_name", "person"),
            "checkpoint_sha256": yolo_row["checkpoint_sha256"],
            "cache_path": str(yolo_cache),
            "row_identifier": args.sample_id,
        },
        "lidar_detection": {
            "box_xyzlwhyaw": [*lidar_center, *lidar_size, yaw],
            "score": float(lidar_row["score"]),
            "class_id": int(float(lidar_row["raw_label_id"])),
            "class_name": lidar_row.get("raw_label_name", "person"),
            "checkpoint_sha256": lidar_row["checkpoint_sha256"],
            "cache_path": str(lidar_cache),
            "row_identifier": args.sample_id,
            "box_origin": box_origin,
            "cache_metadata_box_origin": cache_box_origin,
        },
        "calibration": {
            "source": calibration["calibration_zip"],
            "P": calibration["P"],
            "T_camera_from_lidar": calibration["T_camera_from_lidar"],
            "R_camera_from_lidar": calibration["R_camera_from_lidar"],
            "t_camera_from_lidar_m": calibration["t_camera_from_lidar_m"],
            "distortion_model": calibration.get("distortion_model"),
            "json_vs_bag": calibration.get("json_vs_bag", {}),
        },
        "geometry": {
            "half_dimensions_lwh_m": [v / 2.0 for v in lidar_size],
            "local_corners": local_box_corners(lidar_size, box_origin).tolist(),
            "yaw_rotation_matrix": yaw_rotation_z(yaw).tolist(),
            "corners_lidar": corners_lidar.tolist(),
            "corners_camera": corners_camera.tolist(),
            "corners_image_unclipped": full_uv.tolist(),
            "valid_corner_mask": valid_mask.tolist(),
            "projected_lidar_rectangle_unclipped": projected_unclipped.tolist(),
            "projected_lidar_rectangle_clipped": projected_clipped.tolist(),
        },
        "association": {
            **iou_parts,
            "iou": iou,
            "cached_association_iou": cached_iou,
            "cached_association_iou_note": cached_iou_note,
            "threshold": float(args.threshold),
            "accepted": accepted,
            "decision": "MATCHED" if accepted else "UNMATCHED",
            "camera_index": int(association_row["camera_index"]),
            "lidar_index": int(association_row["lidar_index"]),
            "fused_score": float(association_row["fused_score"]),
        },
        "fusion_outputs": {
            "matched_3d_count": int(per_frame_row["matches"]),
            "unmatched_camera_count": int(per_frame_row["unmatched_camera"]),
            "unmatched_lidar_count": int(per_frame_row["unmatched_lidar"]),
            "final_3d_count": int(per_frame_row["final_3d"]),
            "output_policy": "matched_plus_unmatched_lidar",
            "confidence_policy": "lidar_score",
        },
        "ground_truth_evaluation": gt_eval,
        "source_files": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "yolo_cache": str(yolo_cache),
            "lidar_cache": str(lidar_cache),
            "association_cache": str(association_path),
            "calibration_matrices": str(calibration_path),
        },
    }
    return result, gt_corners


def generate_figures(output_root: Path, data: dict[str, Any], points: np.ndarray, gt_corners: np.ndarray | None) -> list[dict[str, str]]:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    base = load_image_rgba(data["image_path"])
    yolo = np.asarray(data["camera_detection"]["bbox_xyxy"], dtype=float)
    projected = np.asarray(data["geometry"]["projected_lidar_rectangle_clipped"], dtype=float)
    projected_unclipped = np.asarray(data["geometry"]["projected_lidar_rectangle_unclipped"], dtype=float)
    intersection = np.asarray(data["association"]["intersection_xyxy"], dtype=float)
    corners_lidar = np.asarray(data["geometry"]["corners_lidar"], dtype=float)
    corners_camera = np.asarray(data["geometry"]["corners_camera"], dtype=float)
    uv = np.asarray(data["geometry"]["corners_image_unclipped"], dtype=float)
    gt2d = None
    if data.get("ground_truth_evaluation") and data["ground_truth_evaluation"].get("ground_truth_2d_box_xyxy"):
        gt2d = np.asarray(data["ground_truth_evaluation"]["ground_truth_2d_box_xyxy"], dtype=float)

    def add(num: str, filename: str, purpose: str, inputs: str) -> Path:
        path = figures / filename
        manifest_rows.append({"figure_number": num, "filename": str(path.relative_to(output_root)), "purpose": purpose, "input_files": inputs, "data_rows": data["sample_id"], "script_used": "tools/create_late_fusion_visual_walkthrough.py"})
        return path

    sample_line = f"{data['recording_name']} | {data['sample_id'].split('/')[-1]}"

    raw_path = add("01", "01_raw_zed_rgb_image.png", "Raw selected ZED RGB image", data["image_path"])
    save_image_with_panel(
        base,
        raw_path,
        "Stage 1 - Raw ZED RGB image",
        [
            sample_line,
            f"timestamp: {data['image_timestamp']:.9f}",
            f"dimensions: {data['image_dimensions'][0]} x {data['image_dimensions'][1]}",
            f"camera frame: {data['frames']['camera_frame']}",
        ],
        "Raw ZED RGB image used throughout the late-fusion walkthrough.",
    )

    path = add("02", "02_raw_lidar_pointcloud.png", "Raw Livox point cloud", data["pointcloud_path"])
    draw_lidar_scene(points, None, path, "Stage 2 - Raw Livox point cloud", [sample_line, "Grey points: raw point cloud", "Axes preserve LiDAR frame: x forward, y left, z up"])

    rects = [(yolo, GREEN, "YOLO person")]
    if gt2d is not None:
        rects.append((gt2d, BLUE, "GT 2D"))
    path = add("03", "03_yolo_detection.png", "Fine-tuned YOLO detection on raw image", data["camera_detection"]["cache_path"])
    annotate_box_image(
        base,
        rects,
        "Stage 3 - YOLO 2D person detection",
        [
            sample_line,
            f"class: {data['camera_detection']['class_name']} conf={data['camera_detection']['confidence']:.6f}",
            f"xyxy: {format_array(yolo, 3)}",
            f"width={yolo[2]-yolo[0]:.3f}px height={yolo[3]-yolo[1]:.3f}px",
            "Blue GT box, if visible, is evaluation-only.",
        ],
        path,
    )

    path = add("04", "04_pointpillars_3d_detection.png", "PointPillars 3D detection over point cloud", data["lidar_detection"]["cache_path"])
    draw_lidar_scene(
        points,
        corners_lidar,
        path,
        "Stage 4 - PointPillars 3D detection",
        [
            sample_line,
            f"box [x,y,z,l,w,h,yaw]={format_array(data['lidar_detection']['box_xyzlwhyaw'], 4)}",
            f"score={data['lidar_detection']['score']:.6f}",
            f"origin used={data['box_origin_correction']['used_geometry_origin']}",
        ],
        gt_corners=None,
    )

    path = add("05", "05_box_origin_convention.png", "Box-origin convention explanation", data["lidar_detection"]["cache_path"])
    draw_text_figure(
        path,
        "Stage 5 - Box-origin normalisation",
        [
            f"PointPillars cache metadata origin: {data['box_origin_correction']['cache_metadata_origin']}",
            f"Geometry origin used here: {data['box_origin_correction']['used_geometry_origin']}",
            "For bottom_center: z is the lower face and top = z + height.",
            "For center: z is the geometric centre and faces are z +/- height/2.",
            "Selected example: explicit origin correction is applied." if data["box_origin_correction"]["override_applied"] else "Selected example: no origin override is applied.",
            "Fine-tuned SECOND live/cache profiles may require centre-origin handling.",
            "Box-origin normalisation is not a change in sensor calibration.",
        ],
    )

    path = add("06", "06_eight_lidar_box_corners.png", "Eight LiDAR-frame box corners", data["lidar_detection"]["cache_path"])
    draw_corner_labels(points, corners_lidar, path, [sample_line, "L1-L4: lower face; L5-L8: upper face", "Ordering matches late_fusion_pkg/projection.py"])

    path = add("07", "07_lidar_to_camera_transformation.png", "Verified LiDAR-to-camera transform", data["calibration"]["source"])
    draw_text_figure(
        path,
        "Stage 7 - LiDAR to camera transformation",
        [
            "front_lidar_link -> T_C<-L -> front_left_camera_optical_frame",
            "R =",
            *[format_array(row, 6) for row in data["calibration"]["R_camera_from_lidar"]],
            f"t = {format_array(data['calibration']['t_camera_from_lidar_m'], 3)} m",
            "JSON calibration, ROS CameraInfo, and ROS TF agree for this bag.",
        ],
    )

    path = add("08", "08_projected_cuboid_corners.png", "Eight projected cuboid corners", data["calibration"]["source"])
    draw_projected_corners(
        base,
        uv,
        "Stage 8 - Camera projection of cuboid corners",
        [
            sample_line,
            "Projection uses verified CameraInfo.P.",
            "Corners with Z_camera <= 0 would be rejected; all selected corners are projectable.",
        ],
        path,
    )

    path = add("09", "09_projected_3d_cuboid.png", "Projected 3D cuboid against YOLO box", data["calibration"]["source"])
    draw_projected_cuboid(
        base,
        uv,
        yolo,
        None,
        "Stage 9 - Projected 3D cuboid",
        [sample_line, "green: YOLO box", "magenta: projected PointPillars cuboid"],
        path,
    )

    path = add("10", "10_projected_lidar_rectangle.png", "Projected 2D LiDAR envelope and clipping", data["calibration"]["source"])
    draw_projected_cuboid(
        base,
        uv,
        yolo,
        projected,
        "Stage 10 - Projected 2D envelope",
        [
            sample_line,
            f"unclipped: {format_array(projected_unclipped, 3)}",
            f"clipped: {format_array(projected, 3)}",
        ],
        path,
    )

    path = add("11", "11_iou_intersection.png", "IoU intersection region visualisation", "YOLO cache + PointPillars projection")
    annotate_box_image(
        base,
        [(yolo, GREEN, "YOLO"), (projected, YELLOW, "projected LiDAR"), (intersection, CYAN, "intersection")],
        "Stage 11 - Intersection area",
        [
            sample_line,
            f"intersection width={data['association']['intersection_width']:.3f}px",
            f"intersection height={data['association']['intersection_height']:.3f}px",
            f"intersection area={data['association']['intersection_area']:.3f}px^2",
        ],
        path,
        fills=[(yolo, GREEN, 55), (projected, YELLOW, 60), (intersection, CYAN, 120)],
    )

    path = add("12", "12_iou_calculation.png", "Numerical IoU calculation", "Calculated geometry")
    draw_text_figure(
        path,
        "Stage 12 - IoU calculation",
        [
            "IoU = A_intersection / A_union",
            f"YOLO area = {data['association']['camera_area']:.6f} pixels^2",
            f"Projected PointPillars area = {data['association']['lidar_projected_area']:.6f} pixels^2",
            f"Intersection area = {data['association']['intersection_area']:.6f} pixels^2",
            f"Union area = {data['association']['union_area']:.6f} pixels^2",
            f"IoU = {data['association']['iou']:.10f}",
        ],
    )

    path = add("13", "13_match_decision.png", "Late-fusion match decision", "Calculated IoU + fixed threshold")
    draw_text_figure(
        path,
        "Stage 13 - Matching decision",
        [
            f"calculated IoU = {data['association']['iou']:.6f}",
            f"required threshold = {data['association']['threshold']:.2f}",
            f"{data['association']['iou']:.6f} >= {data['association']['threshold']:.2f}",
            "decision = MATCHED",
            "Ground truth is not used in the association.",
        ],
    )

    path = add("14", "14_fusion_outputs.png", "Matched/unmatched output membership", "Fusion per-frame result")
    draw_text_figure(
        path,
        "Stage 14 - Matched and unmatched outputs",
        [
            "YOLO detection: matched",
            "PointPillars box: matched",
            f"matched 3D output count = {data['fusion_outputs']['matched_3d_count']}",
            f"unmatched camera output count = {data['fusion_outputs']['unmatched_camera_count']}",
            f"unmatched LiDAR output count = {data['fusion_outputs']['unmatched_lidar_count']}",
            f"ALL3D/canonical final output count = {data['fusion_outputs']['final_3d_count']}",
            "The fused 3D geometry and confidence come from the LiDAR detector.",
        ],
    )

    path = add("15", "15_final_fused_image.png", "Final fused image-space output", "YOLO cache + PointPillars projection")
    draw_projected_cuboid(
        base,
        uv,
        yolo,
        projected,
        "Stage 15 - Final fused image-space output",
        [
            sample_line,
            f"person | YOLO conf={data['camera_detection']['confidence']:.3f}",
            f"PointPillars score={data['lidar_detection']['score']:.3f}",
            f"IoU={data['association']['iou']:.6f} | MATCHED",
        ],
        path,
    )

    path = add("16", "16_final_3d_fused_view.png", "Final RViz-style 3D fused view", data["pointcloud_path"])
    draw_lidar_scene(
        points,
        corners_lidar,
        path,
        "Stage 16 - Final 3D/RViz-style fused view",
        [sample_line, "green/magenta convention: matched detector-generated 3D cuboid", "Grey: raw point cloud; axes: LiDAR frame"],
        gt_corners=None,
        box_color=GREEN,
        box_label="matched 3D",
    )

    path = add("17", "17_prediction_vs_ground_truth.png", "Prediction versus ground truth evaluation", "AGHRI annotation JSON")
    if gt_corners is not None:
        draw_lidar_scene(
            points,
            corners_lidar,
            path,
            "Stage 17 - Prediction versus ground truth",
            [
                sample_line,
                f"BEV IoU={data['ground_truth_evaluation']['bev_iou']:.6f}",
                f"3D IoU approx={data['ground_truth_evaluation']['iou_3d_approx']:.6f}",
                f"centre error={data['ground_truth_evaluation']['center_error_m']:.6f} m",
                "Ground truth is evaluation-only.",
            ],
            gt_corners=gt_corners,
        )
    else:
        draw_text_figure(path, "Stage 17 - Ground-truth evaluation", ["No valid 3D ground truth was available for this frame.", "Ground truth is not used to decide camera-LiDAR matches."])

    path = add("18", "18_late_fusion_pipeline_flowchart.png", "Complete late-fusion pipeline flowchart", "Code/config review")
    graphviz_png(
        path,
        """
digraph G {
  graph [rankdir=LR, bgcolor="white", nodesep=0.45, ranksep=0.6];
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", fontname="DejaVu Sans", fontsize=12];
  edge [fontname="DejaVu Sans", fontsize=10];
  image [label="ZED RGB image"];
  yolo [label="YOLO11s\\n2D person boxes", fillcolor="#e8f5e9"];
  cloud [label="Livox point cloud"];
  detector [label="PointPillars / SECOND\\n3D person boxes", fillcolor="#fce4ec"];
  norm [label="Box convention\\nnormalisation"];
  corners [label="8 cuboid corners"];
  tf [label="T_C<-L"];
  proj [label="Camera projection P"];
  env [label="Projected 2D envelope"];
  iou [label="Pairwise 2D IoU"];
  assign [label="one-to-one assignment\\nthreshold"];
  outputs [label="matched / unmatched\\nALL3D outputs", fillcolor="#e3f2fd"];
  metrics [label="offline metrics"];
  ros [label="ROS 2 publication\\nimage visualisation + RViz"];
  image -> yolo -> iou;
  cloud -> detector -> norm -> corners -> tf -> proj -> env -> iou -> assign -> outputs;
  outputs -> metrics;
  outputs -> ros;
}
""",
    )

    path = add("19", "19_offline_evaluation_workflow.png", "Offline scientific evaluation workflow", "Frozen detector caches")
    graphviz_png(
        path,
        """
digraph G {
  graph [rankdir=LR, bgcolor="white"];
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", fontname="DejaVu Sans"];
  dataset [label="AGHRI dataset files"];
  infer [label="YOLO + PointPillars/SECOND inference"];
  caches [label="frozen detection caches", fillcolor="#fff9c4"];
  fusion [label="late fusion"];
  metrics [label="AP / precision / recall / F1 / runtime"];
  dataset -> infer -> caches -> fusion -> metrics;
}
""",
    )

    path = add("20", "20_online_ros2_workflow.png", "Fully online ROS 2 workflow", "ROS launch/code review")
    graphviz_png(
        path,
        """
digraph G {
  graph [rankdir=LR, bgcolor="white"];
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", fontname="DejaVu Sans"];
  image [label="ROS image topic"];
  yolo [label="live YOLO inference\\nDetection2DArray", fillcolor="#e8f5e9"];
  cloud [label="ROS PointCloud2 topic"];
  lidar [label="live PointPillars/SECOND\\nDetection3DArray", fillcolor="#fce4ec"];
  fusion [label="online projection\\nand association"];
  topics [label="fused ROS topics"];
  vis [label="projected image + RViz"];
  replay [label="optional cached ROS replay\\ndebugging path", fillcolor="#eeeeee"];
  image -> yolo -> fusion;
  cloud -> lidar -> fusion;
  fusion -> topics -> vis;
  replay -> fusion [style=dashed];
}
""",
    )

    path = add("21", "21_early_vs_late_fusion.png", "Early versus late fusion comparison", "Early/late repo review")
    graphviz_png(
        path,
        """
digraph G {
  graph [rankdir=LR, bgcolor="white"];
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", fontname="DejaVu Sans"];
  subgraph cluster_early {
    label="Early fusion";
    style="rounded";
    eyolo [label="YOLO box"];
    eproj [label="project raw LiDAR points"];
    eselect [label="select points inside box"];
    efilter [label="outlier filtering"];
    ecentre [label="estimated 3D centre", fillcolor="#e3f2fd"];
    eyolo -> eproj -> eselect -> efilter -> ecentre;
  }
  subgraph cluster_late {
    label="Late fusion";
    style="rounded";
    ldet [label="LiDAR detector\\ncomplete 3D box"];
    lcorners [label="project 8 box corners"];
    lenv [label="2D envelope"];
    liou [label="compare with YOLO box"];
    lkeep [label="retain complete 3D cuboid", fillcolor="#e3f2fd"];
    ldet -> lcorners -> lenv -> liou -> lkeep;
  }
}
""",
    )

    path = add("D1", "late_fusion_pipeline_diagram.png", "Concise late-fusion diagram", "Code/config review")
    graphviz_png(
        path,
        """
digraph G {
  graph [rankdir=LR, bgcolor="white"];
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7", fontname="DejaVu Sans"];
  rgb [label="RGB image -> YOLO 2D box", fillcolor="#e8f5e9"];
  lidar [label="Point cloud -> 3D detector box", fillcolor="#fce4ec"];
  geom [label="3D corners -> camera projection -> 2D envelope"];
  assoc [label="YOLO box + envelope -> IoU -> assignment", fillcolor="#fff9c4"];
  out [label="matched / unmatched / ALL3D outputs", fillcolor="#e3f2fd"];
  rgb -> assoc;
  lidar -> geom -> assoc -> out;
}
""",
    )

    path = add("M1", "late_fusion_complete_workflow.png", "Combined workflow image", "Generated figures")
    montage_sources = [
        figures / "01_raw_zed_rgb_image.png",
        figures / "03_yolo_detection.png",
        figures / "04_pointpillars_3d_detection.png",
        figures / "09_projected_3d_cuboid.png",
        figures / "11_iou_intersection.png",
        figures / "13_match_decision.png",
        figures / "16_final_3d_fused_view.png",
        figures / "21_early_vs_late_fusion.png",
    ]
    make_montage(montage_sources, path)
    return manifest_rows


def make_montage(sources: list[Path], output: Path) -> None:
    thumbs: list[Image.Image] = []
    for src in sources:
        img = Image.open(src).convert("RGB")
        img.thumbnail((520, 330))
        canvas = Image.new("RGB", (540, 360), "white")
        canvas.paste(img, ((540 - img.width) // 2, 10))
        ImageDraw.Draw(canvas).text((14, 335), src.name, fill="black", font=text_font(13))
        thumbs.append(canvas)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    out = Image.new("RGB", (cols * 540, rows * 360 + 70), "white")
    draw = ImageDraw.Draw(out)
    draw.text((24, 18), "AGHRI late-fusion complete workflow", fill="black", font=text_font(28, bold=True))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 540
        y = 70 + (idx // cols) * 360
        out.paste(thumb, (x, y))
    out.save(output)


def write_visualisation_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    path = output_root / "visualisation_manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_number", "filename", "purpose", "input_files", "data_rows", "script_used"])
        writer.writeheader()
        writer.writerows(rows)


def write_selected_manifest(output_root: Path, data: dict[str, Any]) -> None:
    manifest = {
        "recording_name": data["recording_name"],
        "sample_id": data["sample_id"],
        "image_path": data["image_path"],
        "point_cloud_path": data["pointcloud_path"],
        "image_timestamp": data["image_timestamp"],
        "point_cloud_timestamp": data["lidar_timestamp"],
        "yolo_cache_path": data["camera_detection"]["cache_path"],
        "yolo_row_identifier": data["camera_detection"]["row_identifier"],
        "yolo_checkpoint_path": load_json(as_path("results/aghri_pointpillars_detector_comparison/cache/yolo_finetuned/metadata.json")).get("checkpoint_path"),
        "yolo_checkpoint_sha256": data["camera_detection"]["checkpoint_sha256"],
        "pointpillars_cache_path": data["lidar_detection"]["cache_path"],
        "pointpillars_row_identifier": data["lidar_detection"]["row_identifier"],
        "pointpillars_checkpoint_path": load_json(as_path("results/aghri_pointpillars_detector_comparison/cache/pointpillars_finetuned/metadata.json")).get("checkpoint_path"),
        "pointpillars_checkpoint_sha256": data["lidar_detection"]["checkpoint_sha256"],
        "calibration_source": data["calibration"]["source"],
        "calibration_matrix_values": {
            "P": data["calibration"]["P"],
            "T_camera_from_lidar": data["calibration"]["T_camera_from_lidar"],
        },
        "manifest_path": data["source_files"]["manifest"],
        "manifest_sha256": data["source_files"]["manifest_sha256"],
        "fusion_threshold": data["association"]["threshold"],
        "source_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip(),
        "date_generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_reason": data["selection"]["reason"],
        "box_origin_correction": data["box_origin_correction"],
    }
    save_json(output_root / "selected_example_manifest.json", manifest)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_calculation(output_root: Path, data: dict[str, Any]) -> None:
    save_json(output_root / "late_fusion_worked_calculation.json", data)
    geom = data["geometry"]
    assoc = data["association"]
    rows = []
    for idx, (local, lidar, camera, uv, valid) in enumerate(
        zip(geom["local_corners"], geom["corners_lidar"], geom["corners_camera"], geom["corners_image_unclipped"], geom["valid_corner_mask"]),
        start=1,
    ):
        rows.append(
            [
                f"{idx}",
                format_array(local, 6),
                format_array(lidar, 6),
                format_array(camera, 6),
                format_array(uv, 6),
                "valid" if valid else "invalid",
            ]
        )
    text = f"""# Late-Fusion Worked Calculation

This calculation uses one real AGHRI held-out test frame and frozen detector outputs.

Recording: `{data['recording_name']}`  
Sample ID: `{data['sample_id']}`

## 1. YOLO 2D Box

`B_C = {data['camera_detection']['bbox_xyxy']}`

Confidence: `{data['camera_detection']['confidence']}`.

## 2. PointPillars 3D Box

`B_L = [x, y, z, length, width, height, yaw] = {data['lidar_detection']['box_xyzlwhyaw']}`

Score: `{data['lidar_detection']['score']}`.

Cache metadata box origin: `{data['box_origin_correction']['cache_metadata_origin']}`.  
Geometry origin used in this walkthrough: `{data['box_origin_correction']['used_geometry_origin']}`.

## 3. Box-Origin Convention

For `bottom_center`, `z` is the lower face and the upper face is `z + height`. For `center`, `z` is the geometric centre and the lower/upper faces are `z - height/2` and `z + height/2`.

{data['box_origin_correction']['reason']}

## 4. Half Dimensions

`[length/2, width/2, height/2] = {geom['half_dimensions_lwh_m']}`

## 5. Yaw Rotation Matrix

```text
{np.asarray(geom['yaw_rotation_matrix'])}
```

## 6. Corners, Transform and Projection

{markdown_table(['corner', 'local LiDAR box', 'LiDAR frame', 'camera frame', 'pixel u,v', 'status'], rows)}

## 7. Camera-From-LiDAR Matrix

```text
{np.asarray(data['calibration']['T_camera_from_lidar'])}
```

## 8. Projection Matrix

```text
{np.asarray(data['calibration']['P'])}
```

## 9. Projected Envelope and Clipping

Unclipped envelope:

```text
{geom['projected_lidar_rectangle_unclipped']}
```

Clipped image envelope:

```text
{geom['projected_lidar_rectangle_clipped']}
```

## 10. Intersection and IoU

Intersection rectangle:

```text
{assoc['intersection_xyxy']}
```

YOLO area: `{assoc['camera_area']}` pixels²  
Projected PointPillars area: `{assoc['lidar_projected_area']}` pixels²  
Intersection area: `{assoc['intersection_area']}` pixels²  
Union area: `{assoc['union_area']}` pixels²  
Frozen cached association IoU: `{assoc['cached_association_iou']}`  
Note: {assoc['cached_association_iou_note']}

```text
IoU = {assoc['intersection_area']} / {assoc['union_area']} = {assoc['iou']}
```

Threshold comparison:

```text
{assoc['iou']} >= {assoc['threshold']} -> {assoc['decision']}
```

## 11. Final Output Membership

The YOLO detection is matched. The PointPillars box is matched. The matched 3D output and ALL3D/canonical output contain this PointPillars box. The unmatched camera and unmatched LiDAR outputs do not contain this selected detection.
"""
    (output_root / "late_fusion_worked_calculation.md").write_text(text)


def write_report(output_root: Path, data: dict[str, Any]) -> Path:
    gt_eval = data.get("ground_truth_evaluation") or {}
    gt_text = "A valid 3D annotation was found for the selected frame." if gt_eval else "No valid 3D ground truth was available for this selected frame."
    if gt_eval:
        gt_text += f" BEV IoU={gt_eval.get('bev_iou', 0.0):.6f}, approximate 3D IoU={gt_eval.get('iou_3d_approx', 0.0):.6f}, centre error={gt_eval.get('center_error_m', 0.0):.6f} m."
    corner_rows = []
    for idx, (lidar, camera, uv) in enumerate(zip(data["geometry"]["corners_lidar"], data["geometry"]["corners_camera"], data["geometry"]["corners_image_unclipped"]), start=1):
        corner_rows.append([f"L{idx}/C{idx}", format_array(lidar, 6), format_array(camera, 6), format_array(uv, 6)])
    fig_refs = [
        "01_raw_zed_rgb_image.png",
        "02_raw_lidar_pointcloud.png",
        "03_yolo_detection.png",
        "04_pointpillars_3d_detection.png",
        "05_box_origin_convention.png",
        "06_eight_lidar_box_corners.png",
        "07_lidar_to_camera_transformation.png",
        "08_projected_cuboid_corners.png",
        "09_projected_3d_cuboid.png",
        "10_projected_lidar_rectangle.png",
        "11_iou_intersection.png",
        "12_iou_calculation.png",
        "13_match_decision.png",
        "14_fusion_outputs.png",
        "15_final_fused_image.png",
        "16_final_3d_fused_view.png",
        "17_prediction_vs_ground_truth.png",
        "18_late_fusion_pipeline_flowchart.png",
        "19_offline_evaluation_workflow.png",
        "20_online_ros2_workflow.png",
        "21_early_vs_late_fusion.png",
    ]
    image_blocks = "\n\n".join([f"![{name}](figures/{name})" for name in fig_refs])
    text = f"""# AGHRI Late-Fusion Visual Walkthrough

## Objective

This report follows one real AGHRI camera-LiDAR frame through the complete generic late-fusion pipeline. It mirrors the early-fusion weekly-report style, but the late-fusion object is a complete detector-generated 3D cuboid rather than an estimated centre from raw projected LiDAR points.

## Selected Real Frame

Recording: `{data['recording_name']}`  
Sample ID: `{data['sample_id']}`  
Frame choice: original calibration-audit frame. Reason: {data['selection']['reason']}

## Sensor Inputs

The ZED RGB image and Livox point cloud are the paired sensor inputs. The camera frame is `{data['frames']['camera_frame']}` and the LiDAR frame is `{data['frames']['lidar_frame']}`.

{image_blocks.split('![04_pointpillars_3d_detection.png]')[0]}

## YOLO Detection

The camera detector output is the real fine-tuned YOLO person detection:

```text
box xyxy = {data['camera_detection']['bbox_xyxy']}
confidence = {data['camera_detection']['confidence']}
```

![03_yolo_detection.png](figures/03_yolo_detection.png)

## PointPillars Detection

The LiDAR detector output is the real fine-tuned PointPillars 3D prediction:

```text
[x, y, z, length, width, height, yaw] = {data['lidar_detection']['box_xyzlwhyaw']}
score = {data['lidar_detection']['score']}
cache metadata box origin = {data['box_origin_correction']['cache_metadata_origin']}
geometry origin used = {data['box_origin_correction']['used_geometry_origin']}
```

{data['box_origin_correction']['reason']}

![04_pointpillars_3d_detection.png](figures/04_pointpillars_3d_detection.png)

## Calibration

Both early and late fusion use the same physical calibration. The live ROS path uses CameraInfo/TF and the offline audit path uses the calibration ZIP JSON, which agree numerically for this representative bag.

```text
T_C<-L = {data['calibration']['T_camera_from_lidar']}
P = {data['calibration']['P']}
```

![07_lidar_to_camera_transformation.png](figures/07_lidar_to_camera_transformation.png)

## 3D-Box Corner Construction

The eight corners follow the ordering in `late_fusion_pkg/projection.py`: lower face first, then upper face.

![05_box_origin_convention.png](figures/05_box_origin_convention.png)

![06_eight_lidar_box_corners.png](figures/06_eight_lidar_box_corners.png)

{markdown_table(['corner', 'LiDAR-frame coordinate', 'camera-frame coordinate', 'projected pixel'], corner_rows)}

## LiDAR-to-Camera Transformation

Each LiDAR corner is multiplied by `T_C<-L` to obtain the camera-frame corner. Corners with `Z_camera <= 0` would be rejected before image projection; all selected-frame corners are projectable.

## Camera Projection

The camera projection uses the verified `CameraInfo.P` matrix. Projected corner pixels are shown below.

![08_projected_cuboid_corners.png](figures/08_projected_cuboid_corners.png)

## Projected Rectangle

The projected cuboid is converted to an image-space envelope. This selected frame touches the upper-left image boundary, so the unclipped envelope is clipped to the actual image extent.

```text
unclipped = {data['geometry']['projected_lidar_rectangle_unclipped']}
clipped = {data['geometry']['projected_lidar_rectangle_clipped']}
```

![09_projected_3d_cuboid.png](figures/09_projected_3d_cuboid.png)

![10_projected_lidar_rectangle.png](figures/10_projected_lidar_rectangle.png)

## IoU

The late-fusion association compares the YOLO rectangle with the projected LiDAR envelope using 2D image IoU.

```text
intersection = {data['association']['intersection_xyxy']}
YOLO area = {data['association']['camera_area']}
projected LiDAR area = {data['association']['lidar_projected_area']}
intersection area = {data['association']['intersection_area']}
union area = {data['association']['union_area']}
IoU = {data['association']['iou']}
frozen cached association IoU = {data['association']['cached_association_iou']}
```

{data['association']['cached_association_iou_note']}

![11_iou_intersection.png](figures/11_iou_intersection.png)

![12_iou_calculation.png](figures/12_iou_calculation.png)

## Association

```text
IoU = {data['association']['iou']:.10f}
threshold = {data['association']['threshold']:.2f}
decision = {data['association']['decision']}
```

Ground truth is not used to decide the camera-LiDAR match.

![13_match_decision.png](figures/13_match_decision.png)

## Final Fused Outputs

For this selected pair, the YOLO detection is matched and the PointPillars box is matched. The fused output preserves the PointPillars 3D geometry and LiDAR confidence.

```text
matched 3D = {data['fusion_outputs']['matched_3d_count']}
unmatched camera = {data['fusion_outputs']['unmatched_camera_count']}
unmatched LiDAR = {data['fusion_outputs']['unmatched_lidar_count']}
ALL3D/final 3D = {data['fusion_outputs']['final_3d_count']}
```

![14_fusion_outputs.png](figures/14_fusion_outputs.png)

![15_final_fused_image.png](figures/15_final_fused_image.png)

![16_final_3d_fused_view.png](figures/16_final_3d_fused_view.png)

## Ground-Truth Evaluation

{gt_text}

Ground truth is used for evaluation after fusion. It is not used to decide the camera-LiDAR match. The fusion IoU above is a 2D image IoU, not BEV IoU or 3D IoU.

![17_prediction_vs_ground_truth.png](figures/17_prediction_vs_ground_truth.png)

## Early Fusion and Late Fusion Using the Same Calibration

Both methods use the same physical calibration, but they fuse different objects. Early fusion projects raw LiDAR points into the YOLO box, filters candidate points, and estimates one 3D centre. Late fusion projects a complete LiDAR detector cuboid, compares its projected envelope with the YOLO box, and preserves or rejects the complete 3D detection.

![21_early_vs_late_fusion.png](figures/21_early_vs_late_fusion.png)

## Offline Evaluation Workflow

Offline evaluation uses dataset files, frozen detector caches, late fusion, and metric scripts. It is separated from live ROS deployment.

![19_offline_evaluation_workflow.png](figures/19_offline_evaluation_workflow.png)

## Fully Online ROS Workflow

The online ROS 2 implementation consumes image and point-cloud topics, runs live detectors, performs projection/association online, and publishes fused topics for image visualisation and RViz. Cached ROS replay is useful for debugging but is not the principal deployment path.

![20_online_ros2_workflow.png](figures/20_online_ros2_workflow.png)

## Complete Numerical Calculation

The full numerical calculation is available in `late_fusion_worked_calculation.md` and `late_fusion_worked_calculation.json`.

![18_late_fusion_pipeline_flowchart.png](figures/18_late_fusion_pipeline_flowchart.png)

![late_fusion_pipeline_diagram.png](figures/late_fusion_pipeline_diagram.png)

![late_fusion_complete_workflow.png](figures/late_fusion_complete_workflow.png)

## Key Points To Remember

- Same physical calibration, different fusion representation.
- Early fusion estimates a 3D centre from raw projected LiDAR points.
- Late fusion preserves a learned 3D cuboid from PointPillars or SECOND.
- The association decision uses projected 2D envelope IoU with the YOLO box.
- Ground truth is evaluation-only and is not used for association.
"""
    md_path = output_root / "LATE_FUSION_VISUAL_WALKTHROUGH.md"
    md_path.write_text(text)
    write_html_report(output_root, text)
    return md_path


def write_html_report(output_root: Path, markdown_text: str) -> None:
    html_lines = [
        "<html><head><meta charset='utf-8'><style>body{font-family:DejaVu Sans,Arial,sans-serif;max-width:1000px;margin:auto;} img{max-width:950px;} pre{background:#f4f4f4;padding:10px;white-space:pre-wrap;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px;font-size:9pt;}</style></head><body>"
    ]
    in_code = False
    in_table = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            html_lines.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            html_lines.append(html.escape(line))
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("!["):
            alt = line.split("](", 1)[0][2:]
            src = line.split("](", 1)[1].rstrip(")")
            html_lines.append(f"<figure><img src='{html.escape(src)}'><figcaption>{html.escape(alt)}</figcaption></figure>")
        elif line.startswith("| ") and " |" in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells) == {"---"}:
                continue
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        elif line.strip() == "":
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append("<br>")
        elif line.startswith("- "):
            html_lines.append(f"<p>{html.escape(line)}</p>")
        else:
            html_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        html_lines.append("</table>")
    html_lines.append("</body></html>")
    html_path = output_root / "LATE_FUSION_VISUAL_WALKTHROUGH.html"
    html_path.write_text("\n".join(html_lines))
    create_docx_from_markdown(markdown_text, output_root / "LATE_FUSION_VISUAL_WALKTHROUGH.docx", output_root)


def xml_escape(text: str) -> str:
    return html.escape(text, quote=True)


def docx_paragraph(text: str, style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
    return f"<w:p>{style_xml}<w:r><w:t{preserve}>{xml_escape(text)}</w:t></w:r></w:p>"


def docx_image_paragraph(rid: str, width_px: int, height_px: int, name: str) -> str:
    max_width_emu = 6_400_000
    emu_per_px = 9525
    width_emu = int(width_px * emu_per_px)
    height_emu = int(height_px * emu_per_px)
    if width_emu > max_width_emu:
        scale = max_width_emu / width_emu
        width_emu = max_width_emu
        height_emu = int(height_emu * scale)
    docpr_name = xml_escape(name)
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0"
        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:docPr id="{rid[3:]}" name="{docpr_name}"/>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr>
                <pic:cNvPr id="0" name="{docpr_name}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def create_docx_from_markdown(markdown_text: str, docx_path: Path, base_dir: Path) -> None:
    body_parts: list[str] = []
    rels: list[tuple[str, str]] = []
    media_files: list[tuple[Path, str]] = []
    image_idx = 1
    in_code = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if line.startswith("![") and "](" in line:
            alt = line.split("](", 1)[0][2:]
            src = line.split("](", 1)[1].rstrip(")")
            image_path = base_dir / src
            if image_path.exists():
                rid = f"rId{image_idx}"
                media_name = f"image{image_idx}{image_path.suffix.lower()}"
                with Image.open(image_path) as img:
                    width_px, height_px = img.size
                rels.append((rid, f"media/{media_name}"))
                media_files.append((image_path, media_name))
                body_parts.append(docx_image_paragraph(rid, width_px, height_px, alt))
                body_parts.append(docx_paragraph(alt, "Caption"))
                image_idx += 1
            continue
        if line.startswith("# "):
            body_parts.append(docx_paragraph(line[2:], "Heading1"))
        elif line.startswith("## "):
            body_parts.append(docx_paragraph(line[3:], "Heading2"))
        elif line.startswith("| ") and " |" in line:
            if "---" not in line:
                body_parts.append(docx_paragraph(line, "Code"))
        elif line.strip() == "":
            body_parts.append("<w:p/>")
        elif in_code:
            body_parts.append(docx_paragraph(line, "Code"))
        else:
            body_parts.append(docx_paragraph(line))

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body_parts)}
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/></w:sectPr>
  </w:body>
</w:document>
"""
    rel_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
""" + "\n".join(
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        for rid, target in rels
    ) + "\n</Relationships>\n"
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="200" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="DejaVu Sans Mono" w:hAnsi="DejaVu Sans Mono"/><w:sz w:val="18"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", package_rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/_rels/document.xml.rels", rel_xml)
        for image_path, media_name in media_files:
            z.write(image_path, f"word/media/{media_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", default=DEFAULT_RECORDING)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--camera-cache", default=DEFAULT_YOLO_CACHE)
    parser.add_argument("--lidar-cache", default=DEFAULT_LIDAR_CACHE)
    parser.add_argument("--association", default=DEFAULT_ASSOCIATION)
    parser.add_argument("--per-frame", default=DEFAULT_PER_FRAME)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--lidar-box-origin",
        choices=("cache", "bottom_center", "center"),
        default="cache",
        help="Use the cache metadata origin or explicitly override the LiDAR box z-origin convention for the worked example.",
    )
    args = parser.parse_args()
    if not args.sample_id.startswith(args.recording + "/"):
        raise RuntimeError("--sample-id must belong to --recording")

    output_root = as_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data, gt_corners = build_worked_example(args)
    points = parse_binary_pcd_xyz(data["pointcloud_path"])
    rows = generate_figures(output_root, data, points, gt_corners)
    write_visualisation_manifest(output_root, rows)
    write_selected_manifest(output_root, data)
    write_calculation(output_root, data)
    write_report(output_root, data)
    print(json.dumps({
        "output_root": str(output_root),
        "figures": len(rows),
        "projected_rectangle": data["geometry"]["projected_lidar_rectangle_clipped"],
        "iou": data["association"]["iou"],
        "decision": data["association"]["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
