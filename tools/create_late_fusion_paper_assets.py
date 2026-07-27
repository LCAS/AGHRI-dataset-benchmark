#!/usr/bin/env python3
"""Create draft late-fusion paper figures and a review notebook."""

from __future__ import annotations

import csv
import base64
import json
import re
import shutil
import textwrap
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches


REPO = Path("/home/prabuddhi/Desktop/late_fusion/deepfusionmot")
OUT = REPO / "results/late_fusion_paper_assets"
ASSETS = OUT / "selected_notebook_assets"
FIGURES = ASSETS / "figures"
DERIVED = ASSETS / "derived_data"
NOTEBOOK = REPO / "notebooks/aghri_late_fusion_paper_assets.ipynb"
REDESIGN_FIGURES = REPO / "results/late_fusion_paper_figure_redesign/figures"

METRICS_CSV = REPO / "results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy_metrics/deepfusionmot_metrics.csv"
PER_RECORDING_CSV = REPO / "results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy_metrics/deepfusionmot_metrics_per_recording.csv"
DETECTOR_SWEEP_CSV = REPO / "results/aghri_deepfusionmot_tracking/diagnostics_val_ft_yolo_ft_pointpillars/detector_output_sweep_results.csv"
SPEED_CSV = REPO / "results/aghri_final_speed/speed_table_aggregate.csv"
PER_RECORDING_SPEED_CSV = REPO / "results/aghri_final_speed/speed_table_per_recording.csv"
WORKED_EXAMPLE_JSON = REPO / "results/late_fusion_visual_walkthrough/late_fusion_worked_calculation.json"
TEST_MANIFEST_CSV = REPO / "data_manifests/aghri_late_fusion_test_manifest.csv"
P4_TEMPORAL_TRACKS_CSV = (
    REPO
    / "results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy/P4"
    / "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label/tracked_3d_results.csv"
)
P4_VINE_TRACKING_DIR = (
    REPO
    / "results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy/P4"
    / "out_vine_4swap+walk_st_ly_11_06_2024_2_label"
)
YOLO_FT_CACHE = REPO / "results/aghri_pointpillars_detector_comparison/cache/yolo_finetuned/detections.csv"
POINTPILLARS_FT_CACHE = REPO / "results/aghri_pointpillars_detector_comparison/cache/pointpillars_finetuned/detections.csv"
RVIZ_PLAYBACK_CAPTURE = REPO / "results/late_fusion_paper_assets/source_rviz/ros2_playback_deepfusionmot_rviz.png"
ROS2_TRACKED_PROJECTION_CAPTURE = REPO / "results/late_fusion_paper_assets/source_rviz/multi_candidate_ros2_close/shortlist/candidate_5_projection_clean_image_only.png"
ROS2_TRACKED_POINTCLOUD_CAPTURE = REPO / "results/late_fusion_paper_assets/source_rviz/multi_candidate_ros2_close/shortlist/candidate_5_pointcloud.png"
FINAL_ROS2_FPS = {
    "S1": 12.73,
    "S2": 12.27,
    "S3": 12.69,
    "S4": 12.43,
    "P1": 13.57,
    "P2": 13.38,
    "P3": 12.15,
    "P4": 12.14,
}

FRAMEWORK_FIG = REPO / "results/deepfusionmot_framework_note/figures/01_framework_overview.png"
VIS_ROOT = REPO / "results/late_fusion_visual_walkthrough/figures"
CONVENTION_ROOT = REPO / "results/aghri_generic_baseline/convention_audit/visualizations"


COLORS = {
    "blue": "#3B6699",
    "green": "#16a34a",
    "red": "#dc2626",
    "cyan": "#0891b2",
    "yellow": "#ca8a04",
    "orange": "#D27228",
    "purple": "#9333ea",
    "gray": "#64748b",
    "ink": "#0f172a",
}


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    REDESIGN_FIGURES.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    for old_figure in FIGURES.glob("*.png"):
        old_figure.unlink()
    for old_table in DERIVED.glob("*.csv"):
        old_table.unlink()


def short_recording(name: str) -> str:
    mapping = {
        "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label": "Footpath moving",
        "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label": "Footpath stationary",
        "in_straw_3pick_diff_st_10_24_2024_5_a_label": "Inside strawberry",
        "out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label": "Outside strawberry",
        "out_vine_1push_3carry_st_ly_11_06_2024_1_label": "Vineyard 1",
        "out_vine_4swap+walk_st_ly_11_06_2024_2_label": "Vineyard 2",
        "COMBINED": "Aggregate",
    }
    return mapping.get(name, name.replace("_label", "")[:22])


def prettify_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(DERIVED / name, index=False)


def plot_metric_summary(metrics: pd.DataFrame) -> Path:
    metrics = metrics.copy()
    metrics["short"] = metrics["combination_id"]
    cols = ["HOTA", "DetA", "AssA", "IDF1"]
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    palette = ["#1d4ed8", "#16a34a", "#9333ea", "#f97316"]
    for idx, col in enumerate(cols):
        ax.bar(x + (idx - 1.5) * width, metrics[col], width=width, label=col, color=palette[idx])
    ax.set_title("AGHRI DeepFusionMOT Tracking Metrics Across Eight Detector Combinations", weight="bold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, max(0.48, float(metrics[cols].max().max()) * 1.18))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["short"])
    ax.legend(ncol=4, frameon=False)
    for i, row in metrics.iterrows():
        ax.text(i, -0.055, row["combination_label"], ha="center", va="top", rotation=35, fontsize=8, transform=ax.get_xaxis_transform())
    prettify_axes(ax)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    path = FIGURES / "F1_tracking_metric_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def plot_second_vs_pointpillars(metrics: pd.DataFrame) -> Path:
    pairs = [
        ("G YOLO + G LiDAR", "S1", "P1"),
        ("FT YOLO + G LiDAR", "S2", "P2"),
        ("G YOLO + FT LiDAR", "S3", "P3"),
        ("FT YOLO + FT LiDAR", "S4", "P4"),
    ]
    rows = []
    by_id = metrics.set_index("combination_id")
    for label, second_id, point_id in pairs:
        rows.append({"condition": label, "SECOND": by_id.loc[second_id, "HOTA"], "PointPillars": by_id.loc[point_id, "HOTA"]})
    df = pd.DataFrame(rows)
    save_table(df, "F2_second_vs_pointpillars_hota.csv")
    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.bar(x - width / 2, df["SECOND"], width=width, label="SECOND", color="#2563eb")
    ax.bar(x + width / 2, df["PointPillars"], width=width, label="PointPillars", color="#9333ea")
    ax.set_title("SECOND vs PointPillars Tracking HOTA Under Matched Camera Settings", weight="bold")
    ax.set_ylabel("HOTA")
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=20, ha="right")
    ax.set_ylim(0, max(0.42, float(df[["SECOND", "PointPillars"]].max().max()) * 1.2))
    ax.legend(frameon=False)
    prettify_axes(ax)
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    path = FIGURES / "F2_second_vs_pointpillars_hota.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def plot_per_recording(per_recording: pd.DataFrame) -> Path:
    df = per_recording[per_recording["combination_id"].isin(["S4", "P4"])].copy()
    df["recording_short"] = df["recording"].map(short_recording)
    pivot = df.pivot(index="recording_short", columns="combination_id", values="HOTA")
    order = [short_recording(x) for x in per_recording["recording"].drop_duplicates() if short_recording(x) in pivot.index]
    pivot = pivot.loc[order]
    save_table(pivot.reset_index(), "F3_per_recording_s4_p4_hota.csv")
    x = np.arange(len(pivot))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(x - width / 2, pivot["S4"], width=width, label="FT YOLO + FT SECOND", color="#2563eb")
    ax.bar(x + width / 2, pivot["P4"], width=width, label="FT YOLO + FT PointPillars", color="#9333ea")
    ax.set_title("Per-Recording Tracking HOTA For The Two Fine-Tuned LiDAR Models", weight="bold")
    ax.set_ylabel("HOTA")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.legend(frameon=False)
    prettify_axes(ax)
    fig.tight_layout()
    path = FIGURES / "F3_per_recording_best_models_hota.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def plot_detector_threshold_sweep(sweep: pd.DataFrame) -> Path:
    df = sweep[
        (sweep["recording"] == "COMBINED")
        & (np.isclose(sweep["min_camera_score"], 0.10))
        & (sweep["output_mode"] == "deepfusionmot")
    ].copy()
    df = df.sort_values("min_lidar_score")
    cols = ["min_lidar_score", "HOTA", "IDF1", "MOTA", "tracker_dets", "CLR_FP", "CLR_FN", "IDSW"]
    save_table(df[cols], "F4_lidar_score_sweep_cam0p10_deepfusionmot.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(df["min_lidar_score"], df["HOTA"], marker="o", label="HOTA", color="#2563eb", linewidth=2.2)
    ax.plot(df["min_lidar_score"], df["IDF1"], marker="o", label="IDF1", color="#16a34a", linewidth=2.2)
    ax.plot(df["min_lidar_score"], df["MOTA"], marker="o", label="MOTA", color="#dc2626", linewidth=2.2)
    ax.axvline(0.20, color="#0f172a", linestyle="--", linewidth=1.2, label="selected 0.20")
    ax.set_title("Validation Confidence Sweep For FT YOLO + FT PointPillars", weight="bold")
    ax.set_xlabel("Minimum LiDAR score")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=False)
    prettify_axes(ax)
    fig.tight_layout()
    path = FIGURES / "F4_lidar_score_threshold_sweep.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#334155", width: int = 3) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    size = 12
    left = (end[0] - size * np.cos(angle - np.pi / 6), end[1] - size * np.sin(angle - np.pi / 6))
    right = (end[0] - size * np.cos(angle + np.pi / 6), end[1] - size * np.sin(angle + np.pi / 6))
    draw.polygon([end, left, right], fill=color)


def draw_polyline_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = "#334155", width: int = 3) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1]):
        draw.line((start, end), fill=color, width=width)
    draw_arrow(draw, points[-2], points[-1], color=color, width=width)


def draw_wrapped_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill: str = COLORS["ink"],
    line_spacing: int = 6,
) -> None:
    x1, y1, x2, y2 = box
    max_chars = max(12, int((x2 - x1) / 8.8))
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=max_chars) or [""])
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) // 2
    for line, line_h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        x = x1 + ((x2 - x1) - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h + line_spacing


def make_pipeline_figure() -> Path:
    path = FIGURES / "F1_complete_late_fusion_tracking_pipeline.png"
    pdf_path = REDESIGN_FIGURES / "01_aghri_late_fusion_deep_association_architecture.pdf"
    png_path = REDESIGN_FIGURES / "01_aghri_late_fusion_deep_association_architecture.png"
    svg_path = REDESIGN_FIGURES / "01_aghri_late_fusion_deep_association_architecture.svg"

    palette = {
        "camera_main": "#4C78A8",
        "camera_fill": "#EAF2FB",
        "lidar_main": "#E69F35",
        "lidar_fill": "#FFF2D6",
        "fusion_main": "#59A14F",
        "fusion_fill": "#EAF6E8",
        "tracking_main": "#9C7BB5",
        "tracking_fill": "#F2ECF7",
        "neutral_fill": "#F5F7FA",
        "line": "#475569",
        "text": "#1F2937",
    }
    fig, ax = plt.subplots(figsize=(18.0, 7.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def group(x: float, y: float, w: float, h: float, title: str) -> None:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=0.9,
                edgecolor=palette["line"],
                facecolor="none",
                linestyle=(0, (4, 3)),
                alpha=0.55,
            )
        )
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", fontsize=13.0, weight="bold", color=palette["text"])

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        fill: str,
        edge: str,
        fontsize: float = 10.0,
        weight: str = "bold",
        lw: float = 1.0,
    ) -> tuple[float, float, float, float]:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.16",
            linewidth=lw,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=palette["text"], linespacing=1.15)
        return (x, y, w, h)

    def center(rect: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = rect
        return (x + w / 2, y + h / 2)

    def port(rect: tuple[float, float, float, float], side: str) -> tuple[float, float]:
        x, y, w, h = rect
        if side == "left":
            return (x, y + h / 2)
        if side == "right":
            return (x + w, y + h / 2)
        if side == "top":
            return (x + w / 2, y + h)
        if side == "bottom":
            return (x + w / 2, y)
        raise ValueError(side)

    def side_port(rect: tuple[float, float, float, float], side: str, fraction: float) -> tuple[float, float]:
        x, y, w, h = rect
        if side == "left":
            return (x, y + h * fraction)
        if side == "right":
            return (x + w, y + h * fraction)
        if side == "top":
            return (x + w * fraction, y + h)
        if side == "bottom":
            return (x + w * fraction, y)
        raise ValueError(side)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = palette["line"], lw: float = 1.05) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "lw": lw, "color": color, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 9},
        )

    def elbow(points: list[tuple[float, float]], color: str = palette["line"], lw: float = 1.05) -> None:
        if len(points) < 2:
            return
        for first, second in zip(points[:-2], points[1:-1]):
            ax.plot([first[0], second[0]], [first[1], second[1]], color=color, lw=lw)
        arrow(points[-2], points[-1], color=color, lw=lw)

    group(1.0, 2.0, 21.0, 38.0, "Inputs and Detectors")
    group(24.0, 2.0, 23.0, 38.0, "Detection-Level Late Fusion")
    group(49.0, 2.0, 37.0, 38.0, "Deep Association")
    group(88.0, 2.0, 11.0, 38.0, "Outputs")

    lane_h = 4.0
    livox = box(2.3, 31.0, 5.9, lane_h, "Livox point\ncloud", palette["lidar_fill"], palette["lidar_main"], 9.0)
    lidar_det = box(8.9, 31.0, 6.1, lane_h, "SECOND /\nPointPillars", palette["lidar_fill"], palette["lidar_main"], 9.2)
    lidar_dets = box(15.7, 31.0, 5.1, lane_h, "3D\ndetections", palette["lidar_fill"], palette["lidar_main"], 9.6)
    zed = box(2.3, 10.2, 5.9, lane_h, "ZED RGB\nimage", palette["camera_fill"], palette["camera_main"], 9.6)
    yolo = box(8.9, 10.2, 6.1, lane_h, "YOLO11s", palette["camera_fill"], palette["camera_main"], 10.0)
    cam_dets = box(15.7, 10.2, 5.1, lane_h, "2D\ndetections", palette["camera_fill"], palette["camera_main"], 9.6)
    for first, second in [(livox, lidar_det), (lidar_det, lidar_dets), (zed, yolo), (yolo, cam_dets)]:
        arrow(port(first, "right"), port(second, "left"))

    fusion = box(
        25.1,
        8.2,
        12.4,
        21.0,
        "Projection-Based\nLate Fusion",
        palette["neutral_fill"],
        palette["line"],
        9.6,
        lw=1.15,
    )
    matched = box(39.2, 25.8, 6.6, 4.0, "Matched\n3D", palette["fusion_fill"], palette["fusion_main"])
    lidar_only = box(39.2, 19.4, 6.6, 4.0, "LiDAR-only\n3D", palette["lidar_fill"], palette["lidar_main"])
    camera_only = box(39.2, 8.2, 6.6, 4.0, "Camera-only\n2D", palette["camera_fill"], palette["camera_main"])

    elbow([port(lidar_dets, "right"), (22.8, center(lidar_dets)[1]), (22.8, side_port(fusion, "left", 0.88)[1]), side_port(fusion, "left", 0.88)], color=palette["lidar_main"])
    arrow(port(cam_dets, "right"), side_port(fusion, "left", 0.19), color=palette["camera_main"])
    arrow(side_port(fusion, "right", 0.93), port(matched, "left"), color=palette["fusion_main"])
    arrow(side_port(fusion, "right", 0.63), port(lidar_only, "left"), color=palette["lidar_main"])
    arrow(side_port(fusion, "right", 0.095), port(camera_only, "left"), color=palette["camera_main"])

    level1 = box(50.2, 25.8, 8.0, 4.0, "Level 1\nassociation", palette["fusion_fill"], palette["fusion_main"])
    level2 = box(50.2, 19.4, 8.0, 4.0, "Level 2\nassociation", palette["lidar_fill"], palette["lidar_main"])
    level3 = box(50.2, 8.2, 8.0, 4.0, "Level 3\nassociation", palette["camera_fill"], palette["camera_main"])
    management = box(62.4, 13.4, 13.0, 16.0, "Track\nManagement", palette["tracking_fill"], palette["tracking_main"], 12.0, lw=1.3)
    odom = box(64.4, 31.4, 9.0, 3.2, "AGHRI\nodometry", palette["neutral_fill"], palette["line"], 9.7)
    active_2d = box(62.4, 8.2, 13.0, 4.0, "Active\n2D tracks", palette["tracking_fill"], palette["tracking_main"])
    remaining_3d = box(77.2, 19.4, 7.5, 4.0, "Remaining\n3D tracks", palette["tracking_fill"], palette["tracking_main"], 9.4)
    level4 = box(77.2, 8.2, 7.5, 4.0, "Level 4\nassociation", palette["neutral_fill"], palette["tracking_main"], 9.4)

    arrow(port(matched, "right"), port(level1, "left"), color=palette["fusion_main"])
    arrow(port(lidar_only, "right"), port(level2, "left"), color=palette["lidar_main"])
    arrow(port(camera_only, "right"), port(level3, "left"), color=palette["camera_main"])
    arrow(port(level1, "right"), side_port(management, "left", 0.90), color=palette["tracking_main"])
    arrow(port(level2, "right"), (62.4, center(level2)[1]), color=palette["tracking_main"])
    arrow(port(level3, "right"), port(active_2d, "left"), color=palette["tracking_main"])
    arrow(port(odom, "bottom"), port(management, "top"), color=palette["tracking_main"])
    arrow(side_port(management, "right", 0.50), port(remaining_3d, "left"), color=palette["tracking_main"])
    arrow(port(remaining_3d, "bottom"), port(level4, "top"), color=palette["tracking_main"])
    arrow(port(active_2d, "right"), port(level4, "left"), color=palette["camera_main"])

    tracks3d_out = box(89.0, 25.2, 8.6, 5.2, "3D tracks\nPerson cuboids", palette["tracking_fill"], palette["tracking_main"], 9.7, lw=1.3)
    tracks2d_out = box(89.0, 7.6, 8.6, 5.2, "2D tracks\nCamera-space", palette["camera_fill"], palette["camera_main"], 9.7)
    arrow(side_port(management, "right", 0.90), port(tracks3d_out, "left"), color=palette["tracking_main"], lw=1.2)
    arrow(port(level4, "right"), port(tracks2d_out, "left"), color=palette["camera_main"])

    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def read_pcd_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                return np.empty((0, 3), dtype=np.float32)
            if line.startswith(b"DATA"):
                break
        data = stream.read()
    arr = np.frombuffer(data, dtype=np.float32)
    arr = arr[: (arr.size // 4) * 4].reshape(-1, 4)
    xyz = arr[:, :3]
    return xyz[np.isfinite(xyz).all(axis=1)]


def add_bbox(ax, xyxy: list[float] | tuple[float, float, float, float], color: str, label: str, lw: float = 2.4) -> None:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=lw, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(
        x1,
        max(2, y1 - 8),
        label,
        color=color,
        fontsize=9,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.78, "pad": 1.5},
    )


def bev_box_corners(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rot.T + np.array([x, y])


def make_same_frame_association_figure() -> Path:
    path = FIGURES / "F2_same_frame_late_fusion_association_example.png"
    data = json.loads(WORKED_EXAMPLE_JSON.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(data["image_path"]).convert("RGB"))
    xyz = read_pcd_xyz(Path(data["pointcloud_path"]))
    cam_box = data["camera_detection"]["bbox_xyxy"]
    cam_score = float(data["camera_detection"]["confidence"])
    lidar_box = data["lidar_detection"]["box_xyzlwhyaw"]
    lidar_score = float(data["lidar_detection"]["score"])
    lidar_rect = data["geometry"]["projected_lidar_rectangle_clipped"]
    inter = data["association"]["intersection_xyxy"]
    iou = float(data["association"]["iou"])
    threshold = float(data["association"]["threshold"])
    decision = data["association"]["decision"]
    outputs = data["fusion_outputs"]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].imshow(image)
    add_bbox(axes[0, 0], cam_box, COLORS["blue"], f"YOLO person {cam_score:.2f}")
    axes[0, 0].set_title("A. Camera detection", loc="left", weight="bold")

    axes[0, 1].set_title("B. LiDAR detection in BEV", loc="left", weight="bold")
    if len(xyz):
        mask = (xyz[:, 0] > 0) & (xyz[:, 0] < 12) & (np.abs(xyz[:, 1]) < 6)
        pts = xyz[mask]
        if len(pts) > 6500:
            pts = pts[np.linspace(0, len(pts) - 1, 6500).astype(int)]
        axes[0, 1].scatter(pts[:, 1], pts[:, 0], s=0.3, c="#64748b", alpha=0.35)
    x, y, z, length, width, height, yaw = [float(v) for v in lidar_box]
    corners = bev_box_corners(x, y, length, width, yaw)
    axes[0, 1].add_patch(patches.Polygon(corners[:, [1, 0]], closed=True, fill=False, edgecolor=COLORS["purple"], linewidth=2.6))
    axes[0, 1].scatter([y], [x], c=COLORS["purple"], s=34)
    axes[0, 1].set_xlim(-3.2, 3.2)
    axes[0, 1].set_ylim(8.5, 0)
    axes[0, 1].set_xlabel("left/right y (m)")
    axes[0, 1].set_ylabel("forward x (m)")
    axes[0, 1].grid(alpha=0.22)
    axes[0, 1].text(
        0.02,
        0.98,
        f"center=({x:.2f}, {y:.2f}, {z:.2f}) m\n"
        f"LWH=({length:.2f}, {width:.2f}, {height:.2f}) m\n"
        f"yaw={yaw:.2f} rad, score={lidar_score:.2f}",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": COLORS["purple"], "alpha": 0.88},
    )

    axes[1, 0].imshow(image)
    add_bbox(axes[1, 0], cam_box, COLORS["blue"], "YOLO box")
    add_bbox(axes[1, 0], lidar_rect, COLORS["purple"], "Projected LiDAR envelope")
    ix1, iy1, ix2, iy2 = [float(v) for v in inter]
    axes[1, 0].add_patch(
        patches.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, linewidth=0, facecolor="#fde047", alpha=0.35)
    )
    axes[1, 0].text(
        0.02,
        0.98,
        f"IoU={iou:.4f}\nthreshold={threshold:.2f}\n{decision}",
        transform=axes[1, 0].transAxes,
        va="top",
        color=COLORS["ink"],
        fontsize=10,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": COLORS["green"], "alpha": 0.88},
    )
    axes[1, 0].set_title("C. Projection and association", loc="left", weight="bold")

    axes[1, 1].axis("off")
    axes[1, 1].set_title("D. Final routed output", loc="left", weight="bold")
    routing = [
        ("Matched 3D", outputs["matched_3d_count"], COLORS["green"]),
        ("Unmatched camera", outputs["unmatched_camera_count"], COLORS["blue"]),
        ("Unmatched LiDAR", outputs["unmatched_lidar_count"], COLORS["red"]),
        ("ALL3D", outputs["final_3d_count"], COLORS["cyan"]),
    ]
    for idx, (label, value, color) in enumerate(routing):
        y0 = 0.78 - idx * 0.16
        axes[1, 1].add_patch(
            patches.FancyBboxPatch((0.08, y0), 0.52, 0.10, boxstyle="round,pad=0.015", edgecolor=color, facecolor="#f8fafc", linewidth=2)
        )
        axes[1, 1].text(0.12, y0 + 0.052, f"{label} = {value}", transform=axes[1, 1].transAxes, va="center", fontsize=14, weight="bold", color=color)
    axes[1, 1].annotate(
        "Preserved LiDAR cuboid\nenters tracker",
        xy=(0.60, 0.35),
        xytext=(0.76, 0.35),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 2, "color": COLORS["ink"]},
        fontsize=13,
        color=COLORS["ink"],
        va="center",
    )
    axes[1, 1].text(
        0.08,
        0.14,
        "Late fusion does not estimate a new 3D box here;\nit routes the detector cuboid according to image-space support.",
        transform=axes[1, 1].transAxes,
        fontsize=11,
        color=COLORS["gray"],
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def manifest_image_by_recording_frame(recording_name: str) -> dict[int, Path]:
    manifest = pd.read_csv(TEST_MANIFEST_CSV)
    rows = manifest[(manifest["recording_name"] == recording_name) & manifest["image_path"].notna()].copy()
    rows = rows.drop_duplicates("image_path").reset_index(drop=True)
    return {idx: Path(row["image_path"]) for idx, row in rows.iterrows()}


def make_temporal_tracking_figure() -> Path:
    path = FIGURES / "F3_temporal_tracking_persistent_id.png"
    if not P4_TEMPORAL_TRACKS_CSV.exists():
        return copy_or_placeholder(P4_TEMPORAL_TRACKS_CSV, path.name, "Temporal tracking example")
    tracks = pd.read_csv(P4_TEMPORAL_TRACKS_CSV)
    recording = "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"
    images_by_frame = manifest_image_by_recording_frame(recording)
    selected = tracks[(tracks["track_id"] == 4) & (tracks["frame_index"].between(108, 112))].sort_values("frame_index")
    if len(selected) < 5:
        selected = tracks[tracks["track_id"] == 0].sort_values("frame_index").head(5)

    fig, axes = plt.subplots(1, len(selected), figsize=(18, 3.55))
    if len(selected) == 1:
        axes = [axes]
    states: list[str] = []
    for ax, (_, row) in zip(axes, selected.iterrows()):
        frame_idx = int(row["frame_index"])
        img_path = images_by_frame.get(frame_idx)
        if img_path and img_path.exists():
            image = np.asarray(Image.open(img_path).convert("RGB"))
            ax.imshow(image)
        else:
            ax.imshow(np.ones((375, 672, 3), dtype=np.uint8) * 245)
        bbox = [row["projected_x1"], row["projected_y1"], row["projected_x2"], row["projected_y2"]]
        if np.isfinite(np.asarray(bbox, dtype=float)).all():
            add_bbox(ax, bbox, COLORS["purple"], f"ID {int(row['track_id'])}", lw=2.6)
        state = str(row["track_state"]).capitalize()
        evidence = str(row["update_source"]).replace("_", " ")
        states.append(state)
        ax.text(
            0.02,
            0.98,
            f"Frame {frame_idx}\n{state}\n{evidence}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": COLORS["purple"], "alpha": 0.82, "pad": 2},
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.035, " -> ".join(states) + "        same track ID retained", ha="center", fontsize=12, color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def make_tentative_unmatched_tracking_figure() -> Path:
    path = FIGURES / "F3b_tentative_unmatched_tracking_example.png"
    recording = "out_vine_4swap+walk_st_ly_11_06_2024_2_label"
    tracks3d_path = P4_VINE_TRACKING_DIR / "tracked_3d_results.csv"
    tracks2d_path = P4_VINE_TRACKING_DIR / "tracked_2d_results.csv"
    if not tracks3d_path.exists():
        return copy_or_placeholder(tracks3d_path, path.name, "Tentative and unmatched tracking example")
    tracks3d = pd.read_csv(tracks3d_path)
    tracks2d = pd.read_csv(tracks2d_path) if tracks2d_path.exists() else pd.DataFrame()
    images_by_frame = manifest_image_by_recording_frame(recording)
    frames = [
        (0, "Tentative tracks created"),
        (1, "Tentative tracks updated"),
        (2, "Tentative tracks continue"),
        (28, "Unmatched LiDAR-only track retained"),
    ]
    fig, axes = plt.subplots(1, len(frames), figsize=(18, 3.8))
    for ax, (frame_idx, caption) in zip(axes, frames):
        img_path = images_by_frame.get(frame_idx)
        if img_path and img_path.exists():
            image = np.asarray(Image.open(img_path).convert("RGB"))
            ax.imshow(image)
        else:
            ax.imshow(np.ones((375, 672, 3), dtype=np.uint8) * 245)
        trk3 = tracks3d[tracks3d["frame_index"] == frame_idx].copy()
        if frame_idx in (0, 1, 2):
            trk3 = trk3[trk3["track_state"] == "tentative"].head(4)
        for _, trk in trk3.iterrows():
            bbox = [trk["projected_x1"], trk["projected_y1"], trk["projected_x2"], trk["projected_y2"]]
            if not np.isfinite(np.asarray(bbox, dtype=float)).all():
                continue
            source = str(trk["update_source"])
            state = str(trk["track_state"])
            color = COLORS["red"] if source == "lidar_only" else COLORS["green"] if source == "matched_camera_lidar" else COLORS["blue"]
            label = f"{state} ID {int(trk['track_id'])}"
            if source == "lidar_only":
                label += " LiDAR-only"
            add_bbox(ax, bbox, color, label, lw=2.0)
        if frame_idx == 28 and len(tracks2d):
            trk2 = tracks2d[tracks2d["frame_index"] == frame_idx].head(2)
            for _, trk in trk2.iterrows():
                add_bbox(ax, [trk["x1"], trk["y1"], trk["x2"], trk["y2"]], COLORS["blue"], f"2D ID {int(trk['track_id'])}", lw=1.6)
        ax.text(
            0.02,
            0.98,
            f"Frame {frame_idx}\n{caption}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86, "pad": 2},
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.025,
        "Green: matched camera-LiDAR 3D track    Red: unmatched LiDAR-only 3D track    Blue: camera-only / 2D support",
        ha="center",
        fontsize=11,
        color=COLORS["ink"],
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def plot_detector_adaptation_interaction(metrics: pd.DataFrame) -> Path:
    path = FIGURES / "F4_detector_adaptation_hota_interaction.png"
    by_id = metrics.set_index("combination_id")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True)
    panels = [
        ("A. SECOND", ["S1", "S3"], ["S2", "S4"], ["Generic SECOND", "Fine-tuned SECOND"]),
        ("B. PointPillars", ["P1", "P3"], ["P2", "P4"], ["Generic PointPillars", "Fine-tuned PointPillars"]),
    ]
    ymax = max(0.42, float(metrics["HOTA"].max()) * 1.18)
    for ax, (title, generic_yolo_ids, ft_yolo_ids, xlabels) in zip(axes, panels):
        x = np.arange(2)
        g_vals = [float(by_id.loc[key, "HOTA"]) for key in generic_yolo_ids]
        ft_vals = [float(by_id.loc[key, "HOTA"]) for key in ft_yolo_ids]
        ax.plot(x, g_vals, marker="o", linewidth=2.8, color=COLORS["blue"], label="Generic YOLO")
        ax.plot(x, ft_vals, marker="o", linewidth=2.8, color=COLORS["green"], label="FT YOLO")
        for xx, yy in zip(x, g_vals):
            ax.text(xx, max(0.012, yy - 0.026), f"{yy:.3f}", ha="center", fontsize=9, color=COLORS["blue"])
        for xx, yy in zip(x, ft_vals):
            ax.text(xx, yy + 0.012, f"{yy:.3f}", ha="center", fontsize=9, color=COLORS["green"])
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("HOTA")
        ax.legend(frameon=False, loc="upper left")
        prettify_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_per_recording_robustness(metrics: pd.DataFrame, per_recording: pd.DataFrame) -> Path:
    path = FIGURES / "F5_per_recording_robustness_s4_p4.png"
    selected = [
        ("Footpath", "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label"),
        ("Polytunnel", "in_straw_3pick_diff_st_10_24_2024_5_a_label"),
        ("Vineyard", "out_vine_4swap+walk_st_ly_11_06_2024_2_label"),
    ]
    rows: list[dict[str, object]] = []
    for scenario, recording in selected:
        sub = per_recording[(per_recording["recording"] == recording) & (per_recording["combination_id"].isin(["S4", "P4"]))]
        values = sub.set_index("combination_id")["HOTA"]
        rows.append(
            {
                "Scenario": scenario,
                "Selected bag used in this figure": recording,
                "S4": float(values.loc["S4"]) if "S4" in values.index else np.nan,
                "P4": float(values.loc["P4"]) if "P4" in values.index else np.nan,
            }
        )
    table_df = pd.DataFrame(rows)
    save_table(table_df, "F5_representative_scenario_s4_p4_hota.csv")

    x = np.arange(len(table_df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    bars_s4 = ax.bar(x - width / 2, table_df["S4"], width=width, color=COLORS["blue"], label="FT YOLO + FT SECOND")
    bars_p4 = ax.bar(x + width / 2, table_df["P4"], width=width, color=COLORS["orange"], label="FT YOLO + FT PointPillars")
    ax.set_ylabel("HOTA")
    ax.set_xticks(x)
    ax.set_xticklabels(table_df["Scenario"])
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, 0.5)
    prettify_axes(ax)
    ax.bar_label(bars_s4, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars_p4, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def add_bev_box(ax, row: pd.Series, color: str, label: str, lw: float = 2.1, alpha: float = 1.0) -> None:
    corners = bev_box_corners(float(row["x"]), float(row["y"]), float(row["length"]), float(row["width"]), float(row["yaw"]))
    ax.add_patch(patches.Polygon(corners[:, [1, 0]], closed=True, fill=False, edgecolor=color, linewidth=lw, alpha=alpha))
    ax.scatter([float(row["y"])], [float(row["x"])], c=color, s=16, alpha=alpha)
    ax.text(float(row["y"]), float(row["x"]), label, color=color, fontsize=8, weight="bold")


def make_ros2_playback_qualitative_figure() -> Path:
    path = FIGURES / "F6_ros2_playback_deepfusionmot_vineyard_example.png"
    recording = "out_vine_4swap+walk_st_ly_11_06_2024_2_label"
    frame_index = 28
    manifest = pd.read_csv(TEST_MANIFEST_CSV)
    rows = manifest[(manifest["recording_name"] == recording) & manifest["image_path"].notna()].drop_duplicates("sample_id").reset_index(drop=True)
    if frame_index >= len(rows):
        return copy_or_placeholder(TEST_MANIFEST_CSV, path.name, "ROS 2 playback qualitative example")

    row = rows.iloc[frame_index]
    sample_id = str(row["sample_id"])
    image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    xyz = read_pcd_xyz(Path(row["lidar_point_cloud_path"]))
    yolo = pd.read_csv(YOLO_FT_CACHE)
    pointpillars = pd.read_csv(POINTPILLARS_FT_CACHE)
    tracks3d = pd.read_csv(P4_VINE_TRACKING_DIR / "tracked_3d_results.csv")
    tracks2d = pd.read_csv(P4_VINE_TRACKING_DIR / "tracked_2d_results.csv")
    yolo_rows = yolo[yolo["sample_id"] == sample_id].copy()
    lidar_rows = pointpillars[pointpillars["sample_id"] == sample_id].copy()
    trk3d_rows = tracks3d[tracks3d["frame_index"] == frame_index].copy()
    trk2d_rows = tracks2d[tracks2d["frame_index"] == frame_index].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.6))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].imshow(image)
    for idx, det in yolo_rows.head(8).iterrows():
        add_bbox(axes[0, 0], [det["x1"], det["y1"], det["x2"], det["y2"]], COLORS["blue"], f"YOLO {float(det['confidence']):.2f}", lw=2.0)
    axes[0, 0].set_title("A. Raw live YOLO 2D detections", loc="left", weight="bold")

    for ax in [axes[0, 1], axes[1, 1]]:
        if len(xyz):
            mask = (xyz[:, 0] > 0) & (xyz[:, 0] < 8) & (np.abs(xyz[:, 1]) < 4)
            pts = xyz[mask]
            if len(pts) > 9000:
                pts = pts[np.linspace(0, len(pts) - 1, 9000).astype(int)]
            ax.scatter(pts[:, 1], pts[:, 0], s=0.25, c="#64748b", alpha=0.35)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(7.2, 0.4)
        ax.set_xlabel("left/right y (m)")
        ax.set_ylabel("forward x (m)")
        ax.grid(alpha=0.18)

    for _, det in lidar_rows.head(12).iterrows():
        add_bev_box(axes[0, 1], det, COLORS["yellow"], f"{float(det['score']):.2f}", lw=1.8, alpha=0.9)
    axes[0, 1].set_title("B. Raw live PointPillars 3D detections", loc="left", weight="bold")

    axes[1, 0].imshow(image)
    for _, trk in trk3d_rows.iterrows():
        bbox = [trk["projected_x1"], trk["projected_y1"], trk["projected_x2"], trk["projected_y2"]]
        if not np.isfinite(np.asarray(bbox, dtype=float)).all():
            continue
        source = str(trk["update_source"])
        color = COLORS["green"] if source == "matched_camera_lidar" else COLORS["red"] if source == "lidar_only" else COLORS["blue"]
        label = f"ID {int(trk['track_id'])}"
        if source == "lidar_only":
            label += " LiDAR-only"
        add_bbox(axes[1, 0], bbox, color, label, lw=2.1)
    for _, trk in trk2d_rows.iterrows():
        add_bbox(axes[1, 0], [trk["x1"], trk["y1"], trk["x2"], trk["y2"]], COLORS["blue"], f"2D ID {int(trk['track_id'])}", lw=1.6)
    axes[1, 0].set_title("C. DeepFusionMOT projection image", loc="left", weight="bold")

    for _, trk in trk3d_rows.iterrows():
        source = str(trk["update_source"])
        color = COLORS["green"] if source == "matched_camera_lidar" else COLORS["red"] if source == "lidar_only" else COLORS["blue"]
        add_bev_box(axes[1, 1], trk, color, f"ID {int(trk['track_id'])}", lw=2.3)
    axes[1, 1].set_title("D. DeepFusionMOT tracks on point cloud", loc="left", weight="bold")
    axes[1, 1].text(
        0.02,
        0.98,
        "Green: matched 3D\nRed: LiDAR-only / unmatched\nBlue: camera-supported",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86},
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def copy_ros2_rviz_playback_figure() -> Path:
    return copy_or_placeholder(
        RVIZ_PLAYBACK_CAPTURE,
        "F6_ros2_playback_deepfusionmot_rviz.png",
        "Real RViz ROS 2 playback capture",
    )


def copy_ros2_tracked_projection_figure() -> Path:
    return copy_or_placeholder(
        ROS2_TRACKED_PROJECTION_CAPTURE,
        "F6_ros2_deepfusionmot_projection_image.png",
        "ROS 2 playback tracked DeepFusionMOT projection image",
    )


def copy_ros2_tracked_pointcloud_figure() -> Path:
    return copy_resized_or_placeholder(
        ROS2_TRACKED_POINTCLOUD_CAPTURE,
        "F7_ros2_deepfusionmot_pointcloud_tracks.png",
        "ROS 2 playback DeepFusionMOT tracks on point cloud",
        max_width=372,
    )



def plot_threshold_appendix(sweep: pd.DataFrame) -> Path:
    path = FIGURES / "A1_validation_lidar_threshold_sensitivity.png"
    df = sweep[
        (sweep["recording"] == "COMBINED")
        & (np.isclose(sweep["min_camera_score"], 0.10))
        & (sweep["output_mode"] == "deepfusionmot")
    ].copy()
    df = df.sort_values("min_lidar_score")
    save_table(df[["min_lidar_score", "HOTA", "IDF1", "MOTA", "tracker_dets", "CLR_FP", "CLR_FN", "IDSW"]], "A1_validation_lidar_threshold_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.plot(df["min_lidar_score"], df["HOTA"], marker="o", color=COLORS["blue"], linewidth=2.5)
    ax.axvline(0.20, color=COLORS["ink"], linestyle="--", linewidth=1.4)
    selected = df[np.isclose(df["min_lidar_score"], 0.20)]
    if len(selected):
        ax.scatter([0.20], [float(selected.iloc[0]["HOTA"])], s=96, color=COLORS["green"], zorder=4)
        ax.text(0.205, float(selected.iloc[0]["HOTA"]) + 0.008, "selected threshold", fontsize=9, color=COLORS["green"])
    ax.set_title("Appendix A1. Validation HOTA sensitivity to LiDAR confidence threshold", weight="bold")
    ax.set_xlabel("Minimum LiDAR confidence threshold")
    ax.set_ylabel("Validation HOTA")
    ax.text(
        0.02,
        0.95,
        "Validation split only; camera score fixed at 0.10.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.86},
    )
    prettify_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_live_fps(speed: pd.DataFrame) -> Path:
    df = speed.copy()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    bars = ax.bar(df["combination_id"], df["live_ros2_end_to_end_fps"], color="#0891b2")
    ax.set_title("Live ROS 2 End-To-End Tracking Throughput Across Detector Combinations", weight="bold")
    ax.set_ylabel("Live ROS 2 end-to-end FPS")
    ax.set_ylim(0, max(8.5, float(df["live_ros2_end_to_end_fps"].max()) * 1.2))
    prettify_axes(ax)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    for i, row in df.iterrows():
        ax.text(i, -0.065, row["combination_label"], ha="center", va="top", rotation=35, fontsize=8, transform=ax.get_xaxis_transform())
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    path = FIGURES / "F5_live_ros2_fps_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_latency_summary(speed: pd.DataFrame) -> Path:
    df = speed.copy()
    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar(x - width, df["yolo_latency_mean_ms"], width=width, label="YOLO mean ms", color="#2563eb")
    ax.bar(x, df["lidar_latency_mean_ms"], width=width, label="LiDAR mean ms", color="#ca8a04")
    ax.bar(x + width, df["tracking_latency_mean_ms"], width=width, label="Tracking mean ms", color="#9333ea")
    ax.set_title("Live ROS 2 Mean Latency Components", weight="bold")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["combination_id"])
    ax.legend(ncol=3, frameon=False)
    prettify_axes(ax)
    fig.tight_layout()
    path = FIGURES / "F6_live_ros2_latency_components.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_contact_sheet(items: list[tuple[Path, str]], out_path: Path, title: str, cols: int = 3, thumb: tuple[int, int] = (560, 315)) -> Path:
    title_font = load_font(34, bold=True)
    label_font = load_font(20, bold=True)
    body_font = load_font(16)
    rows = int(np.ceil(len(items) / cols))
    pad = 28
    label_h = 48
    title_h = 72
    w = cols * thumb[0] + (cols + 1) * pad
    h = title_h + rows * (thumb[1] + label_h + pad) + pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill=COLORS["ink"], font=title_font)
    for idx, (src, label) in enumerate(items):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb[0] + pad)
        y = title_h + row * (thumb[1] + label_h + pad)
        draw.rounded_rectangle((x, y, x + thumb[0], y + thumb[1] + label_h), radius=10, outline="#cbd5e1", width=2, fill="#f8fafc")
        draw.text((x + 12, y + 12), label, fill=COLORS["ink"], font=label_font)
        if src.exists():
            img = Image.open(src).convert("RGB")
            img.thumbnail((thumb[0] - 24, thumb[1] - 12), Image.Resampling.LANCZOS)
            ix = x + (thumb[0] - img.width) // 2
            iy = y + label_h + (thumb[1] - img.height) // 2
            canvas.paste(img, (ix, iy))
        else:
            wrapped = textwrap.fill(f"Missing: {src}", 48)
            draw.text((x + 18, y + label_h + 40), wrapped, fill=COLORS["red"], font=body_font)
    canvas.save(out_path)
    return out_path


def make_color_legend() -> Path:
    path = FIGURES / "F9_ros2_visualization_color_legend.png"
    w, h = 1500, 820
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(36, bold=True)
    label_font = load_font(25, bold=True)
    body_font = load_font(20)
    draw.text((55, 40), "ROS 2 Late-Fusion And Tracking Visualisation Colours", fill=COLORS["ink"], font=title_font)
    entries = [
        ("Blue", "YOLO 2D camera detections", COLORS["blue"]),
        ("Yellow", "Original live LiDAR detector boxes", COLORS["yellow"]),
        ("Green", "Matched camera-supported 3D boxes", COLORS["green"]),
        ("Red", "Unmatched LiDAR boxes", COLORS["red"]),
        ("Cyan", "Canonical fused / ALL3D output", COLORS["cyan"]),
        ("Purple", "DeepFusionMOT tracked boxes with IDs", COLORS["purple"]),
    ]
    for idx, (name, desc, color) in enumerate(entries):
        x = 80 + (idx % 2) * 710
        y = 140 + (idx // 2) * 150
        draw.rounded_rectangle((x, y, x + 610, y + 105), radius=14, outline="#cbd5e1", width=2, fill="#f8fafc")
        draw.rectangle((x + 24, y + 28, x + 92, y + 86), outline=color, width=7)
        draw.text((x + 120, y + 22), name, fill=color, font=label_font)
        draw.text((x + 120, y + 58), desc, fill=COLORS["ink"], font=body_font)
    draw.rounded_rectangle((85, 610, 1415, 750), radius=16, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text((120, 635), "Main visual flow", fill=COLORS["ink"], font=label_font)
    flow = "ZED image + point cloud -> live detectors -> late-fusion match/unmatched/ALL3D -> DeepFusionMOT tracks -> RViz overlays"
    draw.text((120, 682), flow, fill=COLORS["gray"], font=body_font)
    img.save(path)
    return path


def copy_or_placeholder(src: Path, out_name: str, title: str) -> Path:
    dst = FIGURES / out_name
    if src.exists():
        shutil.copy2(src, dst)
    else:
        img = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), title, fill=COLORS["ink"], font=load_font(34, True))
        draw.text((50, 125), f"Missing source: {src}", fill=COLORS["red"], font=load_font(20))
        img.save(dst)
    return dst


def copy_resized_or_placeholder(src: Path, out_name: str, title: str, max_width: int) -> Path:
    dst = FIGURES / out_name
    if src.exists():
        img = Image.open(src).convert("RGB")
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (max_width, int(round(img.height * scale)))
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img = img.resize(new_size, resample)
        img.save(dst)
    else:
        img = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), title, fill=COLORS["ink"], font=load_font(34, True))
        draw.text((50, 125), f"Missing source: {src}", fill=COLORS["red"], font=load_font(20))
        img.save(dst)
    return dst


def write_manifest(rows: list[tuple[str, Path, str]]) -> None:
    with (ASSETS / "visual_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["figure_id", "path", "caption"])
        for figure_id, path, caption in rows:
            writer.writerow([figure_id, path.relative_to(ASSETS), caption])


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    view = df[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{value:.4f}")
    return view.to_markdown(index=False)


def metric_definitions_table() -> str:
    return textwrap.dedent(
        """
        | Metric | What does it mean? | Equation |
        |---|---|---|
        | HOTA (Higher Order Tracking Accuracy) (↑) | Overall tracking quality. Balances correct person detection and correct identity association. | `HOTA ≈ sqrt(DetA × AssA)` |
        | DetA (Detection Accuracy) (↑) | Detection accuracy within tracking. Measures whether people were correctly detected, without too many missed or false boxes. | `DetA = TP / (TP + FP + FN)` |
        | AssA (Association Accuracy) (↑) | Association accuracy. Measures whether each person keeps the correct tracking ID across frames. | Calculated from the proportion of correctly associated detections over time. |
        | IDSW (Identity Switches) (↓) | Number of identity switches. When the same person changes from one predicted track ID to another. | Example: Person A is Track 2 and later becomes Track 6 → ID switch. |
        | MOTP (Multi-Object Tracking Precision) (↑) | Localisation precision of matched tracked boxes. Measures how well predicted boxes overlap the matched GT boxes. | `MOTP = sum of matched IoU values / number of matches` |
        | MOTA (Multi-Object Tracking Accuracy) (↑) | Overall tracking accuracy based on missed detections, false detections, and identity switches. Negative value means the total tracking errors exceeded the number of GT objects. | `MOTA = 1 - (FN + FP + IDSW) / GT` |
        | IDF1 (Identity F1 score) (↑) | Identity-tracking accuracy. Measures how well the predicted identities match the ground-truth identities across the sequence. | `IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)` |
        | FPS (↑) | ROS 2 live playback throughput. Higher means the online system produced tracked outputs faster during playback. | `FPS = tracked output frames / measured playback time` |
        """
    ).strip()


def test_data_table() -> str:
    return textwrap.dedent(
        """
        | Test recording | Scenario | Robot motion |
        |---|---|---|
        | `footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label` | Footpath | Moving |
        | `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` | Footpath | Stationary |
        | `in_straw_3pick_diff_st_10_24_2024_5_a_label` | Polytunnel | Stationary |
        | `out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label` | Polytunnel/outdoor straw | Stationary |
        | `out_vine_1push_3carry_st_ly_11_06_2024_1_label` | Vineyard | Stationary |
        | `out_vine_4swap+walk_st_ly_11_06_2024_2_label` | Vineyard | Stationary |
        """
    ).strip()


def representative_scenario_table() -> str:
    return textwrap.dedent(
        """
        | Scenario | Selected bag used in this figure |
        |---|---|
        | Footpath | `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` |
        | Polytunnel | `in_straw_3pick_diff_st_10_24_2024_5_a_label` |
        | Vineyard | `out_vine_4swap+walk_st_ly_11_06_2024_2_label` |
        """
    ).strip()


def main_results_table(metrics: pd.DataFrame) -> str:
    df = metrics.copy()
    df["FPS"] = df["combination_id"].map(FINAL_ROS2_FPS)
    cols = ["combination_id", "combination_label", "HOTA", "DetA", "AssA", "IDSW", "MOTP", "MOTA", "IDF1", "FPS"]
    best_high = {col: float(df[col].max()) for col in ["HOTA", "DetA", "AssA", "MOTP", "MOTA", "IDF1", "FPS"]}
    best_low = {"IDSW": float(df["IDSW"].min())}
    headers = ["ID", "Combination", "HOTA", "DetA", "AssA", "IDSW", "MOTP", "MOTA", "IDF1", "FPS"]
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df[cols].iterrows():
        values: list[str] = [str(row["combination_id"]), str(row["combination_label"])]
        for col in ["HOTA", "DetA", "AssA"]:
            text = f"{float(row[col]):.4f}"
            if np.isclose(float(row[col]), best_high[col]):
                text = f"**{text}**"
            values.append(text)
        idsw = f"{int(row['IDSW'])}"
        if np.isclose(float(row["IDSW"]), best_low["IDSW"]):
            idsw = f"**{idsw}**"
        values.append(idsw)
        for col in ["MOTP", "MOTA", "IDF1"]:
            text = f"{float(row[col]):.4f}"
            if np.isclose(float(row[col]), best_high[col]):
                text = f"**{text}**"
            values.append(text)
        fps = f"{float(row['FPS']):.2f}"
        if np.isclose(float(row["FPS"]), best_high["FPS"]):
            fps = f"**{fps}**"
        values.append(fps)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def create_notebook(metrics: pd.DataFrame, speed: pd.DataFrame, manifest_rows: list[tuple[str, Path, str]]) -> None:
    nb = nbf.v4.new_notebook()
    rel_base = "../results/late_fusion_paper_assets/selected_notebook_assets"
    result_table = main_results_table(metrics)
    cells = [
        nbf.v4.new_markdown_cell(
            "# AGHRI ZED RGB-Livox Late Fusion and Tracking Results Summary\n\n"
            "This notebook summarises the selected AGHRI camera-LiDAR fusion and tracking figures and tables. "
            "The figures shown here are the selected saved figures from `results/late_fusion_paper_assets/selected_notebook_assets`."
        ),
        nbf.v4.new_markdown_cell(
            "## Methods Compared\n\n"
            "The late-fusion framework evaluates eight detector combinations. The camera detector is either generic (G) YOLO11s or AGHRI fine-tuned (FT) YOLO11s. "
            "The LiDAR detector is either SECOND or PointPillars, each in generic (G) or AGHRI fine-tuned form (FT).\n\n"
            "In each combination, YOLO produces 2D person detections from the ZED RGB image, while SECOND or PointPillars produces 3D person detections from the Livox point cloud. "
            "The late-fusion stage projects each 3D LiDAR box into the camera image, matches it with YOLO using 2D IoU, and forwards matched/unmatched detections into the DeepFusionMOT-style tracker."
        ),
        nbf.v4.new_markdown_cell(
            "## Metrics Used\n\n"
            "Tracking metrics are calculated with TrackEval, an evaluation toolkit for multi-object tracking. TrackEval compares the tracker predictions with ground truth and calculates tracking quality, detection quality, association quality, localisation precision, and identity metrics.\n\n"
            "Abbreviations used in the equations: `GT` is the number of ground-truth objects; `TP`, `FP`, and `FN` are true positives, false positives, and false negatives; "
            "`IDTP`, `IDFP`, and `IDFN` are identity true positives, identity false positives, and identity false negatives.\n\n"
            f"{metric_definitions_table()}"
        ),
        nbf.v4.new_markdown_cell(
            "## Test Data Used\n\n"
            "The frozen held-out benchmark uses all six official AGHRI test recordings.\n\n"
            f"{test_data_table()}\n\n"
            "The main results table and Figure 4 use all six recordings. Figure 5 is a compact scenario-level view using one selected representative bag per scenario, listed under that figure."
        ),
        nbf.v4.new_markdown_cell(
            "## Main Results Table\n\n"
            f"{result_table}\n\n"
            "**Key observation.** The best overall tracking result by HOTA is obtained by **FT YOLO + FT PointPillars**, which also gives the highest DetA and AssA. "
            "**G YOLO + FT PointPillars** gives a very slightly higher IDF1, while both fine-tuned PointPillars combinations have the lowest IDSW. "
            "Fine-tuned PointPillars gives a clear improvement over generic PointPillars, and it also outperforms the SECOND combinations in HOTA and association consistency. "
            "The highest live ROS 2 FPS is obtained by **G YOLO + G PointPillars**, while the best accuracy combination remains **FT YOLO + FT PointPillars**."
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 1: AGHRI Late-Fusion And Deep Association Architecture\n\n"
            f"![Complete late-fusion and tracking pipeline]({rel_base}/figures/F1_complete_late_fusion_tracking_pipeline.png)\n\n"
            "**Overview of the proposed AGHRI camera-LiDAR late-fusion and tracking framework.** YOLO produces camera-based 2D detections, while SECOND or PointPillars produces LiDAR-based 3D cuboids. "
            "After detector-output temporal synchronisation, projected LiDAR envelopes and camera boxes are associated using image-space IoU and Hungarian assignment. "
            "Matched 3D, LiDAR-only and camera-only detections are then processed through the four-level DeepFusionMOT association cascade, with Kalman prediction, AGHRI odometry compensation and lifecycle management producing persistent 3D and 2D tracks."
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 2: Same-Frame Camera-LiDAR Association Example\n\n"
            f"![Same-frame camera-LiDAR association]({rel_base}/figures/F2_same_frame_late_fusion_association_example.png)\n\n"
            "This figure follows one real AGHRI frame through the late-fusion decision. A YOLO 2D person box is compared with the projected LiDAR cuboid envelope; the image-space IoU is approximately `0.5967`, above the `0.10` fusion threshold, so the LiDAR cuboid is routed as a matched camera-supported 3D detection."
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 3: Temporal Tracking And Persistent ID Continuity\n\n"
            f"![Temporal tracking persistent ID]({rel_base}/figures/F3_temporal_tracking_persistent_id.png)\n\n"
            "This qualitative example shows consecutive real frames from the moving footpath recording. The same person remains under the same confirmed DeepFusionMOT track ID across frames, illustrating what the tracking stage adds beyond independent frame-level detections."
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 4: Effect Of Detector Adaptation On HOTA\n\n"
            f"![Detector adaptation HOTA interaction]({rel_base}/figures/F4_detector_adaptation_hota_interaction.png)\n\n"
            "This two-panel interaction plot isolates the effect of detector adaptation. Fine-tuning the LiDAR detector produces the dominant HOTA improvement, while changing only the camera checkpoint gives a smaller change. The strongest HOTA is obtained by the fine-tuned PointPillars setting."
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 5: Per-Recording Robustness Of The Two Fully Fine-Tuned Systems\n\n"
            f"![Per-recording robustness]({rel_base}/figures/F5_per_recording_robustness_s4_p4.png)\n\n"
            "This figure is a compact scenario-level view. It uses one selected representative bag per scenario rather than all six test bags:\n\n"
            f"{representative_scenario_table()}\n\n"
            "It compares the two fully fine-tuned final systems: **FT YOLO + FT SECOND** and **FT YOLO + FT PointPillars**."
        ),
        nbf.v4.new_markdown_cell(
            f"![ROS 2 playback DeepFusionMOT projection image]({rel_base}/figures/F6_ros2_deepfusionmot_projection_image.png)"
        ),
        nbf.v4.new_markdown_cell(
            "## Figure 7: ROS 2 Playback DeepFusionMOT Tracks On Point Cloud\n\n"
            f"![ROS 2 playback DeepFusionMOT point-cloud tracks]({rel_base}/figures/F7_ros2_deepfusionmot_pointcloud_tracks.png)\n\n"
            "This is the corresponding RViz point-cloud view from the same ROS 2 playback workflow. "
            "Only the tracked DeepFusionMOT markers are shown here, so the figure focuses on the final persistent 3D track output rather than the intermediate original LiDAR, matched, or unmatched detector markers."
        ),
        nbf.v4.new_markdown_cell(
            "## Summary\n\n"
            "Across the frozen six-recording AGHRI benchmark, the strongest final tracking result is obtained by **FT YOLO + FT PointPillars**. "
            "The main quantitative gain comes from LiDAR detector adaptation: fine-tuned PointPillars gives the best HOTA, DetA and AssA, while the fine-tuned SECOND setting also improves clearly over generic SECOND. "
            "The ROS 2 playback figures show the same framework running online with live detector nodes, late-fusion association, DeepFusionMOT tracking, and RViz visualisation. "
            "Together, the quantitative tables and qualitative playback examples show that the completed framework supports both offline TrackEval-style benchmarking and live camera-LiDAR tracking visualisation on AGHRI rosbag data."
        ),
    ]
    nb["cells"] = cells
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+/figures/([^)/]+\.png))\)")
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        attachments = cell.setdefault("attachments", {})

        def attach(match: re.Match) -> str:
            alt_text = match.group(1)
            filename = match.group(3)
            image_path = FIGURES / filename
            if image_path.exists():
                attachments[filename] = {"image/png": base64.b64encode(image_path.read_bytes()).decode("ascii")}
                return f"![{alt_text}](attachment:{filename})"
            return match.group(0)

        cell["source"] = image_pattern.sub(attach, cell["source"])
        if not attachments:
            cell.pop("attachments", None)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NOTEBOOK)


def write_readme(manifest_rows: list[tuple[str, Path, str]]) -> None:
    lines = [
        "# AGHRI Late-Fusion Paper Assets Draft",
        "",
        "This folder contains draft figures and derived tables for the late-fusion and DeepFusionMOT tracking paper-assets notebook.",
        "",
        "Notebook:",
        "",
        f"- `{NOTEBOOK.relative_to(REPO)}`",
        "",
        "Selected figures:",
        "",
    ]
    for figure_id, path, caption in manifest_rows:
        lines.append(f"- `{figure_id}`: `{path.relative_to(ASSETS)}` - {caption}")
    lines.extend(
        [
            "",
            "These assets are generated from existing final AGHRI results and visual walkthrough outputs. They are a review draft, not a new evaluation run.",
            "",
        ]
    )
    (ASSETS / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    metrics = pd.read_csv(METRICS_CSV)
    per_recording = pd.read_csv(PER_RECORDING_CSV)
    speed = pd.read_csv(SPEED_CSV)
    per_recording_speed = pd.read_csv(PER_RECORDING_SPEED_CSV)
    save_table(metrics, "final_aghri_tracking_metrics.csv")
    save_table(per_recording, "final_aghri_tracking_metrics_per_recording.csv")
    save_table(speed, "final_live_ros2_speed_aggregate.csv")
    save_table(per_recording_speed, "final_live_ros2_speed_per_recording.csv")

    manifest_rows: list[tuple[str, Path, str]] = []
    manifest_rows.append(("F1", make_pipeline_figure(), "Complete late-fusion and DeepFusionMOT tracking pipeline"))
    manifest_rows.append(("F2", make_same_frame_association_figure(), "Same-frame camera-LiDAR association with routed fusion output"))
    manifest_rows.append(("F3", make_temporal_tracking_figure(), "Temporal tracking example with persistent ID"))
    manifest_rows.append(("F4", plot_detector_adaptation_interaction(metrics), "Two-panel detector adaptation interaction plot for HOTA"))
    manifest_rows.append(("F5", plot_per_recording_robustness(metrics, per_recording), "Per-recording robustness for S4 versus P4"))
    manifest_rows.append(("F6", copy_ros2_tracked_projection_figure(), "Real ROS 2 playback DeepFusionMOT projection image"))
    manifest_rows.append(("F7", copy_ros2_tracked_pointcloud_figure(), "Real ROS 2 playback DeepFusionMOT tracks on point cloud"))

    write_manifest(manifest_rows)
    write_readme(manifest_rows)
    create_notebook(metrics, speed, manifest_rows)
    print(f"Wrote assets to: {ASSETS}")
    print(f"Wrote notebook to: {NOTEBOOK}")


if __name__ == "__main__":
    main()
