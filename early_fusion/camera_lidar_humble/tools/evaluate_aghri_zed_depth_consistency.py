#!/usr/bin/env python3
"""Compare AGHRI fused LiDAR Zc with robust ZED depth-map depth.

This is a consistency check, not a 3D localization benchmark. The ZED depth
map gives per-pixel surface depth, so we compare fused camera-frame Zc against
the median valid depth inside a central region of the detected person box.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RUN = Path(
    "/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2/"
    "runs/finetuned_legacy/detections.json"
)
OUT = Path(
    "/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/results/"
    "aghri_zed_depth_consistency"
)
SELECTED_FIGURES = Path(
    "/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/results/"
    "selected_notebook_assets/figures"
)

MIN_VALID_PIXELS = 25
MIN_DEPTH_M = 0.10
MAX_NON_SATURATED_DEPTH_M = 5.95
NEAREST_DEPTH_TOLERANCE_SEC = 0.04


def timestamp_from_name(path: Path) -> float:
    stem = path.stem
    sec, nsec = stem.split("_", 1)
    return float(sec) + float(f"0.{nsec}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def plot_relative_error_cdf(rel_diffs: list[float]) -> None:
    if not rel_diffs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_pct = np.sort(100.0 * np.asarray(rel_diffs, dtype=np.float32))
    cdf = np.arange(1, values_pct.size + 1, dtype=np.float32) / values_pct.size
    median_pct = float(np.median(values_pct))
    p95_pct = float(np.percentile(values_pct, 95))

    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=160)
    ax.plot(values_pct, cdf, color="#2f5f98", linewidth=2.4)
    ax.axvline(median_pct, color="#d07028", linewidth=2.0, linestyle="--")
    ax.scatter([median_pct], [0.5], color="#d07028", s=38, zorder=3)
    ax.text(
        median_pct + 0.8,
        0.52,
        f"median = {median_pct:.1f}%",
        color="#9d4d14",
        fontsize=10,
        va="bottom",
    )

    ax.set_xlim(0.0, min(max(40.0, p95_pct * 1.18), float(values_pct[-1])))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("relative absolute Z-depth error (%)", labelpad=8)
    ax.set_ylabel("cumulative fraction of detections", labelpad=8)
    ax.set_title("CDF of relative absolute Z-depth error", pad=12, fontsize=13)
    ax.grid(axis="both", color="#d9e0e8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    SELECTED_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUT, SELECTED_FIGURES):
        fig.savefig(directory / "F_ZED_relative_abs_z_error_cdf.png", bbox_inches="tight")
        fig.savefig(directory / "F_ZED_relative_abs_z_error_cdf.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_zed_vs_fused_scatter(valid_rows: list[dict[str, Any]]) -> None:
    pairs = [
        (float(row["zed_depth_median_m"]), float(row["fused_zc_m"]))
        for row in valid_rows
        if row.get("zed_depth_median_m") is not None and row.get("fused_zc_m") is not None
    ]
    if not pairs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zed = np.asarray([p[0] for p in pairs], dtype=np.float32)
    fused = np.asarray([p[1] for p in pairs], dtype=np.float32)

    upper = float(max(6.0, np.percentile(np.concatenate([zed, fused]), 99) * 1.05))
    upper = min(upper, 8.0)

    fig, ax = plt.subplots(figsize=(6.3, 5.8), dpi=160)
    ax.scatter(
        zed,
        fused,
        s=11,
        color="#2f5f98",
        alpha=0.34,
        linewidths=0,
    )
    ax.plot([0.0, upper], [0.0, upper], color="#d07028", linestyle="--", linewidth=2.0)

    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ZED median depth in detected person ROI (m)", labelpad=8)
    ax.set_ylabel("fused Zc depth (m)", labelpad=8)
    ax.set_title("ZED depth vs fused Zc depth", pad=12, fontsize=13)
    ax.grid(axis="both", color="#d9e0e8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    SELECTED_FIGURES.mkdir(parents=True, exist_ok=True)
    for directory in (OUT, SELECTED_FIGURES):
        fig.savefig(directory / "F_ZED_depth_vs_fused_lidar_zc_scatter.png", bbox_inches="tight")
        fig.savefig(directory / "F_ZED_depth_vs_fused_lidar_zc_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_xyz(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or len(value) < 3:
        return None
    xyz = [float(value[0]), float(value[1]), float(value[2])]
    if not all(math.isfinite(v) for v in xyz):
        return None
    return xyz


def central_person_roi(box: list[float], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    height, width = shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    # Avoid box edges/background: central torso-like region of the person box.
    rx1 = int(round(x1 + 0.20 * bw))
    rx2 = int(round(x2 - 0.20 * bw))
    ry1 = int(round(y1 + 0.20 * bh))
    ry2 = int(round(y2 - 0.15 * bh))
    rx1 = max(0, min(width, rx1))
    rx2 = max(0, min(width, rx2))
    ry1 = max(0, min(height, ry1))
    ry2 = max(0, min(height, ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return rx1, ry1, rx2, ry2


class DepthLookup:
    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, Any]] = {}

    def for_recording(self, recording: Path) -> dict[str, Any]:
        if recording in self._cache:
            return self._cache[recording]
        ann = read_json(recording / "annotations/cam_zed_rgb_ann.json")
        depth_dir = recording / "sensor_data/cam_zed_depth"
        depth_files = sorted(depth_dir.glob("*.npy"))
        by_stem = {p.stem: p for p in depth_files}
        by_time = [(timestamp_from_name(p), p) for p in depth_files]
        data = {"ann": ann, "by_stem": by_stem, "by_time": by_time}
        self._cache[recording] = data
        return data

    def depth_path(self, recording: Path, camera_frame_index: int, image_timestamp: float) -> tuple[Path | None, float | None]:
        data = self.for_recording(recording)
        ann = data["ann"]
        image_file = None
        if 0 <= camera_frame_index < len(ann):
            image_file = ann[camera_frame_index].get("File")
        if image_file:
            exact = data["by_stem"].get(Path(image_file).stem)
            if exact is not None:
                return exact, 0.0
        if not data["by_time"]:
            return None, None
        best_t, best_path = min(data["by_time"], key=lambda item: abs(item[0] - image_timestamp))
        delta = abs(best_t - image_timestamp)
        if delta <= NEAREST_DEPTH_TOLERANCE_SEC:
            return best_path, delta
        return None, delta


def evaluate() -> None:
    detections = read_json(RUN)
    lookup = DepthLookup()
    per_rows: list[dict[str, Any]] = []
    attempted = 0

    for row in detections:
        if not row.get("valid"):
            continue
        xyz = parse_xyz(row.get("mean_camera_xyz") or row.get("published_xyz"))
        if xyz is None or xyz[2] <= 0:
            continue
        box = row.get("box")
        if not box:
            continue
        attempted += 1
        recording = Path(row["recording_path"])
        depth_path, depth_delta = lookup.depth_path(
            recording,
            int(row["camera_frame_index"]),
            float(row.get("image_timestamp", 0.0)),
        )
        out = {
            "recording": row["recording"],
            "scene": row.get("scene"),
            "motion": row.get("motion"),
            "frame_number": row.get("frame_number"),
            "camera_frame_index": row.get("camera_frame_index"),
            "detection_index": row.get("detection_index"),
            "confidence": row.get("confidence"),
            "fused_zc_m": xyz[2],
            "depth_map_file": None if depth_path is None else depth_path.name,
            "depth_time_delta_sec": depth_delta,
            "depth_roi_valid": False,
            "depth_roi_valid_pixel_count": 0,
            "zed_depth_median_m": None,
            "zed_depth_iqr_m": None,
            "signed_z_diff_m": None,
            "abs_z_diff_m": None,
            "relative_abs_z_diff": None,
            "skip_reason": None,
        }
        if depth_path is None:
            out["skip_reason"] = "no_matching_depth_map"
            per_rows.append(out)
            continue
        depth = np.load(depth_path)
        if depth.ndim != 2:
            out["skip_reason"] = "depth_map_not_2d"
            per_rows.append(out)
            continue
        roi = central_person_roi(box, depth.shape)
        if roi is None:
            out["skip_reason"] = "invalid_box_roi"
            per_rows.append(out)
            continue
        x1, y1, x2, y2 = roi
        values = depth[y1:y2, x1:x2]
        valid = (
            np.isfinite(values)
            & (values > MIN_DEPTH_M)
            & (values < MAX_NON_SATURATED_DEPTH_M)
        )
        valid_values = values[valid].astype(np.float32)
        out["depth_roi_valid_pixel_count"] = int(valid_values.size)
        if valid_values.size < MIN_VALID_PIXELS:
            out["skip_reason"] = "too_few_valid_depth_pixels"
            per_rows.append(out)
            continue

        zed_depth = float(np.median(valid_values))
        q25, q75 = np.percentile(valid_values, [25, 75])
        signed = float(xyz[2] - zed_depth)
        abs_diff = abs(signed)
        out.update({
            "depth_roi_valid": True,
            "zed_depth_median_m": zed_depth,
            "zed_depth_iqr_m": float(q75 - q25),
            "signed_z_diff_m": signed,
            "abs_z_diff_m": abs_diff,
            "relative_abs_z_diff": float(abs_diff / zed_depth) if zed_depth > 0 else None,
            "skip_reason": "",
        })
        per_rows.append(out)

    valid_rows = [r for r in per_rows if r["depth_roi_valid"]]
    abs_diffs = [float(r["abs_z_diff_m"]) for r in valid_rows]
    signed_diffs = [float(r["signed_z_diff_m"]) for r in valid_rows]
    rel_diffs = [float(r["relative_abs_z_diff"]) for r in valid_rows]
    summary = [{
        "detector": "AGHRI fine-tuned YOLO11s",
        "fusion_method": "legacy_box",
        "attempted_valid_fused_detections": attempted,
        "valid_depth_roi_count": len(valid_rows),
        "valid_depth_roi_rate": f"{len(valid_rows) / attempted:.6f}" if attempted else "",
        "median_abs_z_diff_m": f"{median(abs_diffs):.6f}" if abs_diffs else "",
        "mean_abs_z_diff_m": f"{mean(abs_diffs):.6f}" if abs_diffs else "",
        "p95_abs_z_diff_m": f"{percentile(abs_diffs, 95):.6f}" if abs_diffs else "",
        "mean_signed_z_bias_m": f"{mean(signed_diffs):.6f}" if signed_diffs else "",
        "median_relative_abs_z_diff_percent": f"{100 * median(rel_diffs):.3f}" if rel_diffs else "",
    }]

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "finetuned_zed_depth_consistency_per_detection.csv", per_rows)
    write_csv(OUT / "finetuned_zed_depth_consistency_summary.csv", summary)
    plot_relative_error_cdf(rel_diffs)
    plot_zed_vs_fused_scatter(valid_rows)
    (OUT / "README.md").write_text(
        "# AGHRI ZED Depth Consistency Check\n\n"
        "This is not an absolute 3D localization benchmark. ZED depth maps provide "
        "per-pixel surface depth, not annotated 3D person centres. The evaluation "
        "compares the fused LiDAR camera-frame depth `Zc` with the median valid "
        "ZED depth inside a central region of each fine-tuned YOLO person box.\n\n"
        f"Summary CSV: `{OUT / 'finetuned_zed_depth_consistency_summary.csv'}`\n\n"
        f"Per-detection CSV: `{OUT / 'finetuned_zed_depth_consistency_per_detection.csv'}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary[0], indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    evaluate()
