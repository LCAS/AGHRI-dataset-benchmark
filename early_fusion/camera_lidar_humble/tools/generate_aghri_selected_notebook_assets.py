#!/usr/bin/env python3
"""Generate the reduced notebook-focused AGHRI fusion asset set."""

from __future__ import annotations

import base64
import csv
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
EARLY = REPO.parent
FULL = EARLY / "results" / "aghri_fusion_paper_assets"
OUT = FULL / "selected_notebook_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
FROZEN = EARLY / "results" / "aghri_generic_vs_finetuned"
RICH = EARLY / "results" / "aghri_detector_fusion_2x2"
NOTEBOOK = REPO / "notebooks" / "aghri_fusion_paper_assets.ipynb"

BLUE = (54, 95, 145)
ORANGE = (204, 111, 40)
GREEN = (44, 138, 102)
PURPLE = (121, 90, 155)
TEXT = (32, 36, 43)
MUTED = (91, 101, 117)
GRID = (222, 226, 232)


def font(size: int, bold: bool = False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()


F10 = font(10)
F11 = font(11)
F12 = font(12)
F13 = font(13)
F14 = font(14)
F16 = font(16)
F18 = font(18, True)
F22 = font(22, True)
F26 = font(26, True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def text_size(draw: ImageDraw.ImageDraw, text: str, ft) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=ft)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, ft, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        cand = word if not line else f"{line} {word}"
        if text_size(draw, cand, ft)[0] <= width:
            line = cand
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def yscale(v: float, top: int, bottom: int, vmax: float) -> int:
    return int(bottom - (v / vmax) * (bottom - top)) if vmax else bottom


def draw_panel_bars(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    ymax: float,
    fmt: str = "{:.2f}",
    percent: bool = False,
    yticks: list[float] | None = None,
    tick_fmt: str | None = None,
    bar_gap: int = 5,
    group_spacing: float = 1.0,
    max_category_lines: int = 2,
    error_bars: list[list[float]] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    d.text((x0, y0 - 44), title, fill=TEXT, font=F18)
    tick_values = yticks if yticks is not None else [ymax * i / 4 for i in range(5)]
    for val in tick_values:
        y = yscale(val, y0, y1, ymax)
        d.line([x0, y, x1, y], fill=GRID)
        label = f"{val*100:.0f}%" if percent else (tick_fmt or fmt).format(val)
        d.text((x0 - 48, y - 8), label, fill=MUTED, font=F10)
    d.line([x0, y1, x1, y1], fill=TEXT, width=2)
    d.line([x0, y0, x0, y1], fill=TEXT, width=2)
    group_w = (x1 - x0) / len(categories)
    bar_w = min(32, int(group_w / 4.2))
    actual_bar_w = max(1, bar_w - bar_gap)
    if len(categories) > 1:
        center_step = group_w * group_spacing
        first_center = (x0 + x1) / 2 - center_step * (len(categories) - 1) / 2
    else:
        center_step = 0
        first_center = (x0 + x1) / 2
    for ci, cat in enumerate(categories):
        cx = int(first_center + center_step * ci)
        start = cx - int(len(series) * bar_w / 2)
        for si, (_, vals, color) in enumerate(series):
            val = vals[ci]
            err = error_bars[si][ci] if error_bars else 0.0
            bx = start + si * bar_w
            by = yscale(val, y0, y1, ymax)
            d.rectangle([bx, by, bx + actual_bar_w, y1], fill=color)
            if err > 0.0:
                ebx = bx + actual_bar_w // 2
                ey_top = yscale(min(ymax, val + err), y0, y1, ymax)
                ey_bottom = yscale(max(0.0, val - err), y0, y1, ymax)
                d.line([ebx, ey_top, ebx, ey_bottom], fill=TEXT, width=1)
                d.line([ebx - 5, ey_top, ebx + 5, ey_top], fill=TEXT, width=1)
                d.line([ebx - 5, ey_bottom, ebx + 5, ey_bottom], fill=TEXT, width=1)
            label = f"{val*100:.1f}%" if percent else fmt.format(val)
            tw, _ = text_size(d, label, F10)
            label_y = yscale(min(ymax, val + err), y0, y1, ymax) - 17
            if len(categories) == 1 and len(series) > 1:
                label_y -= si * 13
            d.text((bx + (actual_bar_w - tw) // 2, label_y), label, fill=TEXT, font=F10)
        cat_lines: list[str] = []
        for part in cat.split("\n"):
            cat_lines.extend(wrap(d, part, F10, int(group_w) - 8))
        for li, line in enumerate(cat_lines[:max_category_lines]):
            tw, _ = text_size(d, line, F10)
            d.text((cx - tw // 2, y1 + 9 + 13 * li), line, fill=MUTED, font=F10)


def legend(d: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, tuple[int, int, int]]] | None = None) -> None:
    if items is None:
        items = [("Generic YOLO11s", BLUE), ("AGHRI fine-tuned", ORANGE)]
    for label, color in items:
        d.rectangle([x, y + 3, x + 18, y + 15], fill=color)
        d.text((x + 25, y), label, fill=TEXT, font=F12)
        x += 25 + text_size(d, label, F12)[0] + 28


def draw_vertical_text(
    img: Image.Image,
    text: str,
    xy: tuple[int, int],
    ft,
    fill: tuple[int, int, int],
) -> None:
    probe = ImageDraw.Draw(img)
    tw, th = text_size(probe, text, ft)
    text_img = Image.new("RGBA", (tw + 8, th + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((4, 4), text, font=ft, fill=fill + (255,))
    rotated = text_img.rotate(90, expand=True)
    img.paste(rotated, xy, rotated)


def generate_combined_overall() -> None:
    rows = read_csv(OUT / "tables" / "csv" / "T2_overall_results.csv")
    by = {r["model"]: r for r in rows}
    g = by["Generic YOLO11s"]
    f = by["AGHRI fine-tuned"]
    img = Image.new("RGB", (1600, 500), "white")
    d = ImageDraw.Draw(img)
    legend(d, 42, 42)
    draw_panel_bars(
        d,
        (85, 125, 500, 435),
        "Localization error",
        ["mean", "median", "P95"],
        [
            ("Generic YOLO11s", [float(g["mean_m"]), float(g["median_m"]), float(g["p95_m"])], BLUE),
            ("AGHRI fine-tuned", [float(f["mean_m"]), float(f["median_m"]), float(f["p95_m"])], ORANGE),
        ],
        2.0,
        yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        tick_fmt="{:.1f}",
        group_spacing=1.05,
    )
    draw_panel_bars(
        d,
        (615, 125, 1030, 435),
        "Reliability",
        ["valid rate", "error >1m"],
        [
            ("Generic YOLO11s", [float(g["valid_rate"]), float(g["err_gt_1m"])], BLUE),
            ("AGHRI fine-tuned", [float(f["valid_rate"]), float(f["err_gt_1m"])], ORANGE),
        ],
        1.0,
        percent=True,
        group_spacing=1.10,
    )
    draw_panel_bars(
        d,
        (1155, 125, 1510, 435),
        "FPS",
        ["fps"],
        [
            ("Generic YOLO11s", [float(g["fps"])], BLUE),
            ("AGHRI fine-tuned", [float(f["fps"])], ORANGE),
        ],
        20.0,
        fmt="{:.2f}",
        yticks=[0, 5, 10, 15, 20],
    )
    d.text((190, 470), "3D error (m)", fill=MUTED, font=F12)
    d.text((780, 470), "rate", fill=MUTED, font=F12)
    d.text((1285, 470), "frames/s", fill=MUTED, font=F12)
    path = FIG / "F1_F2_F3_overall_summary_row.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_scenario_f9() -> None:
    selected = [
        ("Footpath", "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label"),
        ("Polytunnel", "in_straw_3pick_diff_st_10_24_2024_5_a_label"),
        ("Vineyard", "out_vine_4swap+walk_st_ly_11_06_2024_2_label"),
    ]
    rows = read_csv(FROZEN / "per_recording_results.csv")
    values = {("Generic YOLO11s", rec): None for _, rec in selected}
    values.update({("AGHRI-fine-tuned YOLO11s", rec): None for _, rec in selected})
    for row in rows:
        key = (row["model"], row["recording"])
        if key in values:
            values[key] = float(row["mean_3d_error_m"])
    img = Image.new("RGB", (720, 490), "white")
    d = ImageDraw.Draw(img)
    legend(d, 48, 32)
    draw_panel_bars(
        d,
        (165, 115, 585, 410),
        "",
        [s for s, _ in selected],
        [
            ("Generic YOLO11s", [values[("Generic YOLO11s", rec)] or 0 for _, rec in selected], BLUE),
            ("AGHRI fine-tuned", [values[("AGHRI-fine-tuned YOLO11s", rec)] or 0 for _, rec in selected], ORANGE),
        ],
        0.5,
        yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        tick_fmt="{:.1f}",
        group_spacing=0.95,
    )
    axis = "mean 3D localization error (m)"
    tw, _ = text_size(d, axis, F13)
    d.text(((720 - tw) // 2, 455), axis, fill=MUTED, font=F13)
    out_rows = [
        {"scenario": scenario, "selected_bag": rec, "generic_mean_3d_error_m": values[("Generic YOLO11s", rec)], "finetuned_mean_3d_error_m": values[("AGHRI-fine-tuned YOLO11s", rec)]}
        for scenario, rec in selected
    ]
    write_csv(OUT / "derived_data" / "F9_selected_representative_scenarios.csv", out_rows)
    path = FIG / "F9_representative_scenario_mean_error.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_f14_depth_bins() -> None:
    rows = [
        r for r in read_csv(RICH / "tables" / "condition_breakdowns.csv")
        if r["breakdown"] == "mean_lidar_error_by_distance_band" and r["fusion_method"] == "legacy_box"
    ]
    order = ["near_lt_2m", "medium_2_5m", "far_ge_5m"]
    vals = {}
    for r in rows:
        vals[(r["detector"], r["group"])] = float(r["mean_3d_error_m"])
    img = Image.new("RGB", (780, 570), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F14. Mean 3D Error by Ground-truth Distance Band", fill=TEXT, font=F22)
    d.text((48, 69), "Depth robustness summary from frozen evaluator condition breakdowns", fill=MUTED, font=F13)
    legend(d, 48, 103)
    labels = ["near <2m", "medium 2-5m", "far >=5m"]
    draw_panel_bars(
        d,
        (180, 165, 600, 475),
        "",
        labels,
        [
            ("Generic YOLO11s", [vals.get(("generic", b), 0) for b in order], BLUE),
            ("AGHRI fine-tuned", [vals.get(("finetuned", b), 0) for b in order], ORANGE),
        ],
        1.2,
        group_spacing=0.95,
    )
    axis = "mean 3D localization error (m)"
    tw, _ = text_size(d, axis, F13)
    d.text(((780 - tw) // 2, 530), axis, fill=MUTED, font=F13)
    path = FIG / "F14_depth_bins_selected.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_f14_depth_bins_percentage() -> None:
    rows = [
        r for r in read_csv(RICH / "tables" / "condition_breakdowns.csv")
        if r["breakdown"] == "mean_lidar_error_by_distance_band" and r["fusion_method"] == "legacy_box"
    ]
    order = ["near_lt_2m", "medium_2_5m", "far_ge_5m"]
    vals = {}
    for r in rows:
        vals[(r["detector"], r["group"])] = float(r["mean_3d_error_m"])
    shares = {}
    for detector in ["generic", "finetuned"]:
        total = sum(vals[(detector, band)] for band in order)
        for band in order:
            shares[(detector, band)] = vals[(detector, band)] / total if total else 0.0
    write_csv(
        OUT / "derived_data" / "F14_depth_bins_selected_percentage.csv",
        [
            {
                "detector": "Generic YOLO11s" if detector == "generic" else "AGHRI fine-tuned",
                "depth_band": band,
                "mean_3d_error_m": f"{vals[(detector, band)]:.6f}",
                "share_of_summed_depth_bin_mean_error_percent": f"{shares[(detector, band)] * 100:.2f}",
            }
            for detector in ["generic", "finetuned"]
            for band in order
        ],
    )
    img = Image.new("RGB", (780, 570), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F14. Mean 3D Error Share by Ground-truth Distance Band", fill=TEXT, font=F22)
    d.text((48, 69), "Each detector sums to 100% across near, medium and far depth-bin mean errors", fill=MUTED, font=F13)
    legend(d, 48, 103)
    labels = ["near <2m", "medium 2-5m", "far >=5m"]
    draw_panel_bars(
        d,
        (180, 165, 600, 475),
        "",
        labels,
        [
            ("Generic YOLO11s", [shares.get(("generic", b), 0) for b in order], BLUE),
            ("AGHRI fine-tuned", [shares.get(("finetuned", b), 0) for b in order], ORANGE),
        ],
        0.70,
        percent=True,
        yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        group_spacing=0.95,
    )
    axis = "share of summed mean 3D localization error across depth bins (%)"
    tw, _ = text_size(d, axis, F13)
    d.text(((780 - tw) // 2, 530), axis, fill=MUTED, font=F13)
    path = FIG / "F14_depth_bins_selected_percentage.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_f14_depth_bins_absolute_improvement() -> None:
    rows = [
        r for r in read_csv(RICH / "tables" / "condition_breakdowns.csv")
        if r["breakdown"] == "mean_lidar_error_by_distance_band" and r["fusion_method"] == "legacy_box"
    ]
    order = ["near_lt_2m", "medium_2_5m", "far_ge_5m"]
    labels = ["near <2m", "medium 2-5m", "far >=5m"]
    vals = {}
    for r in rows:
        vals[(r["detector"], r["group"])] = float(r["mean_3d_error_m"])

    improvements = []
    for band in order:
        generic = vals[("generic", band)]
        finetuned = vals[("finetuned", band)]
        improvements.append((generic - finetuned) / generic if generic else 0.0)

    write_csv(
        OUT / "derived_data" / "F14_depth_bins_absolute_improvement.csv",
        [
            {
                "depth_band": band,
                "generic_mean_3d_error_m": f"{vals[('generic', band)]:.6f}",
                "finetuned_mean_3d_error_m": f"{vals[('finetuned', band)]:.6f}",
                "relative_error_reduction_percent": f"{improvement * 100:.2f}",
            }
            for band, improvement in zip(order, improvements)
        ],
    )

    img = Image.new("RGB", (820, 585), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F14. Mean 3D Error by Ground-truth Distance Band", fill=TEXT, font=F22)
    d.text((48, 69), "Absolute error in metres; labels show AGHRI fine-tuned reduction vs generic", fill=MUTED, font=F13)
    legend(d, 48, 103)
    plot_box = (175, 165, 630, 475)
    draw_panel_bars(
        d,
        plot_box,
        "",
        labels,
        [
            ("Generic YOLO11s", [vals.get(("generic", b), 0) for b in order], BLUE),
            ("AGHRI fine-tuned", [vals.get(("finetuned", b), 0) for b in order], ORANGE),
        ],
        1.2,
        yticks=[0.0, 0.3, 0.6, 0.9, 1.2],
        group_spacing=0.88,
    )
    x0, y0, x1, y1 = plot_box
    group_w = (x1 - x0) / len(labels)
    center_step = group_w * 1.06
    first_center = (x0 + x1) / 2 - center_step * (len(labels) - 1) / 2
    for ci, improvement in enumerate(improvements):
        cx = int(first_center + center_step * ci)
        top_val = max(vals[("generic", order[ci])], vals[("finetuned", order[ci])])
        y = max(y0 + 18, yscale(top_val, y0, y1, 1.2) - 42)
        label = f"{improvement * 100:.1f}% lower"
        tw, th = text_size(d, label, F11)
        d.rounded_rectangle(
            [cx - tw // 2 - 8, y - 4, cx + tw // 2 + 8, y + th + 7],
            radius=4,
            fill=(245, 248, 246),
            outline=GREEN,
        )
        d.text((cx - tw // 2, y), label, fill=GREEN, font=F11)
    axis = "mean 3D localization error (m)"
    tw, _ = text_size(d, axis, F13)
    d.text(((820 - tw) // 2, 535), axis, fill=MUTED, font=F13)
    path = FIG / "F14_depth_bins_absolute_improvement.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_f14_depth_normalized_error() -> None:
    sources = {
        "generic": RICH / "runs" / "generic_legacy" / "detections.json",
        "finetuned": RICH / "runs" / "finetuned_legacy" / "detections.json",
    }
    order = ["near_lt_2m", "medium_2_5m", "far_ge_5m"]
    labels = ["near <2m", "medium 2-5m", "far >=5m"]
    display = {"generic": "Generic YOLO11s", "finetuned": "AGHRI fine-tuned YOLO11s"}
    values: dict[tuple[str, str], list[float]] = {
        (detector, band): [] for detector in sources for band in order
    }
    for detector, path in sources.items():
        for row in json.loads(path.read_text(encoding="utf-8")):
            band = row.get("distance_band")
            if band not in order:
                continue
            if not row.get("valid") or row.get("status") != "matched":
                continue
            error = row.get("errors", {}).get("lidar_center_error")
            depth = (row.get("published_xyz") or [None, None, None])[2]
            if error is None or depth is None:
                continue
            error = float(error)
            depth = float(depth)
            if not (math.isfinite(error) and math.isfinite(depth) and depth > 0.0):
                continue
            values[(detector, band)].append(100.0 * error / depth)

    means = {
        key: (sum(vals) / len(vals) if vals else 0.0)
        for key, vals in values.items()
    }
    counts = {
        key: len(vals)
        for key, vals in values.items()
    }
    reductions = []
    for band in order:
        generic = means[("generic", band)]
        finetuned = means[("finetuned", band)]
        reductions.append((generic - finetuned) / generic if generic else 0.0)

    write_csv(
        OUT / "derived_data" / "F14_depth_bins_depth_normalized_error.csv",
        [
            {
                "detector": display[detector],
                "depth_band": band,
                "mean_relative_3d_error_percent_of_depth": f"{means[(detector, band)]:.6f}",
                "valid_matched_count": len(values[(detector, band)]),
                "visual_encoding": "point_size_scales_with_valid_matched_count",
                "normalization": "100 * lidar_center_error / fused_camera_depth_z",
            }
            for detector in ["generic", "finetuned"]
            for band in order
        ],
    )

    img = Image.new("RGB", (850, 555), "white")
    d = ImageDraw.Draw(img)
    legend(d, 48, 32, [("Generic YOLO11s", BLUE), ("AGHRI fine-tuned YOLO11s", ORANGE)])
    plot_box = (185, 100, 650, 420)
    x0, y0, x1, y1 = plot_box
    ymax = 0.30
    for val in [0.0, 0.1, 0.2, 0.3]:
        y = yscale(val, y0, y1, ymax)
        d.line([x0, y, x1, y], fill=GRID)
        d.text((x0 - 48, y - 8), f"{val * 100:.0f}%", fill=MUTED, font=F10)
    d.line([x0, y1, x1, y1], fill=TEXT, width=2)
    d.line([x0, y0, x0, y1], fill=TEXT, width=2)

    group_w = (x1 - x0) / len(labels)
    center_step = group_w * 0.88
    first_center = (x0 + x1) / 2 - center_step * (len(labels) - 1) / 2
    all_counts = [counts[(detector, band)] for detector in ["generic", "finetuned"] for band in order]
    min_sqrt = math.sqrt(min(all_counts))
    max_sqrt = math.sqrt(max(all_counts))

    def point_radius(n: int) -> int:
        if max_sqrt == min_sqrt:
            return 16
        return int(9 + (math.sqrt(n) - min_sqrt) / (max_sqrt - min_sqrt) * 14)

    for ci, (label, band) in enumerate(zip(labels, order)):
        cx = int(first_center + center_step * ci)
        tw, _ = text_size(d, label, F10)
        d.text((cx - tw // 2, y1 + 9), label, fill=MUTED, font=F10)

        # Offset the two method markers within each depth band so large
        # count-scaled circles remain visibly separated.
        for detector, color, dx in [("generic", BLUE, -27), ("finetuned", ORANGE, 27)]:
            mean = means[(detector, band)] / 100.0
            n = counts[(detector, band)]
            px = cx + dx
            py = yscale(mean, y0, y1, ymax)
            r = point_radius(n)
            d.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=TEXT, width=1)
            value_label = f"{mean * 100:.1f}%"
            tw, _ = text_size(d, value_label, F10)
            d.text((px - tw // 2, py - r - 18), value_label, fill=TEXT, font=F10)

    y_axis = "Mean normalized 3D localization error (% of depth)"
    draw_vertical_text(img, y_axis, (62, 125), F13, MUTED)
    x_axis = "Distance band based on camera-frame depth"
    tw, _ = text_size(d, x_axis, F13)
    d.text(((850 - tw) // 2, 470), x_axis, fill=MUTED, font=F13)
    path = FIG / "F14_depth_bins_depth_normalized_error.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))

    totals = {
        detector: sum(counts[(detector, band)] for band in order)
        for detector in ["generic", "finetuned"]
    }
    contributions = {
        (detector, band): means[(detector, band)] * counts[(detector, band)] / totals[detector]
        for detector in ["generic", "finetuned"]
        for band in order
    }
    write_csv(
        OUT / "derived_data" / "F14b_depth_band_error_contribution.csv",
        [
            {
                "detector": display[detector],
                "depth_band": band,
                "mean_relative_3d_error_percent_of_depth": f"{means[(detector, band)]:.6f}",
                "valid_matched_count": counts[(detector, band)],
                "total_valid_matched_count": totals[detector],
                "band_weight": f"{counts[(detector, band)] / totals[detector]:.6f}",
                "weighted_contribution_percent_points": f"{contributions[(detector, band)]:.6f}",
                "calculation": "mean_relative_3d_error_percent_of_depth * valid_matched_count / total_valid_matched_count",
            }
            for detector in ["generic", "finetuned"]
            for band in order
        ],
    )

    img = Image.new("RGB", (850, 620), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F14b. Distance-band contribution to overall normalized error", fill=TEXT, font=F22)
    d.text((48, 69), "Band mean multiplied by that band's share of valid matched detections", fill=MUTED, font=F13)
    legend(d, 48, 103, [("Generic YOLO11s", BLUE), ("AGHRI fine-tuned YOLO11s", ORANGE)])
    d.text((665, 106), "Lower contribution is better", fill=GREEN, font=F12)
    plot_box = (185, 165, 650, 485)
    count_labels = [
        f"{label}\nG n={counts[('generic', band)]}\nFT n={counts[('finetuned', band)]}"
        for label, band in zip(labels, order)
    ]
    draw_panel_bars(
        d,
        plot_box,
        "",
        count_labels,
        [
            ("Generic YOLO11s", [contributions[("generic", b)] / 100.0 for b in order], BLUE),
            ("AGHRI fine-tuned YOLO11s", [contributions[("finetuned", b)] / 100.0 for b in order], ORANGE),
        ],
        0.08,
        percent=True,
        yticks=[0.0, 0.02, 0.04, 0.06, 0.08],
        group_spacing=0.88,
        max_category_lines=3,
    )
    y_axis = "Contribution to overall normalized error (% points)"
    draw_vertical_text(img, y_axis, (38, 190), F13, MUTED)
    x_axis = "Distance band based on camera-frame depth"
    tw, _ = text_size(d, x_axis, F13)
    d.text(((850 - tw) // 2, 580), x_axis, fill=MUTED, font=F13)
    note = "Contribution view: this answers dataset impact, not per-band difficulty."
    tw, _ = text_size(d, note, F11)
    d.text(((850 - tw) // 2, 140), note, fill=MUTED, font=F11)
    path = FIG / "F14b_depth_band_error_contribution.png"
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def generate_revised_d1() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "D1_fusion_method_overview_revised.svg"
    def box(x, y, w, h, label, fill="#f5f7fb"):
        lines = label.split("\n")
        text = "".join(f'<text x="{x+w/2}" y="{y+24+i*17}" text-anchor="middle" font-size="14" fill="#20242b">{line}</text>' for i, line in enumerate(lines))
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#56616f"/>{text}'
    def arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#384252" stroke-width="2.5" marker-end="url(#a)"/>'
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="975" viewBox="0 65 1180 975">']
    svg.append('<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#384252"/></marker></defs><rect y="65" width="1180" height="975" fill="white"/>')
    svg.append(box(80, 100, 210, 60, "ZED RGB image", "#edf6ff"))
    svg.append(box(80, 220, 210, 60, "Livox point cloud", "#eef8f1"))
    svg.append(box(420, 155, 260, 70, "ApproximateTimeSynchronizer", "#fff3e7"))
    svg.append(arrow(290, 130, 420, 180))
    svg.append(arrow(290, 250, 420, 200))
    svg.append(box(430, 270, 240, 70, "synchronized image-cloud pair", "#fff8df"))
    svg.append(arrow(550, 225, 550, 270))
    svg.append(box(160, 405, 250, 70, "YOLO11s person detection", "#edf6ff"))
    svg.append(box(725, 405, 270, 70, "Transform LiDAR points\ninto ZED camera frame", "#eef8f1"))
    svg.append(arrow(500, 340, 285, 405))
    svg.append(arrow(600, 340, 860, 405))
    left_steps = [("Original person boxes", 505), ("Shrink association boxes", 605)]
    right_steps = [("Project using CameraInfo.P", 535)]
    for label, y in left_steps:
        svg.append(box(175, y, 220, 60, label, "#edf6ff"))
    for label, y in right_steps:
        svg.append(box(745, y, 230, 60, label, "#eef8f1"))
    svg.append(arrow(285, 475, 285, 505))
    svg.append(arrow(285, 565, 285, 605))
    svg.append(arrow(860, 475, 860, 535))
    svg.append(box(430, 700, 300, 65, "Select projected points inside boxes", "#f4f0ff"))
    svg.append(arrow(285, 665, 520, 700))
    svg.append(arrow(860, 595, 640, 700))
    chain = [
        ("Modified Z-score filtering, tau = 3.5", 805),
        ("Retained LiDAR points", 885),
        ("Mean of retained 3D points", 955),
    ]
    last_y = 765
    for label, y in chain:
        svg.append(box(430, y, 300, 50, label, "#f9f9ed"))
        svg.append(arrow(580, last_y, 580, y))
        last_y = y + 50
    svg.append(box(790, 830, 280, 70, "Estimated camera-frame centre\n[Xc, Yc, Zc]", "#eef8f1"))
    svg.append(box(790, 945, 280, 60, "Annotated image:\nperson, Zc m", "#fff3e7"))
    svg.append(arrow(730, 980, 790, 865))
    svg.append(arrow(930, 900, 930, 945))
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")

    img = Image.new("RGB", (1180, 1040), "white")
    d = ImageDraw.Draw(img)

    def pbox(x: int, y: int, w: int, h: int, label: str, fill: tuple[int, int, int]) -> None:
        d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=(86, 97, 111), width=2)
        lines = label.split("\n")
        total_h = len(lines) * 18
        for i, line in enumerate(lines):
            tw, _ = text_size(d, line, F13)
            d.text((x + w // 2 - tw // 2, y + h // 2 - total_h // 2 + i * 18), line, font=F13, fill=TEXT)

    def parr(x1: int, y1: int, x2: int, y2: int) -> None:
        d.line([x1, y1, x2, y2], fill=(56, 66, 82), width=3)
        # Small arrow head pointing roughly along the dominant direction.
        if abs(y2 - y1) >= abs(x2 - x1):
            sign = 1 if y2 >= y1 else -1
            d.polygon([(x2, y2), (x2 - 7, y2 - sign * 12), (x2 + 7, y2 - sign * 12)], fill=(56, 66, 82))
        else:
            sign = 1 if x2 >= x1 else -1
            d.polygon([(x2, y2), (x2 - sign * 12, y2 - 7), (x2 - sign * 12, y2 + 7)], fill=(56, 66, 82))

    blue = (237, 246, 255)
    green = (238, 248, 241)
    orange = (255, 243, 231)
    yellow = (255, 248, 223)
    purple = (244, 240, 255)
    cream = (249, 249, 237)
    pbox(80, 100, 210, 60, "ZED RGB image", blue)
    pbox(80, 220, 210, 60, "Livox point cloud", green)
    pbox(420, 155, 260, 70, "ApproximateTimeSynchronizer", orange)
    parr(290, 130, 420, 180)
    parr(290, 250, 420, 200)
    pbox(430, 270, 240, 70, "synchronized image-cloud pair", yellow)
    parr(550, 225, 550, 270)
    pbox(160, 405, 250, 70, "YOLO11s person detection", blue)
    pbox(725, 405, 270, 70, "Transform LiDAR points\ninto ZED camera frame", green)
    parr(500, 340, 285, 405)
    parr(600, 340, 860, 405)
    pbox(175, 505, 220, 60, "Original person boxes", blue)
    pbox(175, 605, 220, 60, "Shrink association boxes", blue)
    pbox(745, 535, 230, 60, "Project using CameraInfo.P", green)
    parr(285, 475, 285, 505)
    parr(285, 565, 285, 605)
    parr(860, 475, 860, 535)
    pbox(430, 700, 300, 65, "Select projected points inside boxes", purple)
    parr(285, 665, 520, 700)
    parr(860, 595, 640, 700)
    pbox(430, 805, 300, 50, "Modified Z-score filtering, tau = 3.5", cream)
    pbox(430, 885, 300, 50, "Retained LiDAR points", cream)
    pbox(430, 955, 300, 50, "Mean of retained 3D points", cream)
    parr(580, 765, 580, 805)
    parr(580, 855, 580, 885)
    parr(580, 935, 580, 955)
    pbox(790, 830, 280, 70, "Estimated camera-frame centre\n[Xc, Yc, Zc]", green)
    pbox(790, 945, 280, 60, "Annotated image:\nperson, Zc m", orange)
    parr(730, 980, 790, 865)
    parr(930, 900, 930, 945)
    img.crop((0, 65, 1180, 1040)).save(FIG / "D1_fusion_method_overview_revised.png")


def copy_selected_existing_assets() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    copies = [
        (FULL / "figures/qualitative/Q4_step_by_step_association.png", FIG / "Q4_step_by_step_association_selected.png"),
        (FULL / "figures/training/F19_training_curves.png", FIG / "F19_training_curves_selected.png"),
        (REPO / "results/aghri_zed_depth_consistency/F_ZED_depth_vs_fused_lidar_zc_scatter.png", FIG / "F_ZED_depth_vs_fused_lidar_zc_scatter.png"),
        (REPO / "results/aghri_zed_depth_consistency/F_ZED_depth_vs_fused_lidar_zc_scatter.pdf", FIG / "F_ZED_depth_vs_fused_lidar_zc_scatter.pdf"),
    ]
    for src, dst in copies:
        shutil.copy2(src, dst)
    for table_id in ["T2_overall_results", "T5_metric_definitions", "T7_ros_topics_frames"]:
        for kind in ["csv", "markdown", "latex", "rendered"]:
            src_dir = FULL / "tables" / kind
            dst_dir = TAB / kind
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in src_dir.glob(f"{table_id}.*"):
                shutil.copy2(src, dst_dir / src.name)


def render_markdown_table(table_md: str, out_path: Path) -> None:
    lines = [line for line in table_md.strip().splitlines() if line.strip()]
    rows = []
    for line in lines:
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return
    widths = [max(text_size(ImageDraw.Draw(Image.new("RGB", (1, 1))), row[i], F12)[0] for row in rows) for i in range(len(rows[0]))]
    cell_h = 34
    pad_x = 18
    w = sum(widths) + pad_x * 2 * len(widths) + 2
    h = cell_h * len(rows) + 2
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y = 0
    for ri, row in enumerate(rows):
        x = 0
        if ri == 0:
            d.rectangle([0, 0, w, cell_h], fill=(245, 247, 250))
        for ci, cell in enumerate(row):
            d.rectangle([x, y, x + widths[ci] + pad_x * 2, y + cell_h], outline=GRID)
            d.text((x + pad_x, y + 9), cell, fill=TEXT, font=F12 if ri else F13)
            x += widths[ci] + pad_x * 2
        y += cell_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    img.convert("RGB").save(out_path.with_suffix(".pdf"))


def write_selected_tables() -> None:
    t2_rows = [
        {
            "model": "Generic YOLO11s",
            "detections": 2787,
            "matched": 2561,
            "valid_rate": "0.9218",
            "mean_m": "0.4765",
            "median_m": "0.2038",
            "p95_m": "1.8703",
            "err_gt_1m": "0.0859",
            "fps": "15.15",
        },
        {
            "model": "AGHRI fine-tuned",
            "detections": 2534,
            "matched": 2433,
            "valid_rate": "0.9680",
            "mean_m": "0.3926",
            "median_m": "0.1911",
            "p95_m": "1.3595",
            "err_gt_1m": "0.0748",
            "fps": "15.19",
        },
    ]
    t5_rows = [
        {
            "metric": "valid fusion rate",
            "definition": "matched valid fused detections divided by detector person detections",
            "unit": "ratio",
        },
        {
            "metric": "mean/median/P95 3D error",
            "definition": "Euclidean error between fused person centre and GT 3D person centre",
            "unit": "metres",
        },
        {
            "metric": "error > 1m",
            "definition": "fraction of matched valid fused detections with 3D error above 1 metre",
            "unit": "ratio",
        },
        {
            "metric": "FPS",
            "definition": "mean live ROS output rate on /camera_lidar_fusion/result across the six held-out AGHRI test bags",
            "unit": "frames/s",
        },
    ]
    tables = {"T2_overall_results": t2_rows, "T5_metric_definitions": t5_rows}
    for table_id, rows in tables.items():
        write_csv(TAB / "csv" / f"{table_id}.csv", rows)
        headers = list(rows[0])
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        md = "\n".join(md_lines) + "\n"
        md_path = TAB / "markdown" / f"{table_id}.md"
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md, encoding="utf-8")
        latex_lines = ["\\begin{tabular}{" + "l" * len(headers) + "}", " \\toprule", " & ".join(headers) + r" \\", " \\midrule"]
        for row in rows:
            latex_lines.append(" & ".join(str(row[h]) for h in headers) + r" \\")
        latex_lines += [" \\bottomrule", "\\end{tabular}", ""]
        latex_path = TAB / "latex" / f"{table_id}.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text("\n".join(latex_lines), encoding="utf-8")
        render_markdown_table(md, TAB / "rendered" / f"{table_id}.png")


def crop_selected_copied_figures() -> None:
    q4 = FIG / "Q4_step_by_step_association_selected.png"
    if q4.exists():
        src = Image.open(q4).convert("RGB")
        img = Image.new("RGB", (src.width, src.height - 35), "white")
        img.paste(src.crop((0, 45, src.width, src.height)), (0, 10))
        img.save(q4)

    f19 = FIG / "F19_training_curves_selected.png"
    if f19.exists():
        src = Image.open(f19).convert("RGB")
        img = src.crop((0, 108, src.width, src.height))
        img.save(f19)


def write_readme() -> None:
    text = """# Selected Notebook Assets

This folder contains the reduced set requested for the AGHRI fusion notebook.
It does not replace the full paper-asset collection.

Included figures:
- Revised D1 fusion method overview.
- Combined F1/F2/F3 overall summary row.
- F9 representative-scenario mean error using one selected bag per scenario.
- F14 depth-bin robustness summary.
- ZED depth versus fused Zc consistency scatter plot.
- F19 fine-tuning curves.
- Q4 step-by-step association walkthrough.

Included tables:
- T2 overall results.
- T5 metric definitions.
- T7 ROS topics and frames.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def write_notebook() -> None:
    def table_md(table_id: str) -> str:
        return (TAB / "markdown" / f"{table_id}.md").read_text(encoding="utf-8").strip()

    image_paths = {
        "D1_fusion_method_overview_revised.png": FIG / "D1_fusion_method_overview_revised.png",
        "F1_F2_F3_overall_summary_row.png": FIG / "F1_F2_F3_overall_summary_row.png",
        "F9_representative_scenario_mean_error.png": FIG / "F9_representative_scenario_mean_error.png",
        "F14_depth_bins_selected.png": FIG / "F14_depth_bins_selected.png",
        "F14_depth_bins_selected_percentage.png": FIG / "F14_depth_bins_selected_percentage.png",
        "F14_depth_bins_absolute_improvement.png": FIG / "F14_depth_bins_absolute_improvement.png",
        "F14_depth_bins_depth_normalized_error.png": FIG / "F14_depth_bins_depth_normalized_error.png",
        "F_ZED_depth_vs_fused_lidar_zc_scatter.png": FIG / "F_ZED_depth_vs_fused_lidar_zc_scatter.png",
        "F19_training_curves_selected.png": FIG / "F19_training_curves_selected.png",
        "Q4_step_by_step_association_selected.png": FIG / "Q4_step_by_step_association_selected.png",
        "ROS_out_vine_4people_fusion_axes_04.png": REPO / "results" / "selected_notebook_assets" / "figures" / "ROS_out_vine_4people_fusion_axes_04.png",
        "AGHRI_fusion_failure_06_no_valid_cluster.png": REPO / "results" / "selected_notebook_assets" / "figures" / "failure_cases" / "AGHRI_fusion_failure_06_no_valid_cluster.png",
    }

    def image_md(filename: str, alt: str) -> str:
        return f"![{alt}](attachment:{filename})"

    def attach_referenced_images(cells: list[dict]) -> list[dict]:
        for cell in cells:
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", []))
            attachments = {}
            for filename, path in image_paths.items():
                if f"attachment:{filename}" not in source:
                    continue
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                attachments[filename] = {"image/png": data}
            if attachments:
                cell["attachments"] = attachments
        return cells

    def build_cells() -> list[dict]:
        return [
            md_cell(f"""# AGHRI ZED RGB-Livox Early Fusion Results Summary

This notebook summarises the selected AGHRI camera-LiDAR fusion figures and tables. The figures shown here are the selected saved figures from `results/aghri_fusion_paper_assets/selected_notebook_assets`.
"""),
            md_cell("""## Methods Compared

The fusion framework is the same in both rows below. The comparison changes the 2D person detector checkpoint.

| Method name in figures | Detector checkpoint | Fusion method | Explanation |
|---|---|---|---|
| `Generic YOLO11s` | Public YOLO11s checkpoint | Legacy camera-guided LiDAR box association | Baseline detector used without AGHRI-specific fine-tuning. |
| `AGHRI fine-tuned YOLO11s` | YOLO11s fine-tuned on AGHRI ZED RGB data | Legacy camera-guided LiDAR box association | Detector adapted to the AGHRI viewpoints, outdoor scenes, person scale, and clothing/occlusion patterns. |

The important control is that the LiDAR association logic is not changed between these two detector rows. This keeps the comparison focused on whether better 2D detections improve downstream fused 3D person localization.
"""),
            md_cell(f"""## Metrics Used

The accuracy and reliability metrics come from the offline frozen fusion evaluator. FPS is measured separately as mean live ROS 2 output throughput from `/camera_lidar_fusion/result` across the six held-out AGHRI test bags with RViz disabled.

{table_md("T5_metric_definitions")}
"""),
            md_cell("""## Test Data Used

The frozen held-out benchmark uses all six official AGHRI test recordings.

| Test recording | Scenario | Robot motion |
|---|---|---|
| `footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label` | Footpath | Moving |
| `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` | Footpath | Stationary |
| `in_straw_3pick_diff_st_10_24_2024_5_a_label` | Polytunnel | Stationary |
| `out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label` | Polytunnel/outdoor straw | Stationary |
| `out_vine_1push_3carry_st_ly_11_06_2024_1_label` | Vineyard | Stationary |
| `out_vine_4swap+walk_st_ly_11_06_2024_2_label` | Vineyard | Stationary |

The overall result table, Figure 2, Figure 4a, and Figure 4b use all six recordings. Figure 3 uses one selected bag per scenario only, so that figure is easier to read and should not be confused with the full six-recording aggregate.
"""),
            md_cell(f"""## Main Results Table

{table_md("T2_overall_results")}

**Key observation.** The AGHRI fine-tuned detector improves the fusion output even though the LiDAR association method is kept fixed. Compared with generic YOLO11s, mean 3D error drops from 0.4765 m to 0.3926 m, P95 error drops from 1.8703 m to 1.3595 m, and valid fusion rate increases from 0.9218 to 0.9680. The all-six-bag live ROS output rate remains very similar between the two early-fusion checkpoints: 15.15 FPS for generic YOLO11s and 15.19 FPS for the AGHRI fine-tuned checkpoint.
"""),
            code_cell(f"""# Optional reproducibility cell: show where the selected assets live.
from pathlib import Path

ASSETS = Path("{OUT}")
print(ASSETS)
print("selected asset files:", len(list(ASSETS.rglob("*"))))
"""),
            md_cell(f"""## Figure 1: Early Camera-LiDAR Human Detection Fusion Flow

{image_md("D1_fusion_method_overview_revised.png", "Revised fusion method overview")}

The runtime first synchronizes the ZED RGB image and Livox point cloud with an approximate-time synchronizer. YOLO11s produces person boxes on the synchronized image. In parallel, the point cloud is transformed into the ZED camera frame and projected into image coordinates using `CameraInfo.P`. The projected points inside the shrunken person association boxes become candidates. Modified Z-score filtering with `tau = 3.5` removes outliers, and the retained 3D points are averaged to estimate the camera-frame person centre `[Xc, Yc, Zc]`. The ROS playback overlay displays the forward camera depth `Zc` and horizontal camera offset `Xc`.
"""),
            md_cell(f"""## Figure 2: Overall Test Summary

{image_md("F1_F2_F3_overall_summary_row.png", "Overall frozen-test summary")}

This combined figure places the three core summaries in one row. The localization-error panel shows that the fine-tuned detector reduces mean, median, and P95 3D localization error. The reliability panel shows a higher valid fusion rate and a lower fraction of errors above 1 m. The FPS panel shows that both checkpoints run at similar live ROS output rates.
"""),
            md_cell(f"""## Figure 3: Representative Scenario Mean Error

{image_md("F9_representative_scenario_mean_error.png", "Representative scenario mean error")}

This figure is a compact scenario-level view. It uses one selected representative bag per scenario rather than all six test bags:

| Scenario | Selected bag used in this figure |
|---|---|
| Footpath | `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` |
| Polytunnel | `in_straw_3pick_diff_st_10_24_2024_5_a_label` |
| Vineyard | `out_vine_4swap+walk_st_ly_11_06_2024_2_label` |

The trend is consistent with the full aggregate: the AGHRI fine-tuned detector gives lower mean 3D error in the selected representative bags.
"""),
            md_cell(f"""## Figure 4a: Depth-normalized 3D localization error by distance band

{image_md("F14_depth_bins_depth_normalized_error.png", "Depth-normalized depth-bin error summary")}

This figure reports a depth-normalized error: for each valid matched detection, the 3D localization error is divided by the camera-frame depth before averaging within the near, medium, and far bands. Point position shows how large the mean normalized error is within each distance band. Point size shows how much data supports that value, because larger points represent more valid matched detections. Under this normalized view, the AGHRI fine-tuned detector remains lower than generic YOLO11s in all three bands, with the largest relative reduction in the far range.
"""),
            md_cell(f"""## Figure 4b: ZED Depth Image Consistency

{image_md("F_ZED_depth_vs_fused_lidar_zc_scatter.png", "ZED depth versus fused Zc depth scatter")}

This scatter plot compares the robust ZED depth image reference inside the detected person region with the fused camera-frame `Zc` depth estimated from the LiDAR association output. It is a visual agreement check rather than an absolute 3D localization benchmark, because the ZED depth maps provide surface depth values, not annotated 3D person centres. Points close to the dashed diagonal line indicate good Z-depth agreement. Points above the line mean the fused `Zc` is larger or farther than the ZED depth, while points below the line mean the fused `Zc` is smaller or closer than the ZED depth.

Across this ZED-depth consistency set, the fused LiDAR `Zc` differed from the ZED depth by a median absolute error of 0.1169 m, corresponding to a median relative absolute Z-depth error of 3.96%.
"""),
            md_cell(f"""## Figure 5: YOLO11s Fine-tuning Curves

{image_md("F19_training_curves_selected.png", "YOLO11s fine-tuning curves")}

`F19_training_curves` shows the cluster fine-tuning run used to adapt YOLO11s to AGHRI ZED RGB images. The curves track validation precision, recall, mAP50, and mAP50-95 over training epochs. Epoch 46 is marked as the selected best checkpoint. That checkpoint is then used in the frozen fusion comparison, so the downstream results measure how an AGHRI-adapted 2D detector changes the final fused 3D person localization.
"""),
            md_cell(f"""## Figure 6: Step-by-step LiDAR Association Example

{image_md("Q4_step_by_step_association_selected.png", "Step-by-step LiDAR association walkthrough")}

`Q4_step_by_step_association` is a self-contained qualitative walkthrough. The raw ZED image first shows the target person. The projected-point panel then shows all LiDAR points that land inside the image. The candidate panel isolates the points inside the shrunken person association box. The selected-vs-rejected panel shows the modified Z-score filter in action: 22 candidate points enter the filter, 15 are retained, and 7 are rejected. The final panel averages the retained 3D points and recomputes a LiDAR-centre error of about 0.1337 m.
"""),
            md_cell(f"""## ROS Topics and Frames

{table_md("T7_ros_topics_frames")}

The all-topic ROS 2 bags contain the ZED RGB image and CameraInfo, three fisheye CameraInfo streams, Livox point clouds, `/tf`, and `/tf_static`. The fusion node only needs the ZED RGB stream, ZED CameraInfo, Livox cloud, and TF chain for the published ZED RGB-Livox result.
"""),
            md_cell(f"""## Figure 7: ROS Playback Qualitative Example

{image_md("ROS_out_vine_4people_fusion_axes_04.png", "ROS playback vineyard qualitative fusion example")}

This image was captured directly from the ROS 2 playback output topic `/camera_lidar_fusion/result` while playing `out_vine_4swap+walk_st_ly_11_06_2024_2_label`. It shows the fine-tuned YOLO11s detections, projected Livox points over the ZED RGB image, and the displayed fused camera-frame centre labels. In each label, blue `Zc` is the forward camera depth and red `Xc` is the horizontal offset, matching the small legend in the image. Height `Yc` is intentionally not shown.
"""),
            md_cell(f"""## Figure 8: No Valid Fusion Failure

{image_md("AGHRI_fusion_failure_06_no_valid_cluster.png", "No valid fusion failure example")}

This example shows a no valid fusion failure where projected LiDAR candidate points are present for the detection, but they do not survive the association and filtering logic. The candidate set is too weak or inconsistent to meet the configured association criteria, so no retained point cluster is accepted and the pipeline does not report a fused 3D person centre for this box.
"""),
            md_cell("""## Summary

Across the frozen AGHRI fusion benchmark, AGHRI fine-tuning improves the detector input to the same LiDAR association pipeline. The main quantitative effect is lower 3D localization error and higher valid fusion rate.
"""),
        ]

    cells = attach_referenced_images(build_cells())
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    nb_copy = OUT / "aghri_fusion_paper_assets_selected.ipynb"
    nb_copy_data = dict(nb)
    nb_copy_data["cells"] = attach_referenced_images(build_cells())
    nb_copy.write_text(json.dumps(nb_copy_data, indent=2), encoding="utf-8")


def main() -> None:
    (OUT / "derived_data").mkdir(parents=True, exist_ok=True)
    copy_selected_existing_assets()
    write_selected_tables()
    crop_selected_copied_figures()
    generate_revised_d1()
    generate_combined_overall()
    generate_scenario_f9()
    generate_f14_depth_normalized_error()
    write_readme()
    write_notebook()
    print(f"Wrote selected notebook assets to {OUT}")
    print(f"Updated notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
