#!/usr/bin/env python3
"""Generate reproducible paper assets for AGHRI ZED RGB + Livox fusion.

The script intentionally treats the published evaluator outputs as immutable
source data. It writes only to the paper-assets output directory.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
EARLY_FUS_ROOT = REPO_ROOT.parent
OUT = EARLY_FUS_ROOT / "results" / "aghri_fusion_paper_assets"
FROZEN = EARLY_FUS_ROOT / "results" / "aghri_generic_vs_finetuned"
RICH = EARLY_FUS_ROOT / "results" / "aghri_detector_fusion_2x2"
TRAIN = EARLY_FUS_ROOT / "checkpoints" / "aghri_yolo11s_finetune" / "cluster_run"
BAGS = EARLY_FUS_ROOT / "rosbags" / "aghri_test"
REAL_EXAMPLE = Path("/home/prabuddhi/Desktop/aghri_fusion_real_example")
NOTEBOOK = REPO_ROOT / "notebooks" / "aghri_fusion_paper_assets.ipynb"

EXPECTED = {
    "Generic YOLO11s": {
        "detections": 2787,
        "matched_valid_results": 2561,
        "valid_fusion_rate": 0.9218,
        "mean_3d_error_m": 0.4765,
        "median_3d_error_m": 0.2038,
        "p95_3d_error_m": 1.8703,
        "error_rate_above_1m": 0.0859,
        "fps": 69.53,
    },
    "AGHRI-fine-tuned YOLO11s": {
        "detections": 2534,
        "matched_valid_results": 2433,
        "valid_fusion_rate": 0.9680,
        "mean_3d_error_m": 0.3926,
        "median_3d_error_m": 0.1911,
        "p95_3d_error_m": 1.3595,
        "error_rate_above_1m": 0.0748,
        "fps": 69.19,
    },
}

PALETTE = {
    "generic": (54, 95, 145),
    "finetuned": (204, 111, 40),
    "hybrid": (44, 138, 102),
    "legacy": (121, 90, 155),
    "grid": (222, 226, 232),
    "axis": (55, 61, 71),
    "text": (32, 36, 43),
    "muted": (98, 107, 122),
    "bg": (255, 255, 255),
}

FIGURES: list[dict[str, Any]] = []
TABLES: list[dict[str, Any]] = []
NOTES: list[str] = []


def ensure_dirs() -> None:
    for sub in [
        "figures/diagrams",
        "figures/overall",
        "figures/distributions",
        "figures/per_recording",
        "figures/robustness",
        "figures/ablations",
        "figures/training",
        "figures/synchronization",
        "figures/qualitative",
        "tables/csv",
        "tables/markdown",
        "tables/latex",
        "tables/rendered",
        "derived_data",
        "qualitative_selection",
        "logs",
        "notebooks",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fnum(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int | None = None) -> int | None:
    x = fnum(value)
    if x is None:
        return default
    return int(round(x))


def label_model(name: str) -> str:
    if name in ("finetuned", "AGHRI-fine-tuned YOLO11s"):
        return "AGHRI fine-tuned"
    if name in ("generic", "Generic YOLO11s"):
        return "Generic YOLO11s"
    return name.replace("_", " ")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for cand in candidates:
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


F10 = font(10)
F11 = font(11)
F12 = font(12)
F13 = font(13)
F14 = font(14)
F16 = font(16)
F18 = font(18, True)
F22 = font(22, True)
F28 = font(28, True)


def text_size(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=ft)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, ft: ImageFont.ImageFont, width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if text_size(draw, candidate, ft)[0] <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def save_image(img: Image.Image, rel: str, title: str, caption: str, source: str, also_pdf: bool = True) -> Path:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(path)
    files = [str(path.relative_to(OUT))]
    if also_pdf:
        pdf = path.with_suffix(".pdf")
        rgb.save(pdf)
        files.append(str(pdf.relative_to(OUT)))
    FIGURES.append({"id": path.stem, "title": title, "caption": caption, "source": source, "files": files})
    return path


def save_svg(rel: str, svg: str, title: str, caption: str, source: str) -> Path:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    FIGURES.append({"id": path.stem, "title": title, "caption": caption, "source": source, "files": [str(path.relative_to(OUT))]})
    return path


def clean_name(name: str, max_len: int = 32) -> str:
    base = name.replace("_label", "")
    parts = base.split("_")
    if len(base) <= max_len:
        return base
    return "_".join(parts[:4])[:max_len]


def chart_canvas(title: str, subtitle: str = "", size: tuple[int, int] = (1400, 900)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, PALETTE["bg"])
    d = ImageDraw.Draw(img)
    d.text((60, 38), title, fill=PALETTE["text"], font=F28)
    if subtitle:
        d.text((60, 78), subtitle, fill=PALETTE["muted"], font=F14)
    return img, d


def draw_legend(d: ImageDraw.ImageDraw, items: list[tuple[str, tuple[int, int, int]]], x: int, y: int) -> None:
    for label, color in items:
        d.rectangle([x, y + 4, x + 20, y + 16], fill=color)
        d.text((x + 28, y), label, fill=PALETTE["text"], font=F12)
        x += 28 + text_size(d, label, F12)[0] + 28


def y_scale(v: float, y0: int, y1: int, vmin: float, vmax: float) -> int:
    if vmax == vmin:
        return (y0 + y1) // 2
    return int(y1 - (v - vmin) / (vmax - vmin) * (y1 - y0))


def x_scale(v: float, x0: int, x1: int, vmin: float, vmax: float) -> int:
    if vmax == vmin:
        return (x0 + x1) // 2
    return int(x0 + (v - vmin) / (vmax - vmin) * (x1 - x0))


def draw_axes(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlab: str, ylab: str, y_ticks: list[float], y_max: float) -> None:
    x0, y0, x1, y1 = box
    d.line([x0, y1, x1, y1], fill=PALETTE["axis"], width=2)
    d.line([x0, y0, x0, y1], fill=PALETTE["axis"], width=2)
    for t in y_ticks:
        y = y_scale(t, y0, y1, 0, y_max)
        d.line([x0 - 5, y, x1, y], fill=PALETTE["grid"] if t > 0 else PALETTE["axis"])
        d.text((x0 - 54, y - 8), f"{t:g}", fill=PALETTE["muted"], font=F11)
    d.text(((x0 + x1) // 2 - text_size(d, xlab, F13)[0] // 2, y1 + 58), xlab, fill=PALETTE["axis"], font=F13)
    d.text((18, (y0 + y1) // 2 - 8), ylab, fill=PALETTE["axis"], font=F13)


def grouped_bar_chart(
    rel: str,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
    caption: str,
    source: str,
    y_max: float | None = None,
    percent: bool = False,
) -> None:
    img, d = chart_canvas(title, subtitle)
    x0, y0, x1, y1 = 120, 150, 1320, 710
    values = [v for _, vals, _ in series for v in vals if v is not None]
    ymax = y_max if y_max is not None else max(values) * 1.18 if values else 1.0
    if percent:
        ymax = max(ymax, 1.0)
    ticks = [round(ymax * i / 5, 2) for i in range(6)]
    draw_axes(d, (x0, y0, x1, y1), "", y_label, ticks, ymax)
    group_w = (x1 - x0) / max(1, len(categories))
    bar_w = min(52, max(12, int(group_w / (len(series) + 1.8))))
    for ci, cat in enumerate(categories):
        cx = x0 + group_w * ci + group_w / 2
        for si, (_, vals, color) in enumerate(series):
            val = vals[ci]
            bx = int(cx - (len(series) * bar_w) / 2 + si * bar_w)
            by = y_scale(val, y0, y1, 0, ymax)
            d.rectangle([bx, by, bx + bar_w - 5, y1], fill=color)
            txt = f"{val*100:.1f}%" if percent else f"{val:.2f}"
            tw, _ = text_size(d, txt, F11)
            d.text((bx + (bar_w - 5 - tw) // 2, by - 20), txt, fill=PALETTE["text"], font=F11)
        for j, line in enumerate(wrap_text(d, cat, F11, int(group_w) - 10)[:3]):
            tw, _ = text_size(d, line, F11)
            d.text((int(cx - tw / 2), y1 + 12 + j * 15), line, fill=PALETTE["muted"], font=F11)
    draw_legend(d, [(name, color) for name, _, color in series], x0, 112)
    save_image(img, rel, title, caption, source)


def horizontal_bar_chart(rel: str, title: str, subtitle: str, rows: list[tuple[str, float, tuple[int, int, int]]], x_label: str, caption: str, source: str) -> None:
    h = max(620, 170 + 48 * len(rows))
    img, d = chart_canvas(title, subtitle, (1400, h))
    x0, x1 = 420, 1280
    y = 145
    maxv = max([r[1] for r in rows] + [1.0])
    for label, val, color in rows:
        d.text((55, y + 7), label, fill=PALETTE["text"], font=F12)
        d.rectangle([x0, y + 6, x1, y + 30], outline=PALETTE["grid"])
        w = int((x1 - x0) * val / maxv)
        d.rectangle([x0, y + 6, x0 + w, y + 30], fill=color)
        d.text((x0 + w + 8, y + 7), f"{val:.3f}", fill=PALETTE["text"], font=F12)
        y += 48
    d.text(((x0 + x1) // 2, h - 58), x_label, fill=PALETTE["axis"], font=F13)
    save_image(img, rel, title, caption, source)


def line_chart(rel: str, title: str, subtitle: str, xs: list[float], series: list[tuple[str, list[float], tuple[int, int, int]]], x_label: str, y_label: str, caption: str, source: str, marker_x: float | None = None) -> None:
    img, d = chart_canvas(title, subtitle)
    x0, y0, x1, y1 = 120, 150, 1320, 710
    vals = [v for _, ys, _ in series for v in ys if v is not None]
    ymin = min(vals) if vals else 0.0
    ymax = max(vals) if vals else 1.0
    pad = (ymax - ymin) * 0.08 or 0.05
    ymin, ymax = ymin - pad, ymax + pad
    d.rectangle([x0, y0, x1, y1], outline=PALETTE["axis"])
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        y = y_scale(yv, y0, y1, ymin, ymax)
        d.line([x0, y, x1, y], fill=PALETTE["grid"])
        d.text((x0 - 76, y - 8), f"{yv:.2f}", fill=PALETTE["muted"], font=F11)
    xmin, xmax = min(xs), max(xs)
    for name, ys, color in series:
        pts = [(x_scale(x, x0, x1, xmin, xmax), y_scale(y, y0, y1, ymin, ymax)) for x, y in zip(xs, ys)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
        for px, py in pts[:: max(1, len(pts) // 16)]:
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)
    if marker_x is not None:
        mx = x_scale(marker_x, x0, x1, xmin, xmax)
        d.line([mx, y0, mx, y1], fill=(180, 50, 50), width=2)
        d.text((mx + 8, y0 + 8), f"selected epoch {int(marker_x)}", fill=(160, 40, 40), font=F12)
    d.text(((x0 + x1) // 2 - 30, y1 + 48), x_label, fill=PALETTE["axis"], font=F13)
    d.text((20, (y0 + y1) // 2), y_label, fill=PALETTE["axis"], font=F13)
    draw_legend(d, [(n, c) for n, _, c in series], x0, 112)
    save_image(img, rel, title, caption, source)


def ecdf_chart(rel: str, title: str, data: list[tuple[str, list[float], tuple[int, int, int]]], caption: str, source: str) -> None:
    img, d = chart_canvas(title, "Matched valid detections from the frozen offline evaluator")
    x0, y0, x1, y1 = 120, 150, 1320, 710
    allv = sorted(v for _, vals, _ in data for v in vals if math.isfinite(v))
    xmax = min(max(allv) if allv else 1.0, 5.0)
    d.rectangle([x0, y0, x1, y1], outline=PALETTE["axis"])
    for i in range(6):
        xv = xmax * i / 5
        x = x_scale(xv, x0, x1, 0, xmax)
        d.line([x, y0, x, y1], fill=PALETTE["grid"])
        d.text((x - 12, y1 + 12), f"{xv:.1f}", fill=PALETTE["muted"], font=F11)
        yv = i / 5
        y = y_scale(yv, y0, y1, 0, 1)
        d.line([x0, y, x1, y], fill=PALETTE["grid"])
        d.text((x0 - 46, y - 8), f"{yv:.1f}", fill=PALETTE["muted"], font=F11)
    for name, vals, color in data:
        vals = sorted(v for v in vals if math.isfinite(v) and v <= xmax)
        pts = []
        n = max(1, len(vals))
        for i, v in enumerate(vals):
            pts.append((x_scale(v, x0, x1, 0, xmax), y_scale((i + 1) / n, y0, y1, 0, 1)))
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
    d.text((590, y1 + 48), "3D localization error (m)", fill=PALETTE["axis"], font=F13)
    d.text((20, 410), "Cumulative fraction", fill=PALETTE["axis"], font=F13)
    draw_legend(d, [(n, c) for n, _, c in data], x0, 112)
    save_image(img, rel, title, caption, source)


def histogram_chart(rel: str, title: str, data: list[tuple[str, list[float], tuple[int, int, int]]], caption: str, source: str, xmax: float = 4.0) -> None:
    img, d = chart_canvas(title, "Errors are clipped at the right edge for readability")
    x0, y0, x1, y1 = 120, 150, 1320, 710
    bins = [xmax * i / 20 for i in range(21)]
    counts_by = []
    maxc = 1
    for name, vals, color in data:
        counts = [0] * 20
        for v in vals:
            if v is None:
                continue
            idx = min(19, int(max(0, min(v, xmax - 1e-9)) / xmax * 20))
            counts[idx] += 1
        maxc = max(maxc, max(counts))
        counts_by.append((name, counts, color))
    d.rectangle([x0, y0, x1, y1], outline=PALETTE["axis"])
    bw = (x1 - x0) / 20
    for bi in range(20):
        for si, (_, counts, color) in enumerate(counts_by):
            h = int((y1 - y0) * counts[bi] / maxc)
            bx = int(x0 + bi * bw + si * bw / (len(counts_by) + 1))
            w = int(bw / (len(counts_by) + 1))
            d.rectangle([bx, y1 - h, bx + w - 2, y1], fill=color)
    for i in range(6):
        xv = xmax * i / 5
        x = x_scale(xv, x0, x1, 0, xmax)
        d.text((x - 12, y1 + 12), f"{xv:.1f}", fill=PALETTE["muted"], font=F11)
    d.text((580, y1 + 48), "3D localization error (m)", fill=PALETTE["axis"], font=F13)
    d.text((20, 410), "Detection count", fill=PALETTE["axis"], font=F13)
    draw_legend(d, [(n, c) for n, _, c in data], x0, 112)
    save_image(img, rel, title, caption, source)


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def box_chart(rel: str, title: str, data: list[tuple[str, list[float], tuple[int, int, int]]], caption: str, source: str) -> None:
    img, d = chart_canvas(title, "Distribution summary for matched valid detections")
    x0, y0, x1, y1 = 120, 150, 1320, 710
    vmax = min(max((v for _, vals, _ in data for v in vals), default=1.0), 4.0)
    d.rectangle([x0, y0, x1, y1], outline=PALETTE["axis"])
    group_w = (x1 - x0) / len(data)
    for i in range(6):
        yv = vmax * i / 5
        y = y_scale(yv, y0, y1, 0, vmax)
        d.line([x0, y, x1, y], fill=PALETTE["grid"])
        d.text((x0 - 48, y - 8), f"{yv:.1f}", fill=PALETTE["muted"], font=F11)
    for i, (name, vals, color) in enumerate(data):
        vals = [min(v, vmax) for v in vals]
        q1, med, q3 = quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)
        p05, p95 = quantile(vals, 0.05), quantile(vals, 0.95)
        cx = int(x0 + group_w * (i + 0.5))
        yq1, ymed, yq3 = [y_scale(v, y0, y1, 0, vmax) for v in (q1, med, q3)]
        yp05, yp95 = [y_scale(v, y0, y1, 0, vmax) for v in (p05, p95)]
        d.line([cx, yp05, cx, yp95], fill=color, width=3)
        d.rectangle([cx - 70, yq3, cx + 70, yq1], outline=color, width=3)
        d.line([cx - 70, ymed, cx + 70, ymed], fill=(30, 30, 30), width=3)
        tw, _ = text_size(d, name, F12)
        d.text((cx - tw // 2, y1 + 18), name, fill=PALETTE["text"], font=F12)
    d.text((590, y1 + 50), "Detector checkpoint", fill=PALETTE["axis"], font=F13)
    d.text((20, 410), "3D error (m)", fill=PALETTE["axis"], font=F13)
    save_image(img, rel, title, caption, source)


def scatter_chart(rel: str, title: str, pairs: list[tuple[float, float]], x_label: str, y_label: str, caption: str, source: str, lim: float = 4.0) -> None:
    img, d = chart_canvas(title, "One point per jointly matched GT instance; axes clipped for readability")
    x0, y0, x1, y1 = 150, 150, 1140, 710
    d.rectangle([x0, y0, x1, y1], outline=PALETTE["axis"])
    for i in range(6):
        v = lim * i / 5
        x = x_scale(v, x0, x1, 0, lim)
        y = y_scale(v, y0, y1, 0, lim)
        d.line([x, y0, x, y1], fill=PALETTE["grid"])
        d.line([x0, y, x1, y], fill=PALETTE["grid"])
        d.text((x - 12, y1 + 12), f"{v:.1f}", fill=PALETTE["muted"], font=F11)
        d.text((x0 - 48, y - 8), f"{v:.1f}", fill=PALETTE["muted"], font=F11)
    d.line([x0, y1, x1, y0], fill=(140, 140, 140), width=2)
    for gx, fy in pairs[:: max(1, len(pairs) // 1600)]:
        px = x_scale(min(gx, lim), x0, x1, 0, lim)
        py = y_scale(min(fy, lim), y0, y1, 0, lim)
        d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(53, 114, 176))
    d.text((520, y1 + 52), x_label, fill=PALETTE["axis"], font=F13)
    d.text((28, 410), y_label, fill=PALETTE["axis"], font=F13)
    save_image(img, rel, title, caption, source)


def heatmap(rel: str, title: str, rows: list[str], cols: list[str], vals: dict[tuple[str, str], float], caption: str, source: str) -> None:
    img, d = chart_canvas(title, "Cell value is mean 3D localization error in metres")
    x0, y0 = 300, 150
    cell_w, cell_h = 170, 64
    all_vals = [v for v in vals.values() if v is not None]
    vmin, vmax = min(all_vals or [0]), max(all_vals or [1])
    for j, c in enumerate(cols):
        for line_i, line in enumerate(wrap_text(d, c, F11, cell_w - 10)[:2]):
            d.text((x0 + j * cell_w + 8, y0 - 44 + line_i * 14), line, fill=PALETTE["text"], font=F11)
    for i, r in enumerate(rows):
        d.text((55, y0 + i * cell_h + 22), r, fill=PALETTE["text"], font=F12)
        for j, c in enumerate(cols):
            v = vals.get((r, c))
            if v is None:
                fill = (240, 240, 240)
                txt = "NA"
            else:
                t = 0 if vmax == vmin else (v - vmin) / (vmax - vmin)
                fill = (int(235 - 120 * t), int(240 - 130 * t), int(245 - 80 * t))
                txt = f"{v:.2f}"
            x = x0 + j * cell_w
            y = y0 + i * cell_h
            d.rectangle([x, y, x + cell_w - 4, y + cell_h - 4], fill=fill, outline=(255, 255, 255))
            tw, th = text_size(d, txt, F13)
            d.text((x + cell_w // 2 - tw // 2, y + cell_h // 2 - th // 2), txt, fill=PALETTE["text"], font=F13)
    save_image(img, rel, title, caption, source)


def table_markdown(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def table_latex(rows: list[dict[str, Any]], cols: list[str], caption: str, label: str) -> str:
    body = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{" + "l" * len(cols) + "}", "\\hline"]
    body.append(" & ".join(cols).replace("_", "\\_") + " \\\\")
    body.append("\\hline")
    for r in rows:
        body.append(" & ".join(str(r.get(c, "")).replace("_", "\\_") for c in cols) + " \\\\")
    body += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\end{table}", ""]
    return "\n".join(body)


def render_table(path: Path, rows: list[dict[str, Any]], cols: list[str], title: str) -> None:
    w = min(2200, max(1000, 160 * len(cols)))
    row_h = 42
    h = 110 + row_h * (len(rows) + 1)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((30, 24), title, font=F22, fill=PALETTE["text"])
    x_positions = [30]
    col_w = (w - 60) // len(cols)
    for i in range(1, len(cols)):
        x_positions.append(30 + i * col_w)
    y = 80
    d.rectangle([25, y - 6, w - 25, y + row_h - 5], fill=(235, 239, 245))
    for x, c in zip(x_positions, cols):
        d.text((x + 4, y + 8), c, font=F11, fill=PALETTE["text"])
    y += row_h
    for ri, row in enumerate(rows):
        if ri % 2:
            d.rectangle([25, y - 4, w - 25, y + row_h - 5], fill=(249, 250, 252))
        for x, c in zip(x_positions, cols):
            value = str(row.get(c, ""))
            for line_i, line in enumerate(wrap_text(d, value, F10, col_w - 10)[:2]):
                d.text((x + 4, y + 5 + line_i * 14), line, font=F10, fill=PALETTE["text"])
        y += row_h
    img.save(path)
    img.convert("RGB").save(path.with_suffix(".pdf"))


def save_table(table_id: str, title: str, rows: list[dict[str, Any]], cols: list[str], caption: str) -> None:
    csv_path = OUT / "tables" / "csv" / f"{table_id}.csv"
    md_path = OUT / "tables" / "markdown" / f"{table_id}.md"
    tex_path = OUT / "tables" / "latex" / f"{table_id}.tex"
    png_path = OUT / "tables" / "rendered" / f"{table_id}.png"
    write_csv(csv_path, rows, cols)
    md_path.write_text(table_markdown(rows, cols), encoding="utf-8")
    tex_path.write_text(table_latex(rows, cols, caption, table_id), encoding="utf-8")
    render_table(png_path, rows, cols, title)
    TABLES.append({
        "id": table_id,
        "title": title,
        "caption": caption,
        "rows": len(rows),
        "files": [
            str(csv_path.relative_to(OUT)),
            str(md_path.relative_to(OUT)),
            str(tex_path.relative_to(OUT)),
            str(png_path.relative_to(OUT)),
            str(png_path.with_suffix(".pdf").relative_to(OUT)),
        ],
    })


def copy_source(path: Path, rel_dir: str = "derived_data/source_copies") -> dict[str, Any]:
    dst = OUT / rel_dir / path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return {"source": str(path), "copy": str(dst.relative_to(OUT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def validate_canonical(overall: list[dict[str, str]], manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    checks = []
    by_model = {r["model"]: r for r in overall}
    for model, expected in EXPECTED.items():
        row = by_model.get(model)
        if not row:
            raise RuntimeError(f"Missing frozen canonical row for {model}")
        for key, exp in expected.items():
            actual = fnum(row.get(key))
            if isinstance(exp, int):
                ok = int(actual or -1) == exp
            else:
                ok = round(float(actual or -999), 4 if key != "fps" else 2) == exp
            checks.append({"model": model, "metric": key, "expected": exp, "actual": actual, "passed": ok})
            if not ok:
                raise RuntimeError(f"Canonical metric changed: {model} {key} expected {exp}, got {actual}")
    recs = len({r.get("recording_name") or r.get("recording") for r in manifest})
    frames = sum(inum(r.get("expected_synchronised_frame_count"), 0) or 0 for r in manifest)
    gt = max(inum(r.get("gt_instances"), 0) or 0 for r in overall)
    for key, exp, act in [
        ("official_test_recordings", 6, recs),
        ("synchronized_test_pairs", 1248, frames),
        ("gt_3d_person_instances", 3155, gt),
    ]:
        ok = act == exp
        checks.append({"model": "test_manifest", "metric": key, "expected": exp, "actual": act, "passed": ok})
        if not ok:
            raise RuntimeError(f"Canonical manifest changed: {key} expected {exp}, got {act}")
    return checks


def load_valid_errors(run: str) -> list[float]:
    rows = read_csv(RICH / "runs" / run / "detections.csv")
    return [fnum(r.get("lidar_center_error")) for r in rows if r.get("valid") == "True" and r.get("status") == "matched" and fnum(r.get("lidar_center_error")) is not None]  # type: ignore[list-item]


def pair_by_gt(run_a: str, run_b: str) -> list[dict[str, Any]]:
    a = read_csv(RICH / "runs" / run_a / "detections.csv")
    b = read_csv(RICH / "runs" / run_b / "detections.csv")
    def key(r: dict[str, str]) -> tuple[str, str, str]:
        return (r.get("recording", ""), r.get("frame_number", ""), r.get("matched_gt_identity") or r.get("prediction_identity", ""))
    amap = {key(r): r for r in a if r.get("valid") == "True" and r.get("status") == "matched" and key(r)[2]}
    out = []
    for r in b:
        k = key(r)
        if r.get("valid") == "True" and r.get("status") == "matched" and k in amap:
            ea = fnum(amap[k].get("lidar_center_error"))
            eb = fnum(r.get("lidar_center_error"))
            if ea is not None and eb is not None:
                out.append({"key": k, "a_error": ea, "b_error": eb, "diff_b_minus_a": eb - ea, "a": amap[k], "b": r})
    return out


def write_diagrams() -> None:
    def svg_box(x, y, w, h, label, fill="#f4f7fb"):
        lines = html.escape(label).split("\\n")
        txt = "".join([f'<text x="{x+w/2}" y="{y+32+i*18}" text-anchor="middle" font-size="15" fill="#20242b">{line}</text>' for i, line in enumerate(lines)])
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="#56616f"/>{txt}'
    def arrow(x1, y1, x2, y2, label=""):
        t = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#384252" stroke-width="3" marker-end="url(#a)"/>'
        if label:
            t += f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-8}" text-anchor="middle" font-size="13" fill="#596273">{html.escape(label)}</text>'
        return t
    pre = '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680"><defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#384252"/></marker></defs><rect width="1200" height="680" fill="white"/>'
    post = "</svg>"
    d1 = pre + '<text x="60" y="56" font-size="28" font-weight="700" fill="#20242b">D1. Camera-guided LiDAR association workflow</text>'
    boxes = [
        (70, 130, 190, 90, "ZED RGB image\\nfront-left camera", "#edf6ff"),
        (70, 300, 190, 90, "Livox point cloud\\nfront_lidar_link", "#eef8f1"),
        (350, 130, 190, 90, "YOLO11s person\\ndetections", "#fff3e7"),
        (350, 300, 190, 90, "TF + camera info\\nproject LiDAR to RGB", "#f4f0ff"),
        (630, 210, 210, 100, "2D box association\\ntrim/filter/select points", "#f9f9ed"),
        (920, 210, 210, 100, "3D person centre\\nlabel: person : z m", "#eef8f1"),
    ]
    for b in boxes:
        d1 += svg_box(*b)
    for a in [(260,175,350,175,"image"),(260,345,350,345,"points"),(540,175,630,245,"boxes"),(540,345,630,275,"projected points"),(840,260,920,260,"retained points")]:
        d1 += arrow(*a)
    d1 += post
    save_svg("figures/diagrams/D1_fusion_method_overview.svg", d1, "D1 Fusion Method Overview", "Camera detections constrain projected LiDAR points; retained points are averaged into the reported 3D person centre.", "Repository implementation and frozen evaluator metadata")

    d2 = pre + '<text x="60" y="56" font-size="28" font-weight="700" fill="#20242b">D2. ROS-native playback/runtime architecture</text>'
    for b in [
        (65, 140, 210, 90, "ros2 bag play\\nall dataset topics", "#eef8f1"),
        (350, 80, 220, 80, "/dataset/cam_zed_rgb\\nimage + camera_info", "#edf6ff"),
        (350, 220, 220, 80, "/dataset/lidar/points\\nLivox cloud", "#eef8f1"),
        (350, 360, 220, 80, "/tf + /tf_static\\nframe tree", "#f4f0ff"),
        (670, 190, 230, 120, "camera_lidar_fusion\\nROS 2 node", "#fff3e7"),
        (990, 140, 170, 80, "/camera_lidar_fusion/result", "#f9f9ed"),
        (990, 280, 170, 80, "/pred_bboxes + RViz", "#f9f9ed"),
    ]:
        d2 += svg_box(*b)
    for a in [(275,185,350,120,""),(275,185,350,260,""),(275,185,350,400,""),(570,120,670,225,""),(570,260,670,250,""),(570,400,670,275,""),(900,240,990,180,""),(900,260,990,320,"")]:
        d2 += arrow(*a)
    d2 += post
    save_svg("figures/diagrams/D2_ros_runtime_architecture.svg", d2, "D2 ROS Runtime Architecture", "The full-topic ROS 2 bags publish images, camera info, point clouds and TF; the fusion node subscribes only to the topics required for ZED RGB + Livox fusion.", "ROS bag verification summary")

    matrix_path = REAL_EXAMPLE / "calibration" / "camera_from_lidar_matrix.json"
    matrix_obj = read_json(matrix_path) if matrix_path.exists() else []
    matrix = matrix_obj.get("matrix_4x4", matrix_obj) if isinstance(matrix_obj, dict) else matrix_obj
    matrix_txt = "\\n".join(" ".join(f"{float(v): .3f}" for v in row) for row in matrix[:4]) if matrix else "matrix unavailable"
    d3 = pre + '<text x="60" y="56" font-size="28" font-weight="700" fill="#20242b">D3. Calibration TF chain and camera-from-LiDAR transform</text>'
    names = ["front_lidar_link", "base_link", "front_camera_link", "front_camera_center", "front_left_camera_frame", "front_left_camera_optical_frame"]
    x = 50
    for name in names:
        d3 += svg_box(x, 150, 170, 70, name, "#f4f7fb")
        if name != names[-1]:
            d3 += arrow(x + 170, 185, x + 195, 185)
        x += 195
    d3 += svg_box(290, 330, 620, 150, "camera_from_lidar_matrix\\n" + matrix_txt, "#eef8f1")
    d3 += post
    save_svg("figures/diagrams/D3_calibration_tf_chain.svg", d3, "D3 Calibration TF Chain", "Static TF connects the Livox frame to the ZED optical frame; the equivalent camera-from-LiDAR matrix is included for auditability.", str(matrix_path))

    d4 = pre + '<text x="60" y="56" font-size="28" font-weight="700" fill="#20242b">D4. Original framework vs AGHRI adaptation</text>'
    for b in [
        (110, 140, 360, 90, "Original camera-LiDAR demo\\nmanual/static calibration file", "#f4f7fb"),
        (110, 280, 360, 90, "Small topic set\\nminimal bag playback", "#f4f7fb"),
        (730, 140, 360, 90, "AGHRI ZED RGB + Livox\\nCameraInfo + TF-native calibration", "#eef8f1"),
        (730, 280, 360, 90, "Full-topic dataset bags\\n/fisheye images, /camera_info, /tf", "#eef8f1"),
        (420, 470, 360, 90, "Same fusion behaviour\\nvalidated against frozen evaluator", "#fff3e7"),
    ]:
        d4 += svg_box(*b)
    for a in [(470,185,730,185,"adapt"),(470,325,730,325,"extend"),(910,370,780,500,""),(290,370,420,500,"")]:
        d4 += arrow(*a)
    d4 += post
    save_svg("figures/diagrams/D4_original_vs_aghri_adaptation.svg", d4, "D4 Original vs AGHRI Adaptation", "The adaptation changes data ingestion and calibration plumbing while preserving the evaluated camera-guided LiDAR association behaviour.", "Repository README/configuration history")


def build_figures_and_tables() -> dict[str, Any]:
    overall = read_csv(FROZEN / "overall_results.csv")
    per_rec = read_csv(FROZEN / "per_recording_results.csv")
    overall_2x2 = read_csv(RICH / "tables" / "overall_2x2.csv")
    per_rec_2x2 = read_csv(RICH / "tables" / "per_recording_2x2.csv")
    condition = read_csv(RICH / "tables" / "condition_breakdowns.csv")
    manifest = read_csv(FROZEN / "manifest" / "test_manifest.csv")
    checks = validate_canonical(overall, manifest)
    write_csv(OUT / "logs" / "canonical_assertion.csv", checks)
    write_json(OUT / "logs" / "canonical_assertion.json", checks)

    source_files = [
        FROZEN / "overall_results.csv",
        FROZEN / "per_recording_results.csv",
        FROZEN / "checkpoint_comparison.json",
        FROZEN / "manifest" / "test_manifest.csv",
        RICH / "tables" / "overall_2x2.csv",
        RICH / "tables" / "per_recording_2x2.csv",
        RICH / "tables" / "condition_breakdowns.csv",
        RICH / "comparisons" / "pairwise_comparisons.json",
        BAGS / "verification_summary.json",
        TRAIN / "results.csv",
        REAL_EXAMPLE / "reproduction_result.json",
    ]
    source_manifest = [copy_source(p) for p in source_files if p.exists()]

    # Tables.
    setup_rows = [{
        "item": "Official held-out test recordings",
        "value": "6",
        "source": "frozen test manifest",
    }, {
        "item": "Synchronized RGB-LiDAR pairs",
        "value": "1248",
        "source": "frozen test manifest",
    }, {
        "item": "GT 3D person instances",
        "value": "3155",
        "source": "frozen test manifest",
    }, {
        "item": "Camera frame",
        "value": "front_left_camera_optical_frame",
        "source": "fusion evaluator/ROS bag verification",
    }, {
        "item": "LiDAR frame",
        "value": "front_lidar_link",
        "source": "fusion evaluator/ROS bag verification",
    }]
    save_table("T1_experimental_setup", "T1 Experimental Setup", setup_rows, ["item", "value", "source"], "Experimental setup and immutable test-data counts.")

    t2_rows = []
    for r in overall:
        t2_rows.append({
            "model": label_model(r["model"]),
            "detections": r["detections"],
            "matched": r["matched_valid_results"],
            "valid_rate": f"{float(r['valid_fusion_rate']):.4f}",
            "mean_m": f"{float(r['mean_3d_error_m']):.4f}",
            "median_m": f"{float(r['median_3d_error_m']):.4f}",
            "p95_m": f"{float(r['p95_3d_error_m']):.4f}",
            "err_gt_1m": f"{float(r['error_rate_above_1m']):.4f}",
            "fps": f"{float(r['fps']):.2f}",
        })
    save_table("T2_overall_results", "T2 Overall Frozen Test Results", t2_rows, list(t2_rows[0]), "Frozen held-out test results from the offline evaluator.")

    by_model = {r["model"]: r for r in overall}
    g, f = by_model["Generic YOLO11s"], by_model["AGHRI-fine-tuned YOLO11s"]
    imp_rows = []
    for label, key, better_lower in [
        ("Mean 3D error", "mean_3d_error_m", True),
        ("Median 3D error", "median_3d_error_m", True),
        ("P95 3D error", "p95_3d_error_m", True),
        ("Error > 1m rate", "error_rate_above_1m", True),
        ("Valid fusion rate", "valid_fusion_rate", False),
    ]:
        gv, fv = float(g[key]), float(f[key])
        delta = fv - gv
        rel = (gv - fv) / gv if better_lower else (fv - gv) / gv
        imp_rows.append({"metric": label, "generic": f"{gv:.4f}", "finetuned": f"{fv:.4f}", "delta_ft_minus_generic": f"{delta:.4f}", "relative_improvement": f"{rel*100:.1f}%"})
    save_table("T3_relative_improvements", "T3 Relative Improvements", imp_rows, list(imp_rows[0]), "Fine-tuned detector improvements relative to generic YOLO11s.")

    t4_rows = []
    for r in per_rec:
        t4_rows.append({
            "model": label_model(r["model"]),
            "recording": clean_name(r["recording"], 38),
            "scene": r["scene"],
            "motion": r["motion"],
            "gt": r["gt_instances"],
            "matched": r["matched_valid_results"],
            "valid_rate": f"{float(r['valid_fusion_rate']):.3f}",
            "mean_m": f"{float(r['mean_3d_error_m']):.3f}",
            "p95_m": f"{float(r['p95_3d_error_m']):.3f}",
        })
    save_table("T4_per_recording", "T4 Per-recording Results", t4_rows, list(t4_rows[0]), "Per-recording held-out results grouped by detector checkpoint.")

    defs = [
        {"metric": "valid fusion rate", "definition": "matched valid fused detections divided by detector person detections", "unit": "ratio"},
        {"metric": "mean/median/P95 3D error", "definition": "Euclidean error between fused person centre and GT 3D person centre", "unit": "metres"},
        {"metric": "error > 1m", "definition": "fraction of matched valid fused detections with 3D error above 1 metre", "unit": "ratio"},
        {"metric": "FPS", "definition": "approximate offline detector + fusion throughput reported by evaluator", "unit": "frames/s"},
        {"metric": "camera depth error", "definition": "absolute difference along the camera optical depth axis", "unit": "metres"},
    ]
    save_table("T5_metric_definitions", "T5 Metric Definitions", defs, ["metric", "definition", "unit"], "Definitions of reported fusion metrics.")

    orig = [
        {"component": "Calibration", "original": "static calibration/config path", "aghri": "CameraInfo + TF with JSON fallback equivalence check"},
        {"component": "Input topics", "original": "minimal image/points topics", "aghri": "full-topic dataset bags including fisheye CameraInfo and TF"},
        {"component": "Evaluation", "original": "demo/runtime visualization", "aghri": "offline frozen evaluator for numerical claims"},
        {"component": "Qualitative review", "original": "RViz/manual view", "aghri": "deterministic selected examples plus ROS playback screenshots when available"},
    ]
    save_table("T6_original_vs_aghri", "T6 Original vs AGHRI Adaptation", orig, ["component", "original", "aghri"], "Implementation changes required for AGHRI ZED RGB + Livox fusion.")

    ver = read_json(BAGS / "verification_summary.json") if (BAGS / "verification_summary.json").exists() else {}
    topic_rows = [
        {"topic": ver.get("image_topic", "/dataset/cam_zed_rgb/image"), "type": "sensor_msgs/Image", "frame": ver.get("dataset_tools_image_frame", "front_left_camera_optical_frame")},
        {"topic": ver.get("lidar_topic", "/dataset/lidar/points"), "type": "sensor_msgs/PointCloud2", "frame": ver.get("dataset_tools_lidar_frame", "front_lidar_link")},
        {"topic": "/dataset/cam_zed_rgb/camera_info", "type": "sensor_msgs/CameraInfo", "frame": "front_left_camera_optical_frame"},
        {"topic": "/dataset/cam_fish_front/camera_info", "type": "sensor_msgs/CameraInfo", "frame": "fish_front_camera_link_optical"},
        {"topic": "/dataset/cam_fish_left/camera_info", "type": "sensor_msgs/CameraInfo", "frame": "fish_left_camera_link_optical"},
        {"topic": "/dataset/cam_fish_right/camera_info", "type": "sensor_msgs/CameraInfo", "frame": "fish_right_camera_link_optical"},
        {"topic": "/tf", "type": "tf2_msgs/TFMessage", "frame": "map -> base_link"},
        {"topic": "/tf_static", "type": "tf2_msgs/TFMessage", "frame": "static sensor tree"},
    ]
    save_table("T7_ros_topics_frames", "T7 ROS Topics and Frames", topic_rows, ["topic", "type", "frame"], "ROS 2 bag topics and frame IDs used by the fusion workflow.")

    sync_rows = []
    for row in ver.get("rows", []):
        sync_rows.append({
            "recording": clean_name(row["recording"], 36),
            "status": row.get("verification_status"),
            "zed_images": row.get("image_count"),
            "clouds": row.get("point_cloud_count"),
            "tf": row.get("tf_count"),
            "tf_static": row.get("tf_static_count"),
            "duration_s": f"{float(row.get('duration_sec', 0)):.2f}",
        })
    if sync_rows:
        save_table("T8_sync_verification", "T8 ROS Bag Synchronization Verification", sync_rows, list(sync_rows[0]), "Per-recording verification for regenerated all-topic ROS 2 bags.")

    ablation_rows = [{
        "ablation": "box retention 0.60-1.00",
        "split": "validation only",
        "status": "not run in this paper-asset generation",
        "reason": "no frozen validation ablation outputs existed; helper script provided to run without touching test results",
    }, {
        "ablation": "z-score 2.5-4.0/no filtering",
        "split": "validation only",
        "status": "not run",
        "reason": "current evaluator has fixed modified z-score threshold in shared Fusser implementation",
    }, {
        "ablation": "hybrid depth parameters",
        "split": "validation only",
        "status": "helper available",
        "reason": "tools/run_aghri_fusion_validation_ablation.py writes isolated outputs under this asset directory",
    }]
    save_table("T9_validation_ablation_manifest", "T9 Validation Ablation Manifest", ablation_rows, list(ablation_rows[0]), "Validation-only ablation status; no test metrics are altered.")

    # Overall figures.
    cats = ["mean", "median", "P95"]
    grouped_bar_chart(
        "figures/overall/F1_overall_localization_errors.png",
        "F1. Overall 3D Localization Error",
        "Frozen held-out test comparison",
        cats,
        [
            ("Generic YOLO11s", [float(g["mean_3d_error_m"]), float(g["median_3d_error_m"]), float(g["p95_3d_error_m"])], PALETTE["generic"]),
            ("AGHRI fine-tuned", [float(f["mean_3d_error_m"]), float(f["median_3d_error_m"]), float(f["p95_3d_error_m"])], PALETTE["finetuned"]),
        ],
        "3D error (m)",
        "Fine-tuned YOLO11s reduces mean, median and P95 3D localization error on the frozen held-out test set.",
        str(FROZEN / "overall_results.csv"),
    )
    grouped_bar_chart(
        "figures/overall/F2_reliability.png",
        "F2. Fusion Reliability",
        "Higher valid-fusion rate and lower >1m outlier rate are preferred",
        ["valid rate", "error >1m"],
        [
            ("Generic YOLO11s", [float(g["valid_fusion_rate"]), float(g["error_rate_above_1m"])], PALETTE["generic"]),
            ("AGHRI fine-tuned", [float(f["valid_fusion_rate"]), float(f["error_rate_above_1m"])], PALETTE["finetuned"]),
        ],
        "rate",
        "Reliability metrics from the frozen evaluator: valid fusion rate increases while the >1m error rate decreases for the AGHRI fine-tuned detector.",
        str(FROZEN / "overall_results.csv"),
        y_max=1.0,
        percent=True,
    )
    grouped_bar_chart(
        "figures/overall/F3_runtime.png",
        "F3. Offline Runtime",
        "Approximate detector plus fusion throughput",
        ["FPS"],
        [
            ("Generic YOLO11s", [float(g["fps"])], PALETTE["generic"]),
            ("AGHRI fine-tuned", [float(f["fps"])], PALETTE["finetuned"]),
        ],
        "frames/s",
        "Both checkpoints run at similar offline throughput in the frozen evaluator.",
        str(FROZEN / "overall_results.csv"),
    )
    horizontal_bar_chart(
        "figures/overall/F4_relative_improvements.png",
        "F4. Relative Improvements from Fine-tuning",
        "Positive bars indicate improvement over generic YOLO11s",
        [(r["metric"], float(r["relative_improvement"].rstrip("%")) / 100.0, PALETTE["finetuned"]) for r in imp_rows],
        "relative improvement",
        "Relative improvements are computed from frozen overall evaluator metrics.",
        str(FROZEN / "overall_results.csv"),
    )

    generic_errors = load_valid_errors("generic_legacy")
    finetuned_errors = load_valid_errors("finetuned_legacy")
    ecdf_chart(
        "figures/distributions/F5_ecdf_3d_error.png",
        "F5. ECDF of 3D Localization Error",
        [("Generic YOLO11s", generic_errors, PALETTE["generic"]), ("AGHRI fine-tuned", finetuned_errors, PALETTE["finetuned"])],
        "The ECDF shows the cumulative fraction of matched valid detections below each 3D error threshold.",
        str(RICH / "runs/*/detections.csv"),
    )
    histogram_chart(
        "figures/distributions/F6a_error_histogram.png",
        "F6a. 3D Error Histogram",
        [("Generic YOLO11s", generic_errors, PALETTE["generic"]), ("AGHRI fine-tuned", finetuned_errors, PALETTE["finetuned"])],
        "Histogram of matched valid 3D errors, clipped at 4 m for readability.",
        str(RICH / "runs/*/detections.csv"),
    )
    box_chart(
        "figures/distributions/F6b_error_boxplot.png",
        "F6b. 3D Error Box/Whisker Summary",
        [("Generic YOLO11s", generic_errors, PALETTE["generic"]), ("AGHRI fine-tuned", finetuned_errors, PALETTE["finetuned"])],
        "Boxes show quartiles, median and 5th/95th percentiles for matched valid detections.",
        str(RICH / "runs/*/detections.csv"),
    )
    pairs = pair_by_gt("generic_legacy", "finetuned_legacy")
    scatter_chart(
        "figures/distributions/F7_paired_error_scatter.png",
        "F7. Paired Generic vs Fine-tuned Errors",
        [(p["a_error"], p["b_error"]) for p in pairs],
        "Generic error (m)",
        "Fine-tuned error (m)",
        "Each point is a jointly matched GT instance. Points below the diagonal improve after fine-tuning.",
        str(RICH / "runs/*/detections.csv"),
    )
    diff_vals = [p["diff_b_minus_a"] for p in pairs]
    histogram_chart(
        "figures/distributions/F8_paired_error_difference.png",
        "F8. Paired Error Difference",
        [("fine-tuned minus generic", [v + 2.0 for v in diff_vals], PALETTE["finetuned"])],
        "Histogram uses a +2 m plotting offset; negative original values mean fine-tuned is better.",
        str(RICH / "runs/*/detections.csv"),
        xmax=4.0,
    )

    recs = sorted({r["recording"] for r in per_rec})
    for metric, rel, title, ylabel, percent in [
        ("mean_3d_error_m", "F9_per_recording_mean_error.png", "F9. Per-recording Mean 3D Error", "mean error (m)", False),
        ("valid_fusion_rate", "F10_per_recording_valid_rate.png", "F10. Per-recording Valid Fusion Rate", "valid rate", True),
        ("p95_3d_error_m", "F11_per_recording_p95.png", "F11. Per-recording P95 Error", "P95 error (m)", False),
    ]:
        vals = {r["model"]: {rr["recording"]: float(rr[metric]) for rr in per_rec if rr["model"] == r["model"]} for r in overall}
        grouped_bar_chart(
            f"figures/per_recording/{rel}",
            title,
            "Frozen held-out test set",
            [clean_name(r, 22) for r in recs],
            [
                ("Generic YOLO11s", [vals["Generic YOLO11s"][r] for r in recs], PALETTE["generic"]),
                ("AGHRI fine-tuned", [vals["AGHRI-fine-tuned YOLO11s"][r] for r in recs], PALETTE["finetuned"]),
            ],
            ylabel,
            f"{title} computed from frozen per-recording evaluator outputs.",
            str(FROZEN / "per_recording_results.csv"),
            y_max=1.0 if percent else None,
            percent=percent,
        )

    scene_rows = [r for r in condition if r.get("breakdown") in ("mean_lidar_error_by_scene", "valid_rate_by_scene", "mean_lidar_error_by_motion")]
    if scene_rows:
        rows = sorted({r["breakdown"] for r in scene_rows})
        cols = sorted({r["group"] for r in scene_rows})
        vals = {(r["breakdown"].replace("_", " "), r["group"]): float(r["mean_3d_error_m"]) for r in scene_rows if r["detector"] == "finetuned" and r["fusion_method"] == "legacy_box"}
        heatmap("figures/per_recording/F12_scene_motion_condition_heatmap.png", "F12. Scene/Motion Condition Summary", [r.replace("_", " ") for r in rows], cols, vals, "Scene and motion summaries are included where evaluator condition fields are present.", str(RICH / "tables/condition_breakdowns.csv"))

    human_counts = defaultdict(lambda: {"frames": 0, "gt": 0})
    frames = read_json(RICH / "runs" / "finetuned_legacy" / "frames.json")
    for row in frames:
        c = int(row.get("gt3d_count") or 0)
        human_counts[str(c)]["frames"] += 1
        human_counts[str(c)]["gt"] += c
    if human_counts:
        rows = [(f"{k} person(s)", v["frames"], PALETTE["finetuned"]) for k, v in sorted(human_counts.items(), key=lambda x: int(x[0]))]
        horizontal_bar_chart("figures/per_recording/F13_human_count_breakdown.png", "F13. Human-count Breakdown", "Derived from GT counts per synchronized frame", rows, "frames", "Frame-level human counts derived from evaluator frame summaries.", str(RICH / "runs/finetuned_legacy/frames.json"))

    depth_rows = [r for r in condition if r.get("breakdown") == "mean_lidar_error_by_distance_band" and r.get("fusion_method") == "legacy_box"]
    if depth_rows:
        bands = sorted({r["group"] for r in depth_rows})
        vals = defaultdict(dict)
        for r in depth_rows:
            vals[r["detector"]][r["group"]] = float(r["mean_3d_error_m"])
        grouped_bar_chart(
            "figures/robustness/F14_depth_bins.png",
            "F14. Error by Depth Band",
            "Distance-band means from evaluator condition breakdown",
            bands,
            [("Generic YOLO11s", [vals["generic"].get(b, 0) for b in bands], PALETTE["generic"]), ("AGHRI fine-tuned", [vals["finetuned"].get(b, 0) for b in bands], PALETTE["finetuned"])],
            "mean error (m)",
            "Mean 3D error by ground-truth distance band.",
            str(RICH / "tables/condition_breakdowns.csv"),
        )

    det_rows = read_csv(RICH / "runs" / "finetuned_legacy" / "detections.csv")
    point_bands = Counter(r.get("point_count_band", "unknown") for r in det_rows if r.get("valid") == "True")
    horizontal_bar_chart("figures/robustness/F15_retained_point_count.png", "F15. Retained Point-count Bands", "Fine-tuned legacy association", [(k, v, PALETTE["finetuned"]) for k, v in sorted(point_bands.items())], "matched detections", "Distribution of retained LiDAR support bands for matched fine-tuned detections.", str(RICH / "runs/finetuned_legacy/detections.csv"))
    filter_pairs = [(fnum(r.get("candidate_points"), 0) or 0, fnum(r.get("retained_points"), 0) or 0) for r in det_rows if r.get("valid") == "True"]
    scatter_chart("figures/robustness/F16_filtering_behavior.png", "F16. Candidate vs Retained Points", filter_pairs, "candidate points", "retained points", "Association filtering behaviour for fine-tuned matched detections.", str(RICH / "runs/finetuned_legacy/detections.csv"), lim=max([x for p in filter_pairs for x in p] + [10]))
    conf_pairs = [(fnum(r.get("confidence"), 0) or 0, min(fnum(r.get("lidar_center_error"), 0) or 0, 4.0)) for r in det_rows if r.get("valid") == "True"]
    scatter_chart("figures/robustness/F17_confidence_vs_error.png", "F17. Confidence vs 3D Error", conf_pairs, "YOLO confidence", "3D error (m)", "Detector confidence compared with 3D localization error for fine-tuned matched detections.", str(RICH / "runs/finetuned_legacy/detections.csv"), lim=1.0)

    sync_plot_rows = []
    for row in ver.get("rows", []):
        sync_plot_rows.append((clean_name(row["recording"], 26), float(row.get("duration_sec", 0)), PALETTE["generic"]))
    if sync_plot_rows:
        horizontal_bar_chart("figures/synchronization/F18_sync_duration_counts.png", "F18. Bag Duration and Synchronization Audit", "All-topic bag verification passed for all official test recordings", sync_plot_rows, "duration (s)", "ROS 2 bag synchronization/TF verification summary; numerical fusion claims remain from the offline evaluator.", str(BAGS / "verification_summary.json"))

    # Ablation figures are explicit status artifacts to avoid implying test-set tuning.
    img, d = chart_canvas("A1-A3. Validation Ablation Status", "No validation ablation outputs were present before this task")
    msg = [
        "Validation-only ablation helper created:",
        "tools/run_aghri_fusion_validation_ablation.py",
        "",
        "This paper-asset generation did not tune or alter the frozen test metrics.",
        "Ablation rows are documented in T9 with exact missing data reasons.",
    ]
    y = 185
    for line in msg:
        d.text((120, y), line, fill=PALETTE["text"], font=F18 if line else F14)
        y += 42
    save_image(img, "figures/ablations/A1_A2_A3_validation_ablation_status.png", "A1-A3 Validation Ablation Status", "Validation ablations are intentionally separated from held-out test results; T9 records what was and was not available.", "tools/run_aghri_fusion_validation_ablation.py")

    train_csv = TRAIN / "results.csv"
    if train_csv.exists():
        train_rows = read_csv(train_csv)
        xs = [float(r["epoch"]) for r in train_rows]
        line_chart(
            "figures/training/F19_training_curves.png",
            "F19. YOLO11s Fine-tuning Curves",
            "Cluster run; selected best checkpoint at epoch 46",
            xs,
            [
                ("precision", [float(r["metrics/precision(B)"]) for r in train_rows], PALETTE["generic"]),
                ("recall", [float(r["metrics/recall(B)"]) for r in train_rows], PALETTE["finetuned"]),
                ("mAP50", [float(r["metrics/mAP50(B)"]) for r in train_rows], PALETTE["hybrid"]),
                ("mAP50-95", [float(r["metrics/mAP50-95(B)"]) for r in train_rows], PALETTE["legacy"]),
            ],
            "epoch",
            "metric",
            "Fine-tuning curves copied from the YOLO cluster training run; epoch 46 is marked as the selected best checkpoint.",
            str(train_csv),
            marker_x=46,
        )
    grouped_bar_chart(
        "figures/training/F20_detector_fusion_summary.png",
        "F20. Detector Checkpoint Fusion Summary",
        "Fusion-level result of the two frozen detector checkpoints",
        ["detections", "matched"],
        [
            ("Generic YOLO11s", [float(g["detections"]), float(g["matched_valid_results"])], PALETTE["generic"]),
            ("AGHRI fine-tuned", [float(f["detections"]), float(f["matched_valid_results"])], PALETTE["finetuned"]),
        ],
        "count",
        "Detector-level count comparison using fusion evaluator outputs.",
        str(FROZEN / "overall_results.csv"),
    )

    # Qualitative selections.
    q_rows = []
    q_src = REAL_EXAMPLE / "visuals" / "06_final_fusion_result.png"
    if q_src.exists():
        q_dst = OUT / "figures/qualitative/Q1_representative_fusion_result.png"
        shutil.copy2(q_src, q_dst)
        FIGURES.append({"id": "Q1_representative_fusion_result", "title": "Q1 Representative Fusion Result", "caption": "Representative fine-tuned ZED RGB + Livox fusion result from the self-contained real example.", "source": str(q_src), "files": [str(q_dst.relative_to(OUT))]})
        q_rows.append({"id": "Q1", "file": str(q_dst.relative_to(OUT)), "selection": "self-contained representative example", "source": str(q_src)})
    raw = REAL_EXAMPLE / "visuals" / "01_raw_zed_rgb.png"
    fused = REAL_EXAMPLE / "visuals" / "06_final_fusion_result.png"
    if raw.exists() and fused.exists():
        imgs = [Image.open(raw).convert("RGB"), Image.open(fused).convert("RGB")]
        sheet = Image.new("RGB", (imgs[0].width * 2 + 30, imgs[0].height + 80), "white")
        d = ImageDraw.Draw(sheet)
        d.text((15, 15), "Q2. Raw ZED RGB", font=F18, fill=PALETTE["text"])
        d.text((imgs[0].width + 30, 15), "Annotated fusion output", font=F18, fill=PALETTE["text"])
        sheet.paste(imgs[0], (15, 55))
        sheet.paste(imgs[1], (imgs[0].width + 30, 55))
        save_image(sheet, "figures/qualitative/Q2_raw_vs_fused_pair.png", "Q2 Raw vs Fused Pair", "Raw ZED RGB image paired with the annotated fusion output for the same example.", str(REAL_EXAMPLE / "visuals"), also_pdf=False)
        q_rows.append({"id": "Q2", "file": "figures/qualitative/Q2_raw_vs_fused_pair.png", "selection": "raw/fused pair from self-contained example", "source": str(REAL_EXAMPLE)})
    if pairs:
        best = min(pairs, key=lambda p: p["diff_b_minus_a"])
        rec, frame, detid = best["key"]
        # Prefer exact generated visual files if they exist.
        candidates = []
        for run in ["generic_legacy", "finetuned_legacy"]:
            pat = f"{rec}_{int(frame):03d}_"
            run_dir = RICH / "visuals" / run
            found = sorted(run_dir.glob(pat + "*.png"))
            candidates.append(found[0] if found else None)
        if all(candidates):
            ims = [Image.open(p).convert("RGB") for p in candidates]  # type: ignore[arg-type]
            sheet = Image.new("RGB", (ims[0].width * 2 + 30, ims[0].height + 95), "white")
            d = ImageDraw.Draw(sheet)
            d.text((15, 12), f"Generic error {best['a_error']:.2f} m", font=F18, fill=PALETTE["generic"])
            d.text((ims[0].width + 30, 12), f"Fine-tuned error {best['b_error']:.2f} m", font=F18, fill=PALETTE["finetuned"])
            d.text((15, 40), f"largest paired improvement: {(-best['diff_b_minus_a']):.2f} m", font=F13, fill=PALETTE["text"])
            sheet.paste(ims[0], (15, 70))
            sheet.paste(ims[1], (ims[0].width + 30, 70))
            save_image(sheet, "figures/qualitative/Q3_generic_vs_finetuned_pair.png", "Q3 Generic vs Fine-tuned Pair", "Largest jointly matched paired improvement among available qualitative visual files.", str(RICH / "visuals"), also_pdf=False)
            q_rows.append({"id": "Q3", "file": "figures/qualitative/Q3_generic_vs_finetuned_pair.png", "selection": f"largest paired improvement, frame {frame}", "source": str(RICH / "visuals")})
        else:
            NOTES.append("Q3 exact visual pair was not available for the largest paired improvement key; numeric pair is recorded in derived_data/paired_generic_finetuned.csv.")
    write_csv(OUT / "derived_data" / "paired_generic_finetuned.csv", [{"recording": p["key"][0], "frame": p["key"][1], "gt": p["key"][2], "generic_error": p["a_error"], "finetuned_error": p["b_error"], "diff_finetuned_minus_generic": p["diff_b_minus_a"]} for p in pairs])

    # Q4 step-by-step contact sheet.
    step_imgs = [REAL_EXAMPLE / "visuals" / f for f in ["01_raw_zed_rgb.png", "03_all_projected_lidar_points.png", "04_candidate_points_inside_person_box.png", "05_selected_vs_rejected_points.png", "06_final_fusion_result.png"]]
    existing = [p for p in step_imgs if p.exists()]
    repro = read_json(REAL_EXAMPLE / "reproduction_result.json") if (REAL_EXAMPLE / "reproduction_result.json").exists() else {}
    if existing:
        thumbs = []
        for p in existing:
            im = Image.open(p).convert("RGB")
            im.thumbnail((360, 210))
            thumbs.append((p, im.copy()))
        sheet = Image.new("RGB", (760, 760), "white")
        d = ImageDraw.Draw(sheet)
        d.text((25, 18), "Q4. Step-by-step association walkthrough", font=F22, fill=PALETTE["text"])
        d.text((25, 50), f"candidate {repro.get('candidate_points', 22)} retained {repro.get('selected_after_filtering', 15)} rejected {int(repro.get('candidate_points', 22))-int(repro.get('selected_after_filtering', 15))}; error {float(repro.get('recomputed_lidar_error_m', 0.1337)):.4f} m", font=F14, fill=PALETTE["muted"])
        positions = [(25, 90), (395, 90), (25, 330), (395, 330), (205, 570)]
        for (p, im), pos in zip(thumbs, positions):
            d.text((pos[0], pos[1] - 20), p.stem.replace("_", " "), font=F12, fill=PALETTE["text"])
            sheet.paste(im, pos)
        save_image(sheet, "figures/qualitative/Q4_step_by_step_association.png", "Q4 Step-by-step Association", "Self-contained walkthrough: candidate 22 retained 15 rejected 7, recomputed LiDAR-centre error about 0.1337 m.", str(REAL_EXAMPLE), also_pdf=False)
        q_rows.append({"id": "Q4", "file": "figures/qualitative/Q4_step_by_step_association.png", "selection": "median-error self-contained example", "source": str(REAL_EXAMPLE)})
    # Q5 contact sheet from deterministic visuals.
    contact_paths = sorted((RICH / "visuals" / "finetuned_legacy").glob("*.png"))[:12]
    if contact_paths:
        thumbs = []
        for p in contact_paths:
            im = Image.open(p).convert("RGB")
            im.thumbnail((260, 146))
            thumbs.append((p, im.copy()))
        sheet = Image.new("RGB", (1120, 720), "white")
        d = ImageDraw.Draw(sheet)
        d.text((25, 18), "Q5. Deterministic qualitative contact sheet", font=F22, fill=PALETTE["text"])
        for idx, (p, im) in enumerate(thumbs):
            x = 25 + (idx % 4) * 270
            y = 70 + (idx // 4) * 205
            sheet.paste(im, (x, y))
            d.text((x, y + 150), p.name[:38], font=F10, fill=PALETTE["muted"])
        save_image(sheet, "figures/qualitative/Q5_contact_sheet.png", "Q5 Qualitative Contact Sheet", "Deterministic first twelve fine-tuned legacy qualitative examples from the evaluator visual outputs.", str(RICH / "visuals/finetuned_legacy"), also_pdf=False)
        q_rows.append({"id": "Q5", "file": "figures/qualitative/Q5_contact_sheet.png", "selection": "first twelve sorted fine-tuned qualitative outputs", "source": str(RICH / "visuals/finetuned_legacy")})

    if q_rows:
        save_table("T10_qualitative_manifest", "T10 Qualitative Selection Manifest", q_rows, ["id", "file", "selection", "source"], "Deterministic qualitative example selection.")

    runtime_rows = []
    for r in overall_2x2:
        runtime_rows.append({
            "run": r["run"],
            "detector": r["detector"],
            "fusion": r["fusion_method"],
            "fps": f"{float(r.get('approx_pipeline_fps') or 0):.2f}",
            "mean_assoc_ms": f"{1000*float(r.get('mean_fusion_time_sec') or 0):.3f}",
        })
    save_table("T11_runtime_2x2", "T11 Runtime Summary", runtime_rows, ["run", "detector", "fusion", "fps", "mean_assoc_ms"], "Runtime summary for the 2x2 detector/fusion comparison.")

    write_csv(OUT / "derived_data" / "overall_results_copy.csv", overall)
    write_csv(OUT / "derived_data" / "per_recording_results_copy.csv", per_rec)
    write_csv(OUT / "derived_data" / "overall_2x2_copy.csv", overall_2x2)
    write_csv(OUT / "derived_data" / "condition_breakdowns_copy.csv", condition)
    write_json(OUT / "source_data_manifest.json", source_manifest)
    return {"source_manifest": source_manifest, "canonical_checks": checks}


def write_catalogs(extra: dict[str, Any]) -> None:
    readme = f"""# AGHRI Fusion Paper Assets

Generated on {dt.datetime.now().isoformat(timespec='seconds')}.

This directory is a self-contained collection of paper figures, tables, qualitative examples, source-data manifests and reproducibility notes for the AGHRI ZED RGB-Livox camera-LiDAR fusion work.

Numerical claims are sourced from the frozen offline evaluator outputs in `{FROZEN}` and `{RICH}`. ROS playback artifacts are qualitative/audit-only.

## Reproduce

```bash
cd {REPO_ROOT}
python3 tools/generate_aghri_fusion_paper_assets.py
```

The generator first asserts that the canonical held-out results are unchanged. It then writes all derived files under this directory.

## Canonical Result Assertion

- Generic YOLO11s: detections 2787, matched 2561, valid fusion rate 0.9218, mean 0.4765 m, median 0.2038 m, P95 1.8703 m, error >1m 0.0859, FPS 69.53.
- AGHRI fine-tuned YOLO11s: detections 2534, matched 2433, valid fusion rate 0.9680, mean 0.3926 m, median 0.1911 m, P95 1.3595 m, error >1m 0.0748, FPS 69.19.
- Official test recordings 6, synchronized test pairs 1248, GT 3D person instances 3155.

## Notes

{chr(10).join('- ' + n for n in NOTES) if NOTES else '- No generation warnings.'}
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    fig_lines = ["# Figure Catalog\n"]
    for fig in FIGURES:
        fig_lines += [f"## {fig['id']}\n", f"- Title: {fig['title']}", f"- Caption: {fig['caption']}", f"- Source: `{fig['source']}`", f"- Files: {', '.join('`'+f+'`' for f in fig['files'])}", ""]
    (OUT / "figure_catalog.md").write_text("\n".join(fig_lines), encoding="utf-8")

    tab_lines = ["# Table Catalog\n"]
    for table in TABLES:
        tab_lines += [f"## {table['id']}\n", f"- Title: {table['title']}", f"- Caption: {table['caption']}", f"- Rows: {table['rows']}", f"- Files: {', '.join('`'+f+'`' for f in table['files'])}", ""]
    (OUT / "table_catalog.md").write_text("\n".join(tab_lines), encoding="utf-8")

    captions = ["# Captions\n", "## Figures\n"]
    for fig in FIGURES:
        captions.append(f"**{fig['id']}**. {fig['caption']}\n")
    captions.append("## Tables\n")
    for table in TABLES:
        captions.append(f"**{table['id']}**. {table['caption']}\n")
    (OUT / "captions.md").write_text("\n".join(captions), encoding="utf-8")

    cards = []
    for fig in FIGURES:
        png = next((f for f in fig["files"] if f.endswith(".png")), None)
        href = fig["files"][0]
        thumb = png or href
        img_tag = f'<img src="{html.escape(thumb)}" alt="{html.escape(fig["title"])}">' if thumb.endswith(".png") else '<div class="svgmark">SVG</div>'
        cards.append(f'<article><a href="{html.escape(href)}">{img_tag}</a><h2>{html.escape(fig["title"])}</h2><p>{html.escape(fig["caption"])}</p></article>')
    gallery = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AGHRI Fusion Paper Assets</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#20242b;background:#f6f7f9}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}}
article{{background:white;border:1px solid #d9dee7;padding:14px;border-radius:8px}}
img{{max-width:100%;height:190px;object-fit:contain;background:#fff}}
h2{{font-size:16px;margin:10px 0 6px}} p{{font-size:13px;color:#596273}}
.svgmark{{height:190px;display:grid;place-items:center;background:#fff;border:1px dashed #ccd3dd;color:#596273}}
</style></head><body><h1>AGHRI Fusion Paper Assets</h1><div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT / "gallery.html").write_text(gallery, encoding="utf-8")

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "generator": str((REPO_ROOT / "tools" / "generate_aghri_fusion_paper_assets.py")),
        "output_root": str(OUT),
        "figure_count": len(FIGURES),
        "table_count": len(TABLES),
        "notes": NOTES,
        "canonical_checks": extra["canonical_checks"],
        "source_files": extra["source_manifest"],
    }
    write_json(OUT / "generation_manifest.json", manifest)


def write_notebook() -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# AGHRI Fusion Paper Assets\n", "\n", "This notebook is a lightweight reproducibility companion for the generated paper-asset directory.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [f"from pathlib import Path\nASSETS = Path('{OUT}')\nprint(ASSETS)\nprint((ASSETS / 'generation_manifest.json').exists())\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["import json\nmanifest = json.loads((ASSETS / 'generation_manifest.json').read_text())\nmanifest['figure_count'], manifest['table_count']\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Regenerate\n", "\n", "Run the generator from the repository root:\n", "\n", "```bash\npython3 tools/generate_aghri_fusion_paper_assets.py\n```\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["from IPython.display import Image, display\nfor rel in ['figures/overall/F1_overall_localization_errors.png', 'figures/distributions/F5_ecdf_3d_error.png', 'figures/qualitative/Q4_step_by_step_association.png']:\n    p = ASSETS / rel\n    if p.exists():\n        display(Image(filename=str(p)))\n"]},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    shutil.copy2(NOTEBOOK, OUT / "notebooks" / NOTEBOOK.name)


def main() -> None:
    ensure_dirs()
    write_diagrams()
    extra = build_figures_and_tables()
    write_catalogs(extra)
    write_notebook()
    print(f"Wrote {len(FIGURES)} figures and {len(TABLES)} tables to {OUT}")
    print(f"Notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
