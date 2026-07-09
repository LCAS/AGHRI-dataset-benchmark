#!/usr/bin/env python3
"""Generate the reduced notebook-focused AGHRI fusion asset set."""

from __future__ import annotations

import base64
import csv
import json
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
) -> None:
    x0, y0, x1, y1 = box
    d.text((x0, y0 - 44), title, fill=TEXT, font=F18)
    for i in range(5):
        val = ymax * i / 4
        y = yscale(val, y0, y1, ymax)
        d.line([x0, y, x1, y], fill=GRID)
        label = f"{val*100:.0f}%" if percent else fmt.format(val)
        d.text((x0 - 48, y - 8), label, fill=MUTED, font=F10)
    d.line([x0, y1, x1, y1], fill=TEXT, width=2)
    d.line([x0, y0, x0, y1], fill=TEXT, width=2)
    group_w = (x1 - x0) / len(categories)
    bar_w = min(32, int(group_w / 4.2))
    for ci, cat in enumerate(categories):
        cx = int(x0 + group_w * ci + group_w / 2)
        start = cx - int(len(series) * bar_w / 2)
        for si, (_, vals, color) in enumerate(series):
            val = vals[ci]
            bx = start + si * bar_w
            by = yscale(val, y0, y1, ymax)
            d.rectangle([bx, by, bx + bar_w - 5, y1], fill=color)
            label = f"{val*100:.1f}%" if percent else fmt.format(val)
            tw, _ = text_size(d, label, F10)
            d.text((bx + (bar_w - 5 - tw) // 2, by - 17), label, fill=TEXT, font=F10)
        for li, line in enumerate(wrap(d, cat, F10, int(group_w) - 8)[:2]):
            tw, _ = text_size(d, line, F10)
            d.text((cx - tw // 2, y1 + 9 + 13 * li), line, fill=MUTED, font=F10)


def legend(d: ImageDraw.ImageDraw, x: int, y: int) -> None:
    items = [("Generic YOLO11s", BLUE), ("AGHRI fine-tuned", ORANGE)]
    for label, color in items:
        d.rectangle([x, y + 3, x + 18, y + 15], fill=color)
        d.text((x + 25, y), label, fill=TEXT, font=F12)
        x += 25 + text_size(d, label, F12)[0] + 28


def generate_combined_overall() -> None:
    rows = read_csv(FROZEN / "overall_results.csv")
    by = {r["model"]: r for r in rows}
    g = by["Generic YOLO11s"]
    f = by["AGHRI-fine-tuned YOLO11s"]
    img = Image.new("RGB", (1800, 640), "white")
    d = ImageDraw.Draw(img)
    d.text((42, 28), "Overall Frozen Test Summary", fill=TEXT, font=F26)
    d.text((42, 63), "F1, F2 and F3 arranged as a single row for compact paper/notebook use", fill=MUTED, font=F13)
    legend(d, 42, 96)
    draw_panel_bars(
        d,
        (95, 170, 555, 500),
        "F1. Localization error",
        ["mean", "median", "P95"],
        [
            ("Generic YOLO11s", [float(g["mean_3d_error_m"]), float(g["median_3d_error_m"]), float(g["p95_3d_error_m"])], BLUE),
            ("AGHRI fine-tuned", [float(f["mean_3d_error_m"]), float(f["median_3d_error_m"]), float(f["p95_3d_error_m"])], ORANGE),
        ],
        2.05,
    )
    draw_panel_bars(
        d,
        (710, 170, 1170, 500),
        "F2. Reliability",
        ["valid rate", "error >1m"],
        [
            ("Generic YOLO11s", [float(g["valid_fusion_rate"]), float(g["error_rate_above_1m"])], BLUE),
            ("AGHRI fine-tuned", [float(f["valid_fusion_rate"]), float(f["error_rate_above_1m"])], ORANGE),
        ],
        1.0,
        percent=True,
    )
    draw_panel_bars(
        d,
        (1325, 170, 1700, 500),
        "F3. Runtime",
        ["offline FPS"],
        [
            ("Generic YOLO11s", [float(g["fps"])], BLUE),
            ("AGHRI fine-tuned", [float(f["fps"])], ORANGE),
        ],
        80.0,
    )
    d.text((220, 560), "3D error (m)", fill=MUTED, font=F12)
    d.text((875, 560), "rate", fill=MUTED, font=F12)
    d.text((1450, 560), "frames/s", fill=MUTED, font=F12)
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
    img = Image.new("RGB", (1150, 720), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F9. Representative Scenario Mean 3D Error", fill=TEXT, font=F26)
    d.text((48, 69), "One selected representative bag per scenario; labels show scenario names only", fill=MUTED, font=F13)
    legend(d, 48, 103)
    draw_panel_bars(
        d,
        (115, 180, 1060, 555),
        "",
        [s for s, _ in selected],
        [
            ("Generic YOLO11s", [values[("Generic YOLO11s", rec)] or 0 for _, rec in selected], BLUE),
            ("AGHRI fine-tuned", [values[("AGHRI-fine-tuned YOLO11s", rec)] or 0 for _, rec in selected], ORANGE),
        ],
        0.95,
    )
    d.text((450, 630), "mean 3D localization error (m)", fill=MUTED, font=F13)
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
    img = Image.new("RGB", (1150, 720), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), "F14. Mean 3D Error by Ground-truth Distance Band", fill=TEXT, font=F26)
    d.text((48, 69), "Depth robustness summary from frozen evaluator condition breakdowns", fill=MUTED, font=F13)
    legend(d, 48, 103)
    labels = ["near <2m", "medium 2-5m", "far >=5m"]
    draw_panel_bars(
        d,
        (115, 180, 1060, 555),
        "",
        labels,
        [
            ("Generic YOLO11s", [vals.get(("generic", b), 0) for b in order], BLUE),
            ("AGHRI fine-tuned", [vals.get(("finetuned", b), 0) for b in order], ORANGE),
        ],
        1.2,
    )
    d.text((450, 630), "mean 3D localization error (m)", fill=MUTED, font=F13)
    path = FIG / "F14_depth_bins_selected.png"
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
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="1040" viewBox="0 0 1180 1040">']
    svg.append('<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#384252"/></marker></defs><rect width="1180" height="1040" fill="white"/>')
    svg.append('<text x="50" y="50" font-size="28" font-weight="700" fill="#20242b">D1. Revised camera-guided LiDAR fusion flow</text>')
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
    d.text((50, 28), "D1. Revised camera-guided LiDAR fusion flow", font=F26, fill=TEXT)

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
    img.save(FIG / "D1_fusion_method_overview_revised.png")


def copy_selected_existing_assets() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    copies = [
        (FULL / "figures/qualitative/Q4_step_by_step_association.png", FIG / "Q4_step_by_step_association_selected.png"),
        (FULL / "figures/training/F19_training_curves.png", FIG / "F19_training_curves_selected.png"),
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


def write_readme() -> None:
    text = """# Selected Notebook Assets

This folder contains the reduced set requested for the AGHRI fusion notebook.
It does not replace the full paper-asset collection.

Included figures:
- Revised D1 fusion method overview.
- Combined F1/F2/F3 overall summary row.
- F9 representative-scenario mean error using one selected bag per scenario.
- F14 depth-bin robustness summary.
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
        "F19_training_curves_selected.png": FIG / "F19_training_curves_selected.png",
        "Q4_step_by_step_association_selected.png": FIG / "Q4_step_by_step_association_selected.png",
        "ROS_out_vine_4people_fusion_05.png": FIG / "ROS_out_vine_4people_fusion_05.png",
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
            md_cell(f"""# AGHRI ZED RGB-Livox Fusion Results Summary

This notebook summarises the selected AGHRI camera-LiDAR fusion figures and tables using the frozen offline evaluator outputs. The focus is intentionally narrow: detector checkpoint comparison, fusion reliability, depth robustness, training context, and one reproducible qualitative LiDAR-association example.

The figures shown here are the selected saved figures from `results/aghri_fusion_paper_assets/selected_notebook_assets`. They are linked relative to this notebook so they display when Jupyter is opened from the `notebooks/` folder.
"""),
            md_cell("""## Methods Compared

The fusion framework is the same in both rows below. The comparison changes the 2D person detector checkpoint and keeps the camera-guided LiDAR association/evaluation path fixed.

| Method name in figures | Detector checkpoint | Fusion method | Explanation |
|---|---|---|---|
| `Generic YOLO11s` | Public YOLO11s checkpoint | Legacy camera-guided LiDAR box association | Baseline detector used without AGHRI-specific fine-tuning. |
| `AGHRI fine-tuned` | YOLO11s fine-tuned on AGHRI ZED RGB data | Same LiDAR association and evaluator | Detector adapted to the AGHRI viewpoint, outdoor/polytunnel scenes, person scale, and clothing/occlusion patterns. |

The important control is that the LiDAR association logic is not changed between these two detector rows. This keeps the comparison focused on whether better 2D detections improve downstream fused 3D person localization.
"""),
            md_cell(f"""## Metrics Used

The numerical metrics come from the offline fusion evaluator, not from RViz playback screenshots.

{table_md("T5_metric_definitions")}
"""),
            md_cell("""## Test Data Used

The frozen held-out benchmark uses all six official AGHRI test recordings.

| Dataset part | Test recording | Scenario | Robot motion |
|---|---|---|---|
| `dataset_part1` | `footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label` | Footpath | Moving |
| `dataset_part1` | `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` | Footpath | Stationary |
| `dataset_part3` | `in_straw_3pick_diff_st_10_24_2024_5_a_label` | Polytunnel | Stationary |
| `dataset_part3` | `out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label` | Polytunnel/outdoor straw | Stationary |
| `dataset_part4` | `out_vine_1push_3carry_st_ly_11_06_2024_1_label` | Vineyard | Stationary |
| `dataset_part4` | `out_vine_4swap+walk_st_ly_11_06_2024_2_label` | Vineyard | Stationary |

The overall result table and the F1/F2/F3 summary use all six recordings. The representative-scenario figure uses one selected bag per scenario only, so that figure is easier to read and should not be confused with the full six-recording aggregate.
"""),
            md_cell(f"""## Main Results Table

{table_md("T2_overall_results")}

**Key observation.** The AGHRI fine-tuned detector improves the fusion output even though the LiDAR association method is kept fixed. Compared with generic YOLO11s, mean 3D error drops from 0.4765 m to 0.3926 m, P95 error drops from 1.8703 m to 1.3595 m, and valid fusion rate increases from 0.9218 to 0.9680. Runtime remains essentially unchanged.
"""),
            code_cell(f"""# Optional reproducibility cell: show where the selected assets live.
from pathlib import Path

ASSETS = Path("{OUT}")
print(ASSETS)
print("selected asset files:", len(list(ASSETS.rglob("*"))))
"""),
            md_cell(f"""## Figure 1: Fusion Method Overview

{image_md("D1_fusion_method_overview_revised.png", "Revised fusion method overview")}

The runtime first synchronizes the ZED RGB image and Livox point cloud with an approximate-time synchronizer. YOLO11s produces person boxes on the synchronized image. In parallel, the point cloud is transformed into the ZED camera frame and projected into image coordinates using `CameraInfo.P`. The projected points inside the shrunken person association boxes become candidates. Modified Z-score filtering with `tau = 3.5` removes outliers, and the retained 3D points are averaged to estimate the camera-frame person centre `[Xc, Yc, Zc]`. The displayed distance label is the camera-frame depth component `Zc`.
"""),
            md_cell(f"""## Figure 2: Overall Frozen-Test Summary

{image_md("F1_F2_F3_overall_summary_row.png", "Overall frozen-test summary")}

This combined figure places the three core summaries in one row. `F1_overall_localization_errors` shows that the fine-tuned detector reduces mean, median, and P95 3D localization error. `F2_reliability` shows a higher valid fusion rate and a lower fraction of errors above 1 m. `F3_runtime` shows that both checkpoints run at similar offline throughput, so the accuracy improvement is not bought by a meaningful runtime penalty.
"""),
            md_cell(f"""## Figure 3: Representative Scenario Mean Error

{image_md("F9_representative_scenario_mean_error.png", "Representative scenario mean error")}

This figure is a compact scenario-level view. It uses one selected representative bag per scenario rather than all six test bags:

| Scenario | Selected bag used in this figure |
|---|---|
| Footpath | `dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label` |
| Polytunnel | `dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label` |
| Vineyard | `dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label` |

The x-axis intentionally uses only the scenario names. The trend is consistent with the full aggregate: the AGHRI fine-tuned detector gives lower mean 3D error in the selected representative bags.
"""),
            md_cell(f"""## Figure 4: Depth Robustness

{image_md("F14_depth_bins_selected.png", "Depth-bin robustness summary")}

`F14_depth_bins` reports mean 3D error by ground-truth distance band. This is useful because camera-LiDAR fusion can behave differently at near, medium, and far ranges: nearby people may occupy large image regions, while far people have fewer retained LiDAR points and smaller projected boxes. The fine-tuned detector reduces the mean error especially in the farther range, which is where detector quality and point selection become more fragile.
"""),
            md_cell(f"""## Figure 5: YOLO11s Fine-tuning Curves

{image_md("F19_training_curves_selected.png", "YOLO11s fine-tuning curves")}

`F19_training_curves` shows the cluster fine-tuning run used to adapt YOLO11s to AGHRI ZED RGB images. The curves track validation precision, recall, mAP50, and mAP50-95 over training epochs. Epoch 46 is marked as the selected best checkpoint. That checkpoint is then used in the frozen fusion comparison, so the downstream results measure how an AGHRI-adapted 2D detector changes the final fused 3D person localization.
"""),
            md_cell(f"""## Figure 6: Step-by-step LiDAR Association Example

{image_md("Q4_step_by_step_association_selected.png", "Step-by-step LiDAR association walkthrough")}

`Q4_step_by_step_association` is a self-contained qualitative walkthrough. The raw ZED image first shows the target person. The projected-point panel then shows all LiDAR points that land inside the image. The candidate panel isolates the points inside the shrunken person association box. The selected-vs-rejected panel shows the modified Z-score filter in action: 22 candidate points enter the filter, 15 are retained, and 7 are rejected. The final panel averages the retained 3D points and recomputes a LiDAR-centre error of about 0.1337 m. This example is included because every step can be reproduced from `/home/prabuddhi/Desktop/aghri_fusion_real_example`.
"""),
            md_cell(f"""## ROS Topics and Frames

{table_md("T7_ros_topics_frames")}

The all-topic ROS 2 bags contain the ZED RGB image and CameraInfo, three fisheye CameraInfo streams, Livox point clouds, `/tf`, and `/tf_static`. The fusion node only needs the ZED RGB stream, ZED CameraInfo, Livox cloud, and TF chain for the published ZED RGB-Livox result; the extra topics remain in the bag for completeness and reproducibility.
"""),
            md_cell(f"""## Figure 7: ROS Playback Qualitative Example

{image_md("ROS_out_vine_4people_fusion_05.png", "ROS playback vineyard qualitative fusion example")}

This image was captured directly from the ROS 2 playback output topic `/camera_lidar_fusion/result` while playing `dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label`. It shows the fine-tuned YOLO11s detections, projected Livox points over the ZED RGB image, and the displayed person depth labels. The numeric label is the camera-frame depth `Zc` of the fused person centre, so it should be read as forward depth from the ZED camera rather than full Euclidean range.
"""),
            md_cell("""## Summary

Across the frozen AGHRI fusion benchmark, AGHRI fine-tuning improves the detector input to the same LiDAR association pipeline. The main quantitative effect is lower 3D localization error and higher valid fusion rate, with similar runtime. The representative scenario view gives a simpler environment-level reading, the depth-bin figure checks range sensitivity, and the step-by-step qualitative example explains how projected LiDAR points become the final camera-frame centre shown on the annotated image.
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
    generate_revised_d1()
    generate_combined_overall()
    generate_scenario_f9()
    generate_f14_depth_bins()
    copy_selected_existing_assets()
    write_readme()
    write_notebook()
    print(f"Wrote selected notebook assets to {OUT}")
    print(f"Updated notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
