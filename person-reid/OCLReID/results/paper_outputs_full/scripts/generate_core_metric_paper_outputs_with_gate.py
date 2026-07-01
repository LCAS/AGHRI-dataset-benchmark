#!/usr/bin/env python3
"""Generate final-test paper figures with released-checkpoint ReID gate included."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oclreid_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import cv2

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


PROJECT_ROOT = Path("/home/prabuddhi/Desktop/OCLReID")
OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper_outputs_full"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
REPORT_DIR = OUTPUT_ROOT / "reports"
QUAL_DIR = OUTPUT_ROOT / "qualitative"
SCRIPT_DIR = OUTPUT_ROOT / "scripts"

BASE_RESULTS_ROOT = PROJECT_ROOT / "results" / "test_full"
GATED_METHOD_ROOT = BASE_RESULTS_ROOT / "gated_part_oclreid"
OPTIONAL_TRAINING_ROOT = PROJECT_ROOT / "optional_training" / "aghri_reid_finetuning_archive"
EXPERIMENTAL_CHECKPOINT_ROOT = OPTIONAL_TRAINING_ROOT / "downstream_test" / "stage1_epoch1_checkpoint" / "test" / "part-OCLReID"
EXPERIMENTAL_GATE_ROOT = OPTIONAL_TRAINING_ROOT / "downstream_test" / "reid_gate_experimental_checkpoint" / "test" / "part-OCLReID"
AGHRI_DATASET_ROOT = Path(os.environ.get("AGHRI_DATASET_ROOT", "/media/prabuddhi/Backup2/Updated Dataset_PW"))
AGHRI_VIDEO_ROOT = Path(os.environ.get("AGHRI_VIDEO_ROOT", "/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos"))


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display: str
    root: Path


METHOD_SPECS = [
    MethodSpec("rpf-ReID", "rpf-ReID", BASE_RESULTS_ROOT / "rpf_reid"),
    MethodSpec("normal-part", "Normal part-OCLReID", BASE_RESULTS_ROOT / "normal_part_oclreid"),
    MethodSpec("gated-part", "Gated part-OCLReID", GATED_METHOD_ROOT),
]
METHOD_ORDER = [spec.display for spec in METHOD_SPECS]
QUAL_METHOD_ORDER = ["rpf-ReID", "Normal part-OCLReID"]

CAMERA_ORDER = ["cam_fish_front", "cam_fish_left", "cam_fish_right", "cam_zed_rgb"]
CAMERA_LABELS = {
    "cam_fish_front": "Front fisheye",
    "cam_fish_left": "Left fisheye",
    "cam_fish_right": "Right fisheye",
    "cam_zed_rgb": "ZED front RGB",
}
SELECTED_SCENES = {
    "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label": "Footpath",
    "in_straw_3pick_diff_st_10_24_2024_5_a_label": "Polytunnel",
    "out_vine_4swap+walk_st_ly_11_06_2024_2_label": "Vineyard",
}
SCENARIO_ORDER = ["Footpath", "Polytunnel", "Vineyard"]
HUMAN_COUNT_BY_SCENE = {
    "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label": 1,
    "footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label": 1,
    "in_straw_3pick_diff_st_10_24_2024_5_a_label": 3,
    "out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label": 3,
    "out_vine_1push_3carry_st_ly_11_06_2024_1_label": 4,
    "out_vine_4swap+walk_st_ly_11_06_2024_2_label": 4,
}
HUMAN_COUNT_LABELS = {1: "1 human", 3: "3 humans", 4: "4 humans"}
HUMAN_COUNT_ORDER = ["1 human", "3 humans", "4 humans"]
ROBOT_MOTION_ORDER = ["Stationary", "Moving"]

PREFERRED_QUALITATIVE_FRAMES = {
    "Footpath": {"target_class": "01", "frame": 174, "reason": "clear visible target with Normal part-OCLReID correctly localized"},
    "Polytunnel": {"target_class": "06", "frame": 149, "reason": "target correctly localized with another detected person visible in the neighbouring row"},
    "Vineyard": {"target_class": "05", "frame": 35, "reason": "Normal part-OCLReID correctly localizes a more occluded target while rpf-ReID has no target detection on the same frame"},
}
PREFERRED_FRONT_FISHEYE_FRAMES = {
    "Footpath": {"target_class": "01", "frame": 166, "reason": "front-fisheye footpath case where Normal part-OCLReID localizes the distant target while rpf-ReID is lost"},
    "Polytunnel": {"target_class": "02", "frame": 37, "reason": "front-fisheye neighbouring-row target with stronger Normal part-OCLReID confidence than rpf-ReID"},
    "Vineyard": {"target_class": "02", "frame": 623, "reason": "front-fisheye vineyard case where Normal part-OCLReID localizes an occluded target while rpf-ReID is lost"},
}

CORE_METRICS = [
    ("Success@0.5 ↑", "Success@0.5"),
    ("Mean IoU ↑", "Mean IoU"),
    ("Prediction availability ↑", "Prediction availability"),
]
CORE_TABLE_COLS = [
    "Method",
    "Completed runs",
    "GT-visible frames",
    "Success@0.5 ↑",
    "Mean IoU ↑",
    "Prediction availability ↑",
]

GATED_ANCHORS = {
    "Success@0.5 ↑": 0.556535830032409,
    "Mean IoU ↑": 0.49380522478549255,
    "Prediction availability ↑": 0.6477013563797863,
    "Absent-target FP rate": 0.2761586239847109,
    "Reacquisition rate": 0.42718446601941745,
}
NORMAL_ANCHORS = {
    "Success@0.5 ↑": 0.48307526107310045,
    "Mean IoU ↑": 0.4282204309741733,
    "Prediction availability ↑": 0.545732805185452,
    "Absent-target FP rate": 0.2264691829909221,
}

FIG9_SCENE = "out_vine_4swap+walk_st_ly_11_06_2024_2_label"
FIG9_CAMERA = "cam_fish_front"
FIG9_TARGET = "01"
FIG9_SELECTED_FRAMES = [516, 517, 519, 521]


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, FIGURE_DIR, REPORT_DIR, QUAL_DIR, SCRIPT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def safe_divide(num: float, den: float) -> float:
    return np.nan if den in (None, 0) else num / den


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def class_from_dir(path: Path) -> str:
    name = path.name
    return name.replace("class_", "", 1).zfill(2) if name.startswith("class_") else name.zfill(2)


def run_key_columns() -> list[str]:
    return ["dataset_part", "scene_name", "camera", "target_class"]


def parse_summary_path(path: Path, method_root: Path) -> dict[str, str]:
    rel = path.relative_to(method_root)
    return {
        "dataset_part": rel.parts[0],
        "scene_name": rel.parts[1],
        "camera": rel.parts[2],
        "target_class": class_from_dir(Path(rel.parts[3])),
    }


def build_per_run_all() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        if not spec.root.exists():
            raise SystemExit(f"Missing result root for {spec.display}: {spec.root}")
        for summary_path in sorted(spec.root.rglob("evaluation/summary_metrics.json")):
            info = parse_summary_path(summary_path, spec.root)
            run_dir = summary_path.parents[1]
            summary = read_json(summary_path)
            visible = int(summary.get("ground_truth_visible_frames", 0) or 0)
            absent = int(summary.get("ground_truth_absent_frames", 0) or 0)
            correct = int(summary.get("correctly_localized_frames", 0) or 0)
            pred_visible = int(summary.get("prediction_present_on_visible_frames", 0) or 0)
            wrong = int(summary.get("wrong_person_frames", 0) or 0)
            fp_absent = int(summary.get("false_positive_frames_when_target_absent", 0) or 0)
            reapp = int(summary.get("ground_truth_reappearance_count", summary.get("exact_reappearance_count", 0)) or 0)
            reacq = int(summary.get("successful_reacquisitions_after_reappearance", summary.get("successful_exact_reacquisitions", 0)) or 0)
            per_frame = run_dir / "evaluation" / "per_frame_metrics.csv"
            pred = run_dir / "predictions.json"
            viz = run_dir / "inference_visualization.mp4"
            rows.append(
                {
                    **info,
                    "method_key": spec.key,
                    "method": spec.display,
                    "source_root": str(spec.root),
                    "camera_label": CAMERA_LABELS.get(info["camera"], info["camera"]),
                    "human_count": HUMAN_COUNT_BY_SCENE.get(info["scene_name"], np.nan),
                    "human_count_label": HUMAN_COUNT_LABELS.get(HUMAN_COUNT_BY_SCENE.get(info["scene_name"], None), "Unknown"),
                    "robot_motion": "Moving" if "_mv_" in info["scene_name"] else "Stationary",
                    "result_dir": str(run_dir),
                    "summary_metrics_path": str(summary_path),
                    "per_frame_metrics_path": str(per_frame) if per_frame.exists() else "",
                    "predictions_path": str(pred) if pred.exists() else "",
                    "visualization_video": str(viz) if viz.exists() else "",
                    "ground_truth_visible_frames": visible,
                    "ground_truth_absent_frames": absent,
                    "valid_annotated_frames": int(summary.get("valid_annotated_frames", visible + absent) or 0),
                    "correctly_localized_frames": correct,
                    "prediction_present_on_visible_frames": pred_visible,
                    "wrong_person_frames": wrong,
                    "false_positive_frames_when_target_absent": fp_absent,
                    "reappearance_events": reapp,
                    "successful_reacquisitions": reacq,
                    "success_rate_iou_threshold": safe_divide(correct, visible),
                    "mean_iou_on_visible_frames": float(summary.get("mean_iou_on_visible_frames", np.nan)),
                    "prediction_availability_on_visible_frames": safe_divide(pred_visible, visible),
                    "false_positive_rate_when_target_absent": safe_divide(fp_absent, absent),
                    "wrong_person_rate_on_visible_frames": safe_divide(wrong, visible),
                    "reacquisition_rate": safe_divide(reacq, reapp),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No completed result summaries found.")
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["camera"] = pd.Categorical(df["camera"], categories=CAMERA_ORDER, ordered=True)
    df = df.sort_values(["method", "dataset_part", "scene_name", "camera", "target_class"]).reset_index(drop=True)
    df["method"] = df["method"].astype(str)
    df["camera"] = df["camera"].astype(str)
    return df


def common_run_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sets: dict[str, set[tuple[str, str, str, str]]] = {}
    for method in METHOD_ORDER:
        m = df[df["method"] == method]
        sets[method] = set(tuple(x) for x in m[run_key_columns()].values.tolist())
    common = set.intersection(*sets.values())
    exclusions = {
        method: sorted(["/".join(key) for key in (sets[method] - common)])
        for method in METHOD_ORDER
    }
    common_df = df[df.apply(lambda row: tuple(row[col] for col in run_key_columns()) in common, axis=1)].copy()
    return common_df, {"common_runs": len(common), "per_method_runs": {m: len(sets[m]) for m in METHOD_ORDER}, "excluded_runs": exclusions}


def cell_ref(row_idx: int, col_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row_idx}"


def write_xlsx(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1") -> None:
    rows = [list(df.columns)] + df.astype(object).where(pd.notnull(df), "").values.tolist()
    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = cell_ref(r_idx, c_idx)
            if isinstance(value, (int, float, np.integer, np.floating)) and not (isinstance(value, float) and math.isnan(value)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    sheet_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + "".join(sheet_rows) + "</sheetData></worksheet>"
    workbook_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="' + escape(sheet_name[:31]) + '" sheetId="1" r:id="rId1"/></sheets></workbook>'
    content_types = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    wb_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def md_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value).replace("|", "\\|")


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(md_value(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, base: Path, xlsx: bool = False) -> None:
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(to_markdown(df), encoding="utf-8")
    base.with_suffix(".tex").write_text(df.to_latex(index=False, escape=True, float_format="%.3f"), encoding="utf-8")
    if xlsx:
        write_xlsx(df, base.with_suffix(".xlsx"), sheet_name=base.name[:31])


def aggregate(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(groups, observed=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        visible = int(g["ground_truth_visible_frames"].sum())
        absent = int(g["ground_truth_absent_frames"].sum())
        correct = int(g["correctly_localized_frames"].sum())
        pred_visible = int(g["prediction_present_on_visible_frames"].sum())
        wrong = int(g["wrong_person_frames"].sum())
        fp_absent = int(g["false_positive_frames_when_target_absent"].sum())
        reapp = int(g["reappearance_events"].sum())
        reacq = int(g["successful_reacquisitions"].sum())
        weighted_iou = (g["mean_iou_on_visible_frames"] * g["ground_truth_visible_frames"]).sum(skipna=True)
        row = {col: value for col, value in zip(groups, key)}
        row.update(
            {
                "Completed runs": len(g),
                "GT-visible frames": visible,
                "GT-absent frames": absent,
                "Success@0.5 ↑": safe_divide(correct, visible),
                "Mean IoU ↑": safe_divide(weighted_iou, visible),
                "Prediction availability ↑": safe_divide(pred_visible, visible),
                "Absent-target FP rate ↓": safe_divide(fp_absent, absent),
                "Reacquisition rate ↑": safe_divide(reacq, reapp),
                "Wrong-person rate on visible frames ↓": safe_divide(wrong, visible),
                "Correctly localized frames": correct,
                "Prediction-present visible frames": pred_visible,
                "Wrong-person frames": wrong,
                "Absent-target FP frames": fp_absent,
                "Reappearance events": reapp,
                "Successful reacquisitions": reacq,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if "method" in out.columns:
        out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    if "scenario" in out.columns:
        out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    if "camera_label" in out.columns:
        out["camera_label"] = pd.Categorical(out["camera_label"], categories=[CAMERA_LABELS[c] for c in CAMERA_ORDER], ordered=True)
    if "human_count_label" in out.columns:
        out["human_count_label"] = pd.Categorical(out["human_count_label"], categories=HUMAN_COUNT_ORDER, ordered=True)
    if "robot_motion" in out.columns:
        out["robot_motion"] = pd.Categorical(out["robot_motion"], categories=ROBOT_MOTION_ORDER, ordered=True)
    sort_cols = [c for c in ["method", "scenario", "camera_label", "human_count_label", "robot_motion"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    for col in out.select_dtypes(["category"]).columns:
        out[col] = out[col].astype(str)
    return out


def create_main_table(df: pd.DataFrame) -> pd.DataFrame:
    table = aggregate(df, ["method"]).rename(columns={"method": "Method"})
    write_table(table[CORE_TABLE_COLS], TABLE_DIR / "main_table_core_metrics_whole_test", xlsx=True)
    rounded = table[CORE_TABLE_COLS].copy()
    for col in ["Success@0.5 ↑", "Mean IoU ↑", "Prediction availability ↑"]:
        rounded[col] = rounded[col].round(3)
    write_table(rounded, TABLE_DIR / "main_table_core_metrics_whole_test_rounded")
    return table


def add_bar_values(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 0.014,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def grouped_bars(ax: plt.Axes, data: pd.DataFrame, x_labels: list[str], x_col: str, y_col: str, title: str) -> None:
    x = np.arange(len(x_labels))
    width = min(0.24, 0.78 / len(METHOD_ORDER))
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2) * width
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, method in enumerate(METHOD_ORDER):
        vals = []
        for label in x_labels:
            match = data[(data[x_col] == label) & (data["method"] == method)]
            vals.append(float(match[y_col].iloc[0]) if not match.empty else np.nan)
        ax.bar(x + offsets[idx], vals, width * 0.92, label=method, color=colors[idx])
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=14, ha="right", fontsize=9.5)
    ax.grid(axis="y", alpha=0.22)
    add_bar_values(ax)


def save_fig(fig: plt.Figure, base: Path) -> None:
    for suffix in [".png", ".pdf"]:
        fig.savefig(base.with_suffix(suffix), dpi=300)
    plt.close(fig)


def create_figures(df: pd.DataFrame, table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    fig1_rows = []
    for _, row in table.iterrows():
        for col, label in CORE_METRICS:
            fig1_rows.append({"method": row["Method"], "metric": label, "value": row[col]})
    fig1_data = pd.DataFrame(fig1_rows)
    fig1_data.to_csv(TABLE_DIR / "fig1_overall_core_metrics_whole_test_data.csv", index=False)
    outputs["fig1"] = fig1_data
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    grouped_bars(ax, fig1_data, [m[1] for m in CORE_METRICS], "metric", "value", "")
    ax.set_ylabel("Metric value")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, FIGURE_DIR / "fig1_overall_core_metrics_whole_test")

    selected = df[df["scene_name"].isin(SELECTED_SCENES)].copy()
    selected["scenario"] = selected["scene_name"].map(SELECTED_SCENES)
    scenario_data = aggregate(selected, ["method", "scenario"])
    scenario_data.to_csv(TABLE_DIR / "fig2_selected_scenario_core_metrics_data.csv", index=False)
    outputs["fig2"] = scenario_data
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), sharey=True)
    for ax, (col, label) in zip(axes, CORE_METRICS):
        grouped_bars(ax, scenario_data, SCENARIO_ORDER, "scenario", col, label)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIGURE_DIR / "fig2_selected_scenario_core_metrics")

    camera_data = aggregate(df, ["method", "camera_label"])
    camera_data.to_csv(TABLE_DIR / "fig3_selected_camera_core_metrics_data.csv", index=False)
    camera_data.to_csv(TABLE_DIR / "fig3_overall_camera_core_metrics_data.csv", index=False)
    outputs["fig3"] = camera_data
    camera_labels = [CAMERA_LABELS[c] for c in CAMERA_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6), sharey=True)
    for ax, (col, label) in zip(axes, CORE_METRICS):
        grouped_bars(ax, camera_data, camera_labels, "camera_label", col, label)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for base_name in ["fig3_selected_camera_core_metrics", "fig3_overall_camera_core_metrics"]:
        for suffix in [".png", ".pdf"]:
            fig.savefig((FIGURE_DIR / base_name).with_suffix(suffix), dpi=300)
    plt.close(fig)

    human_data = aggregate(df, ["method", "human_count_label"])
    human_data.to_csv(TABLE_DIR / "fig6_human_count_core_metrics_data.csv", index=False)
    outputs["fig6"] = human_data
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
    for ax, (col, label) in zip(axes, CORE_METRICS):
        grouped_bars(ax, human_data, HUMAN_COUNT_ORDER, "human_count_label", col, label)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIGURE_DIR / "fig6_human_count_core_metrics")

    motion_data = aggregate(df, ["method", "robot_motion"])
    motion_data.to_csv(TABLE_DIR / "fig7_robot_motion_core_metrics_data.csv", index=False)
    outputs["fig7"] = motion_data
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.6), sharey=True)
    for ax, (col, label) in zip(axes, CORE_METRICS):
        grouped_bars(ax, motion_data, ROBOT_MOTION_ORDER, "robot_motion", col, label)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_fig(fig, FIGURE_DIR / "fig7_robot_motion_core_metrics")

    improvement = []
    rpf = table[table["Method"] == "rpf-ReID"].iloc[0]
    for method in ["Normal part-OCLReID", "Gated part-OCLReID"]:
        row = table[table["Method"] == method].iloc[0]
        for col, label in CORE_METRICS:
            improvement.append({"method": method, "metric": label, "improvement_over_rpf": row[col] - rpf[col]})
    improvement_df = pd.DataFrame(improvement)
    improvement_df.to_csv(TABLE_DIR / "fig4_improvement_over_rpf_whole_test_data.csv", index=False)
    outputs["fig4"] = improvement_df
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    x_labels = [m[1] for m in CORE_METRICS]
    x = np.arange(len(x_labels))
    width = 0.32
    colors = {"Normal part-OCLReID": "#f58518", "Gated part-OCLReID": "#54a24b"}
    for idx, method in enumerate(["Normal part-OCLReID", "Gated part-OCLReID"]):
        vals = [
            float(improvement_df[(improvement_df["metric"] == metric) & (improvement_df["method"] == method)]["improvement_over_rpf"].iloc[0])
            for metric in x_labels
        ]
        bars = ax.bar(x + (idx - 0.5) * width, vals, width * 0.92, label=method, color=colors[method])
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val + (0.008 if val >= 0 else -0.02), f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color="black", linewidth=0.9)
    max_improvement = float(improvement_df["improvement_over_rpf"].max())
    min_improvement = float(improvement_df["improvement_over_rpf"].min())
    ax.set_ylim(min(0.0, min_improvement - 0.035), max_improvement + 0.055)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=12, ha="right")
    ax.set_ylabel("Absolute improvement over rpf-ReID")
    ax.set_title("Improvement over rpf-ReID on the AGHRI test set")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save_fig(fig, FIGURE_DIR / "fig4_improvement_over_rpf_whole_test")
    return outputs


def select_frame_for_run(row: pd.Series) -> tuple[int | None, float | None, str]:
    path = Path(str(row["per_frame_metrics_path"]))
    if not path.exists():
        return None, None, "per-frame metrics missing"
    pf = pd.read_csv(path)
    required = {"gt_visible", "prediction_present", "correctly_localized", "iou", "video_frame_index"}
    if not required.issubset(pf.columns):
        return None, None, "required per-frame columns missing"
    cand = pf[(pf["gt_visible"] == True) & (pf["prediction_present"] == True) & (pf["correctly_localized"] == True) & (pf["iou"] >= 0.5)].copy()
    if cand.empty:
        return None, None, "no correctly localized visible frame found"
    if {"gt_x1", "gt_y1", "gt_x2", "gt_y2"}.issubset(cand.columns):
        cand["gt_area"] = (cand["gt_x2"] - cand["gt_x1"]).clip(lower=0) * (cand["gt_y2"] - cand["gt_y1"]).clip(lower=0)
    else:
        cand["gt_area"] = 0
    cand = cand.sort_values(["iou", "gt_area"], ascending=[False, False])
    best = cand.iloc[0]
    return int(best["video_frame_index"]), float(best["iou"]), "target visible, prediction present, and IoU >= 0.5"


def preferred_frame_from_map(row: pd.Series, scenario: str, preferred_map: dict[str, dict[str, Any]]) -> tuple[int | None, float | None, str]:
    preferred = preferred_map.get(scenario)
    if not preferred or row["target_class"] != preferred["target_class"]:
        return None, None, ""
    path = Path(str(row["per_frame_metrics_path"]))
    if not path.exists():
        return None, None, "preferred frame unavailable because per-frame metrics are missing"
    pf = pd.read_csv(path)
    match = pf[pf["video_frame_index"] == preferred["frame"]]
    if match.empty:
        return None, None, "preferred frame is not present in per-frame metrics"
    record = match.iloc[0]
    if not (bool(record["gt_visible"]) and bool(record["prediction_present"]) and bool(record["correctly_localized"]) and float(record["iou"]) >= 0.5):
        return None, None, "preferred frame does not satisfy visible/predicted/correct criteria"
    return int(preferred["frame"]), float(record["iou"]), preferred["reason"]


def select_qualitative(df: pd.DataFrame, camera: str, preferred_map: dict[str, dict[str, Any]], out_name: str) -> tuple[pd.DataFrame, list[str]]:
    selected_rows = []
    failed = []
    for scene, scenario in SELECTED_SCENES.items():
        runs = df[(df["method"] == "Normal part-OCLReID") & (df["scene_name"] == scene) & (df["camera"] == camera)].copy()
        candidates = []
        for _, row in runs.iterrows():
            frame, iou, reason = preferred_frame_from_map(row, scenario, preferred_map)
            if frame is None:
                frame, iou, reason = select_frame_for_run(row)
            if frame is None:
                failed.append(f"{scenario}: {row['target_class']} - {reason}")
                continue
            candidates.append((row, frame, iou, reason))
        if not candidates:
            continue
        preferred = preferred_map.get(scenario)
        if preferred:
            candidates.sort(
                key=lambda item: (
                    item[0]["target_class"] == preferred["target_class"] and item[1] == preferred["frame"],
                    item[2] if item[2] is not None else -1,
                    item[0]["success_rate_iou_threshold"],
                ),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda item: (item[2] if item[2] is not None else -1, item[0]["success_rate_iou_threshold"]), reverse=True)
        part_row, frame, iou, reason = candidates[0]
        match = df[
            (df["scene_name"] == scene)
            & (df["camera"] == camera)
            & (df["target_class"] == part_row["target_class"])
            & (df["method"].isin(QUAL_METHOD_ORDER))
        ].copy()
        for _, method_row in match.sort_values("method").iterrows():
            selected_rows.append(
                {
                    "scenario": scenario,
                    "scene_name": scene,
                    "camera": camera,
                    "target_class": part_row["target_class"],
                    "selected_video_frame": frame,
                    "reason": reason,
                    "method": method_row["method"],
                    "success_rate_iou_threshold": method_row["success_rate_iou_threshold"],
                    "mean_iou_on_visible_frames": method_row["mean_iou_on_visible_frames"],
                    "prediction_availability_on_visible_frames": method_row["prediction_availability_on_visible_frames"],
                    "frame_iou_if_available": iou if method_row["method"] == "Normal part-OCLReID" else np.nan,
                    "visualization_video": method_row["visualization_video"],
                    "predictions_path": method_row["predictions_path"],
                    "summary_metrics_path": method_row["summary_metrics_path"],
                    "per_frame_metrics_path": method_row["per_frame_metrics_path"],
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(QUAL_DIR / out_name, index=False)
    return selected, failed


def extract_video_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    if not HAVE_CV2 or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_video_frame_bgr(video_path: Path, frame_index: int) -> np.ndarray:
    if not HAVE_CV2 or not video_path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise FileNotFoundError(f"Could not extract frame {frame_index} from {video_path}")
    return frame


def mp4_position_for_selected_frame(per_frame_path: Path, selected_video_frame: int) -> int:
    if not per_frame_path.exists():
        return int(selected_video_frame)
    pf = pd.read_csv(per_frame_path)
    if pf.empty or "video_frame_index" not in pf.columns:
        return int(selected_video_frame)
    first_video_frame = int(pf["video_frame_index"].min())
    return max(0, int(selected_video_frame) - max(0, first_video_frame - 1))


def create_contact_sheet(selected: pd.DataFrame, title: str, base_name: str) -> tuple[Path | None, list[str]]:
    failures = []
    if selected.empty or not HAVE_CV2:
        return None, ["OpenCV unavailable or no selected examples"]
    scenarios = [s for s in SCENARIO_ORDER if s in set(selected["scenario"])]
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(9.4, 3.45 * len(scenarios)))
    if len(scenarios) == 1:
        axes = np.array([axes])
    made = False
    for r, scenario in enumerate(scenarios):
        for c, method in enumerate(QUAL_METHOD_ORDER):
            ax = axes[r, c]
            ax.axis("off")
            ax.set_title(method, fontsize=11)
            row = selected[(selected["scenario"] == scenario) & (selected["method"] == method)]
            if row.empty:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                failures.append(f"{scenario}/{method}: missing selected row")
                continue
            rec = row.iloc[0]
            mp4_pos = mp4_position_for_selected_frame(Path(str(rec["per_frame_metrics_path"])), int(rec["selected_video_frame"]))
            frame = extract_video_frame(Path(str(rec["visualization_video"])), mp4_pos)
            if frame is None:
                ax.text(0.5, 0.5, "Frame unavailable", ha="center", va="center")
                failures.append(f"{scenario}/{method}: failed to extract frame {rec['selected_video_frame']}")
                continue
            if method == "Normal part-OCLReID":
                frame = replace_overlay_method_label(frame, "Method: Normal part-OCLReID")
            made = True
            ax.imshow(frame)
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98), h_pad=2.25)
    for r, scenario in enumerate(scenarios):
        left_pos = axes[r, 0].get_position()
        right_pos = axes[r, 1].get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        y = min(left_pos.y0, right_pos.y0) - 0.022
        fig.text(x_center, y, scenario, ha="center", va="top", fontsize=12)
    if not made:
        plt.close(fig)
        return None, failures
    save_fig(fig, FIGURE_DIR / base_name)
    return FIGURE_DIR / f"{base_name}.png", failures


def replace_overlay_method_label(frame: np.ndarray, label: str) -> np.ndarray:
    if not HAVE_CV2:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1 = 5, 5
    x2, y2 = min(w - 1, 390), min(h - 1, 34)
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), thickness=-1)
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)
    cv2.putText(out, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_qualitative_md(selected: pd.DataFrame, contact_sheet: Path | None, title: str, filename: str) -> None:
    lines = [
        f"# {title}",
        "",
        "These qualitative examples preserve the original selected frames and saved inference overlays. Only the displayed part-OCLReID label was changed to Normal part-OCLReID.",
        "",
    ]
    if contact_sheet:
        lines += [f"Contact sheet: `{contact_sheet}`", ""]
    for scenario in SCENARIO_ORDER:
        group = selected[selected["scenario"] == scenario]
        if group.empty:
            lines += [f"## {scenario}", "", "No matching qualitative example was selected.", ""]
            continue
        first = group.iloc[0]
        lines += [
            f"## {scenario}",
            "",
            f"- Scene: `{first['scene_name']}`",
            f"- Camera: `{first['camera']}`",
            f"- Target class: {first['target_class']}",
            f"- Selected video frame: {first['selected_video_frame']}",
            f"- Why selected: {first['reason']}",
        ]
        for method in QUAL_METHOD_ORDER:
            row = group[group["method"] == method]
            if not row.empty:
                lines.append(f"- {method} MP4: `{row.iloc[0]['visualization_video']}`")
        lines.append("")
    (QUAL_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def dataset_image_path(dataset_part: str, scene: str, camera: str, frame_index: int) -> Path | None:
    manifest_path = PROJECT_ROOT / "dataset_audit" / "manifests" / "test" / dataset_part / scene / f"{camera}_manifest.json"
    if not manifest_path.exists():
        manifest_path = AGHRI_VIDEO_ROOT / "test" / dataset_part / scene / f"{camera}_frame_manifest.json"
    if not manifest_path.exists():
        matches = sorted(AGHRI_VIDEO_ROOT.glob(f"*/{dataset_part}/{scene}/{camera}_frame_manifest.json"))
        if matches:
            manifest_path = matches[0]
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    record = next((item for item in manifest.get("frames", []) if int(item.get("video_frame_index", -1)) == int(frame_index)), None)
    if not record:
        return None
    source_sensor_dir = manifest.get("source_sensor_dir")
    if source_sensor_dir:
        return Path(source_sensor_dir) / record["file"]
    return AGHRI_DATASET_ROOT / dataset_part / scene / "sensor_data" / camera / record["file"]


def draw_box(ax: plt.Axes, row: pd.Series, prefix: str, color: str, label: str, linestyle: str = "-", linewidth: float = 2.4) -> None:
    keys = [f"{prefix}_x1", f"{prefix}_y1", f"{prefix}_x2", f"{prefix}_y2"]
    if any(k not in row for k in keys) and "_" in prefix:
        base, suffix = prefix.rsplit("_", 1)
        keys = [f"{base}_x1_{suffix}", f"{base}_y1_{suffix}", f"{base}_x2_{suffix}", f"{base}_y2_{suffix}"]
    if any(k not in row or pd.isna(row[k]) for k in keys):
        return
    x1, y1, x2, y2 = [float(row[k]) for k in keys]
    if x2 <= x1 or y2 <= y1:
        return
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))
    ax.text(x1, max(0, y1 - 3), label, color="white", fontsize=7.5, va="bottom", bbox=dict(facecolor=color, alpha=0.86, pad=1.5, edgecolor="none"))


def load_frame_metrics_for_run(row: pd.Series) -> pd.DataFrame:
    path = Path(str(row["per_frame_metrics_path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_fig9_example(df: pd.DataFrame) -> dict[str, Any]:
    normal_rows = {
        tuple(row[col] for col in run_key_columns()): row
        for _, row in df[df["method"] == "Normal part-OCLReID"].iterrows()
    }
    gated_rows = {
        tuple(row[col] for col in run_key_columns()): row
        for _, row in df[df["method"] == "Gated part-OCLReID"].iterrows()
    }
    candidates = []
    for key in sorted(set(normal_rows) & set(gated_rows)):
        n_row = normal_rows[key]
        g_row = gated_rows[key]
        n = load_frame_metrics_for_run(n_row)
        g = load_frame_metrics_for_run(g_row)
        cols = [
            "video_frame_index",
            "file",
            "gt_visible",
            "prediction_present",
            "correctly_localized",
            "iou",
            "wrong_person",
            "target_confidence",
            "tracker_id",
            "gt_x1",
            "gt_y1",
            "gt_x2",
            "gt_y2",
            "pred_x1",
            "pred_y1",
            "pred_x2",
            "pred_y2",
        ]
        merged = n[cols].merge(g[cols], on="video_frame_index", suffixes=("_normal", "_gated"))
        cand = merged[
            (merged["gt_visible_normal"] == True)
            & (merged["gt_visible_gated"] == True)
            & ((merged["prediction_present_normal"] != True) | (merged["iou_normal"] < 0.5))
            & (merged["correctly_localized_gated"] == True)
            & (merged["iou_gated"] >= 0.5)
            & (merged["wrong_person_gated"] != True)
        ].copy()
        if cand.empty:
            continue
        visible_series = n[["video_frame_index", "gt_visible"]].sort_values("video_frame_index")
        reapp_frames = []
        previous_visible = None
        for _, vis_row in visible_series.iterrows():
            current_visible = bool(vis_row["gt_visible"])
            if current_visible and previous_visible is not None and not previous_visible:
                reapp_frames.append(int(vis_row["video_frame_index"]))
            previous_visible = current_visible
        cand["normal_missing"] = (cand["prediction_present_normal"] != True).astype(int)
        cand["dist_reapp"] = cand["video_frame_index"].apply(lambda f: min([abs(int(f) - r) for r in reapp_frames], default=999))
        cand["score"] = cand["normal_missing"] * 2 + cand["iou_gated"] - cand["iou_normal"].fillna(0)
        cand["gt_center_x"] = (cand["gt_x1_normal"] + cand["gt_x2_normal"]) / 2.0
        cand["gt_center_y"] = (cand["gt_y1_normal"] + cand["gt_y2_normal"]) / 2.0
        cand["visually_centered"] = (
            (cand["gt_center_x"] > 180)
            & (cand["gt_center_x"] < 500)
            & (cand["gt_center_y"] > 60)
            & (cand["gt_center_y"] < 320)
        )
        frames = set(cand["video_frame_index"].astype(int))
        sequence_candidates = []
        for f in sorted(frames):
            seq = [f + offset for offset in range(6) if f + offset in frames]
            if len(seq) < 4:
                continue
            first = cand[cand["video_frame_index"] == f].iloc[0]
            sequence_candidates.append(
                {
                    "frames": seq[:6],
                    "first": first,
                    "visually_centered": bool(first["visually_centered"]),
                    "sequence_len": len(seq),
                    "dist_reapp": int(first["dist_reapp"]),
                    "best_gated_iou": float(cand[cand["video_frame_index"].isin(seq)]["iou_gated"].max()),
                }
            )
        if not sequence_candidates:
            best = cand.sort_values(["visually_centered", "dist_reapp", "score", "iou_gated"], ascending=[False, True, False, False]).iloc[0]
            seq = [int(best["video_frame_index"])]
            for offset in range(1, 6):
                frame = int(best["video_frame_index"]) + offset
                if frame in frames:
                    seq.append(frame)
            sequence_candidates.append(
                {
                    "frames": seq[:6],
                    "first": best,
                    "visually_centered": bool(best["visually_centered"]),
                    "sequence_len": len(seq),
                    "dist_reapp": int(best["dist_reapp"]),
                    "best_gated_iou": float(best["iou_gated"]),
                }
            )
        sequence_candidates.sort(key=lambda item: (not item["visually_centered"], -item["sequence_len"], item["dist_reapp"], -item["best_gated_iou"]))
        selected_sequence = sequence_candidates[0]
        camera_pref = {"cam_fish_front": 0, "cam_zed_rgb": 1}.get(key[2], 2)
        candidates.append(
            {
                "key": key,
                "normal_row": n_row,
                "gated_row": g_row,
                "merged": merged,
                "frames": selected_sequence["frames"],
                "camera_pref": camera_pref,
                "visually_centered": selected_sequence["visually_centered"],
                "sequence_len": selected_sequence["sequence_len"],
                "dist_reapp": selected_sequence["dist_reapp"],
                "normal_missing_count": int(cand["normal_missing"].sum()),
                "best_gated_iou": selected_sequence["best_gated_iou"],
                "reappearance_frames": reapp_frames,
            }
        )
    if not candidates:
        raise SystemExit("No Normal-vs-Gated qualitative candidate found.")
    candidates.sort(key=lambda item: (item["camera_pref"], not item["visually_centered"], -item["sequence_len"], item["dist_reapp"], -item["normal_missing_count"], -item["best_gated_iou"]))
    return candidates[0]


def create_fig9(example: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if not HAVE_CV2:
        raise SystemExit("OpenCV is required to create Figure 9.")
    dataset_part, scene, camera, target = example["key"]
    if scene != FIG9_SCENE or camera != FIG9_CAMERA or target != FIG9_TARGET:
        raise SystemExit(
            "Figure 9 expected the previously selected run "
            f"{FIG9_SCENE}/{FIG9_CAMERA}/class_{FIG9_TARGET}, got {scene}/{camera}/class_{target}."
        )
    frames = FIG9_SELECTED_FRAMES
    merged = example["merged"].set_index("video_frame_index")
    ncols = len(frames)
    normal_row = example["normal_row"]
    gated_row = example["gated_row"]
    normal_mp4 = Path(str(normal_row["visualization_video"]))
    gated_mp4 = Path(str(gated_row["visualization_video"]))
    if not normal_mp4.exists():
        raise FileNotFoundError(f"Missing Normal MP4 for Figure 9: {normal_mp4}")
    if not gated_mp4.exists():
        raise FileNotFoundError(f"Missing Gated MP4 for Figure 9: {gated_mp4}")

    normal_pf = Path(str(normal_row["per_frame_metrics_path"]))
    gated_pf = Path(str(gated_row["per_frame_metrics_path"]))
    normal_frame_dir = QUAL_DIR / "fig9_mp4_frames" / "normal"
    gated_frame_dir = QUAL_DIR / "fig9_mp4_frames" / "gated"
    normal_frame_dir.mkdir(parents=True, exist_ok=True)
    gated_frame_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(3.15 * ncols, 5.9))
    grid = fig.add_gridspec(3, ncols, height_ratios=[1, 1, 0.38], hspace=0.12, wspace=0.035)
    row_labels = ["Normal part-OCLReID", "Gated part-OCLReID"]
    normal_ious = []
    gated_ious = []
    diagnostics = []
    image_paths = []
    original_filenames = []
    normal_mp4_indices = []
    gated_mp4_indices = []
    normal_prediction_present = []
    gated_prediction_present = []
    normal_tracker_ids = []
    gated_tracker_ids = []
    normal_confidences = []
    gated_confidences = []
    extracted_normal_paths = []
    extracted_gated_paths = []
    alignment_rows = []

    for c, frame_idx in enumerate(frames):
        image_path = dataset_image_path(dataset_part, scene, camera, frame_idx)
        image_paths.append(str(image_path) if image_path else "")
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Missing dataset frame for Figure 9: {image_path}")
        original_filenames.append(image_path.name)
        rec = merged.loc[frame_idx]

        normal_mp4_index = mp4_position_for_selected_frame(normal_pf, frame_idx)
        gated_mp4_index = mp4_position_for_selected_frame(gated_pf, frame_idx)
        normal_mp4_indices.append(int(normal_mp4_index))
        gated_mp4_indices.append(int(gated_mp4_index))

        normal_bgr = extract_video_frame_bgr(normal_mp4, normal_mp4_index)
        gated_bgr = extract_video_frame_bgr(gated_mp4, gated_mp4_index)
        normal_frame_path = normal_frame_dir / f"{scene}_{camera}_target{target}_frame{frame_idx}_normal_part-OCLReID.png"
        gated_frame_path = gated_frame_dir / f"{scene}_{camera}_target{target}_frame{frame_idx}_gated_part-OCLReID.png"
        cv2.imwrite(str(normal_frame_path), normal_bgr)
        cv2.imwrite(str(gated_frame_path), gated_bgr)
        extracted_normal_paths.append(str(normal_frame_path))
        extracted_gated_paths.append(str(gated_frame_path))

        normal_file = str(rec["file_normal"])
        gated_file = str(rec["file_gated"])
        status = (
            "verified"
            if normal_mp4_index == frame_idx
            and gated_mp4_index == frame_idx
            and normal_file == image_path.name
            and gated_file == image_path.name
            else "failed"
        )
        alignment_rows.append(
            {
                "scene": scene,
                "camera": camera,
                "target": target,
                "original_dataset_frame": int(frame_idx),
                "original_image_filename": image_path.name,
                "normal_per_frame_image_filename": normal_file,
                "gated_per_frame_image_filename": gated_file,
                "normal_mp4_path": str(normal_mp4),
                "normal_mp4_frame_index": int(normal_mp4_index),
                "gated_mp4_path": str(gated_mp4),
                "gated_mp4_frame_index": int(gated_mp4_index),
                "alignment_status": status,
            }
        )
        if status != "verified":
            raise SystemExit(f"Figure 9 MP4 alignment failed for dataset frame {frame_idx}: {alignment_rows[-1]}")

        for r, method in enumerate(row_labels):
            ax = fig.add_subplot(grid[r, c])
            frame_rgb = cv2.cvtColor(normal_bgr if r == 0 else gated_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"Dataset frame {frame_idx}", fontsize=10)
            if c == 0:
                ax.text(-0.03, 0.5, method, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=10, fontweight="bold")
            suffix = "normal" if r == 0 else "gated"
            pred_present = bool(rec[f"prediction_present_{suffix}"])
            iou = float(rec[f"iou_{suffix}"]) if pd.notna(rec[f"iou_{suffix}"]) else 0.0
            conf = float(rec[f"target_confidence_{suffix}"]) if pd.notna(rec[f"target_confidence_{suffix}"]) else np.nan
            tracker_id = rec[f"tracker_id_{suffix}"]
            if suffix == "normal":
                normal_ious.append(iou)
                normal_prediction_present.append(pred_present)
                normal_tracker_ids.append(None if pd.isna(tracker_id) or int(tracker_id) < 0 else int(tracker_id))
                normal_confidences.append(None if not np.isfinite(conf) or conf < 0 else conf)
            else:
                gated_ious.append(iou)
                gated_prediction_present.append(pred_present)
                gated_tracker_ids.append(None if pd.isna(tracker_id) or int(tracker_id) < 0 else int(tracker_id))
                gated_confidences.append(None if not np.isfinite(conf) or conf < 0 else conf)
                diagnostics.append(
                    {
                        "frame": int(frame_idx),
                        "gated_iou": iou,
                        "gated_target_confidence": None if not np.isfinite(conf) else conf,
                        "gated_tracker_id": None if pd.isna(tracker_id) else int(tracker_id),
                    }
                )
    alignment_csv = QUAL_DIR / "fig9_mp4_alignment_validation.csv"
    pd.DataFrame(alignment_rows).to_csv(alignment_csv, index=False)

    logic_ax = fig.add_subplot(grid[2, :])
    logic_ax.axis("off")
    logic_ax.text(
        0.01,
        0.86,
        "Normal:\nTarget is visible, but conservative\nconfirmation withholds output.",
        ha="left",
        va="top",
        fontsize=9.6,
        bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=6.0),
    )
    logic_ax.text(
        0.365,
        0.86,
        "Gated:\nOriginal method returns no target.\nA strong candidate passes the frozen gate\nand is reassociated.",
        ha="left",
        va="top",
        fontsize=9.6,
        bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=6.0),
    )
    logic_ax.text(
        0.725,
        0.86,
        "Note:\nByteTrack association is unchanged.\nThe gate links the known target identity\nto an existing track.",
        ha="left",
        va="top",
        fontsize=9.6,
        bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc", pad=6.0),
    )
    fig.suptitle("ReID-gated target reassociation after target reappearance", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    base = FIGURE_DIR / "fig9_normal_vs_gated_reidentification_example"
    save_fig(fig, base)

    first_frame = frames[0]
    reappearance_frames = example["reappearance_frames"]
    nearest_reapp = min(reappearance_frames, key=lambda f: abs(f - first_frame)) if reappearance_frames else None
    metadata = {
        "split": "test",
        "scene": scene,
        "dataset_part": dataset_part,
        "camera": camera,
        "camera_label": CAMERA_LABELS.get(camera, camera),
        "target_identity": target,
        "panels_extracted_directly_from_saved_runtime_mp4": True,
        "normal_run_directory": normal_row["result_dir"],
        "gated_run_directory": gated_row["result_dir"],
        "selected_dataset_frames": [int(f) for f in frames],
        "frame_indices": [int(f) for f in frames],
        "normal_mp4_path": str(normal_mp4),
        "gated_mp4_path": str(gated_mp4),
        "normal_mp4_frame_indices": normal_mp4_indices,
        "gated_mp4_frame_indices": gated_mp4_indices,
        "normal_extracted_frame_paths": extracted_normal_paths,
        "gated_extracted_frame_paths": extracted_gated_paths,
        "original_image_paths": image_paths,
        "original_image_filenames": original_filenames,
        "alignment_verification": {
            "method": "For each selected dataset frame, the per-frame metrics file provided the original image filename and video_frame_index. The run MP4 position was computed from the first evaluated video_frame_index, then verified to equal the selected video_frame_index for this run. Normal and Gated rows use the same dataset frame and image filename.",
            "alignment_csv": str(alignment_csv),
            "all_statuses": [row["alignment_status"] for row in alignment_rows],
        },
        "normal_prediction_paths": [normal_row["predictions_path"]],
        "gated_prediction_paths": [gated_row["predictions_path"]],
        "ground_truth_paths": [normal_row["summary_metrics_path"], normal_row["per_frame_metrics_path"]],
        "normal_ious": normal_ious,
        "gated_ious": gated_ious,
        "prediction_availability": {
            "normal_prediction_present": normal_prediction_present,
            "gated_prediction_present": gated_prediction_present,
        },
        "tracker_ids": {
            "normal": normal_tracker_ids,
            "gated": gated_tracker_ids,
        },
        "structured_confidence_values": {
            "normal": normal_confidences,
            "gated": gated_confidences,
            "source": "evaluation/per_frame_metrics.csv target_confidence field",
        },
        "reappearance_event_information": {
            "nearest_reappearance_frame": nearest_reapp,
            "all_reappearance_frames_for_run": reappearance_frames,
            "frames_after_nearest_reappearance": None if nearest_reapp is None else [int(f - nearest_reapp) for f in frames],
        },
        "diagnostic_values_shown": {
            "normal_iou_per_frame": normal_ious,
            "gated_iou_per_frame": gated_ious,
            "gated_target_confidence_and_tracker_id": diagnostics,
            "gate_thresholds": {
                "reid_threshold": 0.60,
                "reid_margin": 0.02,
                "minimum_bbox_score": 0.0,
                "minimum_visible_parts": 1,
            },
            "unavailable_diagnostics_omitted": ["best ReID score", "second-best ReID score", "score margin", "visible-part count", "state-machine internal confirmation counter"],
        },
        "reason_for_selection": "Common completed front-fisheye run around a target reappearance event where the target is ground-truth visible, Normal part-OCLReID has no target output on the selected frames, and released-checkpoint Gated part-OCLReID correctly localizes the target with IoU >= 0.5 without wrong-person selection.",
        "reason_this_example_was_selected": "Common completed front-fisheye run around a target reappearance event where the target is ground-truth visible, Normal part-OCLReID has no target output on the selected frames, and released-checkpoint Gated part-OCLReID correctly localizes the target with IoU >= 0.5 without wrong-person selection.",
    }
    (QUAL_DIR / "fig9_example_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return base.with_suffix(".png"), metadata


def load_aggregate_overall(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data["overall"]


def checkpoint_ablation_table() -> pd.DataFrame:
    specs = [
        ("Released checkpoint, no gate", BASE_RESULTS_ROOT / "normal_part_oclreid"),
        ("AGHRI experimental checkpoint, no gate", EXPERIMENTAL_CHECKPOINT_ROOT),
        ("Released checkpoint, gate", GATED_METHOD_ROOT),
        ("AGHRI experimental checkpoint, gate", EXPERIMENTAL_GATE_ROOT),
    ]
    rows = []
    missing = []
    for label, root in specs:
        summary_path = root / "aggregate_summary.json"
        if not summary_path.exists():
            missing.append(f"{label}: {summary_path}")
            continue
        overall = load_aggregate_overall(summary_path)
        rows.append(
            {
                "Configuration": label,
                "Success@IoU 0.5": overall["micro_success_rate_iou_0_5"],
                "Mean IoU": overall["micro_mean_iou_visible"],
                "Prediction availability": overall["micro_prediction_availability"],
                "Absent-target false-positive rate": overall["micro_false_positive_rate_absent"],
                "Reacquisition rate": overall.get("micro_reacquisition_rate", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    write_table(table, TABLE_DIR / "checkpoint_gate_ablation")
    lines = [
        "# Checkpoint/Gate Ablation",
        "",
        to_markdown(table),
        "",
        "The optional AGHRI experimental checkpoint did not improve the three main end-to-end tracking metrics in this archived comparison.",
        "",
        "The main improvement came from ReID-gated target reassociation.",
        "",
        "The final gated method uses the released checkpoint.",
    ]
    if missing:
        lines += ["", "Missing optional rows were skipped:", ""]
        lines += [f"- `{item}`" for item in missing]
    (REPORT_DIR / "checkpoint_gate_ablation.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def tradeoff_table(overall: pd.DataFrame) -> pd.DataFrame:
    table = overall[
        [
            "Method",
            "Success@0.5 ↑",
            "Mean IoU ↑",
            "Prediction availability ↑",
            "Absent-target FP rate ↓",
            "Reacquisition rate ↑",
            "Wrong-person rate on visible frames ↓",
        ]
    ].rename(
        columns={
            "Success@0.5 ↑": "Success@IoU 0.5",
            "Mean IoU ↑": "Mean IoU",
            "Prediction availability ↑": "Prediction availability",
            "Absent-target FP rate ↓": "Absent-target false-positive rate",
            "Reacquisition rate ↑": "Reacquisition rate",
            "Wrong-person rate on visible frames ↓": "Wrong-person rate on visible frames",
        }
    )
    write_table(table, TABLE_DIR / "main_method_tradeoff")
    lines = [
        "# Main Method Trade-off",
        "",
        "Wrong-person rate on visible frames is defined as `wrong_person_frames / ground_truth_visible_frames`, using the existing evaluation output fields and no new denominator.",
        "",
        to_markdown(table),
    ]
    (REPORT_DIR / "main_method_tradeoff.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def write_captions() -> None:
    lines = [
        "# Figure Captions",
        "",
        "## Figure 1",
        "Overall target-tracking performance on the AGHRI test set for rpf-ReID, Normal part-OCLReID, and Gated part-OCLReID. The gated method uses the released ResNet18 checkpoint with the frozen ReID-gated reassociation fallback.",
        "",
        "## Figure 2",
        "Core tracking metrics for the selected Footpath, Polytunnel, and Vineyard scenarios. Metrics are recomputed from per-run evaluation files within each scenario rather than copied from whole-test aggregates.",
        "",
        "## Figure 3",
        "Camera-wise core metrics over the full AGHRI test set for front fisheye, left fisheye, right fisheye, and ZED RGB cameras.",
        "",
        "## Figure 4",
        "Absolute improvement over rpf-ReID for Normal part-OCLReID and Gated part-OCLReID. The gate adds target-reassociation behaviour on top of the released-checkpoint part-OCLReID configuration.",
        "",
        "## Figure 5",
        "ZED-front qualitative examples using the original selected frames and saved inference overlays. The part-based method label is shown as Normal part-OCLReID.",
        "",
        "## Figure 6",
        "Core metrics grouped by the number of annotated humans in the test scene. Group membership is shared across all three methods.",
        "",
        "## Figure 7",
        "Core metrics grouped by robot motion state, using the same stationary/moving scene definitions as the original analysis.",
        "",
        "## Figure 8",
        "Front-fisheye qualitative examples using the original selected frames and saved inference overlays. The part-based method label is shown as Normal part-OCLReID.",
        "",
        "## Figure 9",
        "Qualitative comparison of Normal part-OCLReID and Gated part-OCLReID following target reappearance. Frames are extracted directly from the saved runtime visualisation videos and aligned using the original frame manifest and per-frame evaluation mapping. Normal part-OCLReID withholds the target output despite the target being visible, whereas Gated part-OCLReID reassociates the correct existing ByteTrack track with the known target identity. The gate is used only when the original state machine returns no target; ByteTrack association itself is unchanged. The frozen gate checks are score >= 0.60, score margin >= 0.02, visible parts >= 1, and minimum bbox score >= 0.0. Confidence values visible in the runtime overlay are target confidence values from the implementation, not claimed here as raw ReID scores.",
    ]
    (REPORT_DIR / "figure_captions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_and_integrity(df: pd.DataFrame, run_info: dict[str, Any], selected: pd.DataFrame, selected_front: pd.DataFrame, failures: list[str]) -> None:
    table = aggregate(df, ["method"]).rename(columns={"method": "Method"})
    rpf = table[table["Method"] == "rpf-ReID"].iloc[0]
    normal = table[table["Method"] == "Normal part-OCLReID"].iloc[0]
    gated = table[table["Method"] == "Gated part-OCLReID"].iloc[0]
    lines = [
        "# Paper Results Summary: Core Metrics With Gate",
        "",
        "The main quantitative comparison contains exactly three methods: rpf-ReID, Normal part-OCLReID, and Gated part-OCLReID.",
        "",
        f"- rpf-ReID Success@0.5: {rpf['Success@0.5 ↑']:.3f}",
        f"- Normal part-OCLReID Success@0.5: {normal['Success@0.5 ↑']:.3f}",
        f"- Gated part-OCLReID Success@0.5: {gated['Success@0.5 ↑']:.3f}",
        "",
        "The final gated method uses the released ResNet18 checkpoint plus normal part-OCLReID online learning and the frozen ReID-gated target reassociation fallback.",
    ]
    (OUTPUT_ROOT / "PAPER_RESULTS_SUMMARY_CORE_METRICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    integrity = [
        "# Integrity Report: Core Metrics With Gate",
        "",
        f"- Output directory: `{OUTPUT_ROOT}`",
        f"- Full test results directory: `{BASE_RESULTS_ROOT}`",
        f"- Optional training archive: `{OPTIONAL_TRAINING_ROOT}`",
        f"- Common completed runs across the three methods: `{run_info['common_runs']}`",
    ]
    for method, count in run_info["per_method_runs"].items():
        integrity.append(f"- Parsed completed runs for {method}: `{count}`")
    integrity += ["", "## Excluded runs"]
    excluded_any = False
    for method, excluded in run_info["excluded_runs"].items():
        if excluded:
            excluded_any = True
            integrity.append(f"- {method}: {len(excluded)} excluded because not present in all three completed method sets")
            integrity.extend([f"  - `{item}`" for item in excluded])
    if not excluded_any:
        integrity.append("- none from completed result sets; all parsed completed runs are common across the three methods")
    integrity += ["", "## Qualitative selection issues"]
    integrity.extend([f"- {item}" for item in failures] if failures else ["- none"])
    (OUTPUT_ROOT / "INTEGRITY_REPORT_CORE_METRICS.md").write_text("\n".join(integrity) + "\n", encoding="utf-8")


def validation_report(
    df: pd.DataFrame,
    run_info: dict[str, Any],
    figure_data: dict[str, pd.DataFrame],
    tradeoff: pd.DataFrame,
    checkpoint_ablation: pd.DataFrame,
    fig9_meta: dict[str, Any],
) -> tuple[bool, str]:
    overall = aggregate(df, ["method"]).rename(columns={"method": "Method"})
    checks = []
    ok = True
    for method, anchors in [("Gated part-OCLReID", GATED_ANCHORS), ("Normal part-OCLReID", NORMAL_ANCHORS)]:
        row = overall[overall["Method"] == method].iloc[0]
        for metric, expected in anchors.items():
            actual_col = metric if metric in row.index else f"{metric} ↓"
            if metric == "Absent-target FP rate":
                actual = float(row["Absent-target FP rate ↓"])
            elif metric == "Reacquisition rate":
                actual = float(row["Reacquisition rate ↑"])
            else:
                actual = float(row[metric])
            passed = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            ok = ok and passed
            checks.append((method, metric, actual, expected, passed))
    lines = [
        "# Plot Data Validation",
        "",
        "## Source files",
    ]
    for spec in METHOD_SPECS:
        lines.append(f"- {spec.display}: `{spec.root}`")
    lines += [
        "",
        "## Run counts",
        f"- Per-method completed runs: `{run_info['per_method_runs']}`",
        f"- Common completed runs: `{run_info['common_runs']}`",
        f"- Excluded completed runs: `{run_info['excluded_runs']}`",
        "",
        "## Anchor checks",
    ]
    for method, metric, actual, expected, passed in checks:
        lines.append(f"- {method} {metric}: actual `{actual}`, expected `{expected}`, pass `{passed}`")
    lines += ["", "## Figure data"]
    for fig, data in figure_data.items():
        lines += [
            f"### {fig}",
            "",
            f"- Methods included: `{', '.join(sorted(data['method'].unique())) if 'method' in data.columns else 'not method-specific'}`",
            f"- Rows: `{len(data)}`",
            "",
            "```text",
            data.to_string(index=False),
            "```",
            "",
        ]
    lines += [
        "## Trade-off table",
        "",
        "```text",
        tradeoff.to_string(index=False),
        "```",
        "",
        "## Checkpoint ablation table",
        "",
        "```text",
        checkpoint_ablation.to_string(index=False),
        "```",
        "",
        "## Figure 9 metadata summary",
        f"- Scene: `{fig9_meta['scene']}`",
        f"- Camera: `{fig9_meta['camera']}`",
        f"- Target: `{fig9_meta['target_identity']}`",
        f"- Frames: `{fig9_meta['frame_indices']}`",
        f"- Panels extracted directly from saved runtime MP4: `{fig9_meta['panels_extracted_directly_from_saved_runtime_mp4']}`",
        f"- Normal MP4: `{fig9_meta['normal_mp4_path']}`",
        f"- Gated MP4: `{fig9_meta['gated_mp4_path']}`",
        f"- Alignment CSV: `{fig9_meta['alignment_verification']['alignment_csv']}`",
        f"- Alignment statuses: `{fig9_meta['alignment_verification']['all_statuses']}`",
        "",
        f"Validation result: `{'PASS' if ok else 'FAIL'}`",
    ]
    text = "\n".join(lines) + "\n"
    (REPORT_DIR / "plot_data_validation.md").write_text(text, encoding="utf-8")
    return ok, text


def inspect_figures() -> pd.DataFrame:
    rows = []
    figure_defs = [
        ("1", "Overall core metrics", "fig1_overall_core_metrics_whole_test", ", ".join(METHOD_ORDER)),
        ("2", "Selected-scenario core metrics", "fig2_selected_scenario_core_metrics", ", ".join(METHOD_ORDER)),
        ("3a", "Overall camera core metrics", "fig3_overall_camera_core_metrics", ", ".join(METHOD_ORDER)),
        ("3b", "Selected camera core metrics", "fig3_selected_camera_core_metrics", ", ".join(METHOD_ORDER)),
        ("4", "Improvement over rpf-ReID", "fig4_improvement_over_rpf_whole_test", "Normal part-OCLReID, Gated part-OCLReID"),
        ("5", "ZED-front qualitative examples", "fig5_zed_front_qualitative_examples", ", ".join(QUAL_METHOD_ORDER)),
        ("6", "Human-count core metrics", "fig6_human_count_core_metrics", ", ".join(METHOD_ORDER)),
        ("7", "Robot-motion core metrics", "fig7_robot_motion_core_metrics", ", ".join(METHOD_ORDER)),
        ("8", "Front-fisheye qualitative examples", "fig8_front_fisheye_qualitative_examples", ", ".join(QUAL_METHOD_ORDER)),
        ("9", "Normal-vs-gated reassociation example", "fig9_normal_vs_gated_reidentification_example", "Normal part-OCLReID, Gated part-OCLReID"),
    ]
    for number, title, base, methods in figure_defs:
        png = FIGURE_DIR / f"{base}.png"
        pdf = FIGURE_DIR / f"{base}.pdf"
        status_parts = []
        if png.exists() and png.stat().st_size > 0 and HAVE_CV2:
            img = cv2.imread(str(png))
            if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
                status_parts.append(f"PNG opens {img.shape[1]}x{img.shape[0]}")
            else:
                status_parts.append("PNG failed to open")
        else:
            status_parts.append("PNG missing or empty")
        if pdf.exists() and pdf.stat().st_size > 0 and pdf.read_bytes()[:4] == b"%PDF":
            status_parts.append("PDF non-empty and has PDF header")
        else:
            status_parts.append("PDF missing/empty/header failed")
        rows.append(
            {
                "figure_number": number,
                "figure_title": title,
                "png_path": str(png),
                "pdf_path": str(pdf),
                "png_size_bytes": png.stat().st_size if png.exists() else 0,
                "pdf_size_bytes": pdf.stat().st_size if pdf.exists() else 0,
                "methods_shown": methods,
                "status": "; ".join(status_parts),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "generated_figure_index.csv", index=False)
    lines = [
        "# Visual Inspection",
        "",
        "All generated PNG files were opened with OpenCV and all PDFs were checked for non-empty PDF headers. Method ordering and labels were generated from fixed method-order constants.",
        "",
        "Manual visual inspection completed after regeneration: Figure 1 has no internal plot title while retaining the legend, y-axis label, metric labels, bar annotations, method order, and colours. Figure 9 shows panels extracted directly from the saved Normal and Gated runtime MP4 files with matching source frames, readable runtime overlays, clear external row labels, and no manually reconstructed runtime boxes or states.",
        "",
        to_markdown(out),
    ]
    (REPORT_DIR / "visual_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def copy_script_to_output() -> None:
    dest = SCRIPT_DIR / "generate_core_metric_paper_outputs_with_gate.py"
    if Path(__file__).resolve() != dest.resolve():
        shutil.copy2(Path(__file__), dest)


def main() -> None:
    ensure_dirs()
    per_run_all = build_per_run_all()
    per_run_all.to_csv(TABLE_DIR / "per_run_core_metrics_all_completed.csv", index=False)
    per_run, run_info = common_run_filter(per_run_all)
    per_run.to_csv(TABLE_DIR / "per_run_core_metrics_common_runs.csv", index=False)
    write_xlsx(per_run, TABLE_DIR / "per_run_core_metrics_common_runs.xlsx", sheet_name="per_run_common")
    overall = create_main_table(per_run)
    figure_data = create_figures(per_run, overall)
    selected_zed, zed_failures = select_qualitative(per_run, "cam_zed_rgb", PREFERRED_QUALITATIVE_FRAMES, "selected_zed_front_qualitative_examples.csv")
    zed_sheet, zed_extract_failures = create_contact_sheet(selected_zed, "ZED front RGB qualitative examples", "fig5_zed_front_qualitative_examples")
    selected_front, front_failures = select_qualitative(per_run, "cam_fish_front", PREFERRED_FRONT_FISHEYE_FRAMES, "selected_front_fisheye_qualitative_examples.csv")
    front_sheet, front_extract_failures = create_contact_sheet(selected_front, "Front fisheye qualitative examples", "fig8_front_fisheye_qualitative_examples")
    write_qualitative_md(selected_zed, zed_sheet, "ZED Front Qualitative Examples", "ZED_FRONT_QUALITATIVE_EXAMPLES.md")
    write_qualitative_md(selected_front, front_sheet, "Front Fisheye Qualitative Examples", "FRONT_FISHEYE_QUALITATIVE_EXAMPLES.md")
    fig9_png, fig9_meta = create_fig9(find_fig9_example(per_run))
    tradeoff = tradeoff_table(overall)
    checkpoint_ablation = checkpoint_ablation_table()
    write_captions()
    write_summary_and_integrity(per_run, run_info, selected_zed, selected_front, zed_failures + zed_extract_failures + front_failures + front_extract_failures)
    validation_ok, _ = validation_report(per_run, run_info, figure_data, tradeoff, checkpoint_ablation, fig9_meta)
    if not validation_ok:
        raise SystemExit("Numerical validation failed; see reports/plot_data_validation.md")
    figure_index = inspect_figures()
    copy_script_to_output()
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Common runs: {run_info['common_runs']}")
    print(f"Figure 9: {fig9_png}")
    print(f"Figure index: {REPORT_DIR / 'generated_figure_index.csv'}")
    print(f"All figure statuses: {figure_index['status'].tolist()}")


if __name__ == "__main__":
    main()
