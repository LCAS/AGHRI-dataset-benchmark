#!/usr/bin/env python3
"""Audit AGHRI ALL3D metrics against source LiDAR metrics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aghri_all3d_metric_audit"
TOLERANCE = 1e-9

COMBOS = [
    {
        "camera": "Generic YOLO",
        "lidar_detector": "SECOND",
        "lidar_model": "Generic",
        "combination": "Generic YOLO + Generic SECOND",
        "metrics": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_g/metrics.json",
        "overall": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_g/overall_results.csv",
        "per_frame": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_g/per_frame_results.csv",
    },
    {
        "camera": "Fine-tuned YOLO",
        "lidar_detector": "SECOND",
        "lidar_model": "Generic",
        "combination": "Fine-tuned YOLO + Generic SECOND",
        "metrics": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_g/metrics.json",
        "overall": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_g/overall_results.csv",
        "per_frame": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_g/per_frame_results.csv",
    },
    {
        "camera": "Generic YOLO",
        "lidar_detector": "SECOND",
        "lidar_model": "Fine-tuned",
        "combination": "Generic YOLO + Fine-tuned SECOND",
        "metrics": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_ft/metrics.json",
        "overall": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_ft/overall_results.csv",
        "per_frame": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_g_lidar_ft/per_frame_results.csv",
    },
    {
        "camera": "Fine-tuned YOLO",
        "lidar_detector": "SECOND",
        "lidar_model": "Fine-tuned",
        "combination": "Fine-tuned YOLO + Fine-tuned SECOND",
        "metrics": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_ft/metrics.json",
        "overall": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_ft/overall_results.csv",
        "per_frame": ROOT / "results/aghri_finetuned_detector_comparison/combinations/cam_ft_lidar_ft/per_frame_results.csv",
    },
    {
        "camera": "Generic YOLO",
        "lidar_detector": "PointPillars",
        "lidar_model": "Generic",
        "combination": "Generic YOLO + Generic PointPillars",
        "metrics": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_g/metrics.json",
        "overall": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_g/fusion/overall_results.csv",
        "per_frame": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_g/fusion/per_frame_results.csv",
    },
    {
        "camera": "Fine-tuned YOLO",
        "lidar_detector": "PointPillars",
        "lidar_model": "Generic",
        "combination": "Fine-tuned YOLO + Generic PointPillars",
        "metrics": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_g/metrics.json",
        "overall": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_g/fusion/overall_results.csv",
        "per_frame": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_g/fusion/per_frame_results.csv",
    },
    {
        "camera": "Generic YOLO",
        "lidar_detector": "PointPillars",
        "lidar_model": "Fine-tuned",
        "combination": "Generic YOLO + Fine-tuned PointPillars",
        "metrics": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_ft/metrics.json",
        "overall": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_ft/fusion/overall_results.csv",
        "per_frame": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_g_pointpillars_ft/fusion/per_frame_results.csv",
    },
    {
        "camera": "Fine-tuned YOLO",
        "lidar_detector": "PointPillars",
        "lidar_model": "Fine-tuned",
        "combination": "Fine-tuned YOLO + Fine-tuned PointPillars",
        "metrics": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_ft/metrics.json",
        "overall": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_ft/fusion/overall_results.csv",
        "per_frame": ROOT / "results/aghri_pointpillars_detector_comparison/combinations/cam_ft_pointpillars_ft/fusion/per_frame_results.csv",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric(values: dict[str, Any], key: str) -> float:
    return float(values[key])


def status_for(differences: list[float]) -> str:
    return "PASS" if all(abs(value) <= TOLERANCE for value in differences) else "FAIL"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if column not in {"Camera", "Camera model", "LiDAR detector", "LiDAR model", "Status", "Explanation"} else "---" for column in columns) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.12g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    equality_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"tolerance": TOLERANCE, "combinations": []}
    for combo in COMBOS:
        metrics = read_json(combo["metrics"])
        overall = read_csv(combo["overall"])[0]
        per_frame = read_csv(combo["per_frame"])
        lidar = metrics["lidar_second"]
        all3d = metrics["late_fusion_all3d"]
        matched = metrics["late_fusion_matched_only"]
        diffs = [
            metric(all3d, "3d_iou_0.25_ap") - metric(lidar, "3d_iou_0.25_ap"),
            metric(all3d, "3d_iou_0.50_ap") - metric(lidar, "3d_iou_0.50_ap"),
            metric(all3d, "bev_iou_0.25_ap") - metric(lidar, "bev_iou_0.25_ap"),
            float(all3d["predictions"]) - float(lidar["predictions"]),
        ]
        status = status_for(diffs)
        unmatched_lidar = sum(int(row["unmatched_lidar"]) for row in per_frame)
        matched_pairs = int(overall["matched_pairs"])
        lidar_predictions = int(overall["lidar_predictions"])
        final_3d = int(overall["final_3d_predictions"])
        equality = {
            "Camera": combo["camera"],
            "LiDAR detector": combo["lidar_detector"],
            "LiDAR model": combo["lidar_model"],
            "LiDAR BEV AP@0.25": metric(lidar, "bev_iou_0.25_ap"),
            "ALL3D BEV AP@0.25": metric(all3d, "bev_iou_0.25_ap"),
            "LiDAR 3D AP@0.25": metric(lidar, "3d_iou_0.25_ap"),
            "ALL3D 3D AP@0.25": metric(all3d, "3d_iou_0.25_ap"),
            "Difference": diffs[0],
            "ALL3D 3D AP@0.50": metric(all3d, "3d_iou_0.50_ap"),
            "LiDAR prediction count": int(lidar["predictions"]),
            "ALL3D prediction count": int(all3d["predictions"]),
            "Status": status,
        }
        summary = {
            "LiDAR detector": combo["lidar_detector"],
            "Camera model": combo["camera"],
            "LiDAR model": combo["lidar_model"],
            "LiDAR / ALL3D 3D AP@0.25": metric(all3d, "3d_iou_0.25_ap"),
            "Matched-only 3D AP@0.25": metric(matched, "3d_iou_0.25_ap"),
            "Matched pairs": matched_pairs,
            "Unmatched LiDAR count": unmatched_lidar,
            "Final ALL3D count": final_3d,
            "Explanation": "ALL3D keeps matched and unmatched LiDAR boxes; matched-only is the YOLO-associated subset.",
        }
        if matched_pairs + unmatched_lidar != final_3d or final_3d != lidar_predictions:
            status = "FAIL"
            equality["Status"] = status
        equality_rows.append(equality)
        summary_rows.append(summary)
        details["combinations"].append(
            {
                "camera": combo["camera"],
                "lidar_detector": combo["lidar_detector"],
                "lidar_model": combo["lidar_model"],
                "combination": combo["combination"],
                "metrics_path": str(combo["metrics"]),
                "overall_path": str(combo["overall"]),
                "per_frame_path": str(combo["per_frame"]),
                "all3d_equals_lidar_status": status,
                "differences": {
                    "3d_ap_0.25": diffs[0],
                    "3d_ap_0.50": diffs[1],
                    "bev_ap_0.25": diffs[2],
                    "prediction_count": diffs[3],
                },
                "matched_pairs": matched_pairs,
                "unmatched_lidar_count": unmatched_lidar,
                "final_all3d_count": final_3d,
                "lidar_prediction_count": lidar_predictions,
            }
        )
    details["all_passed"] = all(row["Status"] == "PASS" for row in equality_rows)
    return equality_rows, summary_rows, details


def write_report(equality_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], details: dict[str, Any]) -> None:
    report = [
        "# AGHRI ALL3D Metric Audit",
        "",
        "## Purpose",
        "This audit makes the complete final 3D output explicit. Under the current fusion policy, `ALL3D` is the union of matched LiDAR detections and unmatched LiDAR detections.",
        "",
        "## Definitions",
        "- `matched 3D`: LiDAR detections associated with a YOLO box by projected 2D IoU.",
        "- `unmatched LiDAR`: LiDAR detections without camera support at the fixed IoU threshold.",
        "- `ALL3D`: `matched 3D + unmatched LiDAR`, the canonical complete final 3D output.",
        "",
        "## ALL3D Versus LiDAR",
        markdown_table(
            equality_rows,
            [
                "Camera",
                "LiDAR detector",
                "LiDAR model",
                "LiDAR 3D AP@0.25",
                "ALL3D 3D AP@0.25",
                "Difference",
                "ALL3D 3D AP@0.50",
                "ALL3D prediction count",
                "Status",
            ],
        ),
        "",
        "## Matched-Only Versus ALL3D",
        markdown_table(
            summary_rows,
            [
                "LiDAR detector",
                "Camera model",
                "LiDAR model",
                "LiDAR / ALL3D 3D AP@0.25",
                "Matched-only 3D AP@0.25",
                "Matched pairs",
                "Unmatched LiDAR count",
                "Explanation",
            ],
        ),
        "",
        "## Conclusion",
        "The explicit ALL3D evaluation confirms that the complete final 3D output is equivalent to the source LiDAR detector output under the current fusion policy. Matched-only AP is lower because it evaluates only the camera-associated subset and does not include unmatched LiDAR detections.",
        "",
        f"Tolerance: `{TOLERANCE}`. Overall status: `{'PASS' if details['all_passed'] else 'FAIL'}`.",
    ]
    (OUT / "ALL3D_METRIC_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    equality_rows, summary_rows, details = build_rows()
    write_csv(OUT / "all3d_vs_lidar_metrics.csv", equality_rows)
    write_csv(OUT / "matched_vs_all3d_summary.csv", summary_rows)
    (OUT / "all3d_metric_audit.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_report(equality_rows, summary_rows, details)
    print(json.dumps({"output_root": str(OUT), "all_passed": details["all_passed"]}, indent=2))


if __name__ == "__main__":
    main()
