#!/usr/bin/env python3
"""Summarize AGHRI PointPillars detector-family comparison outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from evaluate_aghri_generic_outputs import _eval_2d, _eval_3d, _gt_for_manifest, _load_manifest, _read_camera, _read_lidar, _read_matched_indices  # noqa: E402
from late_fusion_pkg.checkpoints import sha256_file  # noqa: E402


OUT = ROOT / "results" / "aghri_pointpillars_detector_comparison"
SECOND_OUT = ROOT / "results" / "aghri_finetuned_detector_comparison"
MANIFEST = ROOT / "data_manifests" / "aghri_late_fusion_test_manifest.csv"
IMPORTED = ROOT / "checkpoints" / "aghri_pointpillars_finetune" / "cluster_run"
CLUSTER = Path("/home/prabuddhi/cluster/aghri_pointpillars_finetune")

COMBOS = {
    "cam_g_pointpillars_g": ("Generic YOLO + Generic PointPillars", "Generic YOLO", "yolo_generic", "pointpillars_generic"),
    "cam_ft_pointpillars_g": ("Fine-tuned YOLO + Generic PointPillars", "Fine-tuned YOLO", "yolo_finetuned", "pointpillars_generic"),
    "cam_g_pointpillars_ft": ("Generic YOLO + Fine-tuned PointPillars", "Generic YOLO", "yolo_generic", "pointpillars_finetuned"),
    "cam_ft_pointpillars_ft": ("Fine-tuned YOLO + Fine-tuned PointPillars", "Fine-tuned YOLO", "yolo_finetuned", "pointpillars_finetuned"),
}

LIDAR_BOX_ORIGINS = {
    "pointpillars_generic": "bottom_center",
    "pointpillars_finetuned": "bottom_center",
}


def lidar_box_origin(cache_name: str) -> str:
    return LIDAR_BOX_ORIGINS.get(cache_name, "bottom_center")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    best = {}
    for column in columns[1:]:
        values = [row[column] for row in rows if isinstance(row.get(column), (int, float))]
        if values:
            best[column] = max(values)
    align = []
    for column in columns:
        numeric = all(isinstance(row.get(column), (int, float)) for row in rows)
        align.append("---:" if numeric else "---")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(align) + "|",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            if column in best and isinstance(value, (int, float)) and math.isclose(float(value), float(best[column]), rel_tol=1e-12, abs_tol=1e-12):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sum_unique_inference(path: Path) -> float:
    totals: dict[str, float] = {}
    for row in read_csv(path):
        if row.get("sample_id") and row.get("inference_time_sec"):
            totals.setdefault(row["sample_id"], float(row["inference_time_sec"]))
    return float(sum(totals.values()))


def sum_column(path: Path, key: str) -> float:
    return float(sum(float(row[key]) for row in read_csv(path) if row.get(key)))


def cache_path(name: str, filename: str) -> Path:
    return OUT / "cache" / name / filename


def combo_path(name: str, filename: str) -> Path:
    return OUT / "combinations" / name / filename


def detector_metrics() -> dict[str, dict[str, Any]]:
    rows = _load_manifest(MANIFEST)
    gt2d, gt3d = _gt_for_manifest(rows)
    out: dict[str, dict[str, Any]] = {}
    for name in ("yolo_generic", "yolo_finetuned"):
        values = _eval_2d(_read_camera(cache_path(name, "detections.csv")), gt2d)
        values["runtime_total_sec"] = sum_unique_inference(cache_path(name, "detections.csv"))
        values["metadata"] = read_json(cache_path(name, "metadata.json"))
        out[name] = values
    for name in ("pointpillars_generic", "pointpillars_finetuned"):
        values = _eval_3d(_read_lidar(cache_path(name, "detections.csv"), box_origin=lidar_box_origin(name)), gt3d)
        values["runtime_total_sec"] = sum_unique_inference(cache_path(name, "detections.csv"))
        values["metadata"] = read_json(cache_path(name, "metadata.json"))
        out[name] = values
    return out


def write_cache_audit() -> None:
    rows = _load_manifest(MANIFEST)
    manifest_ids = {row["sample_id"] for row in rows}
    image_ids = {row["sample_id"] for row in rows if row.get("image_path") and Path(row["image_path"]).is_file()}
    audit = {
        "manifest_path": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_rows": len(rows),
        "readable_image_rows": len(image_ids),
        "readable_lidar_rows": sum(bool(row.get("lidar_point_cloud_path")) and Path(row["lidar_point_cloud_path"]).is_file() for row in rows),
        "caches": {},
    }
    for name in ("yolo_generic", "yolo_finetuned", "pointpillars_generic", "pointpillars_finetuned"):
        ids = {row["sample_id"] for row in read_csv(cache_path(name, "detections.csv")) if row.get("sample_id")}
        extra_empty_ros_rows = sum(
            1
            for row in read_csv(cache_path(name, "detections.csv"))
            if "/ros_image_" in row.get("sample_id", "") and str(row.get("empty_frame", "")).lower() in {"true", "1", "yes"}
        )
        expected = image_ids if name.startswith("yolo") else manifest_ids
        audit["caches"][name] = {
            "detections_csv": str(cache_path(name, "detections.csv")),
            "detections_sha256": sha256_file(cache_path(name, "detections.csv")),
            "metadata_json": str(cache_path(name, "metadata.json")),
            "metadata_sha256": sha256_file(cache_path(name, "metadata.json")),
            "sample_id_count": len(ids),
            "contains_expected_frame_set": expected.issubset(ids),
            "extra_empty_ros_rows": extra_empty_ros_rows,
        }
    write_csv(OUT / "reports" / "cache_audit.csv", [{"cache": key, **value} for key, value in audit["caches"].items()])
    (OUT / "reports" / "cache_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def write_import_audit() -> dict[str, Any]:
    summary = read_json(IMPORTED / "summaries" / "summary_aghri_pointpillars_finetune.json")[0]
    imported = read_json(IMPORTED / "import_manifest.json")
    best = IMPORTED / "weights" / "best.pth"
    last = IMPORTED / "weights" / "last.pth"
    checkpoints = []
    for path in sorted((CLUSTER / "checkpoints").rglob("*.pth")) + sorted((CLUSTER / "outputs").rglob("*.pth")):
        rel = path.relative_to(CLUSTER).as_posix()
        if rel.startswith("checkpoints/generic/"):
            kind = "generic source checkpoint"
        elif "/smoke/" in rel:
            kind = "smoke checkpoint"
        elif rel.endswith("best_aghri_bev_ap_mean_epoch_126.pth") and "/full/" in rel:
            kind = "full-run best validation checkpoint"
        elif rel.endswith("epoch_160.pth") and "/full/" in rel:
            kind = "full-run latest checkpoint"
        elif "/full/" in rel and Path(rel).name.startswith("epoch_"):
            kind = "full-run epoch checkpoint"
        else:
            kind = "unknown"
        checkpoints.append({"path": str(path), "relative_path": rel, "classification": kind, "sha256": sha256_file(path)})
    audit = {
        "training_completed": True,
        "epochs": summary["epochs"],
        "best_epoch": 126,
        "best_metric": "aghri/bev_ap_mean",
        "best_validation_metric": summary["bev_ap_mean"],
        "best_checkpoint_cluster": str(CLUSTER / "outputs/pointpillars_aghri_seed42/full/best_aghri_bev_ap_mean_epoch_126.pth"),
        "best_checkpoint_local": str(best),
        "best_checkpoint_sha256": sha256_file(best),
        "last_checkpoint_local": str(last),
        "last_checkpoint_sha256": sha256_file(last),
        "generic_source_checkpoint_sha256": "37dc242098f0d3dd7d870e0fe7cd4514f76fd85a04fb1377681279f963b3cbe5",
        "selected_by_validation_only": True,
        "held_out_test_used_for_selection": False,
        "num_classes": summary["num_classes"],
        "class_mapping": {"0": "person"},
        "point_cloud_range": [-10.24, -10.24, -2.0, 25.6, 10.24, 3.0],
        "voxel_size": [0.16, 0.16, 5.0],
        "nms_settings": {"use_rotate_nms": True, "nms_thr": 0.01, "score_thr": 0.05, "nms_pre": 1000, "max_num": 100},
        "import_manifest": imported,
        "checkpoints": checkpoints,
    }
    (OUT / "import_audit").mkdir(parents=True, exist_ok=True)
    (OUT / "import_audit" / "checkpoint_selection.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "import_audit" / "cluster_run_inventory.json").write_text(json.dumps({"checkpoints": checkpoints, "audit": audit}, indent=2), encoding="utf-8")
    (OUT / "import_audit" / "checkpoint_selection.md").write_text(
        "\n".join(
            [
                "# AGHRI Fine-Tuned PointPillars Checkpoint Selection",
                "",
                f"- Training completed: `{audit['training_completed']}`",
                f"- Best epoch: `{audit['best_epoch']}`",
                f"- Validation metric: `{audit['best_metric']}={audit['best_validation_metric']}`",
                f"- Local best checkpoint: `{audit['best_checkpoint_local']}`",
                f"- SHA256: `{audit['best_checkpoint_sha256']}`",
                "- Smoke checkpoints were discovered but not imported as final checkpoints.",
                "- Training and selection used train/validation only; held-out test was reserved for final evaluation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


def build_main_tables(detectors: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for combo, (label, _camera_label, cam, lidar) in COMBOS.items():
        metrics = read_json(combo_path(combo, "metrics.json"))
        overall = read_csv(combo_path(combo, "fusion/overall_results.csv"))[0]
        fusion_time = sum_column(combo_path(combo, "fusion/per_frame_results.csv"), "late_fusion_time_sec")
        complete_time = detectors[cam]["runtime_total_sec"] + detectors[lidar]["runtime_total_sec"] + fusion_time
        matched = int(overall["matched_pairs"])
        lidar_predictions = int(overall["lidar_predictions"])
        row = {
            "combination_id": combo,
            "Combination": label,
            "Camera AP50": metrics["camera"]["ap"],
            "LiDAR BEV AP@0.25": metrics["lidar_second"]["bev_iou_0.25_ap"],
            "LiDAR BEV AP@0.50": metrics["lidar_second"]["bev_iou_0.50_ap"],
            "LiDAR BEV AP@0.75": metrics["lidar_second"]["bev_iou_0.75_ap"],
            "LiDAR 3D AP@0.25": metrics["lidar_second"]["3d_iou_0.25_ap"],
            "LiDAR 3D AP@0.50": metrics["lidar_second"]["3d_iou_0.50_ap"],
            "LiDAR 3D AP@0.75": metrics["lidar_second"]["3d_iou_0.75_ap"],
            "LiDAR / ALL3D 3D AP@0.25": metrics["late_fusion_all3d"]["3d_iou_0.25_ap"],
            "Matched BEV AP@0.25": metrics["late_fusion_matched_only"]["bev_iou_0.25_ap"],
            "Matched 3D AP@0.25": metrics["late_fusion_matched_only"]["3d_iou_0.25_ap"],
            "Matched-only 3D AP@0.25": metrics["late_fusion_matched_only"]["3d_iou_0.25_ap"],
            "Matched 3D AP@0.50": metrics["late_fusion_matched_only"]["3d_iou_0.50_ap"],
            "ALL3D BEV AP@0.25": metrics["late_fusion_all3d"]["bev_iou_0.25_ap"],
            "ALL3D 3D AP@0.25": metrics["late_fusion_all3d"]["3d_iou_0.25_ap"],
            "ALL3D 3D AP@0.50": metrics["late_fusion_all3d"]["3d_iou_0.50_ap"],
            "ALL3D prediction count": metrics["late_fusion_all3d"]["predictions"],
            "Precision": metrics["late_fusion_matched_only"]["3d_iou_0.25_precision"],
            "Recall": metrics["late_fusion_matched_only"]["3d_iou_0.25_recall"],
            "F1": metrics["late_fusion_matched_only"]["3d_iou_0.25_f1"],
            "Matched pairs": matched,
            "Camera match rate": matched / max(int(overall["camera_predictions"]), 1),
            "LiDAR match rate": matched / max(lidar_predictions, 1),
            "LiDAR inference time sec": detectors[lidar]["runtime_total_sec"],
            "Fusion-only time sec": fusion_time,
            "Complete FPS": int(overall["frames"]) / complete_time if complete_time > 0 else 0.0,
        }
        rows.append(row)
    write_csv(OUT / "tables" / "main_comparison.csv", rows)
    write_md_table(
        OUT / "tables" / "main_comparison.md",
        rows,
        [
            "Combination",
            "Camera AP50",
            "LiDAR BEV AP@0.25",
            "LiDAR / ALL3D 3D AP@0.25",
            "Matched BEV AP@0.25",
            "Matched-only 3D AP@0.25",
            "Matched 3D AP@0.50",
            "ALL3D prediction count",
            "Precision",
            "Recall",
            "F1",
            "Matched pairs",
            "LiDAR match rate",
            "Complete FPS",
        ],
    )
    return rows


def per_recording_tables() -> None:
    rows = _load_manifest(MANIFEST)
    recordings = sorted({row["recording_name"] for row in rows})
    out_rows = []
    for combo, (label, _camera_label, cam, lidar) in COMBOS.items():
        camera = _read_camera(cache_path(cam, "detections.csv"))
        lidar_all = _read_lidar(cache_path(lidar, "detections.csv"), box_origin=lidar_box_origin(lidar))
        matched = _read_matched_indices(combo_path(combo, "fusion/association_results.csv"))
        associations = read_csv(combo_path(combo, "fusion/association_results.csv"))
        for recording in recordings:
            subset = [row for row in rows if row["recording_name"] == recording]
            sample_ids = {row["sample_id"] for row in subset}
            gt2d, gt3d = _gt_for_manifest(subset)
            cam_sub = {key: value for key, value in camera.items() if key in sample_ids}
            lidar_sub = {key: value for key, value in lidar_all.items() if key in sample_ids}
            matched_sub = {key: [item for idx, item in enumerate(items) if idx in matched.get(key, set())] for key, items in lidar_sub.items()}
            cam_m = _eval_2d(cam_sub, gt2d)
            lidar_m = _eval_3d(lidar_sub, gt3d)
            matched_m = _eval_3d(matched_sub, gt3d)
            out_rows.append(
                {
                    "recording_name": recording,
                    "combination": label,
                    "frames": len(subset),
                    "camera_ap50": cam_m["ap"],
                    "lidar_bev_ap_0.25": lidar_m["bev_iou_0.25_ap"],
                    "lidar_3d_ap_0.25": lidar_m["3d_iou_0.25_ap"],
                    "matched_bev_ap_0.25": matched_m["bev_iou_0.25_ap"],
                    "matched_3d_ap_0.25": matched_m["3d_iou_0.25_ap"],
                    "matched_pairs": sum(1 for row in associations if row["sample_id"] in sample_ids),
                }
            )
    write_csv(OUT / "tables" / "per_recording_comparison.csv", out_rows)


def second_vs_pointpillars(rows: list[dict[str, Any]]) -> None:
    second_rows = {row["combination_id"]: row for row in read_csv(SECOND_OUT / "tables" / "main_comparison.csv")}
    pp_rows = {row["combination_id"]: row for row in rows}
    mapping = [
        ("Generic YOLO", "cam_g_lidar_g", "cam_g_pointpillars_g"),
        ("Fine-tuned YOLO", "cam_ft_lidar_ft", "cam_ft_pointpillars_ft"),
    ]
    out_rows = []
    for camera, second_id, pp_id in mapping:
        sec = second_rows[second_id]
        pp = pp_rows[pp_id]
        metric_paths = {
            "SECOND": SECOND_OUT / "combinations" / second_id / "metrics.json",
            "PointPillars": OUT / "combinations" / pp_id / "metrics.json",
        }
        for detector, row in (("SECOND", sec), ("PointPillars", pp)):
            lidar_metrics = read_json(metric_paths[detector])["lidar_second"]
            out_rows.append(
                {
                    "Camera": camera,
                    "LiDAR detector": detector,
                    "BEV AP@0.25": float(lidar_metrics["bev_iou_0.25_ap"]),
                    "BEV AP@0.50": float(lidar_metrics["bev_iou_0.50_ap"]),
                    "BEV AP@0.75": float(lidar_metrics["bev_iou_0.75_ap"]),
                    "3D AP@0.25": float(lidar_metrics["3d_iou_0.25_ap"]),
                    "3D AP@0.50": float(lidar_metrics["3d_iou_0.50_ap"]),
                    "3D AP@0.75": float(lidar_metrics["3d_iou_0.75_ap"]),
                    "Matched-only 3D AP@0.25": float(row.get("Matched 3D AP@0.25", row.get("LF matched 3D AP@0.25", 0.0))),
                    "Matched pairs": int(float(row["Matched pairs"])),
                    "LiDAR match rate": float(row["LiDAR match rate"]),
                    "Precision@3D0.25": float(row.get("Precision", row.get("precision", 0.0))),
                    "Recall@3D0.25": float(row.get("Recall", row.get("recall", 0.0))),
                    "F1@3D0.25": float(row.get("F1", row.get("f1", 0.0))),
                }
            )
    write_csv(OUT / "tables" / "second_vs_pointpillars.csv", out_rows)
    write_md_table(
        OUT / "tables" / "second_vs_pointpillars.md",
        out_rows,
        ["Camera", "LiDAR detector", "BEV AP@0.25", "3D AP@0.25", "Matched-only 3D AP@0.25", "Matched pairs", "LiDAR match rate", "F1@3D0.25"],
    )


def write_report(audit: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    main = (OUT / "tables" / "main_comparison.md").read_text(encoding="utf-8")
    svp = (OUT / "tables" / "second_vs_pointpillars.md").read_text(encoding="utf-8")
    best_pp = max(rows, key=lambda row: row["Matched 3D AP@0.25"])
    report = [
        "# AGHRI Fine-Tuned PointPillars Implementation",
        "",
        "Imported the full-run AGHRI fine-tuned PointPillars checkpoint, generated the frozen held-out test cache, and evaluated all four YOLO/PointPillars detector combinations.",
        "",
        "## Checkpoint",
        f"- Cluster checkpoint: `{audit['best_checkpoint_cluster']}`",
        f"- Local checkpoint: `{audit['best_checkpoint_local']}`",
        f"- SHA256: `{audit['best_checkpoint_sha256']}`",
        f"- Best epoch: `{audit['best_epoch']}` selected by `{audit['best_metric']}={audit['best_validation_metric']}` on validation only.",
        "- Class mapping: one class, `person`.",
        "",
        "## Frozen Test Cache",
        f"- Cache: `{OUT / 'cache/pointpillars_finetuned/detections.csv'}`",
        f"- Cache SHA256: `{sha256_file(OUT / 'cache/pointpillars_finetuned/detections.csv')}`",
        f"- Metadata: `{OUT / 'cache/pointpillars_finetuned/metadata.json'}`",
        "",
        "## Four PointPillars Combinations",
        main,
        "## SECOND Versus PointPillars",
        svp,
        "## ROS Cached Replay",
        "Use `lidar_detector:=pointpillars lidar_model:=finetuned` to replay cached fine-tuned PointPillars detections. ROS playback does not run live MMDetection3D inference.",
        "",
        "## Best PointPillars Combination",
        f"`{best_pp['Combination']}` by matched 3D AP@0.25.",
        "",
        "SECOND artifacts and metrics were read for fair comparison only; they were not modified.",
    ]
    (ROOT / "AGHRI_FINETUNED_POINTPILLARS_IMPLEMENTATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    audit = write_import_audit()
    write_cache_audit()
    detectors = detector_metrics()
    rows = build_main_tables(detectors)
    per_recording_tables()
    second_vs_pointpillars(rows)
    write_report(audit, rows)
    print(json.dumps({"main_table": str(OUT / "tables" / "main_comparison.csv"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
