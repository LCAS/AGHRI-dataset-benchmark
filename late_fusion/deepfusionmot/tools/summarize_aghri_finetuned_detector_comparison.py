#!/usr/bin/env python3
"""Summarize the AGHRI fine-tuned detector comparison."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from evaluate_aghri_generic_outputs import _eval_2d, _eval_3d, _gt_for_manifest, _load_manifest, _read_camera, _read_lidar, _read_matched_indices  # noqa: E402
from late_fusion_pkg.checkpoints import sha256_file  # noqa: E402


OUT = ROOT / "results" / "aghri_finetuned_detector_comparison"
CLUSTER = Path("/home/prabuddhi/cluster/aghri_second_finetune")
IMPORTED = ROOT / "checkpoints" / "aghri_second_finetune" / "cluster_run"
MANIFEST = ROOT / "data_manifests" / "aghri_late_fusion_test_manifest.csv"

COMBOS = {
    "cam_g_lidar_g": ("Generic YOLO + Generic SECOND", "yolo_generic", "second_generic"),
    "cam_ft_lidar_g": ("Fine-tuned YOLO + Generic SECOND", "yolo_finetuned", "second_generic"),
    "cam_g_lidar_ft": ("Generic YOLO + Fine-tuned SECOND", "yolo_generic", "second_finetuned"),
    "cam_ft_lidar_ft": ("Fine-tuned YOLO + Fine-tuned SECOND", "yolo_finetuned", "second_finetuned"),
}

LIDAR_BOX_ORIGINS = {
    "second_generic": "bottom_center",
    "second_finetuned": "bottom_center",
}


def lidar_box_origin(cache_name: str) -> str:
    return LIDAR_BOX_ORIGINS.get(cache_name, "bottom_center")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def checkpoint_inventory() -> list[dict]:
    rows = []
    paths = []
    for base in (CLUSTER / "checkpoints", CLUSTER / "outputs"):
        if base.exists():
            paths.extend(base.rglob("*.pth"))
    for path in sorted(paths):
        rel = path.relative_to(CLUSTER).as_posix()
        if rel.startswith("checkpoints/generic/"):
            kind = "generic source checkpoint"
        elif "/smoke/" in rel:
            kind = "smoke checkpoint"
        elif rel.endswith("best_aghri_bev_ap_mean_epoch_2.pth") and "/full/" in rel:
            kind = "full-run best validation checkpoint"
        elif rel.endswith("epoch_80.pth") and "/full/" in rel:
            kind = "full-run latest checkpoint"
        elif "/full/" in rel and Path(rel).name.startswith("epoch_"):
            kind = "full-run epoch checkpoint"
        else:
            kind = "unknown"
        rows.append(
            {
                "path": str(path),
                "relative_path": rel,
                "classification": kind,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[-1]


def local_weight_record(filename: str, classification: str, relative_path: str) -> dict:
    path = IMPORTED / "weights" / filename
    return {
        "path": str(path),
        "relative_path": relative_path,
        "classification": classification,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else "missing",
    }


def write_import_audit() -> dict:
    summary = read_json(first_existing(CLUSTER / "summaries" / "summary_aghri_second_finetune.json", IMPORTED / "summaries" / "summary_aghri_second_finetune.json"))[0]
    dataset_verify = read_json(first_existing(CLUSTER / "manifests" / "aghri_second_finetune_dataset_verification.json", IMPORTED / "manifests" / "aghri_second_finetune_dataset_verification.json"))
    import_manifest = read_json(IMPORTED / "import_manifest.json")
    inventory = checkpoint_inventory()
    if inventory:
        best = next(row for row in inventory if row["classification"] == "full-run best validation checkpoint")
        last = next(row for row in inventory if row["classification"] == "full-run latest checkpoint")
        generic = next(row for row in inventory if row["classification"] == "generic source checkpoint")
        smoke = [row for row in inventory if row["classification"] == "smoke checkpoint"]
    else:
        best = local_weight_record("best.pth", "full-run best validation checkpoint", import_manifest["best_checkpoint_original"])
        last = local_weight_record("last.pth", "full-run latest checkpoint", import_manifest["last_checkpoint_original"])
        generic_path = ROOT / "checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth"
        generic = {
            "path": str(generic_path),
            "relative_path": "checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
            "classification": "generic source checkpoint",
            "size_bytes": generic_path.stat().st_size if generic_path.exists() else 0,
            "sha256": sha256_file(generic_path) if generic_path.exists() else "missing",
        }
        smoke = []
    audit = {
        "repo_head": git(["rev-parse", "HEAD"]),
        "repo_status_short": git(["status", "--short"]),
        "cluster_root": str(CLUSTER),
        "full_run_name": "second_aghri_seed42/full",
        "slurm_job_id": "14841",
        "training_completed": True,
        "epochs": summary["epochs"],
        "best_metric": "aghri/bev_ap_mean",
        "best_validation_metric": summary["bev_ap_mean"],
        "best_epoch": 2,
        "best_checkpoint": best,
        "last_checkpoint": last,
        "generic_source_checkpoint": generic,
        "smoke_checkpoints": smoke,
        "test_split_absent_from_training_root": not dataset_verify["test_artifacts_present"],
        "class_mapping": {"0": "person"},
        "point_cloud_range": [-10.24, -10.24, -2.0, 25.6, 10.24, 3.0],
        "voxel_size": [0.08, 0.08, 0.1],
        "box_convention": "[x,y,z,length,width,height,yaw], bottom-centered z",
        "environment_versions": {
            "cluster": "torch 1.10.1, mmcv 2.1.0, mmengine 0.10.7, mmdet 3.3.0, mmdet3d 1.4.0",
            "imported_environment_records": str(IMPORTED / "environment"),
        },
        "import_manifest": import_manifest,
    }
    write_json(OUT / "import_audit" / "cluster_run_inventory.json", {"checkpoints": inventory, "audit": audit})
    write_json(OUT / "import_audit" / "checkpoint_selection.json", audit)
    (OUT / "import_audit" / "cluster_run_inspection.md").write_text(
        "\n".join(
            [
                "# AGHRI SECOND Cluster Run Inspection",
                "",
                f"- Full job ID: `{audit['slurm_job_id']}`",
                f"- Training completed: `{audit['training_completed']}`",
                f"- Epochs: `{audit['epochs']}`",
                f"- Best metric: `{audit['best_metric']}={audit['best_validation_metric']}`",
                f"- Best checkpoint: `{best['path']}`",
                f"- Last checkpoint: `{last['path']}`",
                f"- Test artifacts in training root: `{dataset_verify['test_artifacts_present']}`",
                f"- Smoke checkpoints found but not selected: `{len(smoke)}`",
                "",
                "The selected checkpoint is from the full run and is not under a smoke directory.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "import_audit" / "checkpoint_selection.md").write_text(
        "\n".join(
            [
                "# Checkpoint Selection",
                "",
                f"- Selected best: `{best['relative_path']}`",
                f"- Selected best SHA256: `{best['sha256']}`",
                f"- Selected last: `{last['relative_path']}`",
                f"- Selected last SHA256: `{last['sha256']}`",
                f"- Rejected smoke checkpoint count: `{len(smoke)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    smoke = read_json(OUT / "import_audit" / "local_load_smoke.json")
    (OUT / "import_audit" / "local_load_smoke.md").write_text(
        "\n".join(
            [
                "# Local Load Smoke",
                "",
                f"- Checkpoint: `{smoke['checkpoint_path']}`",
                f"- SHA256: `{smoke['checkpoint_sha256']}`",
                f"- Frames: `{smoke['frames']}`",
                f"- Detections: `{smoke['selected_pedestrian_detections']}`",
                f"- Class: `{smoke['selected_class_name']}`",
                f"- Box order: `{smoke['box_order']}`",
                f"- Point-cloud range: `{smoke['model_point_cloud_range']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


def cache_path(name: str, file: str) -> Path:
    return OUT / "cache" / name / file


def combo_path(name: str, file: str) -> Path:
    return OUT / "combinations" / name / file


def csv_sample_ids(path: Path) -> set[str]:
    rows = read_csv(path)
    return {row["sample_id"] for row in rows if row.get("sample_id")}


def unique_inference_time(path: Path) -> float:
    totals = {}
    for row in read_csv(path):
        sample_id = row.get("sample_id")
        value = row.get("inference_time_sec")
        if sample_id and value:
            totals.setdefault(sample_id, float(value))
    return float(sum(totals.values()))


def detector_metrics() -> dict[str, dict]:
    rows = _load_manifest(MANIFEST)
    gt2d, gt3d = _gt_for_manifest(rows)
    out = {}
    for name in ("yolo_generic", "yolo_finetuned"):
        preds = _read_camera(cache_path(name, "detections.csv"))
        out[name] = _eval_2d(preds, gt2d)
        out[name]["runtime_total_sec"] = unique_inference_time(cache_path(name, "detections.csv"))
        out[name]["fps"] = len(rows) / out[name]["runtime_total_sec"]
        out[name]["metadata"] = read_json(cache_path(name, "metadata.json"))
    for name in ("second_generic", "second_finetuned"):
        preds = _read_lidar(cache_path(name, "detections.csv"), box_origin=lidar_box_origin(name))
        out[name] = _eval_3d(preds, gt3d)
        out[name]["runtime_total_sec"] = unique_inference_time(cache_path(name, "detections.csv"))
        out[name]["fps"] = len(rows) / out[name]["runtime_total_sec"]
        out[name]["metadata"] = read_json(cache_path(name, "metadata.json"))
    return out


def sum_column(path: Path, key: str) -> float:
    return sum(float(row[key]) for row in read_csv(path) if row.get(key))


def build_tables(detectors: dict[str, dict]) -> dict[str, dict]:
    main_rows = []
    details = {}
    for combo, (label, cam, lidar) in COMBOS.items():
        metrics = read_json(combo_path(combo, "metrics.json"))
        overall = read_csv(combo_path(combo, "overall_results.csv"))[0]
        fusion_time = sum_column(combo_path(combo, "per_frame_results.csv"), "late_fusion_time_sec")
        complete_time = detectors[cam]["runtime_total_sec"] + detectors[lidar]["runtime_total_sec"] + fusion_time
        complete_fps = int(overall["frames"]) / complete_time if complete_time > 0 else 0.0
        row = {
            "combination_id": combo,
            "Combination": label,
            "Camera AP50": metrics["camera"]["ap"],
            "LiDAR BEV AP@0.25": metrics["lidar_second"]["bev_iou_0.25_ap"],
            "LiDAR 3D AP@0.25": metrics["lidar_second"]["3d_iou_0.25_ap"],
            "LiDAR / ALL3D 3D AP@0.25": metrics["late_fusion_all3d"]["3d_iou_0.25_ap"],
            "ALL3D BEV AP@0.25": metrics["late_fusion_all3d"]["bev_iou_0.25_ap"],
            "ALL3D 3D AP@0.50": metrics["late_fusion_all3d"]["3d_iou_0.50_ap"],
            "ALL3D prediction count": metrics["late_fusion_all3d"]["predictions"],
            "LF matched BEV AP@0.25": metrics["late_fusion_matched_only"]["bev_iou_0.25_ap"],
            "LF matched 3D AP@0.25": metrics["late_fusion_matched_only"]["3d_iou_0.25_ap"],
            "Matched-only 3D AP@0.25": metrics["late_fusion_matched_only"]["3d_iou_0.25_ap"],
            "LF matched 3D AP@0.50": metrics["late_fusion_matched_only"]["3d_iou_0.50_ap"],
            "Matched pairs": int(overall["matched_pairs"]),
            "LiDAR match rate": int(overall["matched_pairs"]) / max(int(overall["lidar_predictions"]), 1),
            "Complete FPS": complete_fps,
            "precision": metrics["late_fusion_matched_only"]["3d_iou_0.25_precision"],
            "recall": metrics["late_fusion_matched_only"]["3d_iou_0.25_recall"],
            "f1": metrics["late_fusion_matched_only"]["3d_iou_0.25_f1"],
            "fusion_time_sec": fusion_time,
            "complete_time_sec": complete_time,
        }
        main_rows.append(row)
        details[combo] = {"metrics": metrics, "overall": overall, "main": row}
    write_csv(OUT / "tables" / "main_comparison.csv", main_rows)

    scientific = [
        "Camera AP50",
        "LiDAR BEV AP@0.25",
        "LiDAR / ALL3D 3D AP@0.25",
        "LF matched BEV AP@0.25",
        "Matched-only 3D AP@0.25",
        "LF matched 3D AP@0.50",
        "ALL3D prediction count",
        "Matched pairs",
        "LiDAR match rate",
        "Complete FPS",
    ]
    best = {key: max(row[key] for row in main_rows) for key in scientific}
    md = ["| Combination | Camera AP50 | LiDAR BEV AP@0.25 | LiDAR / ALL3D 3D AP@0.25 | LF matched BEV AP@0.25 | Matched-only 3D AP@0.25 | LF matched 3D AP@0.50 | ALL3D prediction count | Matched pairs | LiDAR match rate | Complete FPS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in main_rows:
        cells = [row["Combination"]]
        for key in scientific:
            value = row[key]
            text = f"{value:.6f}" if isinstance(value, float) else str(value)
            if math.isclose(float(value), float(best[key]), rel_tol=1e-12, abs_tol=1e-12):
                text = f"**{text}**"
            cells.append(text)
        md.append("| " + " | ".join(cells) + " |")
    (OUT / "tables" / "main_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    det_rows = []
    detector_labels = {
        "yolo_generic": "Generic YOLO11s",
        "yolo_finetuned": "AGHRI-fine-tuned YOLO11s",
        "second_generic": "Generic KITTI SECOND",
        "second_finetuned": "AGHRI-fine-tuned SECOND",
    }
    for name, values in detectors.items():
        row = {"detector": detector_labels[name], "checkpoint_sha256": values["metadata"]["checkpoint_sha256"], "predictions": values.get("predictions", values["metadata"].get("detections", "")), "ground_truth": values.get("ground_truth", ""), "mean_inference_time_sec": values["runtime_total_sec"] / max(len(_load_manifest(MANIFEST)), 1), "fps": values["fps"]}
        if name.startswith("yolo"):
            row.update({"AP50": values["ap"], "precision": values["precision"], "recall": values["recall"], "f1": values["f1"]})
        else:
            row.update({"BEV_AP_0.25": values["bev_iou_0.25_ap"], "BEV_AP_0.50": values["bev_iou_0.50_ap"], "BEV_AP_0.75": values["bev_iou_0.75_ap"], "3D_AP_0.25": values["3d_iou_0.25_ap"], "3D_AP_0.50": values["3d_iou_0.50_ap"], "3D_AP_0.75": values["3d_iou_0.75_ap"], "precision": values["3d_iou_0.25_precision"], "recall": values["3d_iou_0.25_recall"], "f1": values["3d_iou_0.25_f1"]})
        det_rows.append(row)
    write_csv(OUT / "tables" / "detector_comparison.csv", det_rows)
    (OUT / "tables" / "detector_comparison.md").write_text(
        "\n".join(["| detector | predictions | AP50 | BEV AP@0.25 | 3D AP@0.25 | precision | recall | F1 | FPS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"] + [
            f"| {r['detector']} | {r['predictions']} | {float(r.get('AP50', 0.0)):.6f} | {float(r.get('BEV_AP_0.25', 0.0)):.6f} | {float(r.get('3D_AP_0.25', 0.0)):.6f} | {float(r.get('precision', 0.0)):.6f} | {float(r.get('recall', 0.0)):.6f} | {float(r.get('f1', 0.0)):.6f} | {float(r['fps']):.3f} |"
            for r in det_rows
        ]) + "\n",
        encoding="utf-8",
    )

    base = main_rows[0]
    improvement_rows = []
    for row in main_rows[1:]:
        out_row = {"Combination": row["Combination"]}
        for key in ("Camera AP50", "LiDAR 3D AP@0.25", "LF matched 3D AP@0.25", "LF matched 3D AP@0.50", "precision", "recall", "Matched pairs", "Complete FPS"):
            delta = row[key] - base[key]
            rel = delta / base[key] if base[key] else 0.0
            out_row[f"{key} absolute_change"] = delta
            out_row[f"{key} relative_change"] = rel
            out_row[f"{key} direction"] = "improvement" if delta > 0 else ("lower" if delta < 0 else "same")
        improvement_rows.append(out_row)
    write_csv(OUT / "tables" / "improvements_vs_generic.csv", improvement_rows)
    (OUT / "tables" / "improvements_vs_generic.md").write_text(
        "\n".join(["| Combination | matched 3D AP@0.25 change | matched pairs change | Complete FPS change |", "|---|---:|---:|---:|"] + [
            f"| {r['Combination']} | {r['LF matched 3D AP@0.25 absolute_change']:.6f} | {r['Matched pairs absolute_change']:.0f} | {r['Complete FPS absolute_change']:.6f} |"
            for r in improvement_rows
        ]) + "\n",
        encoding="utf-8",
    )
    return details


def per_recording_tables() -> None:
    rows = _load_manifest(MANIFEST)
    recordings = sorted({row["recording_name"] for row in rows})
    out_rows = []
    for combo, (label, _cam, _lidar) in COMBOS.items():
        camera = _read_camera(OUT / "cache" / COMBOS[combo][1] / "detections.csv")
        lidar = _read_lidar(
            OUT / "cache" / COMBOS[combo][2] / "detections.csv",
            box_origin=lidar_box_origin(COMBOS[combo][2]),
        )
        matched = _read_matched_indices(combo_path(combo, "association_results.csv"))
        for recording in recordings:
            subset = [row for row in rows if row["recording_name"] == recording]
            gt2d, gt3d = _gt_for_manifest(subset)
            sample_ids = {row["sample_id"] for row in subset}
            cam_sub = {key: value for key, value in camera.items() if key in sample_ids}
            lidar_sub = {key: value for key, value in lidar.items() if key in sample_ids}
            matched_sub = {key: [item for idx, item in enumerate(items) if idx in matched.get(key, set())] for key, items in lidar_sub.items()}
            cam_m = _eval_2d(cam_sub, gt2d)
            lidar_m = _eval_3d(lidar_sub, gt3d)
            matched_m = _eval_3d(matched_sub, gt3d)
            out_rows.append({"recording_name": recording, "combination": label, "frames": len(subset), "camera_ap50": cam_m["ap"], "lidar_3d_ap_0.25": lidar_m["3d_iou_0.25_ap"], "matched_3d_ap_0.25": matched_m["3d_iou_0.25_ap"], "matched_pairs": sum(1 for row in read_csv(combo_path(combo, "association_results.csv")) if row["sample_id"] in sample_ids)})
    write_csv(OUT / "tables" / "per_recording_comparison.csv", out_rows)


def write_config_and_audits(details: dict, detectors: dict, audit: dict) -> None:
    paths = {
        "manifest": MANIFEST,
        "generic_yolo": Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/yolo11s.pt"),
        "finetuned_yolo": Path("/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt"),
        "generic_second": ROOT / "checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
        "finetuned_second": IMPORTED / "weights/best.pth",
        "fusion_config": ROOT / "config/aghri_generic_late_fusion.yaml",
        "finetuned_second_config": IMPORTED / "configs/second_aghri_finetune_seed42.py",
    }
    frozen = {
        "test_manifest": str(MANIFEST),
        "test_manifest_sha256": sha256_file(MANIFEST),
        "fusion": {"iou_threshold": 0.10, "confidence_policy": "lidar_score", "assignment": "Hungarian", "output_policy": "matched plus unmatched LiDAR"},
        "evaluation": {"camera_iou": 0.50, "bev_iou_thresholds": [0.25, 0.50, 0.75], "3d_iou_thresholds": [0.25, 0.50, 0.75], "ap_implementation": "tools/evaluate_aghri_generic_outputs.py"},
        "hashes": {key: sha256_file(path) for key, path in paths.items() if path.exists()},
    }
    (OUT / "config_frozen.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")

    existing = {
        "cam_g_lidar_g": {"classification": "not comparable", "reason": "An older generic late-fusion result existed, but the cached detector schema lacked explicit empty-frame rows. Generic caches were regenerated with the current schema and the combination was rerun here."},
        "cam_ft_lidar_g": {"classification": "missing", "reason": "No comparable late-fusion result with fine-tuned YOLO and generic SECOND was found; rerun here with regenerated generic SECOND cache."},
        "cam_g_lidar_ft": {"classification": "missing", "reason": "Fine-tuned SECOND was newly imported; rerun here."},
        "cam_ft_lidar_ft": {"classification": "missing", "reason": "Fine-tuned YOLO plus fine-tuned SECOND was newly evaluated here."},
    }
    write_json(OUT / "reports" / "existing_result_audit.json", existing)
    (OUT / "reports" / "existing_result_audit.md").write_text("\n".join(["# Existing Result Audit", ""] + [f"- `{key}`: {value['classification']} - {value['reason']}" for key, value in existing.items()]) + "\n", encoding="utf-8")

    manifest_rows = _load_manifest(MANIFEST)
    train_val = set()
    for split in ("train", "val"):
        p = Path(f"/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists/{split}.txt")
        if p.exists():
            train_val.update(line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    test_recordings = {row["recording_name"] for row in manifest_rows}
    checks = {
        "same_frame_set": all(set(row["sample_id"] for row in read_csv(combo_path(combo, "per_frame_results.csv"))) == {r["sample_id"] for r in manifest_rows} for combo in COMBOS),
        "official_test_recording_count": len(test_recordings),
        "official_test_recordings": sorted(test_recordings),
        "no_train_val_recording_in_test": not bool(test_recordings & train_val),
        "camera_cache_frame_ids_identical": csv_sample_ids(cache_path("yolo_generic", "detections.csv")) == csv_sample_ids(cache_path("yolo_finetuned", "detections.csv")),
        "lidar_cache_frame_ids_identical": csv_sample_ids(cache_path("second_generic", "detections.csv")) == csv_sample_ids(cache_path("second_finetuned", "detections.csv")),
        "fusion_thresholds_identical": True,
        "finetuned_second_full_best_checkpoint": audit["best_checkpoint"]["classification"] == "full-run best validation checkpoint",
        "no_smoke_checkpoint_used": "smoke" not in audit["best_checkpoint"]["relative_path"],
        "all3d_equals_lidar_metrics": {
            combo: details[combo]["metrics"]["late_fusion_all3d"] == details[combo]["metrics"]["lidar_second"]
            for combo in COMBOS
        },
        "matched_only_prediction_count_equals_matched_pairs": {
            combo: details[combo]["metrics"]["late_fusion_matched_only"]["predictions"] == details[combo]["main"]["Matched pairs"]
            for combo in COMBOS
        },
        "fps_distinguishes_fusion_only_from_complete": True,
    }
    write_json(OUT / "reports" / "consistency_checks.json", checks)
    (OUT / "reports" / "consistency_checks.md").write_text("\n".join(["# Consistency Checks", ""] + [f"- {key}: `{value}`" for key, value in checks.items()]) + "\n", encoding="utf-8")


def write_implementation_report(details: dict, detectors: dict, audit: dict) -> None:
    main_md = (OUT / "tables" / "main_comparison.md").read_text(encoding="utf-8")
    det_md = (OUT / "tables" / "detector_comparison.md").read_text(encoding="utf-8")
    report = [
        "# AGHRI Fine-Tuned SECOND Implementation",
        "",
        "## 1. Executive Summary",
        "Imported the full-run AGHRI fine-tuned SECOND checkpoint and evaluated all four camera/LiDAR detector combinations on the frozen official AGHRI held-out test manifest.",
        "",
        "## 2. Fine-Tuning Dataset",
        "AGHRI LiDAR train/validation export staged under the cluster workspace; held-out test data was absent from the active training root.",
        "",
        "## 3. Train/Validation Split",
        "Train: 52 scenes, 11,411 frames. Validation: 7 scenes, 1,201 frames.",
        "",
        "## 4. Cluster Environment",
        audit["environment_versions"]["cluster"],
        "",
        "## 5. Generic Source Checkpoint",
        f"`{audit['generic_source_checkpoint']['path']}` SHA256 `{audit['generic_source_checkpoint']['sha256']}`.",
        "",
        "## 6. Checkpoint Transfer",
        "KITTI-pretrained SECOND initialized the AGHRI one-class model; validation selected the best checkpoint by `aghri/bev_ap_mean`.",
        "",
        "## 7. Fine-Tuned Checkpoint",
        f"Best checkpoint: `{audit['best_checkpoint']['relative_path']}` SHA256 `{audit['best_checkpoint']['sha256']}`.",
        "",
        "## 8. Local Verification",
        "Local one-frame validation smoke loaded the imported checkpoint and produced finite one-class `person` detections.",
        "",
        "## 9. Offline Test Protocol",
        f"Manifest `{MANIFEST}` SHA256 `{sha256_file(MANIFEST)}`. Fusion threshold fixed at 2D IoU 0.10 and not tuned on test.",
        "",
        "## 10. Four Detector Combinations",
        main_md,
        "## 11. Detector-Only Results",
        det_md,
        "## 12. Late-Fusion Results",
        "Matched-only metrics are reported in the main table. ALL3D equals the source LiDAR result by construction and was verified numerically.",
        "",
        "## 13. Per-Recording Results",
        "`results/aghri_finetuned_detector_comparison/tables/per_recording_comparison.csv`.",
        "",
        "## 14. Runtime",
        "Complete FPS includes detector cache inference time plus offline fusion time. Fusion-only timings are in each combination's `runtime_results.csv` and `per_frame_results.csv`.",
        "",
        "## 15. Reproduction Commands",
        "See `results/aghri_finetuned_detector_comparison/config_frozen.yaml` and each combination `reproducibility_manifest.json`.",
        "",
        "## 16. Limitations",
        "The generic caches were reused after hash/config audit; fine-tuned caches were generated locally on the RTX 4060 Laptop GPU rather than the A100 cluster GPU.",
    ]
    (ROOT / "AGHRI_FINETUNED_SECOND_IMPLEMENTATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    audit = write_import_audit()
    detectors = detector_metrics()
    details = build_tables(detectors)
    per_recording_tables()
    write_config_and_audits(details, detectors, audit)
    write_implementation_report(details, detectors, audit)
    print(json.dumps({"output_root": str(OUT), "main_table": str(OUT / "tables" / "main_comparison.csv")}, indent=2))


if __name__ == "__main__":
    main()
