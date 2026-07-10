#!/usr/bin/env python3
"""Run and summarise the held-out AGHRI detector/fusion 2x2 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.evaluate_aghri_association as eval_assoc  # noqa: E402

OUTPUT_ROOT = Path("/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2")
DATASET_ROOT = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
GENERIC_CKPT = REPO_ROOT / "checkpoints/generic/yolo11s.pt"
FINETUNED_CKPT = (
    REPO_ROOT / "checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt"
)
GENERIC_SHA = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
FINETUNED_SHA = "bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110"
BOOTSTRAP_SEED = 20260705


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str], log_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(OUTPUT_ROOT / "metadata" / "ultralytics_config"))
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "duration_sec": time.time() - started,
    }
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8")
        result["log_path"] = str(log_path)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with {proc.returncode}: {' '.join(cmd)}\n{proc.stdout[-4000:]}")
    return result


def repo_state(path: Path) -> dict[str, Any]:
    branch = subprocess.check_output(
        ["git", "-C", str(path), "branch", "--show-current"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(path), "status", "--short"],
        text=True,
    )
    diff_stat = subprocess.check_output(
        ["git", "-C", str(path), "diff", "--stat"],
        text=True,
    )
    return {"path": str(path), "branch": branch, "status_short": status, "diff_stat": diff_stat}


def environment_metadata() -> dict[str, Any]:
    import ultralytics

    disk = subprocess.check_output(["df", "-h", str(OUTPUT_ROOT.parent)], text=True)
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "disk_free": disk.strip().splitlines(),
    }


def checkpoint_metadata(path: Path, expected_sha: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"Unexpected checkpoint SHA for {path}: {actual}")
    model = YOLO(str(path))
    return {
        "path": str(path),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "model_task": getattr(model, "task", None),
        "model_class": type(model.model).__name__,
        "names": {str(k): v for k, v in model.names.items()},
    }


def smoke_inference(image_path: Path, checkpoints: dict[str, Path]) -> dict[str, Any]:
    out = {"image_path": str(image_path)}
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read smoke image: {image_path}")
    for name, ckpt in checkpoints.items():
        model = YOLO(str(ckpt))
        result = model(image, conf=0.1, classes=[0], imgsz=640, verbose=False)[0]
        boxes = result.boxes.data.detach().cpu().numpy()
        out[name] = {
            "box_shape": list(boxes.shape),
            "class_ids": sorted({int(row[5]) for row in boxes}) if len(boxes) else [],
            "names": {str(k): v for k, v in model.names.items()},
        }
    return out


def first_validation_image() -> Path:
    root = Path("/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/images/val")
    images = sorted(
        path for path in root.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.exists()
    )
    if not images:
        raise FileNotFoundError(f"No validation images under {root}")
    return images[0]


def evaluator_cmd(run_name: str, model: Path, cache: Path, association: str) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_aghri_association.py"),
        "--manifest",
        str(OUTPUT_ROOT / "manifest" / "test_manifest.csv"),
        "--output-dir",
        str(OUTPUT_ROOT),
        "--cache-dir",
        str(cache),
        "--run-name",
        run_name,
        "--frame-limit",
        "0",
        "--image-slop-sec",
        "0.10",
        "--yolo-model",
        str(model),
        "--yolo-threshold",
        "0.1",
        "--yolo-classes",
        "0",
        "--yolo-imgsz",
        "640",
        "--association-method",
        association,
        "--centre-statistic",
        "mean",
        "--depth-cluster-gap-m",
        "0.35",
        "--depth-cluster-min-points",
        "5",
        "--box-bottom-trim-ratio",
        "0.0",
        "--hybrid-min-cluster-points",
        "5",
        "--hybrid-min-support-ratio",
        "0.20",
        "--hybrid-max-depth-spread-m",
        "0.45",
        "--hybrid-min-cluster-separation-m",
        "0.15",
        "--no-hybrid-fallback-on-single-cluster",
        "--hybrid-fallback-on-sparse-candidates",
        "--visual-limit",
        "18",
    ]
    return cmd


def build_manifest() -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "evaluate_aghri_association.py"),
        "--build-test-manifest",
        "--dataset-root",
        str(DATASET_ROOT),
        "--output-dir",
        str(OUTPUT_ROOT / "manifest"),
        "--image-slop-sec",
        "0.10",
    ]
    result = run(cmd, OUTPUT_ROOT / "logs" / "build_manifest.log")
    csv_path = OUTPUT_ROOT / "manifest" / "test_manifest.csv"
    json_path = OUTPUT_ROOT / "manifest" / "test_manifest.json"
    (OUTPUT_ROOT / "metadata" / "test_manifest_sha256.txt").write_text(
        f"{sha256_file(csv_path)}  {csv_path}\n{sha256_file(json_path)}  {json_path}\n",
        encoding="utf-8",
    )
    manifest = load_json(json_path)
    return {
        "command": result,
        "manifest_count": len(manifest),
        "supported_count": sum(1 for row in manifest if row.get("supported")),
        "train_val_leakage_count": sum(1 for row in manifest if row.get("train_val_leakage")),
        "expected_synchronised_frame_count": sum(
            int(row.get("expected_synchronised_frame_count") or 0)
            for row in manifest if row.get("supported")
        ),
        "recordings": [row["recording_name"] for row in manifest],
    }


def cache_summary(cache_dir: Path) -> dict[str, Any]:
    meta = cache_dir / "yolo_metadata.json"
    npys = sorted(cache_dir.glob("*.npy"))
    jsons = sorted(path for path in cache_dir.glob("*.json") if path.name != "yolo_metadata.json")
    detections = 0
    for path in npys:
        detections += int(np.load(path).shape[0])
    return {
        "cache_dir": str(cache_dir),
        "frame_cache_files": len(npys),
        "metadata_files": len(jsons),
        "total_cached_detections": detections,
        "metadata": load_json(meta) if meta.exists() else None,
        "cache_listing_sha256": hashlib.sha256(
            "\n".join(path.name for path in npys + jsons).encode("utf-8")
        ).hexdigest(),
    }


def metric(summary: dict[str, Any], key: str) -> Any:
    return summary.get("overall", {}).get(key)


def error_gt_rate(summary: dict[str, Any], threshold_key: str) -> Any:
    le_rate = metric(summary, threshold_key)
    return None if le_rate is None else 1.0 - float(le_rate)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    if not fields:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overall_tables(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, item in runs.items():
        summary = item["summary"]
        frame = summary.get("frame_overall", {})
        rows.append({
            "run": name,
            "detector": item["detector"],
            "fusion_method": item["fusion_method"],
            "total_detections": metric(summary, "total_yolo_person_detections"),
            "gt_instances": frame.get("total_gt3d_instances"),
            "matched_detections": metric(summary, "matched_prediction_count"),
            "unmatched_detections": metric(summary, "unmatched_prediction_count"),
            "ambiguous_matches": metric(summary, "ambiguous_match_count"),
            "valid_fusion_rate": metric(summary, "valid_association_rate"),
            "no_point_rate": metric(summary, "no_point_rate"),
            "mean_3d_error_m": metric(summary, "mean_lidar_center_error"),
            "median_3d_error_m": metric(summary, "median_lidar_center_error"),
            "p75_3d_error_m": metric(summary, "p75_lidar_center_error"),
            "p90_3d_error_m": metric(summary, "p90_lidar_center_error"),
            "p95_3d_error_m": metric(summary, "p95_lidar_center_error"),
            "max_3d_error_m": metric(summary, "max_lidar_center_error"),
            "error_gt_0_5m_rate": error_gt_rate(summary, "lidar_error_le_0_5m_rate"),
            "error_gt_1_0m_rate": metric(summary, "lidar_error_gt_1_0m_rate"),
            "error_gt_2_0m_rate": metric(summary, "lidar_error_gt_2_0m_rate"),
            "mean_depth_error_m": metric(summary, "mean_camera_depth_error"),
            "median_depth_error_m": metric(summary, "median_camera_depth_error"),
            "mean_fusion_time_sec": metric(summary, "mean_association_time_sec"),
            "mean_pipeline_frame_time_sec": frame.get("mean_frame_time_sec"),
            "approx_pipeline_fps": frame.get("approx_fps"),
            "failures": len(summary.get("failures", [])),
        })
    return rows


def per_recording_tables(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, item in runs.items():
        for rec in item["summary"].get("recordings", []):
            rows.append({
                "run": name,
                "detector": item["detector"],
                "fusion_method": item["fusion_method"],
                "recording": rec["recording"],
                "scene": rec["scene"],
                "motion": rec["motion"],
                "usable_frames": rec["usable_frames"],
                "gt_instances": rec.get("frame_metrics", {}).get("total_gt3d_instances"),
                "total_detections": rec.get("metrics", {}).get("total_yolo_person_detections"),
                "matched_detections": rec.get("metrics", {}).get("matched_prediction_count"),
                "valid_fusion_rate": rec.get("metrics", {}).get("valid_association_rate"),
                "mean_3d_error_m": rec.get("metrics", {}).get("mean_lidar_center_error"),
                "median_3d_error_m": rec.get("metrics", {}).get("median_lidar_center_error"),
                "p90_3d_error_m": rec.get("metrics", {}).get("p90_lidar_center_error"),
                "error_gt_1_0m_rate": rec.get("metrics", {}).get("lidar_error_gt_1_0m_rate"),
                "approx_pipeline_fps": rec.get("frame_metrics", {}).get("approx_fps"),
            })
    return rows


def gt_rows_from_manifest() -> list[dict[str, Any]]:
    rows = []
    for recording, split in eval_assoc.read_manifest(OUTPUT_ROOT / "manifest" / "test_manifest.csv"):
        pairs = eval_assoc.useful_pairs(recording, 0, 0.10)
        for frame_number, (lidar_record, image_record, image_index, image_delta) in enumerate(pairs):
            gt3d = eval_assoc.labels_to_gt3d(lidar_record)
            gt2d = {item["id"]: item for item in eval_assoc.labels_to_gt2d(image_record)}
            for gt in gt3d:
                gt2 = gt2d.get(gt["id"])
                rows.append({
                    "recording": eval_assoc.sequence_name(recording),
                    "split": split,
                    "frame_number": frame_number,
                    "camera_frame_index": image_index,
                    "image_file": image_record["File"],
                    "lidar_file": lidar_record["File"],
                    "image_timestamp": eval_assoc.timestamp(image_record),
                    "lidar_timestamp": eval_assoc.timestamp(lidar_record),
                    "image_lidar_delta_sec": image_delta,
                    "gt_identity": gt["id"],
                    "gt_label_index": gt["label_index"],
                    "gt_lidar_xyz": gt["center_lidar"].astype(float).tolist(),
                    "gt_2d_box": None if gt2 is None else gt2["box"].astype(float).tolist(),
                    "recording_path": str(recording),
                })
    return rows


def best_detection_for_gt(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        gt_id = row.get("matched_gt_identity")
        if gt_id is None:
            continue
        key = (row["recording"], int(row["frame_number"]), str(gt_id))
        grouped.setdefault(key, []).append(row)
    best = {}
    for key, matches in grouped.items():
        valid = [row for row in matches if row.get("valid") and row.get("errors", {}).get("lidar_center_error") is not None]
        candidates = valid or matches
        candidates.sort(
            key=lambda row: (
                row.get("errors", {}).get("lidar_center_error", math.inf),
                -float(row.get("confidence") or 0.0),
            )
        )
        chosen = candidates[0]
        chosen = dict(chosen)
        chosen["duplicate_detection_count"] = max(0, len(matches) - 1)
        best[key] = chosen
    return best


def detector_comparison(
    gt_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
    finetuned_rows: list[dict[str, Any]],
    method: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generic = best_detection_for_gt(generic_rows)
    finetuned = best_detection_for_gt(finetuned_rows)
    out = []
    diffs = []
    per_recording: dict[str, list[float]] = {}
    for gt in gt_rows:
        key = (gt["recording"], int(gt["frame_number"]), str(gt["gt_identity"]))
        g = generic.get(key)
        f = finetuned.get(key)
        g_err = None if g is None else g.get("errors", {}).get("lidar_center_error")
        f_err = None if f is None else f.get("errors", {}).get("lidar_center_error")
        if g_err is not None and f_err is not None:
            diff = float(f_err) - float(g_err)
            diffs.append(diff)
            per_recording.setdefault(gt["recording"], []).append(diff)
        else:
            diff = None
        out.append({
            **gt,
            "method": method,
            "generic_detected": g is not None,
            "finetuned_detected": f is not None,
            "generic_confidence": None if g is None else g.get("confidence"),
            "finetuned_confidence": None if f is None else f.get("confidence"),
            "generic_duplicate_count": 0 if g is None else g.get("duplicate_detection_count", 0),
            "finetuned_duplicate_count": 0 if f is None else f.get("duplicate_detection_count", 0),
            "generic_fusion_valid": False if g is None else bool(g.get("valid")),
            "finetuned_fusion_valid": False if f is None else bool(f.get("valid")),
            "generic_3d_error_m": g_err,
            "finetuned_3d_error_m": f_err,
            "finetuned_minus_generic_error_m": diff,
            "generic_failure_reason": "missed_person" if g is None else g.get("validity_failure_reason"),
            "finetuned_failure_reason": "missed_person" if f is None else f.get("validity_failure_reason"),
        })
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = []
    for _ in range(2000 if len(diffs_arr) else 0):
        boot.append(float(np.mean(rng.choice(diffs_arr, size=len(diffs_arr), replace=True))))
    summary = {
        "method": method,
        "gt_instances": len(gt_rows),
        "detected_by_both": sum(1 for row in out if row["generic_detected"] and row["finetuned_detected"]),
        "generic_only": sum(1 for row in out if row["generic_detected"] and not row["finetuned_detected"]),
        "finetuned_only": sum(1 for row in out if row["finetuned_detected"] and not row["generic_detected"]),
        "missed_by_both": sum(1 for row in out if not row["generic_detected"] and not row["finetuned_detected"]),
        "fusion_valid_both": sum(1 for row in out if row["generic_fusion_valid"] and row["finetuned_fusion_valid"]),
        "generic_only_valid": sum(1 for row in out if row["generic_fusion_valid"] and not row["finetuned_fusion_valid"]),
        "finetuned_only_valid": sum(1 for row in out if row["finetuned_fusion_valid"] and not row["generic_fusion_valid"]),
        "joint_valid_error_pairs": int(len(diffs_arr)),
        "mean_finetuned_minus_generic_error_m": float(np.mean(diffs_arr)) if len(diffs_arr) else None,
        "median_finetuned_minus_generic_error_m": float(np.median(diffs_arr)) if len(diffs_arr) else None,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "GT instance",
        "mean_difference_bootstrap_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ] if boot else None,
        "per_recording_mean_differences": {
            key: float(np.mean(vals)) for key, vals in sorted(per_recording.items())
        },
    }
    return out, summary


def failure_taxonomy(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run_name, item in runs.items():
        dets = item["detections"]
        categories = {
            "no_projected_lidar_points": sum(1 for row in dets if row.get("status") == "no_points"),
            "too_few_valid_points": sum(1 for row in dets if row.get("status") == "no_valid_cluster"),
            "unmatched_detection_false_positive_candidate": sum(1 for row in dets if row.get("status") == "unmatched"),
            "ambiguous_gt_match": sum(1 for row in dets if row.get("ambiguous")),
            "large_3d_error_gt_1m": sum(
                1 for row in dets
                if row.get("errors", {}).get("lidar_center_error") is not None
                and float(row["errors"]["lidar_center_error"]) > 1.0
            ),
            "large_3d_error_gt_2m": sum(
                1 for row in dets
                if row.get("errors", {}).get("lidar_center_error") is not None
                and float(row["errors"]["lidar_center_error"]) > 2.0
            ),
            "hybrid_fallback": sum(1 for row in dets if row.get("fallback_occurred")),
        }
        for category, count in categories.items():
            rows.append({
                "run": run_name,
                "detector": item["detector"],
                "fusion_method": item["fusion_method"],
                "failure_type": category,
                "count": count,
            })
    return rows


def condition_breakdowns(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    keys = [
        "mean_lidar_error_by_distance_band",
        "mean_lidar_error_by_sparsity_band",
        "mean_lidar_error_by_image_border_status",
        "mean_lidar_error_by_scene_person_count_band",
        "mean_lidar_error_by_scene",
        "mean_lidar_error_by_motion",
        "mean_lidar_error_by_occlusion_category",
    ]
    for run_name, item in runs.items():
        overall = item["summary"].get("overall", {})
        for key in keys:
            for group, value in (overall.get(key) or {}).items():
                rows.append({
                    "run": run_name,
                    "detector": item["detector"],
                    "fusion_method": item["fusion_method"],
                    "breakdown": key,
                    "group": group,
                    "mean_3d_error_m": value,
                })
    return rows


def image_lookup() -> dict[tuple[str, int], Path]:
    lookup = {}
    for recording, _split in eval_assoc.read_manifest(OUTPUT_ROOT / "manifest" / "test_manifest.csv"):
        for frame_number, (_lidar, image_record, _idx, _delta) in enumerate(eval_assoc.useful_pairs(recording, 0, 0.10)):
            lookup[(eval_assoc.sequence_name(recording), frame_number)] = (
                recording / "sensor_data" / "cam_zed_rgb" / image_record["File"]
            )
    return lookup


def visual_audit(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = image_lookup()
    audit_rows = []
    for run_name, item in runs.items():
        candidates = [
            row for row in item["detections"]
            if row.get("errors", {}).get("lidar_center_error") is not None
        ]
        candidates.sort(key=lambda row: float(row["errors"]["lidar_center_error"]), reverse=True)
        selected = [(row, "top_largest_3d_error") for row in candidates[:5]]
        selected += [
            (row, "representative_no_point")
            for row in item["detections"]
            if row.get("status") == "no_points"
        ][:3]
        for idx, (row, reason) in enumerate(selected):
            image_path = lookup.get((row["recording"], int(row["frame_number"])))
            if image_path is None:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            box = [int(round(v)) for v in row["box"]]
            abox = [int(round(v)) for v in row["association_box"]]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.rectangle(image, (abox[0], abox[1]), (abox[2], abox[3]), (0, 255, 255), 1)
            label = (
                f"{run_name} {reason} conf={row.get('confidence', 0):.2f} "
                f"err={row.get('errors', {}).get('lidar_center_error')}"
            )
            cv2.putText(image, label[:120], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            out_path = OUTPUT_ROOT / "visuals" / "audit" / run_name / f"{idx:02d}_{reason}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image)
            audit_rows.append({
                "run": run_name,
                "reason": reason,
                "recording": row["recording"],
                "frame_number": row["frame_number"],
                "detection_index": row["detection_index"],
                "error_m": row.get("errors", {}).get("lidar_center_error"),
                "image": str(out_path),
            })
    return audit_rows


def write_report(
    metadata: dict[str, Any],
    overall: list[dict[str, Any]],
    pairwise: dict[str, Any],
    detector: dict[str, Any],
) -> None:
    def row_for(run_name: str) -> dict[str, Any]:
        return next(row for row in overall if row["run"] == run_name)

    gl = row_for("generic_legacy")
    fl = row_for("finetuned_legacy")
    gh = row_for("generic_hybrid")
    fh = row_for("finetuned_hybrid")
    lines = [
        "# AGHRI YOLO11s Fusion 2x2 Held-Out Test Benchmark",
        "",
        "## 1. Executive Conclusion",
        (
            "The held-out test benchmark completed all four frozen combinations. "
            "The primary comparison is generic YOLO11s + legacy versus AGHRI-fine-tuned YOLO11s + legacy."
        ),
        "",
        "## 2. Checkpoints",
        f"- Generic SHA256: `{metadata['checkpoints']['generic']['sha256']}`",
        f"- Fine-tuned SHA256: `{metadata['checkpoints']['finetuned']['sha256']}`",
        "",
        "## 3. Test Manifest",
        (
            f"- Recordings: {metadata['manifest']['manifest_count']}; "
            f"supported: {metadata['manifest']['supported_count']}; "
            f"frames: {metadata['manifest']['expected_synchronised_frame_count']}; "
            f"train/val leakage: {metadata['manifest']['train_val_leakage_count']}"
        ),
        "",
        "## 4. Overall 2x2 Results",
        "| Run | Detections | Matched | Valid rate | Mean 3D m | Median 3D m | P95 3D m | >1m rate | FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall:
        lines.append(
            f"| {row['run']} | {row['total_detections']} | {row['matched_detections']} | "
            f"{row['valid_fusion_rate']:.4f} | {row['mean_3d_error_m']:.4f} | "
            f"{row['median_3d_error_m']:.4f} | {row['p95_3d_error_m']:.4f} | "
            f"{row['error_gt_1_0m_rate']:.4f} | {row['approx_pipeline_fps']:.2f} |"
        )
    lines += [
        "",
        "## 5. Primary Detector Comparison",
        (
            f"Legacy GT-centric: detected by both {detector['legacy']['detected_by_both']}, "
            f"generic-only {detector['legacy']['generic_only']}, "
            f"fine-tuned-only {detector['legacy']['finetuned_only']}, "
            f"missed by both {detector['legacy']['missed_by_both']}. "
            f"Mean fine-tuned minus generic 3D error on joint-valid GT pairs: "
            f"{detector['legacy']['mean_finetuned_minus_generic_error_m']} m."
        ),
        "",
        "## 6. Secondary Hybrid Detector Comparison",
        (
            f"Hybrid GT-centric mean fine-tuned minus generic 3D error on joint-valid GT pairs: "
            f"{detector['hybrid']['mean_finetuned_minus_generic_error_m']} m."
        ),
        "",
        "## 7. Fusion Comparisons",
        (
            f"Generic hybrid-minus-legacy mean paired error: "
            f"{pairwise['generic_fusion']['mean_paired_error_difference_hybrid_minus_legacy']} m; "
            f"joint pairs: {pairwise['generic_fusion']['jointly_valid_pairs']}."
        ),
        (
            f"Fine-tuned hybrid-minus-legacy mean paired error: "
            f"{pairwise['finetuned_fusion']['mean_paired_error_difference_hybrid_minus_legacy']} m; "
            f"joint pairs: {pairwise['finetuned_fusion']['jointly_valid_pairs']}."
        ),
        "",
        "## 8. Artifact Index",
        f"- Output root: `{OUTPUT_ROOT}`",
        "- Tables: `tables/`",
        "- Comparisons: `comparisons/`",
        "- Raw runs: `runs/`",
        "- Visual audit: `visuals/audit/`",
        "- Reproducibility metadata: `metadata/reproducibility_manifest.json`",
    ]
    (OUTPUT_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess() -> None:
    run_specs = {
        "generic_legacy": ("generic", "legacy_box"),
        "generic_hybrid": ("generic", "hybrid_depth_legacy"),
        "finetuned_legacy": ("finetuned", "legacy_box"),
        "finetuned_hybrid": ("finetuned", "hybrid_depth_legacy"),
    }
    runs = {}
    for run_name, (detector, method) in run_specs.items():
        run_dir = OUTPUT_ROOT / "runs" / run_name
        runs[run_name] = {
            "detector": detector,
            "fusion_method": method,
            "summary": load_json(run_dir / "summary.json"),
            "detections": load_json(run_dir / "detections.json"),
            "frames": load_json(run_dir / "frames.json"),
        }

    overall = overall_tables(runs)
    per_recording = per_recording_tables(runs)
    write_json(OUTPUT_ROOT / "tables" / "overall_2x2.json", overall)
    write_csv(OUTPUT_ROOT / "tables" / "overall_2x2.csv", overall)
    write_json(OUTPUT_ROOT / "tables" / "per_recording_2x2.json", per_recording)
    write_csv(OUTPUT_ROOT / "tables" / "per_recording_2x2.csv", per_recording)

    pairwise = {
        "generic_fusion": eval_assoc.paired_comparison(
            runs["generic_legacy"]["detections"],
            runs["generic_hybrid"]["detections"],
        ),
        "finetuned_fusion": eval_assoc.paired_comparison(
            runs["finetuned_legacy"]["detections"],
            runs["finetuned_hybrid"]["detections"],
        ),
    }
    write_json(OUTPUT_ROOT / "comparisons" / "pairwise_comparisons.json", pairwise)
    write_csv(
        OUTPUT_ROOT / "tables" / "fusion_comparison_generic.csv",
        pairwise["generic_fusion"].get("pairs", []),
    )
    write_csv(
        OUTPUT_ROOT / "tables" / "fusion_comparison_finetuned.csv",
        pairwise["finetuned_fusion"].get("pairs", []),
    )

    gt_rows = gt_rows_from_manifest()
    write_json(OUTPUT_ROOT / "tables" / "gt_instances.json", gt_rows)
    legacy_rows, legacy_summary = detector_comparison(
        gt_rows,
        runs["generic_legacy"]["detections"],
        runs["finetuned_legacy"]["detections"],
        "legacy",
    )
    hybrid_rows, hybrid_summary = detector_comparison(
        gt_rows,
        runs["generic_hybrid"]["detections"],
        runs["finetuned_hybrid"]["detections"],
        "hybrid",
    )
    write_csv(OUTPUT_ROOT / "tables" / "detector_comparison_legacy.csv", legacy_rows)
    write_csv(OUTPUT_ROOT / "tables" / "detector_comparison_hybrid.csv", hybrid_rows)
    detector_summary = {"legacy": legacy_summary, "hybrid": hybrid_summary}
    write_json(OUTPUT_ROOT / "comparisons" / "primary_checkpoint_comparison.json", legacy_summary)
    write_json(OUTPUT_ROOT / "comparisons" / "secondary_checkpoint_comparison_hybrid.json", hybrid_summary)
    write_json(OUTPUT_ROOT / "comparisons" / "bootstrap_results.json", detector_summary)

    failures = failure_taxonomy(runs)
    breakdowns = condition_breakdowns(runs)
    audit_rows = visual_audit(runs)
    write_csv(OUTPUT_ROOT / "tables" / "failure_taxonomy.csv", failures)
    write_csv(OUTPUT_ROOT / "tables" / "condition_breakdowns.csv", breakdowns)
    write_csv(OUTPUT_ROOT / "visuals" / "visual_manifest.csv", audit_rows)
    write_json(OUTPUT_ROOT / "visuals" / "visual_manifest.json", audit_rows)

    metadata = load_json(OUTPUT_ROOT / "metadata" / "reproducibility_manifest.json")
    metadata["caches"] = {
        "generic_yolo11s": cache_summary(OUTPUT_ROOT / "caches" / "generic_yolo11s"),
        "finetuned_yolo11s": cache_summary(OUTPUT_ROOT / "caches" / "finetuned_yolo11s"),
    }
    metadata["table_paths"] = {
        "overall_2x2": str(OUTPUT_ROOT / "tables" / "overall_2x2.csv"),
        "per_recording_2x2": str(OUTPUT_ROOT / "tables" / "per_recording_2x2.csv"),
        "legacy_detector_comparison": str(OUTPUT_ROOT / "tables" / "detector_comparison_legacy.csv"),
        "hybrid_detector_comparison": str(OUTPUT_ROOT / "tables" / "detector_comparison_hybrid.csv"),
    }
    write_json(OUTPUT_ROOT / "metadata" / "reproducibility_manifest.json", metadata)

    tex_lines = [
        "\\begin{tabular}{lrrrrr}",
        "Run & Detections & Valid & Mean 3D & Median 3D & P95 3D \\\\",
        "\\hline",
    ]
    for row in overall:
        tex_lines.append(
            f"{row['run']} & {row['total_detections']} & "
            f"{row['valid_fusion_rate']:.3f} & {row['mean_3d_error_m']:.3f} & "
            f"{row['median_3d_error_m']:.3f} & {row['p95_3d_error_m']:.3f} \\\\"
        )
    tex_lines.append("\\end{tabular}")
    (OUTPUT_ROOT / "tables" / "paper_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    write_report(metadata, overall, pairwise, detector_summary)


def run_benchmark(skip_runs: bool) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["manifest", "metadata", "caches/generic_yolo11s", "caches/finetuned_yolo11s", "runs", "comparisons", "tables", "visuals", "audit", "logs"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_unix_time": time.time(),
        "repo_state_start": {
            "fusion": repo_state(REPO_ROOT),
            "dataset_tools": repo_state(Path("/home/prabuddhi/Desktop/agri-human-dataset-tools")),
            "benchmark": repo_state(Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark")),
        },
        "environment": environment_metadata(),
        "checkpoints": {
            "generic": checkpoint_metadata(GENERIC_CKPT, GENERIC_SHA),
            "finetuned": checkpoint_metadata(FINETUNED_CKPT, FINETUNED_SHA),
        },
        "smoke_inference": smoke_inference(
            first_validation_image(),
            {"generic": GENERIC_CKPT, "finetuned": FINETUNED_CKPT},
        ),
        "generic_result_reuse_decision": {
            "decision": "rerun_generic_results",
            "reason": (
                "Previous generic output did not contain explicit yolo_imgsz cache metadata "
                "or current evaluator source hash, so fresh generic runs were produced."
            ),
        },
    }
    metadata["checkpoints"]["different_sha256"] = (
        metadata["checkpoints"]["generic"]["sha256"]
        != metadata["checkpoints"]["finetuned"]["sha256"]
    )
    metadata["manifest"] = build_manifest()
    write_json(OUTPUT_ROOT / "metadata" / "reproducibility_manifest.json", metadata)
    write_json(OUTPUT_ROOT / "metadata" / "generic_result_reuse_check.json", metadata["generic_result_reuse_decision"])

    if not skip_runs:
        commands = {
            "generic_legacy": evaluator_cmd(
                "generic_legacy",
                GENERIC_CKPT,
                OUTPUT_ROOT / "caches" / "generic_yolo11s",
                "legacy_box",
            ),
            "generic_hybrid": evaluator_cmd(
                "generic_hybrid",
                GENERIC_CKPT,
                OUTPUT_ROOT / "caches" / "generic_yolo11s",
                "hybrid_depth_legacy",
            ),
            "finetuned_legacy": evaluator_cmd(
                "finetuned_legacy",
                FINETUNED_CKPT,
                OUTPUT_ROOT / "caches" / "finetuned_yolo11s",
                "legacy_box",
            ),
            "finetuned_hybrid": evaluator_cmd(
                "finetuned_hybrid",
                FINETUNED_CKPT,
                OUTPUT_ROOT / "caches" / "finetuned_yolo11s",
                "hybrid_depth_legacy",
            ),
        }
        run_results = {}
        for run_name, cmd in commands.items():
            run_results[run_name] = run(cmd, OUTPUT_ROOT / "logs" / f"{run_name}.log")
            write_json(OUTPUT_ROOT / "metadata" / "run_commands.json", run_results)

    postprocess()
    metadata = load_json(OUTPUT_ROOT / "metadata" / "reproducibility_manifest.json")
    metadata["repo_state_end"] = {
        "fusion": repo_state(REPO_ROOT),
        "dataset_tools": repo_state(Path("/home/prabuddhi/Desktop/agri-human-dataset-tools")),
        "benchmark": repo_state(Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark")),
    }
    write_json(OUTPUT_ROOT / "metadata" / "reproducibility_manifest.json", metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postprocess-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(skip_runs=args.postprocess_only)


if __name__ == "__main__":
    main()
