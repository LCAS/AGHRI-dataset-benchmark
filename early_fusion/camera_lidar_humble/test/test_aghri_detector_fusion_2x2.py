"""Focused tests for the AGHRI detector/fusion 2x2 benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from tools.evaluate_aghri_association import yolo_cache_metadata
from tools.run_aghri_detector_fusion_2x2 import (
    detector_comparison,
    overall_tables,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
FINETUNED_BEST = str(
    REPO_ROOT / "checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt"
)


def _params(path: str):
    data = yaml.safe_load((CONFIG_DIR / path).read_text(encoding="utf-8"))
    return data["lidar_fusion"]["ros__parameters"]


def test_finetuned_legacy_config_only_changes_model_path():
    generic = _params("aghri_zed_livox.yaml")
    finetuned = _params("aghri_zed_livox_yolo11s_finetuned.yaml")
    changed = {
        key
        for key in set(generic) | set(finetuned)
        if generic.get(key) != finetuned.get(key)
    }
    assert changed == {"YOLO_model"}
    assert finetuned["YOLO_model"] == FINETUNED_BEST
    assert finetuned["association_method"] == "legacy_box"


def test_finetuned_hybrid_config_has_only_intended_differences():
    legacy = _params("aghri_zed_livox_yolo11s_finetuned.yaml")
    hybrid = _params("aghri_zed_livox_yolo11s_finetuned_hybrid.yaml")
    changed = {
        key
        for key in set(legacy) | set(hybrid)
        if legacy.get(key) != hybrid.get(key)
    }
    assert changed == {
        "association_method",
        "hybrid_min_support_ratio",
        "hybrid_max_depth_spread_m",
        "hybrid_min_cluster_separation_m",
        "hybrid_fallback_on_single_cluster",
    }
    assert hybrid["association_method"] == "hybrid_depth_legacy"
    assert hybrid["centre_statistic"] == "mean"
    assert hybrid["box_bottom_trim_ratio"] == 0.0
    assert hybrid["depth_cluster_gap_m"] == 0.35
    assert hybrid["depth_cluster_min_points"] == 5
    assert hybrid["hybrid_min_cluster_points"] == 5
    assert hybrid["hybrid_min_support_ratio"] == 0.20
    assert hybrid["hybrid_max_depth_spread_m"] == 0.45
    assert hybrid["hybrid_min_cluster_separation_m"] == 0.15
    assert hybrid["hybrid_fallback_on_single_cluster"] is False
    assert hybrid["hybrid_fallback_on_sparse_candidates"] is True


def test_yolo_cache_metadata_records_checkpoint_hash_and_imgsz(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image")
    model = tmp_path / "best.pt"
    model.write_bytes(b"checkpoint")
    metadata = yolo_cache_metadata(
        image,
        np.zeros((10, 20, 3), dtype=np.uint8),
        123.0,
        model,
        sha256_file(model),
        0.1,
        [0],
        640,
        "8.3.154",
    )
    assert metadata["checkpoint_sha256"] == sha256_file(model)
    assert metadata["classes"] == [0]
    assert metadata["confidence_threshold"] == 0.1
    assert metadata["imgsz"] == 640


def test_gt_centric_detector_comparison_preserves_misses_and_duplicates():
    gt_rows = [
        {"recording": "r", "frame_number": 0, "gt_identity": "01"},
        {"recording": "r", "frame_number": 0, "gt_identity": "02"},
    ]
    generic_rows = [
        {
            "recording": "r",
            "frame_number": 0,
            "matched_gt_identity": "01",
            "confidence": 0.8,
            "valid": True,
            "errors": {"lidar_center_error": 0.4},
        },
        {
            "recording": "r",
            "frame_number": 0,
            "matched_gt_identity": "01",
            "confidence": 0.7,
            "valid": True,
            "errors": {"lidar_center_error": 0.5},
        },
    ]
    finetuned_rows = [
        {
            "recording": "r",
            "frame_number": 0,
            "matched_gt_identity": "01",
            "confidence": 0.9,
            "valid": True,
            "errors": {"lidar_center_error": 0.2},
        },
        {
            "recording": "r",
            "frame_number": 0,
            "matched_gt_identity": "02",
            "confidence": 0.6,
            "valid": False,
            "validity_failure_reason": "no_points",
            "errors": {},
        },
    ]
    rows, summary = detector_comparison(gt_rows, generic_rows, finetuned_rows, "legacy")
    first = rows[0]
    second = rows[1]
    assert first["generic_duplicate_count"] == 1
    assert first["finetuned_minus_generic_error_m"] == -0.2
    assert second["generic_detected"] is False
    assert second["finetuned_detected"] is True
    assert summary["detected_by_both"] == 1
    assert summary["finetuned_only"] == 1
    assert summary["joint_valid_error_pairs"] == 1


def test_overall_table_schema_contains_runtime_and_error_columns():
    runs = {
        "generic_legacy": {
            "detector": "generic",
            "fusion_method": "legacy_box",
            "summary": {
                "overall": {
                    "total_yolo_person_detections": 2,
                    "matched_prediction_count": 1,
                    "unmatched_prediction_count": 1,
                    "ambiguous_match_count": 0,
                    "valid_association_rate": 0.5,
                    "no_point_rate": 0.0,
                    "mean_lidar_center_error": 0.3,
                    "median_lidar_center_error": 0.3,
                    "p75_lidar_center_error": 0.3,
                    "p90_lidar_center_error": 0.3,
                    "p95_lidar_center_error": 0.3,
                    "max_lidar_center_error": 0.3,
                    "lidar_error_le_0_5m_rate": 1.0,
                    "lidar_error_gt_1_0m_rate": 0.0,
                    "lidar_error_gt_2_0m_rate": 0.0,
                    "mean_camera_depth_error": 0.1,
                    "median_camera_depth_error": 0.1,
                    "mean_association_time_sec": 0.001,
                },
                "frame_overall": {
                    "total_gt3d_instances": 2,
                    "mean_frame_time_sec": 0.05,
                    "approx_fps": 20.0,
                },
                "failures": [],
            },
        }
    }
    row = overall_tables(runs)[0]
    assert row["total_detections"] == 2
    assert row["gt_instances"] == 2
    assert row["error_gt_0_5m_rate"] == 0.0
    assert row["mean_fusion_time_sec"] == 0.001
    assert row["approx_pipeline_fps"] == 20.0
