from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "create_late_fusion_visual_walkthrough.py"
SPEC = importlib.util.spec_from_file_location("late_fusion_visual_walkthrough", SCRIPT_PATH)
walkthrough = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(walkthrough)


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        recording=walkthrough.DEFAULT_RECORDING,
        sample_id=walkthrough.DEFAULT_SAMPLE_ID,
        camera_cache=walkthrough.DEFAULT_YOLO_CACHE,
        lidar_cache=walkthrough.DEFAULT_LIDAR_CACHE,
        association=walkthrough.DEFAULT_ASSOCIATION,
        per_frame=walkthrough.DEFAULT_PER_FRAME,
        manifest=walkthrough.DEFAULT_MANIFEST,
        calibration=walkthrough.DEFAULT_CALIBRATION,
        output_root=walkthrough.DEFAULT_OUTPUT_ROOT,
        threshold=walkthrough.DEFAULT_THRESHOLD,
        lidar_box_origin="center",
    )


def _skip_if_walkthrough_inputs_missing(args: argparse.Namespace) -> None:
    required = [
        args.camera_cache,
        args.lidar_cache,
        args.association,
        args.per_frame,
        args.manifest,
        args.calibration,
    ]
    missing = [str(path) for path in required if not walkthrough.as_path(path).exists()]
    if missing:
        pytest.skip("archived walkthrough cache inputs are not in the cleaned repo")


def test_corner_order_and_yaw_rotation() -> None:
    size = (4.0, 2.0, 3.0)
    local = walkthrough.local_box_corners(size, "bottom_center")
    np.testing.assert_allclose(local[0], [-2.0, -1.0, 0.0])
    np.testing.assert_allclose(local[3], [-2.0, 1.0, 0.0])
    np.testing.assert_allclose(local[4], [-2.0, -1.0, 3.0])

    rot = walkthrough.yaw_rotation_z(math.pi / 2.0)
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_box_origin_conversion_rules() -> None:
    bottom = walkthrough.local_box_corners((1.0, 1.0, 2.0), "bottom_center")
    center = walkthrough.local_box_corners((1.0, 1.0, 2.0), "center")
    np.testing.assert_allclose(bottom[:, 2], [0, 0, 0, 0, 2, 2, 2, 2])
    np.testing.assert_allclose(center[:, 2], [-1, -1, -1, -1, 1, 1, 1, 1])


def test_transform_projection_clipping_and_iou() -> None:
    args = _default_args()
    _skip_if_walkthrough_inputs_missing(args)
    data, _gt = walkthrough.build_worked_example(args)

    np.testing.assert_allclose(
        data["geometry"]["projected_lidar_rectangle_clipped"],
        walkthrough.EXPECTED_AUDIT_RECT,
        atol=1e-9,
    )
    assert math.isclose(data["association"]["iou"], walkthrough.EXPECTED_AUDIT_IOU, abs_tol=1e-12)
    assert data["association"]["accepted"] is True
    assert data["association"]["decision"] == "MATCHED"

    corners_lidar = np.asarray(data["geometry"]["corners_lidar"], dtype=float)
    corners_camera = np.asarray(data["geometry"]["corners_camera"], dtype=float)
    transform = np.asarray(data["calibration"]["T_camera_from_lidar"], dtype=float)
    projected_camera = walkthrough.transform_points(corners_lidar, transform)
    np.testing.assert_allclose(projected_camera, corners_camera, atol=1e-12)

    p = np.asarray(data["calibration"]["P"], dtype=float)
    uv = walkthrough.project_camera_points(corners_camera, p)
    np.testing.assert_allclose(uv, np.asarray(data["geometry"]["corners_image_unclipped"]), atol=1e-9)

    clipped = walkthrough.clip_xyxy(np.asarray(data["geometry"]["projected_lidar_rectangle_unclipped"]), 672, 376)
    np.testing.assert_allclose(clipped, walkthrough.EXPECTED_AUDIT_RECT, atol=1e-9)


def test_selected_example_output_membership() -> None:
    args = _default_args()
    _skip_if_walkthrough_inputs_missing(args)
    data, _gt = walkthrough.build_worked_example(args)
    assert data["fusion_outputs"]["matched_3d_count"] == 1
    assert data["fusion_outputs"]["unmatched_camera_count"] == 0
    assert data["fusion_outputs"]["unmatched_lidar_count"] == 0
    assert data["fusion_outputs"]["final_3d_count"] == 1
    assert data["fusion_outputs"]["output_policy"] == "matched_plus_unmatched_lidar"
