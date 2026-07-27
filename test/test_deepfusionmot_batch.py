import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_batch_module():
    spec = importlib.util.spec_from_file_location(
        "run_aghri_deepfusionmot_batch",
        REPO_ROOT / "tools" / "run_aghri_deepfusionmot_batch.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_defines_exactly_eight_aghri_detector_combinations():
    module = _load_batch_module()
    combos = module.COMBINATIONS
    assert len(combos) == 8
    assert [combo.combination_id for combo in combos] == ["S1", "S2", "S3", "S4", "P1", "P2", "P3", "P4"]
    assert {(combo.camera_model, combo.lidar_detector, combo.lidar_model) for combo in combos} == {
        ("generic", "second", "generic"),
        ("finetuned", "second", "generic"),
        ("generic", "second", "finetuned"),
        ("finetuned", "second", "finetuned"),
        ("generic", "pointpillars", "generic"),
        ("finetuned", "pointpillars", "generic"),
        ("generic", "pointpillars", "finetuned"),
        ("finetuned", "pointpillars", "finetuned"),
    }


def test_batch_selector_accepts_legacy_and_short_ids():
    module = _load_batch_module()
    selected = module._selected_combinations("S1,cam_ft_pointpillars_ft")
    assert [combo.combination_id for combo in selected] == ["S1", "P4"]


def test_batch_command_passes_deepfusionmot_strict_and_calibration_path(tmp_path):
    module = _load_batch_module()
    args = type(
        "Args",
        (),
        {
            "recording": "rec",
            "limit_frames": 0,
            "ego_motion_mode": "none",
            "max_age": 0.75,
            "max_age_frames": 5,
            "min_hits": 2,
            "cache_tolerance_sec": 0.08,
            "iou_threshold": 0.1,
            "visual_frames": 0,
            "visual_video_fps": 5.0,
            "association_config": "",
            "calibration_path": tmp_path / "calibration",
            "ego_odom_max_delta_sec": 0.25,
            "deepfusionmot_strict": True,
        },
    )()
    command = module._single_run_command(args, module.COMBINATIONS[0], tmp_path / "out")
    assert "--deepfusionmot-strict" in command
    assert f"--calibration-path {tmp_path / 'calibration'}" in command
    assert "--ego-odom-max-delta-sec 0.25" in command
