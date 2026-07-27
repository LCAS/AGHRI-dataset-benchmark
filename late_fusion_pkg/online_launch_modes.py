"""Mode helpers for keeping cached and live ROS workflows separated."""

from __future__ import annotations


VALID_INFERENCE_MODES = ("live", "cached")


def normalize_inference_mode(mode: str | None) -> str:
    value = (mode or "").strip().lower() or "live"
    if value not in VALID_INFERENCE_MODES:
        raise ValueError(f"inference_mode must be one of {', '.join(VALID_INFERENCE_MODES)}; got {mode!r}")
    return value


def assert_online_launch_mode(mode: str | None) -> str:
    value = normalize_inference_mode(mode)
    if value != "live":
        raise RuntimeError(
            "This launch file is the fully online late-fusion workflow. "
            "Use the existing *_generic_late_fusion.launch.py files for cached replay."
        )
    return value


def process_kinds_for_mode(mode: str | None) -> tuple[str, ...]:
    value = normalize_inference_mode(mode)
    if value == "live":
        return ("live_yolo_detector", "live_lidar_detector", "late_fusion", "visualizer")
    return ("cached_yolo_publisher", "cached_lidar_publisher", "late_fusion", "visualizer")
