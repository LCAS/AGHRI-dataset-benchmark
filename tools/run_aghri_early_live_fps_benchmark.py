#!/usr/bin/env python3
"""Measure live ROS 2 throughput for the AGHRI early-fusion node."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


REPO_ROOT = Path("/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble")
BUILD_ROOT = Path("/home/prabuddhi/Desktop/Early_Fus")
DEFAULT_BAG = BUILD_ROOT / "rosbags/aghri_test/out_vine_4swap+walk_st_ly_11_06_2024_2_label"
RESULT_ROOT = REPO_ROOT / "results/aghri_early_live_fps_benchmark"
COMBINATIONS = [
    ("generic", REPO_ROOT / "config/aghri_zed_livox.yaml"),
    ("finetuned", REPO_ROOT / "config/aghri_zed_livox_yolo11s_finetuned.yaml"),
]


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(clean) - 1)
    frac = rank - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def ros_cmd(command: str) -> str:
    return (
        "conda deactivate 2>/dev/null || true; "
        "source /opt/ros/humble/setup.bash; "
        f"source {BUILD_ROOT}/install/setup.bash; "
        f"exec {command}"
    )


def close_process(proc: subprocess.Popen | None, timeout: float = 15.0) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
    log = getattr(proc, "_log_stream", None)
    if log is not None:
        log.close()


def launch_process(command: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=BUILD_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._log_stream = log  # type: ignore[attr-defined]
    return proc


class OutputCounter(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("aghri_early_live_fps_counter")
        self.times: list[float] = []
        self.create_subscription(Image, topic, self._callback, 10)

    def _callback(self, _msg: Image) -> None:
        self.times.append(time.perf_counter())


def parse_processing_times(log_path: Path) -> list[float]:
    pattern = re.compile(r"objects detected in ([0-9.]+) seconds")
    if not log_path.exists():
        return []
    return [float(match.group(1)) for match in pattern.finditer(log_path.read_text(errors="replace"))]


def run_one(args: argparse.Namespace, model_name: str, params_path: Path) -> dict:
    run_dir = args.results_dir / f"{model_name}_{args.bag_path.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    launch_log = run_dir / "launch.log"
    command = ros_cmd(
        "ros2 launch camera_lidar_fusion aghri_test_bag_fusion_rviz.launch.py "
        f"bag_path:={args.bag_path} "
        f"params_file:={params_path} "
        f"playback_rate:={args.playback_rate} "
        f"launch_rviz:=false loop:={'true' if args.loop else 'false'} autoplay:=true"
    )

    counter = OutputCounter(args.output_topic)
    proc = None
    start_wall = time.perf_counter()
    try:
        proc = launch_process(command, launch_log)
        deadline = time.perf_counter() + args.max_runtime_sec
        last_count = 0
        last_seen = None
        while time.perf_counter() < deadline:
            rclpy.spin_once(counter, timeout_sec=0.1)
            if len(counter.times) != last_count:
                last_count = len(counter.times)
                last_seen = time.perf_counter()
                if args.measure_outputs > 0 and len(counter.times) >= args.warmup_outputs + args.measure_outputs:
                    break
            if last_seen is not None and time.perf_counter() - last_seen > args.idle_after_sec:
                break
            if proc.poll() is not None and last_seen is not None:
                break
    finally:
        close_process(proc)
        counter.destroy_node()

    times = counter.times[args.warmup_outputs :]
    processing_times = parse_processing_times(launch_log)
    fps = 0.0
    duration = 0.0
    if len(times) >= 2:
        duration = max(times[-1] - times[0], 1e-9)
        fps = (len(times) - 1) / duration

    measured_processing_times = processing_times[args.warmup_outputs : args.warmup_outputs + len(times)]
    return {
        "model": model_name,
        "bag": args.bag_path.name,
        "playback_rate": args.playback_rate,
        "output_topic": args.output_topic,
        "warmup_outputs": args.warmup_outputs,
        "outputs_total_seen": len(counter.times),
        "outputs_measured": len(times),
        "measurement_duration_s": duration,
        "live_output_fps": fps,
        "pipeline_time_mean_ms": mean(measured_processing_times) * 1000.0,
        "pipeline_time_median_ms": median(measured_processing_times) * 1000.0,
        "pipeline_time_p95_ms": percentile(measured_processing_times, 0.95) * 1000.0,
        "wall_runtime_s": time.perf_counter() - start_wall,
        "launch_log": str(launch_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-path", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--results-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--playback-rate", type=float, default=8.0)
    parser.add_argument("--max-runtime-sec", type=float, default=300.0)
    parser.add_argument("--idle-after-sec", type=float, default=8.0)
    parser.add_argument("--output-topic", default="/camera_lidar_fusion/result")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--warmup-outputs", type=int, default=20)
    parser.add_argument("--measure-outputs", type=int, default=200)
    parser.add_argument("--only", nargs="*", choices=["generic", "finetuned"], default=[])
    args = parser.parse_args()

    selected = [(name, path) for name, path in COMBINATIONS if not args.only or name in set(args.only)]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    try:
        rows = [run_one(args, name, params_path) for name, params_path in selected]
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    fields = list(rows[0].keys()) if rows else []
    with (args.results_dir / "summary_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / "summary_results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
