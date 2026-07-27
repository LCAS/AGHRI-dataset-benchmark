#!/usr/bin/env python3
"""Run the fully online AGHRI ROS 2 FPS benchmark matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from copy import copy


REPO_ROOT = Path("/home/prabuddhi/Desktop/late_fusion/deepfusionmot")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WORKSPACE_ROOT = Path(os.environ.get("LATE_FUSION_WS", "/home/prabuddhi/Desktop/late_fusion_ws"))
RESULT_ROOT = REPO_ROOT / "results/aghri_online_fps_benchmark"
RAW_ROOT = RESULT_ROOT / "raw_runs"
BAG_PATH = Path(
    "/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test/"
    "footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label"
)
QOS_PATH = Path(
    "/home/prabuddhi/Desktop/agri-human-dataset-benchmark/"
    "early_fusion/camera_lidar_humble/config/aghri_rosbag2_qos_overrides.yaml"
)
COMBINATIONS = [
    ("S1", "generic", "second", "generic"),
    ("S2", "finetuned", "second", "generic"),
    ("S3", "generic", "second", "finetuned"),
    ("S4", "finetuned", "second", "finetuned"),
    ("P1", "generic", "pointpillars", "generic"),
    ("P2", "finetuned", "pointpillars", "generic"),
    ("P3", "generic", "pointpillars", "finetuned"),
    ("P4", "finetuned", "pointpillars", "finetuned"),
]
FIELDNAMES = [
    "experiment_id",
    "recording",
    "bag_path",
    "camera_model",
    "lidar_detector",
    "lidar_model",
    "test_mode",
    "playback_rate",
    "run_number",
    "warmup_frames",
    "measurement_duration_s",
    "incoming_images",
    "incoming_pointclouds",
    "yolo_processed_frames",
    "lidar_processed_frames",
    "fusion_callbacks",
    "fused_outputs",
    "tracked_outputs",
    "yolo_fps",
    "lidar_fps",
    "fusion_fps",
    "online_fused_fps",
    "online_tracked_fps",
    "yolo_latency_mean_ms",
    "yolo_latency_median_ms",
    "yolo_latency_p95_ms",
    "lidar_latency_mean_ms",
    "lidar_latency_median_ms",
    "lidar_latency_p95_ms",
    "fusion_latency_mean_ms",
    "fusion_latency_median_ms",
    "fusion_latency_p95_ms",
    "tracking_latency_mean_ms",
    "tracking_latency_median_ms",
    "tracking_latency_p95_ms",
    "end_to_end_latency_mean_ms",
    "end_to_end_latency_median_ms",
    "end_to_end_latency_p95_ms",
    "end_to_end_latency_max_ms",
    "processing_ratio",
    "dropped_images",
    "dropped_pointclouds",
    "synchronisation_failures",
    "tf_failures",
    "camera_info_failures",
    "worker_errors",
    "run_status",
    "failure_reason",
]


def sh(cmd: str, timeout: float | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip()


def ros_cmd(command: str) -> str:
    setup_path = WORKSPACE_ROOT / "install" / "setup.bash"
    return f"source /opt/ros/humble/setup.bash && source {setup_path} && exec {command}"


def installed_executable(name: str) -> Path:
    return WORKSPACE_ROOT / "install" / "late_fusion_pkg" / "lib" / "late_fusion_pkg" / name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(float(v) for v in values if v is not None)
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


def stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--bag-path", type=Path, default=BAG_PATH)
    parser.add_argument(
        "--bag-root",
        type=Path,
        default=Path("/home/prabuddhi/Desktop/Early_Fus/rosbags/aghri_test"),
        help="Root used with --recordings to build AGHRI rosbag paths.",
    )
    parser.add_argument(
        "--recordings",
        nargs="*",
        default=[],
        help="Optional recording names. When provided, each bag is read from --bag-root/<recording>.",
    )
    parser.add_argument("--normal-rate", type=float, default=1.0)
    parser.add_argument("--max-throughput-rate", type=float, default=8.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--ready-timeout-sec", type=float, default=900.0)
    parser.add_argument("--bag-timeout-sec", type=float, default=0.0)
    parser.add_argument("--post-run-cooldown-sec", type=float, default=20.0)
    parser.add_argument("--worker-python", default="/home/prabuddhi/miniconda3/envs/aghri-mmdet3d/bin/python")
    parser.add_argument("--mmdet3d-device", default="cuda:0")
    parser.add_argument("--detector-device", default="auto")
    parser.add_argument("--qos-overrides", type=Path, default=QOS_PATH)
    parser.add_argument("--only", nargs="*", default=[], help="Optional IDs to run, e.g. S4 P4.")
    parser.add_argument("--test-mode", choices=["normal", "max", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", help="Write config/environment without launching ROS.")
    return parser.parse_args()


def launch_process(command: str, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._late_fusion_log = log  # type: ignore[attr-defined]
    return proc


def close_process(proc: subprocess.Popen | None, timeout: float = 20.0) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
    log = getattr(proc, "_late_fusion_log", None)
    if log is not None:
        log.close()


def wait_for_ready(log_path: Path, proc: subprocess.Popen, timeout_sec: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_sec
    saw_yolo = False
    saw_worker = False
    saw_lidar = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f"launch exited before readiness rc={proc.returncode}"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        saw_yolo = saw_yolo or "LIVE YOLO inference enabled" in text
        saw_worker = saw_worker or "worker_ready" in text
        saw_lidar = saw_lidar or "LIVE LiDAR inference enabled" in text
        if saw_yolo and saw_worker and saw_lidar:
            return True, ""
        time.sleep(2.0)
    return False, f"timed out waiting for live readiness yolo={saw_yolo} worker_ready={saw_worker} lidar={saw_lidar}"


def parse_monitor_events(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def diag_events(events: list[dict], source: str) -> list[dict]:
    out = []
    for event in events:
        if event.get("event") != "diagnostic" or event.get("source") != source:
            continue
        data = event.get("data")
        if isinstance(data, dict):
            out.append(event)
    return out


def last_number(events: list[dict], key: str, default: int = 0) -> int:
    values = [int(event.get(key, 0)) for event in events if key in event]
    return values[-1] if values else default


def successful_live_diag(events: list[dict], mode: str) -> list[dict]:
    result = []
    for event in events:
        data = event.get("data", {})
        if data.get("mode") != mode:
            continue
        if data.get("cache_used") is not False:
            continue
        if data.get("error"):
            continue
        result.append(event)
    return result


def summarize_events(events: list[dict], warmup_frames: int, metadata: dict, failure_reason: str = "") -> dict:
    row = {field: "" for field in FIELDNAMES}
    row.update(metadata)
    row["warmup_frames"] = warmup_frames
    if failure_reason:
        row["run_status"] = "failed"
        row["failure_reason"] = failure_reason
        return row
    if not events:
        row["run_status"] = "failed"
        row["failure_reason"] = "no monitor events collected"
        return row

    yolo_diags_all = successful_live_diag(diag_events(events, "camera_live"), "live_yolo_inference")
    lidar_diags_all = successful_live_diag(diag_events(events, "lidar_live"), "live_lidar_inference")
    if len(yolo_diags_all) <= warmup_frames or len(lidar_diags_all) <= warmup_frames:
        row["run_status"] = "failed"
        row["failure_reason"] = (
            f"insufficient post-ready warmup frames: yolo={len(yolo_diags_all)} "
            f"lidar={len(lidar_diags_all)} warmup={warmup_frames}"
        )
        return row
    cutoff_ns = max(
        int(yolo_diags_all[warmup_frames - 1]["monitor_time_ns"]),
        int(lidar_diags_all[warmup_frames - 1]["monitor_time_ns"]),
    )
    post_events = [event for event in events if int(event.get("monitor_time_ns", 0)) >= cutoff_ns]
    yolo_diags = [event for event in yolo_diags_all if int(event.get("monitor_time_ns", 0)) >= cutoff_ns]
    lidar_diags = [event for event in lidar_diags_all if int(event.get("monitor_time_ns", 0)) >= cutoff_ns]
    fusion_diags = [
        event for event in diag_events(post_events, "fusion")
        if isinstance(event.get("data"), dict) and "total_callback_time" in event["data"]
    ]
    tracking_diags = [
        event for event in diag_events(post_events, "tracking")
        if isinstance(event.get("data"), dict) and "tracker_update_time_sec" in event["data"]
    ]
    fused = [event for event in post_events if event.get("event") == "fused_output"]
    tracked = [event for event in post_events if event.get("event") == "tracked_3d_output"]
    images = [event for event in post_events if event.get("event") == "input_image"]
    pointclouds = [event for event in post_events if event.get("event") == "input_pointcloud"]

    end_ns = max(int(event.get("monitor_time_ns", cutoff_ns)) for event in post_events)
    duration_s = max((end_ns - cutoff_ns) / 1_000_000_000.0, 1e-9)
    yolo_lat = [float(event["data"].get("latency_ms", 0.0)) for event in yolo_diags]
    lidar_lat = [float(event["data"].get("latency_ms", 0.0)) for event in lidar_diags]
    fusion_lat = [float(event["data"].get("total_callback_time", 0.0)) * 1000.0 for event in fusion_diags]
    e2e_lat = [float(event.get("end_to_end_latency_ms")) for event in fused if event.get("end_to_end_latency_ms") is not None]

    if len(fused) >= 2:
        first_fused = int(fused[0]["monitor_time_ns"])
        last_fused = int(fused[-1]["monitor_time_ns"])
        online_fused_fps = (len(fused) - 1) / max((last_fused - first_fused) / 1_000_000_000.0, 1e-9)
    else:
        online_fused_fps = 0.0
    if len(tracked) >= 2:
        first_tracked = int(tracked[0]["monitor_time_ns"])
        last_tracked = int(tracked[-1]["monitor_time_ns"])
        online_tracked_fps = (len(tracked) - 1) / max((last_tracked - first_tracked) / 1_000_000_000.0, 1e-9)
    else:
        online_tracked_fps = 0.0

    camera_diag_all = diag_events(events, "camera_live")
    lidar_diag_all = diag_events(events, "lidar_live")
    last_camera_data = camera_diag_all[-1].get("data", {}) if camera_diag_all else {}
    last_lidar_data = lidar_diag_all[-1].get("data", {}) if lidar_diag_all else {}
    worker_errors = sum(1 for event in camera_diag_all + lidar_diag_all if event.get("data", {}).get("error"))
    fusion_raw = [str(event.get("raw", "")) for event in diag_events(post_events, "fusion")]
    tf_failures = sum(1 for raw in fusion_raw if raw.startswith("tf_unavailable"))
    sync_failures = max(0, len(yolo_diags) + len(lidar_diags) - 2 * len(fusion_diags))
    tracking_lat = [float(event["data"].get("tracker_update_time_sec", 0.0)) * 1000.0 for event in tracking_diags]

    row.update({
        "measurement_duration_s": fmt(duration_s),
        "incoming_images": len(images),
        "incoming_pointclouds": len(pointclouds),
        "yolo_processed_frames": len(yolo_diags),
        "lidar_processed_frames": len(lidar_diags),
        "fusion_callbacks": len(fusion_diags),
        "fused_outputs": len(fused),
        "tracked_outputs": len(tracked),
        "yolo_fps": fmt(len(yolo_diags) / duration_s),
        "lidar_fps": fmt(len(lidar_diags) / duration_s),
        "fusion_fps": fmt(len(fusion_diags) / duration_s),
        "online_fused_fps": fmt(online_fused_fps),
        "online_tracked_fps": fmt(online_tracked_fps),
        "yolo_latency_mean_ms": fmt(mean(yolo_lat)),
        "yolo_latency_median_ms": fmt(median(yolo_lat)),
        "yolo_latency_p95_ms": fmt(percentile(yolo_lat, 0.95)),
        "lidar_latency_mean_ms": fmt(mean(lidar_lat)),
        "lidar_latency_median_ms": fmt(median(lidar_lat)),
        "lidar_latency_p95_ms": fmt(percentile(lidar_lat, 0.95)),
        "fusion_latency_mean_ms": fmt(mean(fusion_lat)),
        "fusion_latency_median_ms": fmt(median(fusion_lat)),
        "fusion_latency_p95_ms": fmt(percentile(fusion_lat, 0.95)),
        "tracking_latency_mean_ms": fmt(mean(tracking_lat)),
        "tracking_latency_median_ms": fmt(median(tracking_lat)),
        "tracking_latency_p95_ms": fmt(percentile(tracking_lat, 0.95)),
        "end_to_end_latency_mean_ms": fmt(mean(e2e_lat)),
        "end_to_end_latency_median_ms": fmt(median(e2e_lat)),
        "end_to_end_latency_p95_ms": fmt(percentile(e2e_lat, 0.95)),
        "end_to_end_latency_max_ms": fmt(max(e2e_lat) if e2e_lat else 0.0),
        "processing_ratio": fmt(len(fused) / max(len(pointclouds), 1)),
        "dropped_images": int(last_camera_data.get("dropped", 0) or 0),
        "dropped_pointclouds": int(last_lidar_data.get("dropped", 0) or 0),
        "synchronisation_failures": sync_failures,
        "tf_failures": tf_failures,
        "camera_info_failures": 0,
        "worker_errors": worker_errors,
        "run_status": "ok",
        "failure_reason": "",
    })
    return row


def run_one(args: argparse.Namespace, combo: tuple[str, str, str, str], test_mode: str, playback_rate: float, run_number: int) -> dict:
    exp_id, camera_model, lidar_detector, lidar_model = combo
    recording_slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", getattr(args, "recording", "") or args.bag_path.name)
    run_id = f"{test_mode}_{recording_slug}_{exp_id}_run{run_number}"
    raw_dir = args.results_dir / "raw_runs" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = raw_dir / "monitor_events.jsonl"
    launch_log = raw_dir / "launch.log"
    bag_log = raw_dir / "bag_play.log"
    diag_path = raw_dir / "collector_diagnostics.jsonl"
    metadata = {
        "experiment_id": exp_id,
        "recording": getattr(args, "recording", "") or args.bag_path.name,
        "bag_path": str(args.bag_path),
        "camera_model": camera_model,
        "lidar_detector": lidar_detector,
        "lidar_model": lidar_model,
        "test_mode": test_mode,
        "playback_rate": playback_rate,
        "run_number": run_number,
    }
    launch_cmd = ros_cmd(
        "ros2 launch late_fusion_pkg aghri_online_late_fusion.launch.py "
        "inference_mode:=live play_bag:=false enable_tracking:=true enable_visualisation:=false launch_rviz:=false "
        f"bag_path:={args.bag_path} camera_model:={camera_model} "
        f"lidar_detector:={lidar_detector} lidar_model:={lidar_model} "
        f"detector_device:={args.detector_device} mmdet3d_device:={args.mmdet3d_device} "
        f"worker_python:={args.worker_python} diagnostics_output:={diag_path} "
        "yolo_conf:=0.10 lidar_score_threshold:=0.20 "
        "tracking_min_hits:=3 tracking_max_age_frames:=10 "
        "tracking_association_3d_threshold:=0.03 tracking_association_2d_threshold:=0.40 "
        "tracking_support_2d_to_3d_threshold:=0.40 tracking_ego_motion_mode:=tf"
    )
    monitor_executable = installed_executable("online_performance_monitor")
    monitor_cmd = ros_cmd(
        f"{monitor_executable} --ros-args "
        f"-p output_jsonl:={monitor_path} -p use_sim_time:=true"
    )
    bag_cmd = ros_cmd(
        f"ros2 bag play {args.bag_path} -r {playback_rate} --clock "
        f"--qos-profile-overrides-path {args.qos_overrides}"
    )
    launch_proc = None
    monitor_proc = None
    bag_proc = None
    failure = ""
    try:
        launch_proc = launch_process(launch_cmd, launch_log)
        ready, reason = wait_for_ready(launch_log, launch_proc, args.ready_timeout_sec)
        if not ready:
            failure = reason
            return summarize_events([], args.warmup_frames, metadata, failure)
        monitor_proc = launch_process(monitor_cmd, raw_dir / "monitor.log")
        time.sleep(2.0)
        bag_proc = launch_process(bag_cmd, bag_log)
        try:
            bag_proc.wait(timeout=args.bag_timeout_sec if args.bag_timeout_sec > 0 else None)
        except subprocess.TimeoutExpired:
            failure = f"bag playback timed out after {args.bag_timeout_sec} sec"
        time.sleep(2.0)
    except Exception as exc:
        failure = str(exc)
    finally:
        close_process(bag_proc, timeout=15)
        close_process(monitor_proc, timeout=15)
        close_process(launch_proc, timeout=30)
    events = parse_monitor_events(monitor_path)
    row = summarize_events(events, args.warmup_frames, metadata, failure)
    (raw_dir / "run_summary.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.post_run_cooldown_sec > 0:
        time.sleep(args.post_run_cooldown_sec)
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize_rows(rows: list[dict]) -> list[dict]:
    metrics = [
        "online_tracked_fps",
        "online_fused_fps",
        "lidar_fps",
        "yolo_fps",
        "end_to_end_latency_mean_ms",
        "end_to_end_latency_p95_ms",
        "processing_ratio",
    ]
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["test_mode"], row["recording"], row["camera_model"], row["lidar_detector"], row["lidar_model"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        ok = [row for row in group if row.get("run_status") == "ok"]
        item = {
            "test_mode": key[0],
            "recording": key[1],
            "camera_model": key[2],
            "lidar_detector": key[3],
            "lidar_model": key[4],
            "runs": len(group),
            "ok_runs": len(ok),
            "failed_runs": len(group) - len(ok),
        }
        for metric in metrics:
            values = [float(row.get(metric, 0.0) or 0.0) for row in ok]
            item[f"{metric}_mean"] = fmt(mean(values))
            item[f"{metric}_std"] = fmt(stdev(values))
            item[f"{metric}_mean_pm_std"] = f"{mean(values):.3f} +/- {stdev(values):.3f}" if values else "n/a"
        failures = [row.get("failure_reason", "") for row in group if row.get("run_status") != "ok"]
        item["failure_reasons"] = " | ".join(reason for reason in failures if reason)
        summary.append(item)
    return summary


def write_summary_markdown(path: Path, summary: list[dict], config: dict) -> None:
    lines = [
        "# AGHRI Online FPS Benchmark Summary",
        "",
        f"Generated: {config['generated_at']}",
        "",
        "This benchmark uses `aghri_online_late_fusion.launch.py` with `inference_mode:=live`, no cached detector publishers, visualisation disabled, and RViz disabled.",
        "",
        f"Warm-up rule: exclude the first {config['warmup_frames']} successful live YOLO diagnostics and the first {config['warmup_frames']} successful live LiDAR diagnostics; measurement starts after both warm-up sets complete.",
        "",
    ]
    for mode in ("normal", "max"):
        rows = [row for row in summary if row["test_mode"] == mode]
        if not rows:
            continue
        lines += [
            f"## {mode.title()} Playback",
            "",
            "| Recording | Camera model | LiDAR model | Online tracked FPS | Online fused FPS | LiDAR FPS | YOLO FPS | Mean latency (ms) | P95 latency (ms) | Processing ratio | Runs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lidar_label = ("FT " if row["lidar_model"] == "finetuned" else "G ") + row["lidar_detector"].title()
            camera_label = "FT YOLO" if row["camera_model"] == "finetuned" else "G YOLO"
            lines.append(
                "| "
                + " | ".join([
                    row["recording"],
                    camera_label,
                    lidar_label,
                    row["online_tracked_fps_mean_pm_std"],
                    row["online_fused_fps_mean_pm_std"],
                    row["lidar_fps_mean_pm_std"],
                    row["yolo_fps_mean_pm_std"],
                    row["end_to_end_latency_mean_ms_mean_pm_std"],
                    row["end_to_end_latency_p95_ms_mean_pm_std"],
                    row["processing_ratio_mean_pm_std"],
                    f"{row['ok_runs']}/{row['runs']}",
                ])
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_environment(args: argparse.Namespace) -> dict:
    from late_fusion_pkg.live_detector_profiles import resolve_live_camera_profile, resolve_live_lidar_profile

    code, git_commit = sh("git rev-parse HEAD || true")
    _, git_status = sh("git status --short || true")
    _, ros_distro = sh("printenv ROS_DISTRO || true")
    _, nvidia = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true")
    _, cuda = sh("nvcc --version 2>/dev/null | tail -n 1 || true")
    _, bag_info = sh(ros_cmd(f"ros2 bag info {args.bag_path} || true"), timeout=60)
    cpu = Path("/proc/cpuinfo").read_text(errors="ignore").split("model name", 1)[-1].splitlines()[0].replace(":", "").strip() if Path("/proc/cpuinfo").exists() else ""
    mem = ""
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem = line
                break
    profiles = {}
    for camera_model in ("generic", "finetuned"):
        cam = resolve_live_camera_profile("aghri", camera_model, validate_paths=False)
        profiles[f"camera_{camera_model}"] = {
            "checkpoint_path": str(cam.checkpoint_path),
            "checkpoint_sha256": sha256(cam.checkpoint_path) if cam.checkpoint_path.exists() else "missing",
        }
    for detector in ("second", "pointpillars"):
        for lidar_model in ("generic", "finetuned"):
            lidar = resolve_live_lidar_profile("aghri", detector, lidar_model, validate_paths=False)
            profiles[f"lidar_{detector}_{lidar_model}"] = {
                "config_path": str(lidar.config_path),
                "config_sha256": sha256(lidar.config_path) if lidar.config_path.exists() else "missing",
                "checkpoint_path": str(lidar.checkpoint_path),
                "checkpoint_sha256": sha256(lidar.checkpoint_path) if lidar.checkpoint_path.exists() else "missing",
                "box_origin": lidar.box_origin,
                "score_threshold": lidar.score_threshold,
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit if code == 0 else "",
        "git_status_short": git_status,
        "ros_distro": ros_distro,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "worker_python": args.worker_python,
        "cuda_version": cuda,
        "gpu": nvidia,
        "cpu": cpu,
        "memory": mem,
        "bag_path": str(args.bag_path),
        "bag_metadata": bag_info,
        "qos_overrides": str(args.qos_overrides),
        "profiles": profiles,
    }


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "raw_runs").mkdir(parents=True, exist_ok=True)
    selected = [combo for combo in COMBINATIONS if not args.only or combo[0] in set(args.only)]
    modes = []
    if args.test_mode in ("normal", "both"):
        modes.append(("normal", args.normal_rate))
    if args.test_mode in ("max", "both"):
        modes.append(("max", args.max_throughput_rate))

    recording_names = list(args.recordings) or [args.bag_path.name]
    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bag_path": str(args.bag_path),
        "bag_root": str(args.bag_root),
        "recordings": recording_names,
        "normal_rate": args.normal_rate,
        "max_throughput_rate": args.max_throughput_rate,
        "repetitions": args.repetitions,
        "warmup_frames": args.warmup_frames,
        "visualisation": False,
        "launch_rviz": False,
        "inference_mode": "live",
        "combinations": [
            {"id": exp_id, "camera_model": camera, "lidar_detector": detector, "lidar_model": lidar}
            for exp_id, camera, detector, lidar in selected
        ],
    }
    (args.results_dir / "benchmark_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    environment = collect_environment(args)
    (args.results_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows: list[dict] = []
    if not args.dry_run:
        for test_mode, rate in modes:
            for recording in recording_names:
                bag_path = args.bag_root / recording if args.recordings else args.bag_path
                if not bag_path.exists():
                    raise FileNotFoundError(f"ROS bag directory does not exist: {bag_path}")
                run_args = copy(args)
                run_args.recording = recording
                run_args.bag_path = bag_path
                for combo in selected:
                    for run_number in range(1, args.repetitions + 1):
                        print(f"[benchmark] {test_mode} {recording} {combo[0]} run {run_number}/{args.repetitions} rate={rate}", flush=True)
                        row = run_one(run_args, combo, test_mode, rate, run_number)
                        rows.append(row)
                        write_csv(args.results_dir / "per_run_results.csv", rows, FIELDNAMES)

    if not rows and (args.results_dir / "per_run_results.csv").exists():
        with (args.results_dir / "per_run_results.csv").open("r", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    write_csv(args.results_dir / "per_run_results.csv", rows, FIELDNAMES)
    summary = summarize_rows(rows)
    summary_fields = list(summary[0].keys()) if summary else [
        "test_mode", "camera_model", "lidar_detector", "lidar_model", "runs", "ok_runs", "failed_runs"
    ]
    write_csv(args.results_dir / "summary_results.csv", summary, summary_fields)
    write_summary_markdown(args.results_dir / "summary_results.md", summary, config)
    report_lines = [
        "# AGHRI Online FPS Benchmark Final Report",
        "",
        f"Generated: {config['generated_at']}",
        "",
        "The benchmark runner is `tools/run_aghri_online_fps_matrix.py`.",
        "It launches `launch/aghri_online_late_fusion.launch.py` with `inference_mode:=live`, `play_bag:=false`, `enable_visualisation:=false`, and `launch_rviz:=false`, waits for `worker_ready` and `LIVE LiDAR inference enabled`, then starts `ros2 bag play` separately.",
        "",
        "Detector CSV caches and cached detector publishers are not used by this benchmark.",
        "",
        f"Rows collected: {len(rows)}",
        f"Dry run: {args.dry_run}",
        "",
        "See `per_run_results.csv`, `summary_results.csv`, `summary_results.md`, `benchmark_config.json`, `environment.json`, and `raw_runs/`.",
    ]
    (args.results_dir / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
