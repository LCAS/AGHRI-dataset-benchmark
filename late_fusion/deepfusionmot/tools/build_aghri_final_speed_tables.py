#!/usr/bin/env python3
"""Build final AGHRI speed tables from offline and live benchmark outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


COMBO_ORDER = ["S1", "S2", "S3", "S4", "P1", "P2", "P3", "P4"]
OUT_FIELDS = [
    "combination_id",
    "combination_label",
    "recording",
    "tracker_only_fps",
    "cached_pipeline_fps",
    "live_ros2_end_to_end_fps",
    "yolo_latency_mean_ms",
    "yolo_latency_p95_ms",
    "lidar_latency_mean_ms",
    "lidar_latency_p95_ms",
    "tracking_latency_mean_ms",
    "tracking_latency_p95_ms",
    "live_processed_outputs",
    "live_measurement_duration_s",
    "live_status",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict], fields: list[str] = OUT_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _combo_sort(row: dict) -> tuple[int, str]:
    combo = str(row.get("combination_id", ""))
    return (COMBO_ORDER.index(combo) if combo in COMBO_ORDER else 999, str(row.get("recording", "")))


def _group_live_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("run_status") != "ok":
            continue
        recording = str(row.get("recording", ""))
        combo = str(row.get("experiment_id", ""))
        if not recording or not combo:
            continue
        groups.setdefault((combo, recording), []).append(row)

    out: dict[tuple[str, str], dict] = {}
    for key, group in groups.items():
        outputs = [_float(row, "tracked_outputs") for row in group]
        durations = [_float(row, "measurement_duration_s") for row in group]
        total_outputs = sum(outputs)
        total_duration = sum(durations)
        out[key] = {
            "live_ros2_end_to_end_fps": _stream_fps(group, "online_tracked_fps", "tracked_outputs")
            or (total_outputs / total_duration if total_duration else 0.0),
            "yolo_latency_mean_ms": _weighted_mean(group, "yolo_latency_mean_ms", "yolo_processed_frames"),
            "yolo_latency_p95_ms": statistics.mean([_float(row, "yolo_latency_p95_ms") for row in group]),
            "lidar_latency_mean_ms": _weighted_mean(group, "lidar_latency_mean_ms", "lidar_processed_frames"),
            "lidar_latency_p95_ms": statistics.mean([_float(row, "lidar_latency_p95_ms") for row in group]),
            "tracking_latency_mean_ms": _weighted_mean(group, "tracking_latency_mean_ms", "tracked_outputs"),
            "tracking_latency_p95_ms": statistics.mean([_float(row, "tracking_latency_p95_ms") for row in group]),
            "live_processed_outputs": int(total_outputs),
            "live_measurement_duration_s": total_duration,
            "live_status": f"{len(group)} ok run(s)",
        }
    return out


def _stream_fps(rows: list[dict], fps_key: str, count_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    fallback = []
    for row in rows:
        fps = _float(row, fps_key)
        count = max(0.0, _float(row, count_key) - 1.0)
        if fps <= 0.0:
            continue
        fallback.append(fps)
        if count <= 0.0:
            continue
        numerator += count
        denominator += count / fps
    if numerator > 0.0 and denominator > 0.0:
        return numerator / denominator
    return statistics.mean(fallback) if fallback else 0.0


def _weighted_mean(rows: list[dict], value_key: str, weight_key: str) -> float:
    weighted = sum(_float(row, value_key) * max(1.0, _float(row, weight_key)) for row in rows)
    weights = sum(max(1.0, _float(row, weight_key)) for row in rows)
    return weighted / weights if weights else 0.0


def build_per_recording_rows(offline_rows: list[dict], live_rows: list[dict]) -> list[dict]:
    live_by_key = _group_live_rows(live_rows)
    out = []
    for offline in offline_rows:
        combo = str(offline["combination_id"])
        recording = str(offline["recording"])
        live = live_by_key.get((combo, recording), {})
        row = {
            "combination_id": combo,
            "combination_label": offline.get("combination_label", ""),
            "recording": recording,
            "tracker_only_fps": _fmt(_float(offline, "tracker_only_fps")),
            "cached_pipeline_fps": _fmt(_float(offline, "cached_pipeline_fps")),
            "live_ros2_end_to_end_fps": _fmt(float(live.get("live_ros2_end_to_end_fps", 0.0))),
            "yolo_latency_mean_ms": _fmt(float(live.get("yolo_latency_mean_ms", 0.0))),
            "yolo_latency_p95_ms": _fmt(float(live.get("yolo_latency_p95_ms", 0.0))),
            "lidar_latency_mean_ms": _fmt(float(live.get("lidar_latency_mean_ms", 0.0))),
            "lidar_latency_p95_ms": _fmt(float(live.get("lidar_latency_p95_ms", 0.0))),
            "tracking_latency_mean_ms": _fmt(float(live.get("tracking_latency_mean_ms", 0.0))),
            "tracking_latency_p95_ms": _fmt(float(live.get("tracking_latency_p95_ms", 0.0))),
            "live_processed_outputs": str(live.get("live_processed_outputs", 0)),
            "live_measurement_duration_s": _fmt(float(live.get("live_measurement_duration_s", 0.0))),
            "live_status": str(live.get("live_status", "missing live run")),
        }
        out.append(row)
    return sorted(out, key=_combo_sort)


def build_aggregate_rows(per_rows: list[dict], offline_agg_rows: list[dict]) -> list[dict]:
    offline_by_combo = {row["combination_id"]: row for row in offline_agg_rows}
    groups: dict[str, list[dict]] = {}
    for row in per_rows:
        groups.setdefault(row["combination_id"], []).append(row)

    out = []
    for combo in COMBO_ORDER:
        if combo not in groups:
            continue
        group = groups[combo]
        offline = offline_by_combo.get(combo, {})
        live_outputs = sum(int(float(row["live_processed_outputs"])) for row in group)
        live_duration = sum(float(row["live_measurement_duration_s"]) for row in group)
        row = {
            "combination_id": combo,
            "combination_label": group[0]["combination_label"],
            "recording": "COMBINED_TEST_SET",
            "tracker_only_fps": _fmt(_float(offline, "tracker_only_fps")),
            "cached_pipeline_fps": _fmt(_float(offline, "cached_pipeline_fps")),
            "live_ros2_end_to_end_fps": _fmt(
                _stream_fps(group, "live_ros2_end_to_end_fps", "live_processed_outputs")
                or (live_outputs / live_duration if live_duration else 0.0)
            ),
            "yolo_latency_mean_ms": _fmt(_weighted_from_table(group, "yolo_latency_mean_ms", "live_processed_outputs")),
            "yolo_latency_p95_ms": _fmt(statistics.mean([float(row["yolo_latency_p95_ms"]) for row in group])),
            "lidar_latency_mean_ms": _fmt(_weighted_from_table(group, "lidar_latency_mean_ms", "live_processed_outputs")),
            "lidar_latency_p95_ms": _fmt(statistics.mean([float(row["lidar_latency_p95_ms"]) for row in group])),
            "tracking_latency_mean_ms": _fmt(_weighted_from_table(group, "tracking_latency_mean_ms", "live_processed_outputs")),
            "tracking_latency_p95_ms": _fmt(statistics.mean([float(row["tracking_latency_p95_ms"]) for row in group])),
            "live_processed_outputs": str(live_outputs),
            "live_measurement_duration_s": _fmt(live_duration),
            "live_status": f"{sum(1 for row in group if row['live_status'] != 'missing live run')}/{len(group)} bag rows with live data",
        }
        out.append(row)
    return out


def _weighted_from_table(rows: list[dict], value_key: str, weight_key: str) -> float:
    weighted = sum(float(row[value_key]) * max(1.0, float(row[weight_key])) for row in rows)
    weights = sum(max(1.0, float(row[weight_key])) for row in rows)
    return weighted / weights if weights else 0.0


def _write_md(path: Path, title: str, rows: list[dict]) -> None:
    lines = [
        f"# {title}",
        "",
        "| ID | Combination | Recording | Tracker-only FPS | Cached pipeline FPS | Live ROS2 end-to-end FPS | YOLO mean ms | YOLO p95 ms | LiDAR mean ms | LiDAR p95 ms | Tracking mean ms | Tracking p95 ms | Live status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {combination_id} | {combination_label} | {recording} | {tracker_only_fps} | {cached_pipeline_fps} | "
            "{live_ros2_end_to_end_fps} | {yolo_latency_mean_ms} | {yolo_latency_p95_ms} | "
            "{lidar_latency_mean_ms} | {lidar_latency_p95_ms} | {tracking_latency_mean_ms} | "
            "{tracking_latency_p95_ms} | {live_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `tracker-only FPS` and `cached pipeline FPS` are from the offline cached-detector tracking runs.",
            "- `live ROS2 end-to-end FPS` is measured from `/deepfusionmot/tracked_3d` output timing in live ROS 2 playback.",
            "- Aggregate live FPS is output-count weighted across the six test bags.",
            "- YOLO and LiDAR latency values come from the live detector node diagnostics.",
            "- TrackEval does not compute these speed values; TrackEval only computes tracking metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-per-recording",
        type=Path,
        default=Path("results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy_metrics/deepfusionmot_metrics_per_recording.csv"),
    )
    parser.add_argument(
        "--offline-aggregate",
        type=Path,
        default=Path("results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy_metrics/deepfusionmot_metrics.csv"),
    )
    parser.add_argument(
        "--live-per-run",
        type=Path,
        default=Path("results/aghri_final_speed/live_ros2_tracking_benchmark/per_run_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/aghri_final_speed"))
    args = parser.parse_args()

    offline_per = _read_csv(args.offline_per_recording)
    offline_agg = _read_csv(args.offline_aggregate)
    live_rows = _read_csv(args.live_per_run) if args.live_per_run.exists() else []
    per_rows = build_per_recording_rows(offline_per, live_rows)
    agg_rows = build_aggregate_rows(per_rows, offline_agg)
    _write_csv(args.output_dir / "speed_table_per_recording.csv", per_rows)
    _write_csv(args.output_dir / "speed_table_aggregate.csv", agg_rows)
    _write_md(args.output_dir / "speed_table_per_recording.md", "AGHRI Final Speed Table Per Recording", per_rows)
    _write_md(args.output_dir / "speed_table_aggregate.md", "AGHRI Final Speed Table Aggregate", agg_rows)
    print(f"output_dir={args.output_dir}")
    print(f"per_recording_rows={len(per_rows)}")
    print(f"aggregate_rows={len(agg_rows)}")
    print(f"live_rows={len(live_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
