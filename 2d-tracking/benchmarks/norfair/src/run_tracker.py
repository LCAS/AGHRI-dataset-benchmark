"""
Run a Norfair tracking job from explicit CLI arguments or a YAML config.
Relative paths in configs and CLI arguments resolve from the 2d-tracking root.

Examples:
    python src/run_tracker.py --config configs/tracking/default.yaml
    python src/run_tracker.py --detections-json reports/runs/example/detections.json --out-mot reports/runs/example/tracks.txt
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from norfair import Detection, Tracker, drawing
from norfair.drawing.color import Palette

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = BENCH_ROOT / "configs" / "tracking" / "default.yaml"


@dataclass(frozen=True)
class TrackerConfig:
    """Normalized tracking settings after YAML loading and CLI overrides."""

    detections_json: Path
    frames_dir: Optional[Path] = None
    mot_output: Optional[Path] = None
    runtime_json: Optional[Path] = None
    output_video: Optional[Path] = None
    save_frames_dir: Optional[Path] = None
    frame_rate: float = 30.0
    distance_function: str = "iou"
    distance_threshold: float = 0.75
    hit_counter_max: int = 15
    initialization_delay: Optional[int] = None
    detection_threshold: float = 0.0
    min_detection_confidence: float = 0.25
    box_thickness: int = 2
    id_offset: int = 10


def _resolve_path(value: Optional[Any]) -> Optional[Path]:
    """Resolve repo-relative config paths from the 2d-tracking root."""

    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def _optional_int(value: Optional[Any]) -> Optional[int]:
    """Parse optional integer fields that may be blank or explicitly `None`."""

    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return int(value)


def load_config(path: Path) -> TrackerConfig:
    """Load one tracking YAML file into the internal config dataclass."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    detections_json = raw.get("detections_json")
    if not detections_json:
        raise ValueError("`detections_json` is required in the tracking config.")

    return TrackerConfig(
        detections_json=_resolve_path(detections_json),
        frames_dir=_resolve_path(raw.get("frames_dir")),
        mot_output=_resolve_path(raw.get("mot_output")),
        runtime_json=_resolve_path(raw.get("runtime_json")),
        output_video=_resolve_path(raw.get("output_video")),
        save_frames_dir=_resolve_path(raw.get("save_frames_dir")),
        frame_rate=float(raw.get("frame_rate", 30.0)),
        distance_function=str(raw.get("distance_function", "iou")),
        distance_threshold=float(raw.get("distance_threshold", 0.75)),
        hit_counter_max=int(raw.get("hit_counter_max", 15)),
        initialization_delay=_optional_int(raw.get("initialization_delay")),
        detection_threshold=float(raw.get("detection_threshold", 0.0)),
        min_detection_confidence=float(raw.get("min_detection_confidence", 0.25)),
        box_thickness=int(raw.get("box_thickness", 2)),
        id_offset=int(raw.get("id_offset", 10)),
    )

def _to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert project boxes into Norfair's expected xyxy corner format."""

    # The detector export uses xywh, but older or ad hoc files may already be xyxy.
    x, y, w, h = map(float, box)
    if w > 0 and h > 0:
        return x, y, x + w, y + h
    return x, y, w, h


def _labels_iter(labels_field: Any) -> Iterable[Tuple[str, Tuple[float, float, float, float], float]]:
    """Yield label, box, and original detector confidence."""

    # Accept the small set of record shapes that have appeared in this project so
    # the tracker can consume detector exports without one-off conversion steps.
    if labels_field is None:
        return []

    if isinstance(labels_field, dict):
        label = (
            labels_field.get("Class")
            or labels_field.get("class")
            or labels_field.get("label")
            or labels_field.get("Label")
        )
        box = (
            labels_field.get("BoundingBoxes")
            or labels_field.get("bbox")
            or labels_field.get("box")
        )
        if label is not None and box is not None:
            yield str(label), _to_xyxy(box), float(labels_field.get("Confidence", 1.0))
        return

    if isinstance(labels_field, list):
        for item in labels_field:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("Class")
                or item.get("class")
                or item.get("label")
                or item.get("Label")
            )
            box = item.get("BoundingBoxes") or item.get("bbox") or item.get("box")
            if label is None or box is None:
                continue
            yield str(label), _to_xyxy(box), float(item.get("Confidence", 1.0))
        return

    return []


def detections_from_json_record(record: Dict[str, Any], min_confidence: float = 0.25) -> List[Detection]:
    """Adapt one project detection record into the Norfair `Detection` objects."""

    detections: List[Detection] = []
    dropped = 0

    for label, (x1, y1, x2, y2), score in _labels_iter(record.get("Labels")):
        if score < min_confidence:
            continue
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        # Norfair expects a box as two points: top-left and bottom-right.
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        detections.append(Detection(points=points, scores=None, label=label))

    if dropped:
        warnings.warn(
            f"Dropped {dropped} invalid bbox(es) in record {record.get('File', '<no-file>')}",
            stacklevel=2,
        )

    return detections


class MotWriter:
    """Append tracked boxes to a MOTChallenge-format text file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write_frame(self, frame_idx: int, tracked_objects: Sequence[Any]) -> None:
        lines = []
        for tracked_object in tracked_objects:
            x1, y1 = tracked_object.estimate[0]
            x2, y2 = tracked_object.estimate[1]
            width = x2 - x1
            height = y2 - y1
            lines.append(
                f"{frame_idx},{tracked_object.id},{x1:f},{y1:f},{width:f},{height:f},1,-1,-1,-1\n"
            )

        with self.path.open("a", encoding="utf-8") as file:
            file.writelines(lines)


def build_runtime_summary(frame_count: int, tracking_time_seconds: float) -> Dict[str, object]:
    """Build one runtime summary row for the current tracking run."""

    tracking_time_per_frame_ms = (tracking_time_seconds * 1000.0 / frame_count) if frame_count else 0.0
    tracking_fps = (frame_count / tracking_time_seconds) if tracking_time_seconds > 0.0 else None
    return {
        "tracking_frame_count": frame_count,
        "tracking_time_seconds": tracking_time_seconds,
        "tracking_time_per_frame_ms": tracking_time_per_frame_ms,
        "tracking_fps": tracking_fps,
        "tracking_timing_scope": (
            "Per-frame tracking computation only; excludes frame disk I/O, MOT writing, "
            "rendering, rendered output writes, and MOT evaluation."
        ),
        "tracking_includes_input_adaptation": True,
        "tracking_includes_frame_io": False,
        "tracking_includes_mot_writing": False,
        "tracking_includes_rendering": False,
        "tracking_includes_output_writes": False,
        "tracking_includes_evaluation": False,
        "tracking_includes_process_startup": False,
    }


def write_runtime_summary(path: Optional[Path], frame_count: int, tracking_time_seconds: float) -> Dict[str, object]:
    """Persist one runtime summary when requested and return the summary row."""

    summary = build_runtime_summary(frame_count, tracking_time_seconds)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_config(cfg: TrackerConfig) -> None:
    """Check input/output paths before the tracking loop starts."""

    if not cfg.detections_json.is_file():
        raise FileNotFoundError(f"Detections JSON not found: {cfg.detections_json}")

    if cfg.frames_dir is None and (cfg.output_video is not None or cfg.save_frames_dir is not None):
        raise ValueError("`frames_dir` is required when writing videos or annotated frames.")

    if cfg.frames_dir is not None and not cfg.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {cfg.frames_dir}")


def run(cfg: TrackerConfig) -> None:
    """Run one Norfair pass over the frame-ordered detection export."""

    validate_config(cfg)
    # Read the full frame-ordered detections file once. Each record is one frame.
    records = json.loads(cfg.detections_json.read_text(encoding="utf-8"))

    tracker = Tracker(
        distance_function=cfg.distance_function,
        distance_threshold=cfg.distance_threshold,
        hit_counter_max=cfg.hit_counter_max,
        initialization_delay=cfg.initialization_delay,
        detection_threshold=cfg.detection_threshold,
    )

    mot_writer = MotWriter(cfg.mot_output) if cfg.mot_output else None
    video_writer = None
    first_frame_shape = None
    tracking_time_seconds = 0.0

    if cfg.output_video is not None:
        cfg.output_video.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_frames_dir is not None:
        cfg.save_frames_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, record in enumerate(records, start=1):
        # Convert the project JSON shape into Norfair detections for this frame.
        frame_start = time.perf_counter()
        detections = detections_from_json_record(record, cfg.min_detection_confidence)
        tracked_objects = tracker.update(detections)
        tracking_time_seconds += time.perf_counter() - frame_start

        if mot_writer is not None:
            mot_writer.write_frame(frame_idx, tracked_objects)

        if cfg.frames_dir is None:
            continue

        file_name = Path(record.get("File", f"frame_{frame_idx:05d}.png")).name
        frame_path = cfg.frames_dir / file_name
        frame = cv2.imread(str(frame_path))

        if frame is None:
            # Keep the export running even if a frame is missing on disk.
            if first_frame_shape is None:
                first_frame_shape = (720, 1280, 3)
            frame = np.zeros(first_frame_shape, dtype=np.uint8)
        else:
            first_frame_shape = frame.shape

        drawing.draw_boxes(
            frame,
            drawables=tracked_objects,
            draw_ids=False,
            color="by_id",
            thickness=cfg.box_thickness,
        )

        for tracked_object in tracked_objects:
            x1, y1 = tracked_object.estimate[0]
            color = Palette.choose_color(tracked_object.id)
            cv2.putText(
                frame,
                f"ID{tracked_object.id}",
                (int(x1), max(0, int(y1) - cfg.id_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        if cfg.output_video is not None:
            if video_writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(cfg.output_video),
                    fourcc,
                    cfg.frame_rate,
                    (width, height),
                )
            video_writer.write(frame)

        if cfg.save_frames_dir is not None:
            cv2.imwrite(str(cfg.save_frames_dir / file_name), frame)

    if video_writer is not None:
        video_writer.release()

    write_runtime_summary(cfg.runtime_json, len(records), tracking_time_seconds)


def parse_args() -> argparse.Namespace:
    """Define the CLI that layers ad hoc overrides on top of the YAML config."""

    parser = argparse.ArgumentParser(
        description="Run Norfair tracking from a config or explicit args.",
        epilog=(
            "Examples:\n"
            "  python src/run_tracker.py --config configs/tracking/default.yaml\n"
            "  python src/run_tracker.py --detections-json reports/runs/example/detections.json "
            "--out-mot reports/runs/example/tracks.txt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a tracking YAML config. Relative data and report paths resolve from 2d-tracking.",
    )
    parser.add_argument("--detections-json", type=Path, help="Detections JSON file to track.")
    parser.add_argument("--frames-dir", type=Path, help="Optional directory containing video frames.")
    parser.add_argument("--out-mot", type=Path, help="Output MOT file path.")
    parser.add_argument("--runtime-json", type=Path, help="Optional JSON path for tracker runtime metrics.")
    parser.add_argument("--out-video", type=Path, help="Output MP4 path.")
    parser.add_argument("--save-frames-dir", type=Path, help="Directory for annotated output frames.")
    parser.add_argument("--frame-rate", type=float, help="Output video FPS.")
    parser.add_argument("--distance-function", type=str, help="Norfair distance function.")
    parser.add_argument("--distance-threshold", type=float, help="Tracking match threshold.")
    parser.add_argument("--hit-counter-max", type=int, help="Maximum missed frames before deletion.")
    parser.add_argument(
        "--initialization-delay",
        type=str,
        help="Frames required to confirm a track. Use `None` to keep Norfair default behavior.",
    )
    parser.add_argument("--detection-threshold", type=float, help="Detection confidence threshold.")
    parser.add_argument("--box-thickness", type=int, help="Rendered bounding box thickness.")
    parser.add_argument("--id-offset", type=int, help="Vertical text offset in pixels.")
    return parser.parse_args()


def merge_cli_overrides(cfg: TrackerConfig, args: argparse.Namespace) -> TrackerConfig:
    """Overlay any explicit CLI flags on top of the loaded YAML config."""

    # CLI flags intentionally override YAML so cluster scripts can reuse one base config.
    updates = {}
    if args.detections_json is not None:
        updates["detections_json"] = _resolve_path(args.detections_json)
    if args.frames_dir is not None:
        updates["frames_dir"] = _resolve_path(args.frames_dir)
    if args.out_mot is not None:
        updates["mot_output"] = _resolve_path(args.out_mot)
    if args.runtime_json is not None:
        updates["runtime_json"] = _resolve_path(args.runtime_json)
    if args.out_video is not None:
        updates["output_video"] = _resolve_path(args.out_video)
    if args.save_frames_dir is not None:
        updates["save_frames_dir"] = _resolve_path(args.save_frames_dir)
    if args.frame_rate is not None:
        updates["frame_rate"] = float(args.frame_rate)
    if args.distance_function is not None:
        updates["distance_function"] = args.distance_function
    if args.distance_threshold is not None:
        updates["distance_threshold"] = float(args.distance_threshold)
    if args.hit_counter_max is not None:
        updates["hit_counter_max"] = int(args.hit_counter_max)
    if args.initialization_delay is not None:
        updates["initialization_delay"] = _optional_int(args.initialization_delay)
    if args.detection_threshold is not None:
        updates["detection_threshold"] = float(args.detection_threshold)
    if args.box_thickness is not None:
        updates["box_thickness"] = int(args.box_thickness)
    if args.id_offset is not None:
        updates["id_offset"] = int(args.id_offset)
    return replace(cfg, **updates)


def main() -> None:
    """CLI entrypoint used by local scripts and SLURM launchers."""

    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
