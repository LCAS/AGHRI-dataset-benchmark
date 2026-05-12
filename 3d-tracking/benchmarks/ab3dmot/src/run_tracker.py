"""
Run AB3DMOT 3D tracking on per-scene detection JSONs.

Each scene is a subdirectory under `detections_dir` containing per-frame JSON files
(e.g. 000000.json, 000001.json …) in the format produced by dump_detections_3d.py.

One MOT3D CSV is written per scene to `mot_output_dir`.

Usage:
    python src/run_tracker.py --config configs/tracking/aghri_ab3dmot.yaml
    python src/run_tracker.py --config configs/tracking/aghri_ab3dmot.yaml \\
        --detections-dir /path/to/detections \\
        --mot-output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
TRACKING_ROOT = Path(__file__).resolve().parents[3]
COMMON_ROOT = TRACKING_ROOT / "common" / "mot3d"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(COMMON_ROOT))

from ab3dmot_tracker import AB3DMOT
from mot3d_writer import Mot3dWriter

DEFAULT_CONFIG = BENCH_ROOT / "configs" / "tracking" / "aghri_ab3dmot.yaml"


@dataclass(frozen=True)
class TrackerConfig:
    detections_dir: Path
    mot_output_dir: Path
    runtime_json: Optional[Path] = None
    score_threshold: float = 0.25
    max_age: int = 2
    min_hits: int = 3
    iou_threshold: float = 0.1


def _resolve(value) -> Optional[Path]:
    if value in (None, ""):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else (TRACKING_ROOT / p).resolve()


def load_config(path: Path) -> TrackerConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for field in ("detections_dir", "mot_output_dir"):
        if not raw.get(field):
            raise ValueError(f"`{field}` is required in config.")
    return TrackerConfig(
        detections_dir=_resolve(raw["detections_dir"]),
        mot_output_dir=_resolve(raw["mot_output_dir"]),
        runtime_json=_resolve(raw.get("runtime_json")),
        score_threshold=float(raw.get("score_threshold", 0.25)),
        max_age=int(raw.get("max_age", 2)),
        min_hits=int(raw.get("min_hits", 3)),
        iou_threshold=float(raw.get("iou_threshold", 0.1)),
    )


def _load_frame_detections(json_path: Path):
    """Return (boxes, scores) from a per-frame detection JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not data:
        return np.empty((0, 7), dtype=np.float64), np.empty(0, dtype=np.float64)
    boxes, scores = [], []
    for det in data:
        boxes.append([det["x"], det["y"], det["z"], det["l"], det["w"], det["h"], det["yaw"]])
        scores.append(det["score"])
    return np.array(boxes, dtype=np.float64), np.array(scores, dtype=np.float64)


def run_scene(
    tracker: AB3DMOT,
    scene_dir: Path,
    mot_output: Path,
    cfg: TrackerConfig,
) -> Dict:
    """Track one scene; return runtime stats dict."""
    frame_files = sorted(scene_dir.glob("*.json"))
    if not frame_files:
        return {"frames": 0, "tracking_time_seconds": 0.0}

    tracker.reset()
    writer = Mot3dWriter(mot_output)
    tracking_time = 0.0

    for frame_id, json_path in enumerate(frame_files, start=1):
        boxes, scores = _load_frame_detections(json_path)
        t0 = time.perf_counter()
        tracks = tracker.update(boxes, scores)
        tracking_time += time.perf_counter() - t0
        writer.write_frame(frame_id, tracks)

    return {"frames": len(frame_files), "tracking_time_seconds": tracking_time}


def run(cfg: TrackerConfig) -> None:
    if not cfg.detections_dir.is_dir():
        raise FileNotFoundError(f"detections_dir not found: {cfg.detections_dir}")

    scene_dirs = sorted(d for d in cfg.detections_dir.iterdir() if d.is_dir())
    if not scene_dirs:
        raise RuntimeError(f"No scene subdirectories found in {cfg.detections_dir}")

    tracker = AB3DMOT(
        max_age=cfg.max_age,
        min_hits=cfg.min_hits,
        iou_threshold=cfg.iou_threshold,
        score_threshold=cfg.score_threshold,
    )

    total_frames = 0
    total_time = 0.0
    print(f"[AB3DMOT] Tracking {len(scene_dirs)} scenes -> {cfg.mot_output_dir}")

    for scene_dir in scene_dirs:
        mot_path = cfg.mot_output_dir / f"{scene_dir.name}.csv"
        stats = run_scene(tracker, scene_dir, mot_path, cfg)
        total_frames += stats["frames"]
        total_time += stats["tracking_time_seconds"]
        print(f"  {scene_dir.name}: {stats['frames']} frames")

    fps = total_frames / total_time if total_time > 0 else 0.0
    print(f"[AB3DMOT] Done. {total_frames} frames in {total_time:.2f}s ({fps:.1f} FPS)")

    if cfg.runtime_json is not None:
        summary = {
            "tracker": "ab3dmot",
            "total_frames": total_frames,
            "tracking_time_seconds": total_time,
            "tracking_fps": fps,
            "max_age": cfg.max_age,
            "min_hits": cfg.min_hits,
            "iou_threshold": cfg.iou_threshold,
            "score_threshold": cfg.score_threshold,
        }
        cfg.runtime_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.runtime_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AB3DMOT 3D tracking on per-scene detection JSONs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--detections-dir", type=Path, default=None)
    parser.add_argument("--mot-output-dir", type=Path, default=None)
    parser.add_argument("--runtime-json", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--max-age", type=int, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    return parser.parse_args()


def merge_cli(cfg: TrackerConfig, args: argparse.Namespace) -> TrackerConfig:
    updates = {}
    if args.detections_dir is not None:
        updates["detections_dir"] = _resolve(args.detections_dir)
    if args.mot_output_dir is not None:
        updates["mot_output_dir"] = _resolve(args.mot_output_dir)
    if args.runtime_json is not None:
        updates["runtime_json"] = _resolve(args.runtime_json)
    if args.score_threshold is not None:
        updates["score_threshold"] = args.score_threshold
    if args.max_age is not None:
        updates["max_age"] = args.max_age
    if args.min_hits is not None:
        updates["min_hits"] = args.min_hits
    if args.iou_threshold is not None:
        updates["iou_threshold"] = args.iou_threshold
    return replace(cfg, **updates)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
