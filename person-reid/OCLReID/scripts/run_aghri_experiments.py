#!/usr/bin/env python3
"""
Automate OCLReID inference and evaluation for AGHRI validation/test splits.

For every selected split, scene, camera, method, and annotated target identity:
1. Locate the generated MP4 and matching annotation JSON.
2. Verify image/annotation/video alignment.
3. Find the first frame where the target is annotated.
4. Write its initial x,y,width,height box automatically.
5. Run OCLReID with scripts/run_single_video.py.
6. Evaluate from the frame after initialization.
7. Save per-run metrics and aggregate results.

Run from the OCLReID project root, for example:
    conda activate oclreid
    cd ~/Desktop/OCLReID
    python scripts/run_aghri_experiments.py --splits val --methods part-OCLReID --max-runs 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aghri_alignment import (
    collapse_duplicate_boxes,
    load_frame_manifest,
    validate_manifest_against_sources,
)


# ============================================================
# DEFAULT PATHS AND DATASET SETTINGS
# ============================================================

DEFAULT_DATASET_ROOT = Path(
    "/media/prabuddhi/Backup2/Updated Dataset_PW"
)

DEFAULT_VIDEO_ROOT = Path(
    "/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos"
)

DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "automated_runs"
)

DEFAULT_RUN_VIDEO = PROJECT_ROOT / "scripts" / "run_single_video.py"
DEFAULT_EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_aghri_results.py"
DEFAULT_REID_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "reid" / "resnet18.pth"

DATASET_PARTS = [
    "dataset_part1",
    "dataset_part2",
    "dataset_part3",
    "dataset_part4",
]

CAMERA_FPS = {
    "cam_fish_front": 30.0,
    "cam_fish_left": 30.0,
    "cam_fish_right": 30.0,
    "cam_zed_rgb": 15.0,
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
}

SUPPORTED_METHODS = [
    "part-OCLReID",
    "global-OCLReID",
    "rpf-ReID",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def safe_name(value: str) -> str:
    """
    Convert a value into a filesystem-safe directory name.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value))
    return cleaned.strip("_") or "unknown"


def class_sort_key(value: str) -> Tuple[int, Any]:
    """
    Sort numeric identities numerically and other values alphabetically.
    """
    text = str(value)

    if text.isdigit():
        return 0, int(text)

    return 1, text


def load_split_scene_names(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file does not exist: {split_file}"
        )

    names = []

    with split_file.open("r", encoding="utf-8-sig") as file:
        for line in file:
            name = line.strip()

            if not name or name.startswith("#"):
                continue

            names.append(name)

    names = list(dict.fromkeys(names))

    if not names:
        raise ValueError(
            f"No scene names were found in: {split_file}"
        )

    return names


def build_scene_index(
    dataset_root: Path,
) -> Dict[str, List[Tuple[str, Path]]]:
    index: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for part_name in DATASET_PARTS:
        part_path = dataset_root / part_name

        if not part_path.exists():
            print(f"[WARNING] Dataset part missing: {part_path}")
            continue

        for scene_path in part_path.iterdir():
            if scene_path.is_dir():
                index[scene_path.name].append(
                    (part_name, scene_path)
                )

    return index


def get_sorted_images(camera_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in camera_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ],
        key=lambda path: path.name,
    )


def get_video_metadata(video_path: Path) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to inspect video with ffprobe: {video_path}\n"
                f"{result.stderr}"
            )
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        frame_text = (
            stream.get("nb_read_frames")
            or stream.get("nb_frames")
            or 0
        )
        rate_text = stream.get("r_frame_rate") or "0/1"
        numerator, denominator = [
            float(value)
            for value in rate_text.split("/")
        ]
        return {
            "frame_count": int(frame_text),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": (
                numerator / denominator
                if denominator else 0.0
            ),
        }

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    try:
        frame_count = int(
            round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        width = int(
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        )

        height = int(
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

    finally:
        capture.release()

    return {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
    }


# ============================================================
# ANNOTATION AND ALIGNMENT HELPERS
# ============================================================

def extract_target_classes(
    annotations: Sequence[Dict[str, Any]],
) -> List[str]:
    classes = set()

    for record in annotations:
        for label in record.get("Labels", []):
            target_class = str(label.get("Class", "")).strip()
            bbox = label.get("BoundingBoxes")

            if (
                target_class
                and isinstance(bbox, list)
                and len(bbox) == 4
            ):
                classes.add(target_class)

    return sorted(classes, key=class_sort_key)


def get_target_initialization_candidates(
    annotations: Sequence[Dict[str, Any]],
    target_class: str,
    manifest: Dict[str, Any],
    image_width: int,
    image_height: int,
    min_width: float,
    min_height: float,
    min_area_ratio: float,
    border_margin: float,
    retry_frame_step: int,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """
    Return suitable target-initialization candidates in chronological order.

    A candidate must be sufficiently large and must not be heavily clipped by
    the image border. Nearby eligible frames are spaced apart so retries do not
    repeatedly test almost identical views.
    """
    eligible: List[Dict[str, Any]] = []
    image_area = float(image_width * image_height)

    file_to_video_frame = manifest["file_to_video_frame"]

    for annotation_record_index, record in enumerate(annotations):
        filename = str(record.get("File", ""))
        if filename not in file_to_video_frame:
            continue

        labels = record.get("Labels", [])
        if not isinstance(labels, list):
            labels = []

        gt_status, bbox, _ = collapse_duplicate_boxes(
            labels,
            target_class,
        )

        if gt_status != "visible" or bbox is None:
            continue

        x, y, width, height = [
            float(value)
            for value in bbox
        ]

        if width <= 0 or height <= 0:
            continue

        x2 = x + width
        y2 = y + height
        area_ratio = (
            width * height / image_area
            if image_area > 0
            else 0.0
        )

        sufficiently_large = (
            width >= min_width
            and height >= min_height
            and area_ratio >= min_area_ratio
        )

        inside_image = (
            x >= border_margin
            and y >= border_margin
            and x2 <= image_width - border_margin
            and y2 <= image_height - border_margin
        )

        if not sufficiently_large or not inside_image:
            continue

        eligible.append(
            {
                "annotation_record_index":
                    annotation_record_index,
                "video_frame_index":
                    int(file_to_video_frame[filename]),
                "bbox_xywh": [x, y, width, height],
                "width": width,
                "height": height,
                "area_ratio": area_ratio,
                "file": filename,
            }
        )

    if not eligible:
        return []

    # Keep candidates chronologically ordered while avoiding near-duplicate
    # attempts from immediately adjacent frames.
    selected: List[Dict[str, Any]] = []
    last_selected_frame: Optional[int] = None

    for candidate in eligible:
        frame_index = int(candidate["video_frame_index"])

        if (
            last_selected_frame is not None
            and frame_index - last_selected_frame < retry_frame_step
        ):
            continue

        selected.append(candidate)
        last_selected_frame = frame_index

        if len(selected) >= max_candidates:
            break

    return selected


def prediction_has_valid_initial_target(
    prediction_path: Path,
    frame_index: int,
) -> bool:
    """
    Confirm that OCLReID produced a valid target on the initialization frame.
    """
    if not prediction_path.exists():
        return False

    try:
        predictions = read_json(prediction_path)
    except Exception:
        return False

    possible_keys = [
        f"{frame_index:06d}.jpg",
        f"{frame_index:06d}",
    ]

    frame_result = None

    for key in possible_keys:
        if key in predictions:
            frame_result = predictions[key]
            break

    if not isinstance(frame_result, dict):
        return False

    target_info = frame_result.get("target_info", [])

    if not isinstance(target_info, list) or len(target_info) < 6:
        return False

    try:
        tracker_id = int(target_info[0])
        x1 = float(target_info[1])
        y1 = float(target_info[2])
        x2 = float(target_info[3])
        y2 = float(target_info[4])
    except (TypeError, ValueError):
        return False

    return (
        tracker_id >= 0
        and x2 > x1
        and y2 > y1
    )


def count_target_visible_frames(
    annotations: Sequence[Dict[str, Any]],
    target_class: str,
) -> int:
    count = 0

    for record in annotations:
        found = False

        for label in record.get("Labels", []):
            label_class = str(
                label.get("Class", "")
            ).strip()

            bbox = label.get("BoundingBoxes")

            if (
                label_class == target_class
                and isinstance(bbox, list)
                and len(bbox) == 4
                and float(bbox[2]) > 0
                and float(bbox[3]) > 0
            ):
                found = True
                break

        if found:
            count += 1

    return count


def verify_alignment(
    annotations: Sequence[Dict[str, Any]],
    camera_dir: Path,
    video_path: Path,
    manifest_path: Path,
    expected_fps: float,
) -> Dict[str, Any]:
    """
    Verify filename-based annotation to full-video frame alignment.

    Extra unannotated sensor images are allowed; missing annotation-referenced
    images and inconsistent manifests/videos remain fatal.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing frame manifest: {manifest_path}"
        )

    video_metadata = get_video_metadata(video_path)
    manifest = load_frame_manifest(manifest_path)
    alignment = validate_manifest_against_sources(
        manifest=manifest,
        camera_dir=camera_dir,
        annotations=list(annotations),
        expected_fps=expected_fps,
        video_frame_count=video_metadata["frame_count"],
    )

    return {
        **alignment,
        **video_metadata,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
    }


# ============================================================
# SUBPROCESS HELPERS
# ============================================================

def run_logged_command(
    command: List[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    printable = " ".join(
        subprocess.list2cmdline([item])
        for item in command
    )

    print(f"      Command: {printable}")

    if dry_run:
        return

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(f"Command: {printable}\n\n")
        log_file.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code "
            f"{process.returncode}. See: {log_path}"
        )


# ============================================================
# ONE TARGET RUN
# ============================================================

def run_one_target(
    *,
    split_name: str,
    method: str,
    part_name: str,
    scene_name: str,
    scene_path: Path,
    camera_name: str,
    video_path: Path,
    annotation_path: Path,
    target_class: str,
    annotations: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    results_root: Path,
    run_video_path: Path,
    evaluator_path: Path,
    reid_checkpoint: Optional[Path],
    overwrite: bool,
    save_frames: bool,
    show_live: bool,
    save_visualizations: bool,
    association_mode: str,
    association_reid_threshold: float,
    association_reid_margin: float,
    association_min_bbox_score: float,
    association_min_visible_parts: int,
    dry_run: bool,
    minimum_visible_frames: int,
    min_init_width: float,
    min_init_height: float,
    min_init_area_ratio: float,
    init_border_margin: float,
    max_init_attempts: int,
    init_retry_frame_step: int,
) -> Dict[str, Any]:
    visible_frames = count_target_visible_frames(
        annotations,
        target_class,
    )

    base_record: Dict[str, Any] = {
        "split": split_name,
        "method": method,
        "dataset_part": part_name,
        "scene": scene_name,
        "camera": camera_name,
        "target_class": target_class,
        "visible_frames": visible_frames,
        "video": str(video_path),
        "annotations": str(annotation_path),
        "frame_manifest": metadata.get("manifest_path", ""),
        "reid_checkpoint": str(reid_checkpoint) if reid_checkpoint else "",
        "association_mode": (
            association_mode if method == "part-OCLReID" else "not_applicable"
        ),
        "association_reid_threshold": (
            association_reid_threshold if method == "part-OCLReID" else ""
        ),
        "association_reid_margin": (
            association_reid_margin if method == "part-OCLReID" else ""
        ),
        "association_min_bbox_score": (
            association_min_bbox_score if method == "part-OCLReID" else ""
        ),
        "association_min_visible_parts": (
            association_min_visible_parts if method == "part-OCLReID" else ""
        ),
        "visualization_video": "",
        "visualization_status": (
            "requested"
            if save_visualizations else "not_requested"
        ),
        "status": "pending",
        "error": "",
    }

    if visible_frames < minimum_visible_frames:
        base_record["status"] = "skipped_too_few_visible_frames"
        return base_record

    candidates = get_target_initialization_candidates(
        annotations=annotations,
        target_class=target_class,
        manifest=metadata["manifest"],
        image_width=int(metadata["width"]),
        image_height=int(metadata["height"]),
        min_width=min_init_width,
        min_height=min_init_height,
        min_area_ratio=min_init_area_ratio,
        border_margin=init_border_margin,
        retry_frame_step=init_retry_frame_step,
        max_candidates=max_init_attempts,
    )

    if not candidates:
        base_record["status"] = "skipped_no_suitable_initialization"
        base_record["error"] = (
            "No target box satisfied the initialization filters: "
            f"width>={min_init_width}, height>={min_init_height}, "
            f"area_ratio>={min_init_area_ratio}, "
            f"border_margin>={init_border_margin}."
        )
        return base_record

    target_dir = (
        results_root
        / split_name
        / method
        / part_name
        / scene_name
        / camera_name
        / f"class_{safe_name(target_class)}"
    )

    prediction_path = target_dir / "predictions.json"
    bbox_path = target_dir / "init_bbox.txt"
    metadata_path = target_dir / "run_metadata.json"
    attempts_path = target_dir / "initialization_attempts.json"
    visualization_video_path = target_dir / "inference_visualization.mp4"
    evaluation_dir = target_dir / "evaluation"
    summary_path = evaluation_dir / "summary_metrics.json"
    evaluation_log = target_dir / "evaluation.log"
    frame_output_dir = target_dir / "frames"

    base_record["result_dir"] = str(target_dir)
    if save_visualizations:
        base_record["visualization_video"] = str(visualization_video_path)

    if summary_path.exists() and not overwrite:
        base_record["status"] = "resumed_existing"
        base_record["summary_path"] = str(summary_path)
        base_record["summary"] = read_json(summary_path)
        return base_record

    target_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for stale_path in [
            prediction_path,
            summary_path,
            evaluation_dir / "per_frame_metrics.csv",
            evaluation_log,
            visualization_video_path,
        ]:
            if stale_path.exists() and stale_path.is_file():
                stale_path.unlink()

    attempts: List[Dict[str, Any]] = []
    successful_candidate: Optional[Dict[str, Any]] = None
    successful_log: Optional[Path] = None

    def build_inference_command(init_frame: int) -> List[str]:
        command = [
            sys.executable,
            str(run_video_path),
            "--input",
            str(video_path),
            "--start_frame",
            str(init_frame),
            "--gt_bbox_file",
            str(bbox_path),
            "--method",
            method,
            "--img_width",
            str(metadata["width"]),
            "--img_height",
            str(metadata["height"]),
            "--output_json",
            str(prediction_path),
        ]

        if show_live:
            command.append(
                "--show_result"
            )

        if reid_checkpoint is not None:
            command.extend(
                [
                    "--reid-checkpoint",
                    str(reid_checkpoint),
                ]
            )

        if method == "part-OCLReID" and association_mode != "baseline":
            command.extend(
                [
                    "--association-mode",
                    association_mode,
                    "--association-reid-threshold",
                    str(association_reid_threshold),
                    "--association-reid-margin",
                    str(association_reid_margin),
                    "--association-min-bbox-score",
                    str(association_min_bbox_score),
                    "--association-min-visible-parts",
                    str(association_min_visible_parts),
                ]
            )

        if save_visualizations:
            command.extend(
                [
                    "--visualization-video",
                    str(visualization_video_path),
                ]
            )

        if save_frames:
            command.extend(
                [
                    "--output",
                    str(frame_output_dir),
                ]
            )

        return command

    print(
        f"      Target {target_class}: "
        f"{visible_frames} visible frames, "
        f"{len(candidates)} initialization candidate(s)"
    )

    if dry_run:
        planned_commands = []
        for attempt_number, candidate in enumerate(
            candidates,
            start=1,
        ):
            init_frame = int(candidate["video_frame_index"])
            command = build_inference_command(init_frame)
            planned_commands.append(command)
            print(
                f"      Candidate {attempt_number}: "
                f"annotation {candidate['annotation_record_index']} "
                f"-> video frame {init_frame}, "
                f"box {candidate['bbox_xywh']}"
            )
            print(
                "      Planned scripts/run_single_video.py command: "
                + " ".join(command)
            )

        base_record["status"] = "dry_run"
        base_record["planned_inference_commands"] = planned_commands
        return base_record

    for attempt_number, candidate in enumerate(
        candidates,
        start=1,
    ):
        init_frame = int(candidate["video_frame_index"])
        init_bbox_xywh = candidate["bbox_xywh"]

        bbox_path.write_text(
            ",".join(
                f"{value:.10g}"
                for value in init_bbox_xywh
            )
            + "\n",
            encoding="utf-8",
        )

        if prediction_path.exists():
            prediction_path.unlink()

        attempt_log = (
            target_dir
            / (
                f"inference_attempt_{attempt_number:02d}"
                f"_frame_{init_frame:06d}.log"
            )
        )

        inference_command = build_inference_command(init_frame)

        attempt_record: Dict[str, Any] = {
            "attempt": attempt_number,
            "annotation_record_index":
                candidate["annotation_record_index"],
            "frame_index": init_frame,
            "video_frame_index": init_frame,
            "bbox_xywh": init_bbox_xywh,
            "width": candidate["width"],
            "height": candidate["height"],
            "area_ratio": candidate["area_ratio"],
            "file": candidate["file"],
            "visualization_video": (
                str(visualization_video_path)
                if save_visualizations else ""
            ),
            "command": inference_command,
            "log": str(attempt_log),
            "status": "pending",
            "error": "",
        }

        print(
            f"      Attempt {attempt_number}/"
            f"{len(candidates)}: annotation "
            f"{candidate['annotation_record_index']} "
            f"-> video frame {init_frame}, "
            f"box {init_bbox_xywh}"
        )

        try:
            run_logged_command(
                inference_command,
                PROJECT_ROOT,
                attempt_log,
                False,
            )

            if not prediction_has_valid_initial_target(
                prediction_path,
                init_frame,
            ):
                raise RuntimeError(
                    "OCLReID did not produce a valid target "
                    "on the initialization frame."
                )

            attempt_record["status"] = "success"
            if save_visualizations:
                if visualization_video_path.exists():
                    attempt_record["visualization_status"] = "created"
                    base_record["visualization_status"] = "created"
                else:
                    attempt_record["visualization_status"] = "not_created"
                    base_record["visualization_status"] = "not_created"
            attempts.append(attempt_record)
            successful_candidate = candidate
            successful_log = attempt_log
            break

        except Exception as error:
            attempt_record["status"] = "failed"
            attempt_record["error"] = str(error)
            if save_visualizations:
                attempt_record["visualization_status"] = (
                    "created"
                    if visualization_video_path.exists()
                    else "not_created"
                )
            attempts.append(attempt_record)

            print(
                f"      [RETRY] Initialization failed "
                f"at frame {init_frame}: {error}"
            )

    write_json(
        attempts_path,
        attempts,
    )

    if successful_candidate is None:
        base_record["status"] = "failed_initialization"
        base_record["error"] = (
            f"All {len(candidates)} initialization "
            "attempts failed."
        )

        failure_path = target_dir / "failure.txt"
        failure_path.write_text(
            base_record["error"]
            + "\nSee initialization_attempts.json and "
            "the attempt-specific inference logs.\n",
            encoding="utf-8",
        )

        return base_record

    init_frame = int(
        successful_candidate["video_frame_index"]
    )

    init_bbox_xywh = (
        successful_candidate["bbox_xywh"]
    )

    evaluation_annotation_frames = [
        int(metadata["manifest"]["file_to_video_frame"][str(record.get("File", ""))])
        for record in annotations
        if str(record.get("File", "")) in metadata["manifest"]["file_to_video_frame"]
        and int(metadata["manifest"]["file_to_video_frame"][str(record.get("File", ""))]) > init_frame
    ]

    if not evaluation_annotation_frames:
        base_record["status"] = "skipped_no_evaluation_frames"
        return base_record

    evaluation_start_frame = min(evaluation_annotation_frames)

    # Preserve a convenient canonical copy of the successful log.
    if successful_log is not None:
        shutil.copy2(
            successful_log,
            target_dir / "inference.log",
        )

    base_record.update(
        {
            "init_frame": init_frame,
            "initialization_video_frame_index":
                init_frame,
            "initialization_annotation_record_index":
                successful_candidate["annotation_record_index"],
            "initialization_file":
                successful_candidate["file"],
            "evaluation_start_frame":
                evaluation_start_frame,
            "evaluation_first_annotated_video_frame":
                evaluation_start_frame,
            "initialization_attempts":
                len(attempts),
        }
    )

    run_metadata = {
        **base_record,
        "init_bbox_xywh": init_bbox_xywh,
        "selected_initialization_candidate":
            successful_candidate,
        "initialization_filters": {
            "min_width": min_init_width,
            "min_height": min_init_height,
            "min_area_ratio": min_init_area_ratio,
            "border_margin": init_border_margin,
            "max_attempts": max_init_attempts,
            "retry_frame_step": init_retry_frame_step,
        },
        "video_metadata": metadata,
        "reid_checkpoint": str(reid_checkpoint) if reid_checkpoint else "",
        "association": {
            "mode": association_mode if method == "part-OCLReID" else "not_applicable",
            "reid_threshold": association_reid_threshold,
            "reid_margin": association_reid_margin,
            "min_bbox_score": association_min_bbox_score,
            "min_visible_parts": association_min_visible_parts,
            "forwarded_to_run_video": (
                method == "part-OCLReID" and association_mode != "baseline"
            ),
        },
        "camera_fps_expected":
            CAMERA_FPS[camera_name],
        "evaluation_excludes_initialization_frame":
            True,
    }

    write_json(
        metadata_path,
        run_metadata,
    )

    evaluation_command = [
        sys.executable,
        str(evaluator_path),
        "--predictions",
        str(prediction_path),
        "--annotations",
        str(annotation_path),
        "--frame_manifest",
        str(metadata["manifest_path"]),
        "--target_class",
        target_class,
        "--initialization_video_frame",
        str(init_frame),
        "--fps",
        str(CAMERA_FPS[camera_name]),
        "--iou_threshold",
        "0.5",
        "--center_threshold",
        "20",
        "--output_dir",
        str(evaluation_dir),
    ]

    try:
        run_logged_command(
            evaluation_command,
            PROJECT_ROOT,
            evaluation_log,
            False,
        )

        if not summary_path.exists():
            raise FileNotFoundError(
                "Evaluation completed without creating "
                f"{summary_path}"
            )

        summary = read_json(summary_path)

        base_record["status"] = "completed"
        base_record["summary_path"] = str(summary_path)
        base_record["summary"] = summary

        return base_record

    except Exception as error:
        base_record["status"] = "failed_evaluation"
        base_record["error"] = str(error)

        failure_path = target_dir / "failure.txt"
        failure_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )

        print(
            f"      [FAILED EVALUATION] "
            f"Target {target_class}: {error}"
        )

        return base_record


# ============================================================
# AGGREGATION
# ============================================================

SUMMARY_COLUMNS = [
    "split",
    "method",
    "dataset_part",
    "scene",
    "camera",
    "target_class",
    "init_frame",
    "initialization_annotation_record_index",
    "initialization_video_frame_index",
    "initialization_file",
    "evaluation_start_frame",
    "evaluation_first_annotated_video_frame",
    "visible_frames",
    "visualization_video",
    "visualization_status",
    "status",
    "valid_annotated_frames",
    "ambiguous_duplicate_target_frames",
    "total_evaluated_frames",
    "ground_truth_visible_frames",
    "ground_truth_absent_frames",
    "prediction_present_on_visible_frames",
    "correctly_localized_frames",
    "missed_or_incorrect_visible_frames",
    "mean_iou_on_visible_frames",
    "mean_iou_when_prediction_present",
    "success_rate_iou_threshold",
    "prediction_availability_on_visible_frames",
    "precision_center_threshold",
    "center_success_rate_50px",
    "mean_center_error_pixels",
    "mean_center_error_pixels_when_prediction_present",
    "wrong_person_frames",
    "wrong_person_rate",
    "wrong_person_rate_on_predicted_visible_frames",
    "wrong_person_rate_on_all_visible_frames",
    "false_positive_frames_when_target_absent",
    "false_positive_rate_when_target_absent",
    "correct_rejection_rate_when_target_absent",
    "annotated_frame_target_state_accuracy",
    "exact_reappearance_count",
    "uncertain_reappearance_count",
    "successful_exact_reacquisitions",
    "exact_reacquisition_rate",
    "average_exact_reacquisition_delay_frames",
    "ground_truth_reappearance_count",
    "successful_reacquisitions_after_reappearance",
    "reacquisition_rate_after_reappearance",
    "average_tracker_recovery_delay_frames",
    "average_tracker_recovery_delay_seconds",
    "result_dir",
    "error",
]


def flatten_result_record(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        key: record.get(key, "")
        for key in SUMMARY_COLUMNS
    }

    summary = record.get("summary")

    if isinstance(summary, dict):
        for key in SUMMARY_COLUMNS:
            if key in summary:
                row[key] = summary[key]

    return row


def write_run_manifest(
    records: Sequence[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        flatten_result_record(record)
        for record in records
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=SUMMARY_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def aggregate_group(
    records: Sequence[Dict[str, Any]],
    group_name: str,
) -> Dict[str, Any]:
    completed = [
        record
        for record in records
        if (
            record.get("status")
            in {"completed", "resumed_existing"}
            and isinstance(record.get("summary"), dict)
        )
    ]

    aggregate: Dict[str, Any] = {
        "group": group_name,
        "completed_runs": len(completed),
    }

    if not completed:
        return aggregate

    summaries = [
        record["summary"]
        for record in completed
    ]

    total_visible = sum(
        int(summary.get(
            "ground_truth_visible_frames",
            0,
        ))
        for summary in summaries
    )

    total_valid_annotated = sum(
        int(summary.get(
            "valid_annotated_frames",
            int(summary.get("ground_truth_visible_frames", 0))
            + int(summary.get("ground_truth_absent_frames", 0)),
        ))
        for summary in summaries
    )

    total_ambiguous = sum(
        int(summary.get(
            "ambiguous_duplicate_target_frames",
            0,
        ))
        for summary in summaries
    )

    total_absent = sum(
        int(summary.get(
            "ground_truth_absent_frames",
            0,
        ))
        for summary in summaries
    )

    total_correct = sum(
        int(summary.get(
            "correctly_localized_frames",
            0,
        ))
        for summary in summaries
    )

    total_predicted_visible = sum(
        int(summary.get(
            "prediction_present_on_visible_frames",
            0,
        ))
        for summary in summaries
    )

    total_wrong = sum(
        int(summary.get(
            "wrong_person_frames",
            0,
        ))
        for summary in summaries
    )

    total_false_positive_absent = sum(
        int(summary.get(
            "false_positive_frames_when_target_absent",
            0,
        ))
        for summary in summaries
    )

    total_reappearance = sum(
        int(summary.get(
            "exact_reappearance_count",
            summary.get("ground_truth_reappearance_count", 0),
        ))
        for summary in summaries
    )

    total_uncertain_reappearance = sum(
        int(summary.get(
            "uncertain_reappearance_count",
            0,
        ))
        for summary in summaries
    )

    total_reacquired = sum(
        int(summary.get(
            "successful_exact_reacquisitions",
            summary.get("successful_reacquisitions_after_reappearance", 0),
        ))
        for summary in summaries
    )

    weighted_iou_sum = sum(
        float(summary.get(
            "mean_iou_on_visible_frames",
            0.0,
        ))
        * int(summary.get(
            "ground_truth_visible_frames",
            0,
        ))
        for summary in summaries
    )

    valid_center_count = sum(
        int(summary.get(
            "prediction_present_on_visible_frames",
            0,
        ))
        for summary in summaries
        if finite_number(
            summary.get("mean_center_error_pixels")
        ) is not None
    )

    weighted_center_error_sum = sum(
        float(summary["mean_center_error_pixels"])
        * int(summary.get(
            "prediction_present_on_visible_frames",
            0,
        ))
        for summary in summaries
        if finite_number(
            summary.get("mean_center_error_pixels")
        ) is not None
    )

    center_success_approx = sum(
        float(summary.get(
            "precision_center_threshold",
            0.0,
        ))
        * int(summary.get(
            "ground_truth_visible_frames",
            0,
        ))
        for summary in summaries
    )

    recovery_delays_frames = []
    exact_reacquisition_delays_frames = []

    for summary in summaries:
        for event in summary.get(
            "exact_reappearance_events",
            summary.get("ground_truth_reappearance_events", []),
        ):
            delay = finite_number(
                event.get("delay_frames")
            )

            if delay is not None:
                exact_reacquisition_delays_frames.append(delay)

        for event in summary.get(
            "tracker_recovery_events",
            [],
        ):
            delay = finite_number(
                event.get("delay_frames")
            )

            if delay is not None:
                recovery_delays_frames.append(delay)

    aggregate.update(
        {
            "ground_truth_visible_frames":
                total_visible,
            "ground_truth_absent_frames":
                total_absent,
            "valid_annotated_frames":
                total_valid_annotated,
            "ambiguous_duplicate_target_frames":
                total_ambiguous,
            "correctly_localized_frames":
                total_correct,
            "prediction_present_on_visible_frames":
                total_predicted_visible,
            "wrong_person_frames":
                total_wrong,
            "false_positive_frames_when_target_absent":
                total_false_positive_absent,
            "reappearance_events":
                total_reappearance,
            "exact_reappearance_count":
                total_reappearance,
            "uncertain_reappearance_count":
                total_uncertain_reappearance,
            "successful_reacquisitions":
                total_reacquired,
            "successful_exact_reacquisitions":
                total_reacquired,

            "micro_success_rate_iou_0_5": (
                total_correct / total_visible
                if total_visible else None
            ),

            "micro_prediction_availability": (
                total_predicted_visible / total_visible
                if total_visible else None
            ),

            "micro_mean_iou_visible": (
                weighted_iou_sum / total_visible
                if total_visible else None
            ),

            "micro_center_precision_20px": (
                center_success_approx / total_visible
                if total_visible else None
            ),

            "weighted_mean_center_error_pixels": (
                weighted_center_error_sum
                / valid_center_count
                if valid_center_count else None
            ),

            "micro_false_positive_rate_absent": (
                total_false_positive_absent
                / total_absent
                if total_absent else None
            ),

            "micro_reacquisition_rate": (
                total_reacquired
                / total_reappearance
                if total_reappearance else None
            ),

            "micro_exact_reacquisition_rate": (
                total_reacquired
                / total_reappearance
                if total_reappearance else None
            ),

            "micro_annotated_frame_target_state_accuracy": (
                sum(
                    int(summary.get(
                        "annotated_frame_target_state_correct_frames",
                        0,
                    ))
                    for summary in summaries
                )
                / total_valid_annotated
                if total_valid_annotated else None
            ),

            "mean_recovery_delay_frames": (
                sum(recovery_delays_frames)
                / len(recovery_delays_frames)
                if recovery_delays_frames else None
            ),

            "average_exact_reacquisition_delay_frames": (
                sum(exact_reacquisition_delays_frames)
                / len(exact_reacquisition_delays_frames)
                if exact_reacquisition_delays_frames else None
            ),

            "macro_success_rate_iou_0_5": (
                sum(
                    float(summary.get(
                        "success_rate_iou_threshold",
                        0.0,
                    ))
                    for summary in summaries
                )
                / len(summaries)
            ),

            "macro_mean_iou_visible": (
                sum(
                    float(summary.get(
                        "mean_iou_on_visible_frames",
                        0.0,
                    ))
                    for summary in summaries
                )
                / len(summaries)
            ),
        }
    )

    return aggregate


def write_aggregates(
    records: Sequence[Dict[str, Any]],
    results_root: Path,
) -> None:
    grouped: Dict[
        Tuple[str, str],
        List[Dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        grouped[
            (
                str(record.get("split", "")),
                str(record.get("method", "")),
            )
        ].append(record)

    for (
        split_name,
        method,
    ), group_records in grouped.items():
        output_dir = (
            results_root
            / split_name
            / method
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        overall = aggregate_group(
            group_records,
            "overall",
        )

        by_camera = []

        for camera_name in CAMERA_FPS:
            camera_records = [
                record
                for record in group_records
                if record.get("camera") == camera_name
            ]

            by_camera.append(
                aggregate_group(
                    camera_records,
                    camera_name,
                )
            )

        write_json(
            output_dir / "aggregate_summary.json",
            {
                "split": split_name,
                "method": method,
                "overall": overall,
                "by_camera": by_camera,
            },
        )

        camera_csv_path = (
            output_dir / "aggregate_by_camera.csv"
        )

        fieldnames = sorted(
            {
                key
                for row in by_camera
                for key in row.keys()
            }
        )

        with camera_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(by_camera)


# ============================================================
# MAIN BATCH PROCESSING
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically run and evaluate OCLReID "
            "on AGHRI dataset splits."
        )
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        choices=["val", "test"],
        help=(
            "Splits to process. Begin with val; use test "
            "only after the pipeline is fixed."
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["part-OCLReID"],
        choices=SUPPORTED_METHODS,
        help="OCLReID methods to evaluate.",
    )

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(CAMERA_FPS.keys()),
        choices=list(CAMERA_FPS.keys()),
        help="Camera streams to process.",
    )

    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=None,
        help=(
            "Optional annotation target classes to evaluate, for example "
            "--target-classes 03 04."
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
    )

    parser.add_argument(
        "--run-video",
        type=Path,
        default=DEFAULT_RUN_VIDEO,
    )

    parser.add_argument(
        "--reid-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional initial ReID checkpoint passed to scripts/run_single_video.py. "
            f"Defaults inside scripts/run_single_video.py to {DEFAULT_REID_CHECKPOINT}."
        ),
    )

    parser.add_argument(
        "--association-mode",
        choices=["baseline", "reid_gate"],
        default="baseline",
        help=(
            "ReID-gated target-association mode forwarded to scripts/run_single_video.py "
            "for part-OCLReID only. Default preserves baseline behavior."
        ),
    )

    parser.add_argument(
        "--association-reid-threshold",
        type=float,
        default=0.6,
        help=(
            "ReID-gated ReID-gated association threshold forwarded to "
            "scripts/run_single_video.py for part-OCLReID when association mode is enabled."
        ),
    )

    parser.add_argument(
        "--association-reid-margin",
        type=float,
        default=0.02,
        help=(
            "ReID-gated minimum margin between the best and second-best ReID "
            "candidate, forwarded for part-OCLReID when association mode is enabled."
        ),
    )

    parser.add_argument(
        "--association-min-bbox-score",
        type=float,
        default=0.0,
        help=(
            "ReID-gated minimum detector/track bbox score for ReID-gated "
            "association, forwarded for part-OCLReID when association mode is enabled."
        ),
    )

    parser.add_argument(
        "--association-min-visible-parts",
        type=int,
        default=1,
        help=(
            "ReID-gated minimum visible body-part count for ReID-gated "
            "association, forwarded for part-OCLReID when association mode is enabled."
        ),
    )

    parser.add_argument(
        "--evaluator",
        type=Path,
        default=DEFAULT_EVALUATOR,
    )

    parser.add_argument(
        "--minimum-visible-frames",
        type=int,
        default=2,
        help=(
            "Skip identities annotated in fewer than this "
            "number of frames."
        ),
    )

    parser.add_argument(
        "--min-init-width",
        type=float,
        default=30.0,
        help=(
            "Minimum initialization-box width in pixels. "
            "Tiny boxes are unsuitable for detector-based initialization."
        ),
    )

    parser.add_argument(
        "--min-init-height",
        type=float,
        default=70.0,
        help="Minimum initialization-box height in pixels.",
    )

    parser.add_argument(
        "--min-init-area-ratio",
        type=float,
        default=0.005,
        help=(
            "Minimum initialization-box area divided by image area."
        ),
    )

    parser.add_argument(
        "--init-border-margin",
        type=float,
        default=2.0,
        help=(
            "Required pixel margin between the initialization box "
            "and every image border."
        ),
    )

    parser.add_argument(
        "--max-init-attempts",
        type=int,
        default=6,
        help=(
            "Maximum number of later suitable frames tried when "
            "OCLReID cannot initialize."
        ),
    )

    parser.add_argument(
        "--init-retry-frame-step",
        type=int,
        default=5,
        help=(
            "Minimum frame gap between automatic initialization retries."
        ),
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. For example, "
            "--max-runs 2."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun targets that already have summary metrics.",
    )

    parser.add_argument(
        "--save-frames",
        action="store_true",
        help=(
            "Also save annotated JPEG frames. This can use "
            "substantial disk space."
        ),
    )

    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help=(
            "Save the annotated inference visualization as "
            "inference_visualization.mp4 in each target result directory."
        ),
    )

    parser.add_argument(
        "--show-live",
        action="store_true",
        help=(
            "Show the live OpenCV scripts/run_single_video.py window during "
            "automated inference. Disabled automatically when no "
            "graphical display is available."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )

    return parser.parse_args()


def validate_required_paths(
    arguments: argparse.Namespace,
) -> None:
    required = {
        "dataset root": arguments.dataset_root,
        "video root": arguments.video_root,
        "scripts/run_single_video.py": arguments.run_video,
        "scripts/evaluate_aghri_results.py": arguments.evaluator,
    }

    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required {label} not found: {path}"
            )

    if arguments.reid_checkpoint is not None:
        checkpoint = arguments.reid_checkpoint.expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"ReID checkpoint not found: {checkpoint}"
            )
        arguments.reid_checkpoint = checkpoint


def main() -> None:
    arguments = parse_arguments()
    validate_required_paths(arguments)

    print("=" * 80)
    print("AGHRI OCLReID AUTOMATED BENCHMARK")
    print("=" * 80)
    print(f"Python:       {sys.executable}")
    print(f"Dataset:      {arguments.dataset_root}")
    print(f"Videos:       {arguments.video_root}")
    print(f"Results:      {arguments.results_root}")
    print(f"Splits:       {arguments.splits}")
    print(f"Methods:      {arguments.methods}")
    print(f"Cameras:      {arguments.cameras}")
    print(f"Targets:      {arguments.target_classes or 'all annotated classes'}")
    print(
        "ReID ckpt:    "
        f"{arguments.reid_checkpoint or DEFAULT_REID_CHECKPOINT}"
    )
    print(f"Association:  mode={arguments.association_mode}")
    print(
        "Association thresholds: "
        f"reid={arguments.association_reid_threshold}, "
        f"margin={arguments.association_reid_margin}, "
        f"min_bbox_score={arguments.association_min_bbox_score}, "
        f"min_visible_parts={arguments.association_min_visible_parts}"
    )
    if arguments.show_live and not (
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    ):
        print(
            "[WARNING] --show-live was requested, but DISPLAY is not set. "
            "Continuing headlessly."
        )
        arguments.show_live = False

    print(
        "Initialization rule: earliest sufficiently large, "
        "unclipped target box; retry later frames if needed."
    )
    print(
        "Evaluation rule: map annotation File values through "
        "the frame manifest and evaluate annotated frames after "
        "the initialization video frame."
    )

    scene_index = build_scene_index(
        arguments.dataset_root
    )

    records: List[Dict[str, Any]] = []
    attempted_runs = 0
    stop_requested = False

    for split_name in arguments.splits:
        split_file = (
            arguments.dataset_root
            / "split_lists"
            / f"{split_name}.txt"
        )

        scene_names = load_split_scene_names(
            split_file
        )

        print("\n" + "=" * 80)
        print(
            f"SPLIT: {split_name.upper()} "
            f"({len(scene_names)} scenes)"
        )
        print("=" * 80)

        for scene_name in scene_names:
            locations = scene_index.get(
                scene_name,
                [],
            )

            if not locations:
                print(
                    f"[MISSING SCENE] {scene_name}"
                )
                continue

            if len(locations) > 1:
                print(
                    f"[WARNING] Scene appears in "
                    f"{len(locations)} dataset parts: "
                    f"{scene_name}"
                )

            for part_name, scene_path in locations:
                print(
                    f"\nScene: {scene_name}\n"
                    f"Part:  {part_name}"
                )

                for camera_name in arguments.cameras:
                    video_path = (
                        arguments.video_root
                        / split_name
                        / part_name
                        / scene_name
                        / f"{camera_name}.mp4"
                    )

                    manifest_path = (
                        arguments.video_root
                        / split_name
                        / part_name
                        / scene_name
                        / f"{camera_name}_frame_manifest.json"
                    )

                    annotation_path = (
                        scene_path
                        / "annotations"
                        / f"{camera_name}_ann.json"
                    )

                    camera_dir = (
                        scene_path
                        / "sensor_data"
                        / camera_name
                    )

                    print(
                        f"  Camera: {camera_name}"
                    )

                    if not video_path.exists():
                        print(
                            f"    [SKIP] Missing video: "
                            f"{video_path}"
                        )
                        continue

                    if not annotation_path.exists():
                        print(
                            f"    [SKIP] Missing annotation: "
                            f"{annotation_path}"
                        )
                        continue

                    if not camera_dir.exists():
                        print(
                            f"    [SKIP] Missing image folder: "
                            f"{camera_dir}"
                        )
                        continue

                    try:
                        annotations = read_json(
                            annotation_path
                        )

                        if not isinstance(annotations, list):
                            raise ValueError(
                                "Annotation JSON is not a list."
                            )

                        metadata = verify_alignment(
                            annotations,
                            camera_dir,
                            video_path,
                            manifest_path,
                            CAMERA_FPS[camera_name],
                        )

                        expected_fps = (
                            CAMERA_FPS[camera_name]
                        )

                        if abs(
                            metadata["fps"] - expected_fps
                        ) > 0.1:
                            raise ValueError(
                                f"Video FPS is "
                                f"{metadata['fps']}, expected "
                                f"{expected_fps}."
                            )

                        target_classes = (
                            extract_target_classes(
                                annotations
                            )
                        )

                        if arguments.target_classes is not None:
                            allowed_targets = {
                                str(target)
                                for target in arguments.target_classes
                            }
                            target_classes = [
                                target
                                for target in target_classes
                                if str(target) in allowed_targets
                            ]

                        print(
                            f"    Frames: "
                            f"{metadata['frame_count']}"
                            f" | Annotations: "
                            f"{metadata['annotation_count']}"
                            f" | Unannotated: "
                            f"{metadata['unannotated_frame_count']}"
                            f" | Resolution: "
                            f"{metadata['width']}x"
                            f"{metadata['height']}"
                            f" | Targets: "
                            f"{target_classes}"
                        )

                        print(
                            "    Annotation mapping: verified"
                        )

                    except Exception as error:
                        print(
                            f"    [ALIGNMENT ERROR] {error}"
                        )
                        continue

                    for method in arguments.methods:
                        for target_class in target_classes:
                            if (
                                arguments.max_runs is not None
                                and attempted_runs
                                >= arguments.max_runs
                            ):
                                stop_requested = True
                                break

                            attempted_runs += 1

                            record = run_one_target(
                                split_name=split_name,
                                method=method,
                                part_name=part_name,
                                scene_name=scene_name,
                                scene_path=scene_path,
                                camera_name=camera_name,
                                video_path=video_path,
                                annotation_path=annotation_path,
                                target_class=target_class,
                                annotations=annotations,
                                metadata=metadata,
                                results_root=arguments.results_root,
                                run_video_path=arguments.run_video,
                                evaluator_path=arguments.evaluator,
                                reid_checkpoint=arguments.reid_checkpoint,
                                overwrite=arguments.overwrite,
                                save_frames=arguments.save_frames,
                                show_live=arguments.show_live,
                                save_visualizations=(
                                    arguments.save_visualizations
                                ),
                                association_mode=(
                                    arguments.association_mode
                                ),
                                association_reid_threshold=(
                                    arguments.association_reid_threshold
                                ),
                                association_reid_margin=(
                                    arguments.association_reid_margin
                                ),
                                association_min_bbox_score=(
                                    arguments.association_min_bbox_score
                                ),
                                association_min_visible_parts=(
                                    arguments.association_min_visible_parts
                                ),
                                dry_run=arguments.dry_run,
                                minimum_visible_frames=(
                                    arguments.minimum_visible_frames
                                ),
                                min_init_width=(
                                    arguments.min_init_width
                                ),
                                min_init_height=(
                                    arguments.min_init_height
                                ),
                                min_init_area_ratio=(
                                    arguments.min_init_area_ratio
                                ),
                                init_border_margin=(
                                    arguments.init_border_margin
                                ),
                                max_init_attempts=(
                                    arguments.max_init_attempts
                                ),
                                init_retry_frame_step=(
                                    arguments.init_retry_frame_step
                                ),
                            )

                            records.append(record)

                        if stop_requested:
                            break

                    if stop_requested:
                        break

                if stop_requested:
                    break

            if stop_requested:
                break

        if stop_requested:
            break

    manifest_path = (
        arguments.results_root
        / "batch_manifest.csv"
    )

    write_run_manifest(
        records,
        manifest_path,
    )

    write_aggregates(
        records,
        arguments.results_root,
    )

    completed = sum(
        record.get("status")
        in {"completed", "resumed_existing"}
        for record in records
    )

    failed = sum(
        str(record.get("status", "")).startswith("failed")
        for record in records
    )

    skipped = len(records) - completed - failed

    print("\n" + "=" * 80)
    print("AUTOMATION FINISHED")
    print("=" * 80)
    print(f"Target runs recorded: {len(records)}")
    print(f"Completed/resumed:     {completed}")
    print(f"Failed:                {failed}")
    print(f"Skipped/dry runs:      {skipped}")
    print(f"Manifest:              {manifest_path}")
    print(
        "Aggregate files are stored inside each "
        "results/<split>/<method>/ directory."
    )


if __name__ == "__main__":
    main()
