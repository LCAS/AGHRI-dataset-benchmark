#!/usr/bin/env python3
"""Generate AGHRI OCLReID videos and frame manifests."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aghri_alignment import (
    build_frame_manifest,
    load_annotations,
    sorted_sensor_images,
    validate_manifest_against_sources,
    write_json,
)


BASE_DIR = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
OUTPUT_ROOT = Path("/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos")
DATASET_PARTS = ["dataset_part1", "dataset_part2", "dataset_part3", "dataset_part4"]
CAMERA_FPS = {
    "cam_fish_front": 30.0,
    "cam_fish_left": 30.0,
    "cam_fish_right": 30.0,
    "cam_zed_rgb": 15.0,
}


def load_split_scene_names(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8-sig") as handle:
        names = [
            line.strip()
            for line in handle
            if line.strip() and not line.strip().startswith("#")
        ]
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError(f"No scene names were found in: {split_file}")
    return names


def build_scene_index(base_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    index: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for part_name in DATASET_PARTS:
        part_path = base_dir / part_name
        if not part_path.exists():
            print(f"[WARNING] Dataset part missing: {part_path}")
            continue
        for scene_path in sorted(part_path.iterdir(), key=lambda p: p.name):
            if scene_path.is_dir():
                index[scene_path.name].append((part_name, scene_path))
    return dict(index)


def escape_ffmpeg_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def get_video_metadata(video_path: Path, count_frames: bool = False) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        if count_frames:
            command.insert(5, "-count_frames")
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Unable to inspect video with ffprobe: {video_path}\n{result.stderr}")
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        frame_text = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
        rate_text = stream.get("r_frame_rate") or "0/1"
        numerator, denominator = [float(value) for value in rate_text.split("/")]
        return {
            "frame_count": int(frame_text),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": numerator / denominator if denominator else 0.0,
        }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        return {
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def read_image_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3"}:
                    length = int.from_bytes(handle.read(2), "big")
                    precision_height_width = handle.read(5)
                    height = int.from_bytes(precision_height_width[1:3], "big")
                    width = int.from_bytes(precision_height_width[3:5], "big")
                    return width, height
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                length = int.from_bytes(handle.read(2), "big")
                handle.seek(length - 2, 1)
    raise ValueError(f"Unsupported or unreadable image dimensions: {image_path}")


def validate_image_dimensions(images: list[Path]) -> tuple[int, int]:
    expected: tuple[int, int] | None = None
    for image_path in images:
        current = read_image_size(image_path)
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(
                f"Inconsistent image dimensions in {image_path.parent}: "
                f"{current} versus first image {expected}"
            )
    if expected is None:
        raise ValueError("No images found.")
    return expected


def generate_video(camera_path: Path, output_video: Path, fps: float, overwrite: bool) -> dict[str, Any]:
    images = sorted_sensor_images(camera_path)
    if not images:
        raise ValueError(f"No images found in: {camera_path}")

    width, height = validate_image_dimensions(images)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if output_video.exists() and not overwrite:
        metadata = get_video_metadata(output_video, count_frames=True)
        if metadata["frame_count"] != len(images):
            raise ValueError(
                f"Existing video has {metadata['frame_count']} frames, expected {len(images)}."
            )
        return {"status": "skipped", **metadata, "width": width, "height": height}

    temporary_list_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="aghri_oclreid_ffmpeg_",
            delete=False,
            encoding="utf-8",
        ) as temporary_list:
            temporary_list_path = Path(temporary_list.name)
            for image_path in images:
                temporary_list.write(f"file '{escape_ffmpeg_path(image_path)}'\n")

        command = [
            "ffmpeg",
            "-y" if overwrite else "-n",
            "-r",
            str(fps),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(temporary_list_path),
            # The environment pins FFmpeg 4.2.2, which does not support the
            # newer `-fps_mode passthrough` option. `-vsync 0` is the compatible
            # pass-through setting and keeps one output frame per source image.
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_video),
        ]
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed for {camera_path}:\n{result.stderr}")

        metadata = get_video_metadata(output_video)
        if metadata["frame_count"] != len(images):
            raise ValueError(
                f"Generated video frame count {metadata['frame_count']} "
                f"does not match sensor image count {len(images)}."
            )
        if abs(metadata["fps"] - fps) > 0.1:
            raise ValueError(f"Generated video FPS {metadata['fps']} does not match {fps}.")
        return {"status": "created", **metadata}
    finally:
        if temporary_list_path is not None and temporary_list_path.exists():
            temporary_list_path.unlink()


def process_camera(
    *,
    split: str,
    dataset_part: str,
    scene_path: Path,
    camera: str,
    output_dir: Path,
    overwrite: bool,
    manifest_only: bool,
) -> dict[str, Any]:
    camera_dir = scene_path / "sensor_data" / camera
    annotation_path = scene_path / "annotations" / f"{camera}_ann.json"
    video_path = output_dir / f"{camera}.mp4"
    manifest_path = output_dir / f"{camera}_frame_manifest.json"
    fps = CAMERA_FPS[camera]

    row = {
        "split": split,
        "dataset_part": dataset_part,
        "scene": scene_path.name,
        "camera": camera,
        "video_path": str(video_path),
        "manifest_path": str(manifest_path),
        "fps": fps,
        "sensor_images": 0,
        "annotation_records": 0,
        "unannotated_frames": 0,
        "video_frame_count": "",
        "status": "pending",
        "error": "",
    }

    try:
        if not camera_dir.is_dir():
            raise FileNotFoundError(f"Missing camera folder: {camera_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        manifest = build_frame_manifest(
            split=split,
            dataset_part=dataset_part,
            scene=scene_path.name,
            camera=camera,
            fps=fps,
            camera_dir=camera_dir,
            annotation_path=annotation_path,
        )
        write_json(manifest_path, manifest)

        annotations = load_annotations(annotation_path)
        row["sensor_images"] = manifest["total_sensor_images"]
        row["annotation_records"] = manifest["total_annotation_records"]
        row["unannotated_frames"] = row["sensor_images"] - row["annotation_records"]

        if manifest_only:
            video_metadata = get_video_metadata(video_path) if video_path.exists() else None
            validate_manifest_against_sources(
                manifest=manifest,
                camera_dir=camera_dir,
                annotations=annotations,
                expected_fps=fps,
                video_frame_count=video_metadata["frame_count"] if video_metadata else None,
            )
            row["video_frame_count"] = video_metadata["frame_count"] if video_metadata else ""
            row["status"] = "manifest_verified"
        else:
            video_metadata = generate_video(camera_dir, video_path, fps, overwrite)
            validate_manifest_against_sources(
                manifest=manifest,
                camera_dir=camera_dir,
                annotations=annotations,
                expected_fps=fps,
                video_frame_count=video_metadata["frame_count"],
            )
            row["video_frame_count"] = video_metadata["frame_count"]
            row["status"] = video_metadata["status"]

        print(
            f"      [OK] {camera}: Frames: {row['sensor_images']} | "
            f"Annotations: {row['annotation_records']} | "
            f"Unannotated frames: {row['unannotated_frames']} | "
            "Annotation mapping: verified"
        )
    except Exception as error:
        row["status"] = "failed"
        row["error"] = str(error)
        print(f"      [ERROR] {camera}: {error}")

    return row


def write_generation_summary(rows: list[dict[str, Any]], output_root: Path) -> None:
    csv_path = output_root / "generation_summary.csv"
    json_path = output_root / "generation_summary.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "dataset_part",
        "scene",
        "camera",
        "fps",
        "sensor_images",
        "annotation_records",
        "unannotated_frames",
        "video_frame_count",
        "status",
        "video_path",
        "manifest_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])
    write_json(
        json_path,
        {
            "total_sequences": len(rows),
            "status_counts": {
                status: sum(1 for row in rows if row["status"] == status)
                for status in sorted({row["status"] for row in rows})
            },
            "rows": rows,
        },
    )
    print(f"\nGeneration summary CSV:  {csv_path}")
    print(f"Generation summary JSON: {json_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=BASE_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--cameras", nargs="+", default=list(CAMERA_FPS), choices=list(CAMERA_FPS))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    scene_index = build_scene_index(args.dataset_root)
    rows: list[dict[str, Any]] = []
    print("=" * 72)
    print("AGHRI OCLReID VIDEO GENERATOR")
    print("=" * 72)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Splits:       {args.splits}")
    print(f"Manifest only: {args.manifest_only}")

    for split in args.splits:
        split_file = args.dataset_root / "split_lists" / f"{split}.txt"
        scenes = load_split_scene_names(split_file)
        print("\n" + "=" * 72)
        print(f"SPLIT {split.upper()}: {len(scenes)} scenes")
        print("=" * 72)
        for scene in scenes:
            locations = scene_index.get(scene, [])
            if not locations:
                print(f"[MISSING SCENE] {scene}")
                continue
            for dataset_part, scene_path in locations:
                print(f"\nScene: {scene}\nPart:  {dataset_part}")
                output_dir = args.output_root / split / dataset_part / scene
                for camera in args.cameras:
                    rows.append(
                        process_camera(
                            split=split,
                            dataset_part=dataset_part,
                            scene_path=scene_path,
                            camera=camera,
                            output_dir=output_dir,
                            overwrite=not args.no_overwrite,
                            manifest_only=args.manifest_only,
                        )
                    )

    write_generation_summary(rows, args.output_root)
    failed = sum(1 for row in rows if row["status"] == "failed")
    if failed:
        raise SystemExit(f"{failed} camera sequence(s) failed.")


if __name__ == "__main__":
    main()
