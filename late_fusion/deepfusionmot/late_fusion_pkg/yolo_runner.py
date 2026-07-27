"""ROS-independent YOLO11s runner for AGHRI ZED RGB images."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from late_fusion_pkg.checkpoints import sha256_file


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(
    image_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    imgsz: int,
    conf: float,
    classes: list[int],
    device: str,
    nms: str = "ultralytics_default",
) -> dict:
    import ultralytics

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "ultralytics_version": ultralytics.__version__,
        "imgsz": imgsz,
        "confidence_threshold": conf,
        "nms": nms,
        "class_filter": classes,
        "input_image_path": str(image_path),
        "input_image_sha256": file_hash(image_path),
        "device": device,
    }


def run_yolo_images(
    images: Iterable[tuple[str, float | None, Path]],
    checkpoint_path: Path,
    output_csv: Path,
    output_metadata: Path,
    imgsz: int = 640,
    conf: float = 0.10,
    classes: list[int] | None = None,
    device: str = "auto",
) -> dict:
    from ultralytics import YOLO
    import torch

    classes = [0] if classes is None else classes
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    checkpoint_sha = sha256_file(checkpoint_path)
    model_load_start = time.perf_counter()
    model = YOLO(str(checkpoint_path)).to(resolved_device)
    model_load_time = time.perf_counter() - model_load_start

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "timestamp",
        "image_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
        "class_id",
        "class_name",
        "checkpoint_sha256",
        "inference_time_sec",
        "empty_frame",
    ]
    count = 0
    frame_count = 0
    identities = []
    inference_times = []
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample_id, timestamp, image_path in images:
            frame_count += 1
            identity = cache_identity(image_path, checkpoint_path, checkpoint_sha, imgsz, conf, classes, resolved_device)
            identities.append(identity)
            start = time.perf_counter()
            result = model(str(image_path), imgsz=imgsz, conf=conf, classes=classes, verbose=False)[0]
            inference_time = time.perf_counter() - start
            inference_times.append(inference_time)
            boxes = result.boxes.data.detach().cpu().numpy() if result.boxes is not None else []
            for box in boxes:
                x1, y1, x2, y2, score, cls_id = box[:6]
                class_id = int(cls_id)
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "timestamp": "" if timestamp is None else f"{timestamp:.9f}",
                        "image_path": str(image_path),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(score),
                        "class_id": class_id,
                        "class_name": result.names.get(class_id, str(class_id)),
                        "checkpoint_sha256": checkpoint_sha,
                        "inference_time_sec": inference_time,
                        "empty_frame": False,
                    }
                )
                count += 1
            if len(boxes) == 0:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "timestamp": "" if timestamp is None else f"{timestamp:.9f}",
                        "image_path": str(image_path),
                        "class_name": "__EMPTY__",
                        "checkpoint_sha256": checkpoint_sha,
                        "inference_time_sec": inference_time,
                        "empty_frame": True,
                    }
                )
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "model_load_time_sec": model_load_time,
        "frames": frame_count,
        "detections": count,
        "imgsz": imgsz,
        "confidence_threshold": conf,
        "classes": classes,
        "device": resolved_device,
        "cache_identities": identities[:20],
        "cache_identity_count": len(identities),
        "inference_time_total_sec": float(sum(inference_times)),
        "inference_time_mean_sec": float(np.mean(inference_times)) if inference_times else 0.0,
        "inference_time_p95_sec": float(np.percentile(inference_times, 95)) if inference_times else 0.0,
        "inference_fps": float(len(inference_times) / sum(inference_times)) if sum(inference_times) > 0.0 else 0.0,
    }
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
