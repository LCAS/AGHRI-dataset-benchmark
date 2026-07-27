"""ROS-independent MMDetection3D runner for cached LiDAR detector replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable

import numpy as np

from late_fusion_pkg.checkpoints import PEDESTRIAN_CLASS_ID, KITTI_CLASS_MAPPING, sha256_file


def _parse_pcd_header(path: Path) -> tuple[dict[str, list[str]], int]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD file has no DATA line: {path}")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                return header, stream.tell()


def _pcd_to_xyzi_array(path: Path) -> np.ndarray:
    header, data_offset = _parse_pcd_header(path)
    fields = header.get("FIELDS", [])
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"PCD file is missing x/y/z fields: {path}")
    data_mode = (header.get("DATA") or [""])[0].lower()
    if header.get("POINTS"):
        points = int(header["POINTS"][0])
    else:
        points = int((header.get("WIDTH") or ["0"])[0]) * int((header.get("HEIGHT") or ["1"])[0])
    raw_rgb = None
    if data_mode == "ascii":
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            lines = stream.readlines()
        first_data_line = next(i for i, line in enumerate(lines) if line.upper().startswith("DATA")) + 1
        raw = np.loadtxt(lines[first_data_line:], dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        columns = {name: raw[:, idx] for idx, name in enumerate(fields)}
        raw_rgb = columns.get("rgb")
    elif data_mode == "binary":
        sizes = [int(v) for v in header.get("SIZE", ["4"] * len(fields))]
        types = header.get("TYPE", ["F"] * len(fields))
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
        dtype_fields = []
        for name, size, kind, count in zip(fields, sizes, types, counts):
            if kind == "F" and size == 4:
                dtype = np.float32
            elif kind == "F" and size == 8:
                dtype = np.float64
            elif kind == "U" and size == 1:
                dtype = np.uint8
            elif kind == "U" and size == 2:
                dtype = np.uint16
            elif kind == "U" and size == 4:
                dtype = np.uint32
            elif kind == "I" and size == 1:
                dtype = np.int8
            elif kind == "I" and size == 2:
                dtype = np.int16
            elif kind == "I" and size == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported PCD field type {kind}{size} in {path}")
            dtype_fields.append((name, dtype, (count,)) if count > 1 else (name, dtype))
        raw = np.fromfile(path, dtype=np.dtype(dtype_fields), count=points, offset=data_offset)
        columns = {name: np.asarray(raw[name], dtype=np.float32).reshape(points, -1)[:, 0] for name in fields}
        raw_rgb = raw["rgb"] if "rgb" in fields else None
    else:
        raise ValueError(f"Unsupported PCD DATA mode {data_mode!r} in {path}")
    if "intensity" in columns:
        intensity = columns["intensity"]
    elif "reflectivity" in columns:
        intensity = columns["reflectivity"]
    elif raw_rgb is not None:
        rgb_float = np.asarray(raw_rgb, dtype=np.float32)
        rgb_uint = rgb_float.view(np.uint32)
        r = ((rgb_uint >> 16) & 255).astype(np.float32)
        g = ((rgb_uint >> 8) & 255).astype(np.float32)
        b = (rgb_uint & 255).astype(np.float32)
        intensity = ((r + g + b) / 3.0) / 255.0
    else:
        intensity = np.zeros(points, dtype=np.float32)
    return np.column_stack([columns["x"], columns["y"], columns["z"], intensity]).astype(np.float32)


def prepare_xyzi_bin(pointcloud_path: Path) -> tuple[Path, Path | None]:
    if pointcloud_path.suffix.lower() == ".bin":
        return pointcloud_path, None
    if pointcloud_path.suffix.lower() != ".pcd":
        raise ValueError(f"Unsupported point cloud format for MMDetection3D: {pointcloud_path}")
    xyzi = _pcd_to_xyzi_array(pointcloud_path)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as stream:
        temp_path = Path(stream.name)
        xyzi.tofile(stream)
    return temp_path, temp_path


def count_points_outside_kitti_range(bin_path: Path, point_range=(0.0, -39.68, -3.0, 69.12, 39.68, 1.0)) -> dict:
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        return {"total": 0, "outside": 0, "error": "point file does not contain xyzi float32 records"}
    pts = points.reshape(-1, 4)
    x1, y1, z1, x2, y2, z2 = point_range
    inside = (
        (pts[:, 0] >= x1)
        & (pts[:, 0] <= x2)
        & (pts[:, 1] >= y1)
        & (pts[:, 1] <= y2)
        & (pts[:, 2] >= z1)
        & (pts[:, 2] <= z2)
    )
    return {"total": int(pts.shape[0]), "outside": int((~inside).sum())}


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as exc:
        return f"unavailable: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def run_mmdet3d_pointclouds(
    pointclouds: Iterable[tuple[str, float | None, Path]],
    config_path: Path,
    checkpoint_path: Path,
    output_csv: Path,
    output_metadata: Path,
    device: str = "cuda:0",
    score_threshold: float = 0.0,
    selected_class_id: int = PEDESTRIAN_CLASS_ID,
    selected_class_ids: list[int] | tuple[int, ...] | set[int] | None = None,
    selected_class_name: str | None = None,
    class_mapping: dict[int, str] | None = None,
    model_point_cloud_range: tuple[float, float, float, float, float, float] = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0),
    voxel_size: tuple[float, float, float] | None = None,
    nms_settings: dict | None = None,
    manifest_path: Path | None = None,
    model_type: str = "generic",
    box_frame: str = "front_lidar_link",
    detector_family: str = "second",
    box_origin: str = "bottom_center",
) -> dict:
    bench_root = Path("/home/prabuddhi/Desktop/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d")
    sys.path.insert(0, str(bench_root))
    from mmdet3d.apis import inference_detector, init_model
    import torch

    resolved_device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    checkpoint_sha = sha256_file(checkpoint_path)
    class_mapping = KITTI_CLASS_MAPPING if class_mapping is None else class_mapping
    selected_ids = {int(selected_class_id)} if selected_class_ids is None else {int(value) for value in selected_class_ids}
    selected_class_name = selected_class_name or class_mapping.get(selected_class_id, str(selected_class_id))
    load_start = time.perf_counter()
    model = init_model(str(config_path), str(checkpoint_path), device=resolved_device)
    model_load_time = time.perf_counter() - load_start

    fields = [
        "sample_id",
        "timestamp",
        "pointcloud_path",
        "x",
        "y",
        "z",
        "length",
        "width",
        "height",
        "yaw",
        "score",
        "raw_label_id",
        "raw_label_name",
        "selected_pedestrian",
        "checkpoint_sha256",
        "inference_time_sec",
        "points_total",
        "points_outside_model_range",
        "empty_frame",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    det_count = 0
    inference_times = []
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample_id, timestamp, pointcloud_path in pointclouds:
            frame_count += 1
            prepared_path, temp_path = prepare_xyzi_bin(pointcloud_path)
            try:
                counts = count_points_outside_kitti_range(prepared_path, point_range=model_point_cloud_range)
                start = time.perf_counter()
                result = inference_detector(model, str(prepared_path))
                inference_time = time.perf_counter() - start
                inference_times.append(inference_time)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            if isinstance(result, (list, tuple)):
                result = result[0]
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()
            selected_in_frame = 0
            for box, score, label in zip(boxes, scores, labels):
                label = int(label)
                selected = label in selected_ids and float(score) >= score_threshold
                selected_in_frame += int(selected)
                x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
                if selected:
                    values = np.asarray([x, y, z, length, width, height, yaw, float(score)], dtype=np.float64)
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"Non-finite selected prediction for {sample_id}: {values.tolist()}")
                    if length <= 0.0 or width <= 0.0 or height <= 0.0:
                        raise ValueError(f"Non-positive selected box dimensions for {sample_id}: {[length, width, height]}")
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "timestamp": "" if timestamp is None else f"{timestamp:.9f}",
                        "pointcloud_path": str(pointcloud_path),
                        "x": x,
                        "y": y,
                        "z": z,
                        "length": length,
                        "width": width,
                        "height": height,
                        "yaw": yaw,
                        "score": float(score),
                        "raw_label_id": label,
                        "raw_label_name": class_mapping.get(label, str(label)),
                        "selected_pedestrian": selected,
                        "checkpoint_sha256": checkpoint_sha,
                        "inference_time_sec": inference_time,
                        "points_total": counts.get("total", 0),
                        "points_outside_model_range": counts.get("outside", 0),
                        "empty_frame": False,
                    }
                )
                det_count += int(selected)
            if selected_in_frame == 0:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "timestamp": "" if timestamp is None else f"{timestamp:.9f}",
                        "pointcloud_path": str(pointcloud_path),
                        "x": "",
                        "y": "",
                        "z": "",
                        "length": "",
                        "width": "",
                        "height": "",
                        "yaw": "",
                        "score": "",
                        "raw_label_id": "",
                        "raw_label_name": "__EMPTY__",
                        "selected_pedestrian": False,
                        "checkpoint_sha256": checkpoint_sha,
                        "inference_time_sec": inference_time,
                        "points_total": counts.get("total", 0),
                        "points_outside_model_range": counts.get("outside", 0),
                        "empty_frame": True,
                    }
                )
    metadata = {
        "detector_family": detector_family,
        "model_type": model_type,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "test_manifest_path": str(manifest_path) if manifest_path is not None else "",
        "test_manifest_sha256": sha256_file(manifest_path) if manifest_path is not None and manifest_path.exists() else "",
        "class_mapping": class_mapping,
        "selected_class_id": selected_class_id,
        "selected_class_ids": sorted(selected_ids),
        "selected_class_name": selected_class_name,
        "score_threshold": score_threshold,
        "nms_settings": nms_settings or {},
        "box_order": "x,y,z,length,width,height,yaw",
        "box_frame": f"LiDAR/{box_frame}",
        "box_origin": f"MMDetection3D LiDAR box convention; recorded as {box_origin} for downstream fusion",
        "cache_box_origin": box_origin,
        "model_point_cloud_range": list(model_point_cloud_range),
        "voxel_size": list(voxel_size) if voxel_size is not None else [],
        "point_features": ["x", "y", "z", "intensity"],
        "units": "metres",
        "device": resolved_device,
        "environment_versions": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", "unknown"),
            "mmdet3d": _module_version("mmdet3d"),
            "mmdet": _module_version("mmdet"),
            "mmcv": _module_version("mmcv"),
            "mmengine": _module_version("mmengine"),
        },
        "model_load_time_sec": model_load_time,
        "frames": frame_count,
        "selected_pedestrian_detections": det_count,
        "inference_time_total_sec": float(sum(inference_times)),
        "inference_time_mean_sec": float(np.mean(inference_times)) if inference_times else 0.0,
        "inference_time_p95_sec": float(np.percentile(inference_times, 95)) if inference_times else 0.0,
        "inference_fps": float(len(inference_times) / sum(inference_times)) if sum(inference_times) > 0.0 else 0.0,
    }
    output_metadata.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
