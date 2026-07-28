#!/usr/bin/env python3
"""Shared helpers for the AGHRI SECOND fine-tuning workflow."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


SPLITS = ("train", "val", "test")
EXPECTED_GENERIC_SECOND_SHA256 = "b086d0a369d0b955b3ff45b6849a4370c46dea2b95445671837a989839eb2bbd"


@dataclass(frozen=True)
class SceneRecord:
    name: str
    root: Path
    ann_path: Path
    lidar_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_split_scenes(splits_dir: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for split in SPLITS:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        scenes = []
        for raw_line in split_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            first = line.split()[0].replace("\\", "/")
            scene = first.split("/", 1)[0]
            scenes.append(scene)
        result[split] = scenes
    return result


def discover_scenes(raw_root: Path) -> Dict[str, SceneRecord]:
    records: Dict[str, SceneRecord] = {}
    for ann_path in raw_root.glob("dataset_part*/**/annotations/lidar_ann.json"):
        scene_root = ann_path.parents[1]
        lidar_dir = scene_root / "sensor_data" / "lidar"
        if lidar_dir.is_dir():
            records[scene_root.name] = SceneRecord(scene_root.name, scene_root, ann_path, lidar_dir)
    direct_ann = raw_root / "annotations" / "lidar_ann.json"
    if direct_ann.exists() and (raw_root / "sensor_data" / "lidar").is_dir():
        records[raw_root.name] = SceneRecord(raw_root.name, raw_root, direct_ann, raw_root / "sensor_data" / "lidar")
    return dict(sorted(records.items()))


def normalize_box(raw_box: Sequence[float]) -> List[float]:
    box = raw_box
    if (
        isinstance(box, list)
        and len(box) == 2
        and all(isinstance(item, list) and len(item) == 9 for item in box)
    ):
        box = box[0]
    if not isinstance(box, list) or len(box) != 9:
        raise ValueError(f"Unsupported bounding box payload: {raw_box!r}")
    x, y, z, dx, dy, dz, _rx, _ry, rz = [float(v) for v in box]
    return [x, y, z, dx, dy, dz, float(rz)]


def iter_scene_frames(scene: SceneRecord) -> Iterator[dict]:
    data = load_json(scene.ann_path)
    for frame in data:
        filename = str(frame.get("File", ""))
        if not filename.endswith(".pcd"):
            continue
        pcd_path = scene.lidar_dir / filename
        yield {
            "scene": scene.name,
            "timestamp": float(frame.get("Timestamp", 0.0)),
            "pcd_path": pcd_path,
            "pcd_stem": Path(filename).stem,
            "labels": frame.get("Labels", []),
        }


def read_pcd_header(path: Path) -> Tuple[Dict[str, List[str]], int]:
    header_lines: List[str] = []
    bytes_read = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PCD header: {path}")
            bytes_read += len(line)
            decoded = line.decode("ascii", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA "):
                break
    header: Dict[str, List[str]] = {}
    for line in header_lines:
        parts = line.split()
        if parts:
            header[parts[0]] = parts[1:]
    return header, bytes_read


def pcd_dtype_fields(header: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    dtype_fields: List[Tuple[str, str]] = []
    for field, size, value_type, count in zip(fields, sizes, types, counts):
        if count != 1:
            raise ValueError(f"Unsupported COUNT={count} for field {field}")
        if value_type == "F" and size == 4:
            dtype_fields.append((field, "<f4"))
        elif value_type == "U" and size == 4:
            dtype_fields.append((field, "<u4"))
        elif value_type == "I" and size == 4:
            dtype_fields.append((field, "<i4"))
        else:
            raise ValueError(f"Unsupported PCD field {field}: TYPE={value_type}, SIZE={size}")
    return dtype_fields


def decode_rgb_float_to_intensity(rgb_value: float) -> float:
    packed = struct.unpack("<I", struct.pack("<f", float(rgb_value)))[0]
    red = (packed >> 16) & 255
    green = (packed >> 8) & 255
    blue = packed & 255
    return ((red + green + blue) / 3.0) / 255.0


def ensure_no_split_overlap(split_scenes: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    overlaps: Dict[str, List[str]] = {}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            shared = sorted(set(split_scenes.get(left, [])) & set(split_scenes.get(right, [])))
            if shared:
                overlaps[f"{left}_vs_{right}"] = shared
    return overlaps


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "mean": None, "p95": None, "max": None}
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "min": min(vals),
        "p05": percentile(vals, 0.05),
        "mean": sum(vals) / len(vals),
        "p95": percentile(vals, 0.95),
        "max": max(vals),
    }
