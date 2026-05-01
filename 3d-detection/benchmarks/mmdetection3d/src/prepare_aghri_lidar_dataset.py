import argparse
import json
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


CLASS_NAME = 'human'


@dataclass(frozen=True)
class SceneFrame:
    scene_name: str
    timestamp: float
    source_pcd: Path
    source_annotation: dict


def _scene_group(scene_name: str) -> str:
    prefixes = (
        'footpath1',
        'footpath2',
        'in_straw',
        'in_vine',
        'out_straw',
        'out_vine',
    )
    for prefix in prefixes:
        if scene_name.startswith(prefix):
            return prefix
    return scene_name.split('_', 1)[0]


def _load_json(path: Path) -> list:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _normalize_box(raw_box) -> List[float]:
    box = raw_box
    if (
        isinstance(box, list)
        and len(box) == 2
        and all(isinstance(item, list) and len(item) == 9 for item in box)
    ):
        box = box[0]
    if not isinstance(box, list) or len(box) != 9:
        raise ValueError(f'Unsupported bounding box payload: {raw_box!r}')
    x, y, z, dx, dy, dz, rx, ry, rz = [float(v) for v in box]
    yaw = float(rz)
    return [x, y, z, dx, dy, dz, yaw]


def _parse_pcd(path: Path) -> np.ndarray:
    with path.open('rb') as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f'Unexpected EOF while reading PCD header: {path}')
            decoded = line.decode('ascii', errors='ignore').strip()
            header_lines.append(decoded)
            if decoded.startswith('DATA '):
                break

        header = {}
        for line in header_lines:
            parts = line.split()
            if parts:
                header[parts[0]] = parts[1:]

        fields = header['FIELDS']
        sizes = [int(v) for v in header['SIZE']]
        types = header['TYPE']
        counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
        num_points = int(header['POINTS'][0])
        storage = header['DATA'][0].lower()
        if storage != 'binary':
            raise ValueError(f'Only binary PCD is supported, found {storage!r} in {path}')

        dtype_fields = []
        for field, size, value_type, count in zip(fields, sizes, types, counts):
            if count != 1:
                raise ValueError(f'Unsupported COUNT={count} for field {field} in {path}')
            if value_type == 'F' and size == 4:
                dtype_fields.append((field, '<f4'))
            elif value_type == 'U' and size == 4:
                dtype_fields.append((field, '<u4'))
            elif value_type == 'I' and size == 4:
                dtype_fields.append((field, '<i4'))
            else:
                raise ValueError(
                    f'Unsupported field layout for {field}: TYPE={value_type}, SIZE={size}, COUNT={count} in {path}')

        structured = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=num_points)
    return structured


def _decode_intensity(points: np.ndarray) -> np.ndarray:
    if 'intensity' in points.dtype.names:
        return points['intensity'].astype(np.float32)
    if 'rgb' in points.dtype.names:
        rgb_float = points['rgb'].astype(np.float32)
        rgb_uint = rgb_float.view(np.uint32)
        r = ((rgb_uint >> 16) & 255).astype(np.float32)
        g = ((rgb_uint >> 8) & 255).astype(np.float32)
        b = (rgb_uint & 255).astype(np.float32)
        return ((r + g + b) / 3.0) / 255.0
    return np.zeros(points.shape[0], dtype=np.float32)


def _convert_pcd_to_bin(src_path: Path, dst_path: Path, overwrite: bool = False) -> None:
    if dst_path.exists() and not overwrite:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    structured = _parse_pcd(src_path)
    converted = np.column_stack([
        structured['x'].astype(np.float32),
        structured['y'].astype(np.float32),
        structured['z'].astype(np.float32),
        _decode_intensity(structured),
    ]).astype(np.float32)
    converted.tofile(dst_path)


def _discover_scenes(raw_root: Path, limit_scenes: int = None) -> List[Path]:
    scenes = sorted(
        path for path in raw_root.iterdir()
        if path.is_dir() and path.name != 'splits'
    )
    if limit_scenes is not None:
        scenes = scenes[:limit_scenes]
    return scenes


def _build_scene_frames(scene_dir: Path) -> List[SceneFrame]:
    ann_path = scene_dir / 'annotations' / 'lidar_ann.json'
    lidar_dir = scene_dir / 'sensor_data' / 'lidar'
    ann_data = _load_json(ann_path)
    frames: List[SceneFrame] = []
    for frame_ann in ann_data:
        source_pcd = lidar_dir / frame_ann['File']
        if not source_pcd.exists():
            raise FileNotFoundError(f'Missing point cloud referenced by annotation: {source_pcd}')
        frames.append(
            SceneFrame(
                scene_name=scene_dir.name,
                timestamp=float(frame_ann['Timestamp']),
                source_pcd=source_pcd,
                source_annotation=frame_ann,
            ))
    return frames


def _split_scenes(scene_names: Sequence[str], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for scene_name in scene_names:
        grouped.setdefault(_scene_group(scene_name), []).append(scene_name)

    rng = random.Random(seed)
    split_map = {'train': [], 'val': [], 'test': []}
    for group_name in sorted(grouped):
        names = sorted(grouped[group_name])
        rng.shuffle(names)
        train_count = max(1, int(round(len(names) * train_ratio)))
        val_count = max(1, int(round(len(names) * val_ratio))) if len(names) >= 3 else 0
        if train_count + val_count >= len(names):
            val_count = max(0, len(names) - train_count - 1)
        train_names = names[:train_count]
        val_names = names[train_count:train_count + val_count]
        test_names = names[train_count + val_count:]
        if not test_names:
            if val_names:
                test_names = [val_names.pop()]
            else:
                test_names = [train_names.pop()]
        split_map['train'].extend(train_names)
        split_map['val'].extend(val_names)
        split_map['test'].extend(test_names)

    for split_name in split_map:
        split_map[split_name] = sorted(split_map[split_name])
    return split_map


def _frame_to_info(frame: SceneFrame) -> dict:
    relative_bin = Path('points') / frame.scene_name / (frame.source_pcd.stem + '.bin')
    info = dict(
        sample_idx=f'{frame.scene_name}/{frame.source_pcd.stem}',
        scene_name=frame.scene_name,
        timestamp=frame.timestamp,
        lidar_points=dict(
            lidar_path=relative_bin.as_posix(),
            num_pts_feats=4,
        ),
        instances=[],
    )

    for raw_instance in frame.source_annotation.get('Labels', []):
        category = str(raw_instance.get('Class', '')).lower()
        if not category.startswith('human'):
            continue
        info['instances'].append(
            dict(
                bbox_3d=_normalize_box(raw_instance.get('BoundingBoxes', [])),
                bbox_label_3d=0,
                bbox_3d_isvalid=True,
                num_lidar_pts=-1,
            )
        )
    return info


def _write_split_files(image_sets_dir: Path, split_to_scenes: Dict[str, List[str]], split_to_samples: Dict[str, List[str]]) -> None:
    image_sets_dir.mkdir(parents=True, exist_ok=True)
    for split_name, scene_names in split_to_scenes.items():
        (image_sets_dir / f'{split_name}_scenes.txt').write_text(
            '\n'.join(scene_names) + '\n',
            encoding='utf-8',
        )
    for split_name, sample_ids in split_to_samples.items():
        (image_sets_dir / f'{split_name}.txt').write_text(
            '\n'.join(sample_ids) + '\n',
            encoding='utf-8',
        )


def _write_infos(path: Path, data_list: List[dict]) -> None:
    payload = dict(
        metainfo=dict(
            dataset='aghri_lidar',
            info_version='1.0',
            classes=[CLASS_NAME],
        ),
        data_list=data_list,
    )
    with path.open('wb') as handle:
        pickle.dump(payload, handle)


def _summarize_infos(split_to_infos: Dict[str, List[dict]], split_to_scenes: Dict[str, List[str]]) -> dict:
    box_dims = []
    box_bottoms = []
    total_instances = 0
    for infos in split_to_infos.values():
        for info in infos:
            for instance in info['instances']:
                total_instances += 1
                bbox = instance['bbox_3d']
                box_dims.append(bbox[3:6])
                box_bottoms.append(bbox[2])
    dims_mean = np.mean(np.asarray(box_dims, dtype=np.float32), axis=0).tolist() if box_dims else [0, 0, 0]
    bottom_mean = float(np.mean(np.asarray(box_bottoms, dtype=np.float32))) if box_bottoms else 0.0

    return dict(
        classes=[CLASS_NAME],
        split_scenes=split_to_scenes,
        split_frames={split_name: len(infos) for split_name, infos in split_to_infos.items()},
        total_frames=sum(len(infos) for infos in split_to_infos.values()),
        total_instances=total_instances,
        anchor_size_mean=dict(length=float(dims_mean[0]), width=float(dims_mean[1]), height=float(dims_mean[2])),
        anchor_bottom_height_mean=bottom_mean,
        point_cloud_range_recommendation=[-10.24, -10.24, -2.0, 25.6, 10.24, 3.0],
    )


def prepare_dataset(
    raw_root: Path,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    workers: int,
    overwrite_points: bool,
    limit_scenes: int = None,
) -> None:
    raw_root = raw_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = _discover_scenes(raw_root, limit_scenes=limit_scenes)
    if not scene_dirs:
        raise RuntimeError(f'No labelled scenes found under {raw_root}')

    scene_frames = {scene_dir.name: _build_scene_frames(scene_dir) for scene_dir in scene_dirs}
    split_to_scenes = _split_scenes(
        scene_names=sorted(scene_frames.keys()),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    convert_jobs = []
    for frames in scene_frames.values():
        for frame in frames:
            dst_path = out_root / 'points' / frame.scene_name / (frame.source_pcd.stem + '.bin')
            convert_jobs.append((frame.source_pcd, dst_path))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_convert_pcd_to_bin, src, dst, overwrite_points)
                for src, dst in convert_jobs
            ]
            for future in futures:
                future.result()
    else:
        for src, dst in convert_jobs:
            _convert_pcd_to_bin(src, dst, overwrite_points)

    split_to_infos: Dict[str, List[dict]] = {'train': [], 'val': [], 'test': []}
    split_to_samples: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}

    for split_name, scene_names in split_to_scenes.items():
        for scene_name in scene_names:
            for frame in scene_frames[scene_name]:
                info = _frame_to_info(frame)
                split_to_infos[split_name].append(info)
                split_to_samples[split_name].append(info['sample_idx'])

    _write_infos(out_root / 'aghri_infos_train.pkl', split_to_infos['train'])
    _write_infos(out_root / 'aghri_infos_val.pkl', split_to_infos['val'])
    _write_infos(out_root / 'aghri_infos_test.pkl', split_to_infos['test'])
    _write_infos(out_root / 'agri_person_infos_train.pkl', split_to_infos['train'])
    _write_infos(out_root / 'agri_person_infos_val.pkl', split_to_infos['val'])
    _write_infos(out_root / 'agri_person_infos_test.pkl', split_to_infos['test'])
    _write_split_files(out_root / 'ImageSets', split_to_scenes, split_to_samples)

    summary = _summarize_infos(split_to_infos, split_to_scenes)
    (out_root / 'dataset_stats.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(f'Prepared dataset at: {out_root}')
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert agri-human LiDAR PCD data to MMDetection3D-ready BIN and PKL files.',
    )
    parser.add_argument(
        '--raw-root',
        default=r'D:\AOC\datasets\agri-human-sensing\labelled_dataset',
        help='Path to the raw labelled_dataset directory.',
    )
    parser.add_argument(
        '--out-root',
        default=r'D:\AOC\datasets\agri-human-sensing\mmdet3d_lidar_aghri',
        help='Output directory for processed points, ImageSets, and info pickles.',
    )
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        '--overwrite-points',
        action='store_true',
        help='Rebuild .bin files even when they already exist.',
    )
    parser.add_argument(
        '--limit-scenes',
        type=int,
        default=None,
        help='Only process the first N scenes. Useful for quick validation.',
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    remaining = 1.0 - args.train_ratio - args.val_ratio
    if args.train_ratio <= 0 or args.val_ratio < 0 or remaining <= 0:
        raise ValueError(
            'train_ratio must be > 0, val_ratio must be >= 0, and train_ratio + val_ratio must be < 1.0.')

    prepare_dataset(
        raw_root=Path(args.raw_root),
        out_root=Path(args.out_root),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        workers=max(1, int(args.workers)),
        overwrite_points=bool(args.overwrite_points),
        limit_scenes=args.limit_scenes,
    )


if __name__ == '__main__':
    main()
