"""Export scored 3D detections for one split/model; verify frame coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/work/users/pwariyapperuma/Dataset_paper')
CODE = BASE / 'aghri_second_finetune/code/mmdetection3d'
MODELS = {
    'pointpillars_scratch': (
        CODE / 'configs/models/pointpillars_aghri.py',
        BASE / 'aghri_3d_scratch_stage5/outputs/full/pointpillars_scratch/best_aghri_bev_ap_mean_epoch_108.pth',
    ),
    'second_scratch': (
        CODE / 'configs/models/second_aghri.py',
        BASE / 'aghri_3d_scratch_stage5/outputs/full/second_scratch/best_aghri_bev_ap_mean_epoch_36.pth',
    ),
}
POINT_ROOTS = {
    'val': BASE / 'aghri_second_finetune/data/aghri_mmdet3d/points',
    'test': BASE / 'aghri_3d_test_stage6/data/aghri_mmdet3d/points',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=POINT_ROOTS, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    names = [s.strip() for s in (ROOT / 'split_lists' / f'{args.split}.txt').read_text().splitlines() if s.strip()]
    assert len(names) == (7 if args.split == 'val' else 6) and len(set(names)) == len(names)
    config, checkpoint = MODELS[args.model]
    script = CODE / 'src/dump_detections_3d.py'
    for path in (config, checkpoint, script):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    output_root = ROOT / 'detections' / args.split / args.model
    config_root = ROOT / 'configs' / args.split / args.model
    config_root.mkdir(parents=True, exist_ok=True)
    frame_total = 0
    for name in names:
        points = POINT_ROOTS[args.split] / name
        expected = {p.stem for p in points.glob('*.bin')}
        if not expected:
            raise RuntimeError(f'No prepared point clouds: {points}')
        ann_path = ROOT / 'annotations' / name / 'annotations/lidar_ann.json'
        frames = json.loads(ann_path.read_text())
        annotated = {Path(f['File']).stem for f in frames if str(f.get('File', '')).endswith('.pcd')}
        if expected != annotated or len(annotated) != len(frames):
            raise RuntimeError(f'Annotation/point-cloud frame mismatch: {name}')
        scene_output = output_root / name
        existing = {p.stem for p in scene_output.glob('*.json')} if scene_output.exists() else set()
        if existing != expected:
            if existing:
                raise RuntimeError(f'Partial export at {scene_output}; inspect it before retrying')
            cfg_path = config_root / f'{name}.yaml'
            cfg_path.write_text(yaml.safe_dump({
                'name': 'aghri_3d_tracking',
                'device': 'cuda:0',
                'confidence': 0.05,
                'points_dir': str(points),
                'output_dir': str(output_root),
                'models': [{'name': name, 'config': str(config), 'checkpoint': str(checkpoint)}],
            }, sort_keys=False))
            subprocess.run([sys.executable, str(script), '--config', str(cfg_path)], check=True, cwd=CODE)
        produced = {p.stem for p in scene_output.glob('*.json')}
        if produced != expected:
            raise RuntimeError(f'Incomplete export: {name}: {len(produced)}/{len(expected)}')
        for file in scene_output.glob('*.json'):
            rows = json.loads(file.read_text())
            if not isinstance(rows, list) or any(not isinstance(d.get('score'), (int, float)) for d in rows):
                raise ValueError(f'Invalid detection JSON: {file}')
        frame_total += len(expected)
        print(f'{args.split} {args.model} {name}: {len(expected)} frames', flush=True)
    if args.split == 'test' and frame_total != 1308:
        raise RuntimeError(f'Expected 1308 test frames, found {frame_total}')
    manifest_dir = ROOT / 'manifests' / args.split
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'split': args.split, 'model': args.model, 'sequences': names,
        'frames': frame_total, 'confidence_export_floor': 0.05,
        'config': str(config), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'split_list_sha256': sha256(ROOT / 'split_lists' / f'{args.split}.txt'),
        'output_root': str(output_root),
    }
    (manifest_dir / f'{args.model}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Export complete:', json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
