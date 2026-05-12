"""
Export per-frame 3D detections to JSON for downstream tracking pipelines.
Each model in the config produces one JSON file per point cloud, containing
a list of 3D bounding boxes consumable by a 3D tracking pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

# The aghri3d custom module lives one level above this src/ directory.
# MMEngine's custom_imports needs its parent on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: str
    checkpoint: str


@dataclass(frozen=True)
class PredictConfig3D:
    name: str
    points_dir: Path
    output_dir: Path
    confidence: float
    device: str
    models: List[ModelSpec]


def load_config(path: Path) -> PredictConfig3D:
    with path.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    for field in ('points_dir', 'output_dir', 'models'):
        if not raw.get(field):
            raise ValueError(f'`{field}` is required in config.')
    models = []
    for item in raw['models']:
        if not item.get('name') or not item.get('config') or not item.get('checkpoint'):
            raise ValueError('Each model entry must have `name`, `config`, and `checkpoint`.')
        models.append(ModelSpec(
            name=str(item['name']),
            config=str(item['config']),
            checkpoint=str(item['checkpoint']),
        ))
    return PredictConfig3D(
        name=str(raw.get('name', 'detection3d')),
        points_dir=Path(raw['points_dir']),
        output_dir=Path(raw['output_dir']),
        confidence=float(raw.get('confidence', 0.25)),
        device=str(raw.get('device', 'cuda:0')),
        models=models,
    )


def _install_open3d_stub() -> None:
    if 'open3d' in sys.modules:
        return
    open3d_module = types.ModuleType('open3d')
    geometry_module = types.ModuleType('open3d.geometry')
    utility_module = types.ModuleType('open3d.utility')
    visualization_module = types.ModuleType('open3d.visualization')

    class _Placeholder:
        pass

    visualization_module.Visualizer = _Placeholder
    open3d_module.geometry = geometry_module
    open3d_module.utility = utility_module
    open3d_module.visualization = visualization_module
    sys.modules['open3d'] = open3d_module
    sys.modules['open3d.geometry'] = geometry_module
    sys.modules['open3d.utility'] = utility_module
    sys.modules['open3d.visualization'] = visualization_module


def run_inference(model_spec: ModelSpec, cfg: PredictConfig3D) -> None:
    _install_open3d_stub()
    from mmdet3d.apis import inference_detector, init_model

    model = init_model(model_spec.config, model_spec.checkpoint, device=cfg.device)
    model.eval()

    out_dir = cfg.output_dir / model_spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(cfg.points_dir.glob('*.bin'))
    if not bin_files:
        print(f'[{model_spec.name}] No .bin files found in {cfg.points_dir}', flush=True)
        return

    print(f'[{model_spec.name}] Running on {len(bin_files)} point clouds → {out_dir}', flush=True)

    for bin_path in bin_files:
        result = inference_detector(model, str(bin_path))
        if isinstance(result, (list, tuple)):
            result = result[0]
        pred = result.pred_instances_3d

        boxes = pred.bboxes_3d.tensor.cpu().numpy()   # (N, 7): cx, cy, cz, l, w, h, yaw
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()

        keep = scores >= cfg.confidence
        detections = [
            {
                'x': float(box[0]),
                'y': float(box[1]),
                'z': float(box[2]),
                'l': float(box[3]),
                'w': float(box[4]),
                'h': float(box[5]),
                'yaw': float(box[6]),
                'score': float(score),
                'label': int(label),
            }
            for box, score, label in zip(boxes[keep], scores[keep], labels[keep])
        ]

        out_file = out_dir / (bin_path.stem + '.json')
        out_file.write_text(json.dumps(detections), encoding='utf-8')

    print(f'[{model_spec.name}] Done.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Dump per-frame 3D detections to JSON for tracking pipelines.',
    )
    parser.add_argument('--config', required=True, help='Path to predict YAML config.')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f'Config not found: {cfg_path}')

    cfg = load_config(cfg_path)
    for model_spec in cfg.models:
        run_inference(model_spec, cfg)


if __name__ == '__main__':
    main()
