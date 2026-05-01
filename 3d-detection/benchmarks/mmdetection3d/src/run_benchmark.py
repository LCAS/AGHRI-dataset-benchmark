import argparse
import csv
import json
import os
import pickle
import types
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import yaml

ROOT_3D = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
BENCH_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FIELDS = [
    'model',
    'train_time_sec',
    'inf_time_per_frame_ms',
    'preprocess_time_per_frame_ms',
    'postprocess_time_per_frame_ms',
    'total_time_per_frame_ms',
    'checkpoint_used',
    'eval_split',
    'num_classes',
    'bev_ap_mean',
    '3d_ap_mean',
    'bev_precision_mean',
    '3d_precision_mean',
    'bev_recall_mean',
    '3d_recall_mean',
    'bev_f1_mean',
    '3d_f1_mean',
    'bev_ap_0.25',
    'bev_ap_0.50',
    'bev_ap_0.75',
    'bev_precision_0.25',
    'bev_precision_0.50',
    'bev_precision_0.75',
    '3d_ap_0.25',
    '3d_ap_0.50',
    '3d_ap_0.75',
    '3d_precision_0.25',
    '3d_precision_0.50',
    '3d_precision_0.75',
    'bev_recall_0.25',
    'bev_recall_0.50',
    'bev_recall_0.75',
    '3d_recall_0.25',
    '3d_recall_0.50',
    '3d_recall_0.75',
    'bev_f1_0.25',
    'bev_f1_0.50',
    'bev_f1_0.75',
    '3d_f1_0.25',
    '3d_f1_0.50',
    '3d_f1_0.75',
    'precision',
    'recall',
    'f1',
    'per_class_precision',
    'per_class_recall',
    'per_class_f1',
    'epochs',
    'batch',
    'eval_batch',
    'dataset',
    'config_device',
    'fixed_input_inference_ms',
    'fixed_input_sample',
    'fixed_input_num_points',
    'fixed_input_dtype',
    'fixed_input_device',
    'fixed_input_iters',
    'fixed_input_warmup',
    'train_results_dir',
    'eval_results_dir',
]


def load_config(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _set_cuda_device(device_cfg: Any) -> None:
    if device_cfg is None:
        return
    if isinstance(device_cfg, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_cfg)
        return
    if isinstance(device_cfg, str):
        normalized = device_cfg.strip().lower()
        if normalized == 'cpu':
            return
        if normalized.startswith('cuda:'):
            os.environ['CUDA_VISIBLE_DEVICES'] = normalized.split(':', 1)[1]
            return
        if normalized.isdigit():
            os.environ['CUDA_VISIBLE_DEVICES'] = normalized
            return
    if isinstance(device_cfg, (list, tuple)) and device_cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(item) for item in device_cfg)


def _device_to_mmdet_str(device_cfg: Any) -> str:
    try:
        import torch
    except Exception:
        torch = None

    cuda_available = bool(torch and torch.cuda.is_available())
    if device_cfg is None:
        return 'cuda:0' if cuda_available else 'cpu'
    if isinstance(device_cfg, int):
        return 'cuda:0' if cuda_available else 'cpu'
    if isinstance(device_cfg, str):
        value = device_cfg.strip().lower()
        if value == 'cpu':
            return 'cpu'
        if value.startswith('cuda'):
            return value if cuda_available else 'cpu'
        if value.isdigit():
            return 'cuda:0' if cuda_available else 'cpu'
    if isinstance(device_cfg, (list, tuple)):
        return 'cuda:0' if cuda_available else 'cpu'
    return 'cuda:0' if cuda_available else 'cpu'


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _is_remote_ref(path_str: str) -> bool:
    parsed = urlparse(str(path_str))
    return parsed.scheme in {'http', 'https'}


def _resolve_checkpoint_ref(path_str: str, repo_root: Path) -> str:
    if _is_remote_ref(path_str):
        return str(path_str)
    return str(_resolve_path(path_str, repo_root))


def _unwrap_dataset(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    current = dataset_cfg
    while isinstance(current, dict) and 'dataset' in current:
        current = current['dataset']
    return current


def _set_nested_data_root(dataset_cfg: Dict[str, Any], data_root: str) -> None:
    current = dataset_cfg
    while isinstance(current, dict):
        current['data_root'] = data_root
        if 'dataset' not in current:
            break
        current = current['dataset']


def _override_evaluator_ann_file(evaluator_cfg: Any, data_root: str) -> None:
    if isinstance(evaluator_cfg, list):
        for item in evaluator_cfg:
            _override_evaluator_ann_file(item, data_root)
        return
    if not isinstance(evaluator_cfg, dict):
        return
    ann_file = evaluator_cfg.get('ann_file')
    if not ann_file:
        return
    ann_name = Path(str(ann_file)).name
    evaluator_cfg['ann_file'] = f'{data_root}/infos/{ann_name}'


def _apply_dataset_root_override(cfg: Any) -> None:
    data_root_override = os.environ.get('AGRIHUMAN_3D_DATA_ROOT')
    if not data_root_override:
        return
    for dataloader_name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataloader_cfg = cfg.get(dataloader_name)
        if not dataloader_cfg or 'dataset' not in dataloader_cfg:
            continue
        _set_nested_data_root(dataloader_cfg['dataset'], data_root_override)
    _override_evaluator_ann_file(cfg.get('val_evaluator'), data_root_override)
    _override_evaluator_ann_file(cfg.get('test_evaluator'), data_root_override)


def _install_open3d_stub() -> None:
    if os.environ.get('AGRIHUMAN_DISABLE_OPEN3D_STUB', '0') == '1':
        return
    if 'open3d' in sys.modules:
        return

    open3d_module = types.ModuleType('open3d')
    geometry_module = types.ModuleType('open3d.geometry')
    utility_module = types.ModuleType('open3d.utility')
    visualization_module = types.ModuleType('open3d.visualization')

    class _Open3DPlaceholder:
        pass

    visualization_module.Visualizer = _Open3DPlaceholder
    open3d_module.geometry = geometry_module
    open3d_module.utility = utility_module
    open3d_module.visualization = visualization_module

    sys.modules['open3d'] = open3d_module
    sys.modules['open3d.geometry'] = geometry_module
    sys.modules['open3d.utility'] = utility_module
    sys.modules['open3d.visualization'] = visualization_module


def _disable_visualization(cfg: Any) -> None:

    if 'default_hooks' not in cfg or cfg.default_hooks is None:
        cfg.default_hooks = {}
    cfg.default_hooks['visualization'] = None
    if 'custom_hooks' in cfg and isinstance(cfg.custom_hooks, list):
        cfg.custom_hooks = [
            hook for hook in cfg.custom_hooks
            if not (
                isinstance(hook, dict)
                and 'Visualization' in str(hook.get('type', ''))
            )
        ]
    cfg['visualizer'] = None
    cfg.visualizer = None
    if 'vis_backends' in cfg:
        cfg['vis_backends'] = []


def _join_dataset_path(dataset_cfg: Dict[str, Any]) -> Optional[str]:
    ann_file = dataset_cfg.get('ann_file')
    if ann_file is None:
        return None
    ann_path = Path(str(ann_file))
    if ann_path.is_absolute():
        return str(ann_path)
    data_root = dataset_cfg.get('data_root')
    if data_root:
        return str((Path(str(data_root)) / ann_path).as_posix())
    return str(ann_path.as_posix())


def _infer_split_name(ann_file: Optional[str]) -> Optional[str]:
    if not ann_file:
        return None
    ann_file = ann_file.lower()
    for split_name in ('test', 'val', 'train'):
        if split_name in ann_file:
            return split_name
    return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value


def _extract_sample_pcd(test_dataset_cfg: Dict[str, Any]) -> Optional[Path]:
    ann_file = test_dataset_cfg.get('ann_file')
    data_root = test_dataset_cfg.get('data_root')
    if not ann_file or not data_root:
        return None

    ann_path = Path(str(ann_file))
    if not ann_path.is_absolute():
        ann_path = Path(str(data_root)) / ann_path
    if not ann_path.exists():
        return None

    with ann_path.open('rb') as handle:
        payload = pickle.load(handle)
    data_list = payload.get('data_list', [])
    if not data_list:
        return None

    first_item = data_list[0]
    lidar_rel_path = first_item.get('lidar_points', {}).get('lidar_path')
    if not lidar_rel_path:
        return None
    sample_path = Path(str(data_root)) / lidar_rel_path
    return sample_path.resolve()


def _count_points(bin_path: Path) -> Optional[int]:
    try:
        return int(bin_path.stat().st_size // (4 * 4))
    except Exception:
        return None


def benchmark_fixed_input(
    cfg_path: Path,
    checkpoint_path: Optional[Union[str, Path]],
    device: str,
    sample_pcd_path: Optional[Path],
    iters: int,
    warmup: int,
) -> Dict[str, Any]:
    if checkpoint_path is None or sample_pcd_path is None or not sample_pcd_path.exists():
        return {'fixed_input_inference_ms': None}

    try:
        import torch
        from mmdet3d.apis import inference_detector, init_model
    except Exception:
        return {'fixed_input_inference_ms': None}

    iters = max(int(iters), 0)
    warmup = max(int(warmup), 0)
    if iters == 0:
        return {'fixed_input_inference_ms': None}

    model = init_model(str(cfg_path), str(checkpoint_path), device=device)
    sample_path_str = str(sample_pcd_path)

    for _ in range(warmup):
        inference_detector(model, sample_path_str)

    if device.startswith('cuda'):
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        inference_detector(model, sample_path_str)
    if device.startswith('cuda'):
        torch.cuda.synchronize()

    elapsed_ms = (time.time() - start) * 1000.0 / iters
    first_param = next(model.parameters())
    return {
        'fixed_input_inference_ms': round(elapsed_ms, 3),
        'fixed_input_sample': sample_path_str,
        'fixed_input_num_points': _count_points(sample_pcd_path),
        'fixed_input_dtype': str(first_param.dtype),
        'fixed_input_device': str(first_param.device),
        'fixed_input_iters': iters,
        'fixed_input_warmup': warmup,
    }


def _latest_checkpoint(work_dir: Path) -> Optional[Path]:
    latest = work_dir / 'latest.pth'
    if latest.exists():
        return latest
    candidates = sorted(work_dir.glob('*.pth'), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _best_checkpoint(work_dir: Path, metric_name: Optional[str]) -> Optional[Path]:
    candidates = sorted(
        work_dir.glob('best*.pth'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    if metric_name:
        metric_token = metric_name.replace('/', '_')
        preferred = [path for path in candidates if metric_token in path.name]
        if preferred:
            return preferred[0]
    return candidates[0]


def _select_eval_checkpoint(
    work_dir: Path,
    explicit_checkpoint: Optional[Union[str, Path]],
    prefer_best: bool,
    best_metric_name: Optional[str],
) -> Optional[Union[str, Path]]:
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    if prefer_best:
        best = _best_checkpoint(work_dir, best_metric_name)
        if best is not None:
            return best
    return _latest_checkpoint(work_dir)


def _configure_checkpoint_hook(cfg: Any, bench_cfg: Dict[str, Any]) -> None:
    if 'default_hooks' not in cfg or cfg.default_hooks is None:
        cfg.default_hooks = {}
    checkpoint_hook = cfg.default_hooks.get('checkpoint', {}) or {}

    save_best_metric = bench_cfg.get('save_best_metric')
    if save_best_metric:
        checkpoint_hook['save_best'] = save_best_metric
        checkpoint_hook['rule'] = bench_cfg.get('save_best_rule', 'greater')
    if bench_cfg.get('max_keep_ckpts') is not None:
        checkpoint_hook['max_keep_ckpts'] = int(bench_cfg['max_keep_ckpts'])

    cfg.default_hooks['checkpoint'] = checkpoint_hook


def train_and_eval_model(model_item: Dict[str, Any], bench_cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    _install_open3d_stub()

    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import RUNNERS

    cfg_path = _resolve_path(model_item['config'], repo_root)
    if not cfg_path.exists():
        raise FileNotFoundError(f'Model config not found: {cfg_path}')

    model_name = model_item.get('name', cfg_path.stem)
    cfg = Config.fromfile(str(cfg_path))
    cfg.launcher = 'none'
    _apply_dataset_root_override(cfg)
    _disable_visualization(cfg)

    if bench_cfg.get('cfg_options'):
        cfg.merge_from_dict(bench_cfg['cfg_options'])
    if model_item.get('cfg_options'):
        cfg.merge_from_dict(model_item['cfg_options'])
    _disable_visualization(cfg)

    seed = model_item.get('seed', bench_cfg.get('seed'))
    if seed is not None:
        cfg.randomness = dict(seed=int(seed))

    default_work_dir = (
        Path(bench_cfg.get('project', '3d-detection/reports/benchmarks/mmdetection3d/runs'))
        / bench_cfg.get('name', 'mmdetection3d-benchmark')
        / model_name
    )
    work_dir = _resolve_path(model_item.get('work_dir', str(default_work_dir)), repo_root)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    _configure_checkpoint_hook(cfg, bench_cfg)
    _disable_visualization(cfg)

    train_enabled = bool(model_item.get('train', True))
    checkpoint = model_item.get('checkpoint')
    if checkpoint:
        cfg.load_from = _resolve_checkpoint_ref(checkpoint, repo_root)
    else:
        checkpoint = None

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    train_time_sec = None
    if train_enabled:
        start = time.time()
        runner.train()
        train_time_sec = round(time.time() - start, 2)
    elif checkpoint is None and cfg.get('load_from') is None:
        raise ValueError(f"Model '{model_name}' has train=false but no checkpoint or load_from configured.")

    prefer_best = bool(bench_cfg.get('prefer_best_checkpoint', True))
    best_metric_name = bench_cfg.get('save_best_metric')
    checkpoint_for_eval = checkpoint if not train_enabled else None
    if checkpoint_for_eval is None and not train_enabled and cfg.get('load_from') is not None:
        checkpoint_for_eval = str(cfg.load_from)

    checkpoint = _select_eval_checkpoint(
        work_dir=work_dir,
        explicit_checkpoint=checkpoint_for_eval,
        prefer_best=prefer_best,
        best_metric_name=best_metric_name,
    )
    if checkpoint is None:
        raise FileNotFoundError(f'No checkpoint found in {work_dir}. Cannot run evaluation.')

    cfg.load_from = str(checkpoint)
    cfg.resume = False
    _disable_visualization(cfg)
    if 'runner_type' not in cfg:
        eval_runner = Runner.from_cfg(cfg)
    else:
        eval_runner = RUNNERS.build(cfg)
    metrics = _to_jsonable(eval_runner.test() or {})

    test_dataset_cfg = _unwrap_dataset(cfg.test_dataloader['dataset'])
    metainfo = test_dataset_cfg.get('metainfo', cfg.get('metainfo', {}))
    classes = metainfo.get('classes') if isinstance(metainfo, dict) else None
    primary_mode = str(bench_cfg.get('summary_primary_mode', '3d')).lower()
    primary_iou = float(bench_cfg.get('summary_primary_iou', 0.50))
    primary_iou_key = f'{primary_iou:.2f}'
    primary_precision_key = f'{primary_mode}_precision_{primary_iou_key}'
    primary_recall_key = f'{primary_mode}_recall_{primary_iou_key}'
    primary_f1_key = f'{primary_mode}_f1_{primary_iou_key}'

    per_class_precision = None
    per_class_recall = None
    per_class_f1 = None
    if classes:
        per_class_precision = {
            class_name: _to_float(metrics.get(f'{class_name}_{primary_precision_key}'))
            for class_name in classes
        }
        per_class_recall = {
            class_name: _to_float(metrics.get(f'{class_name}_{primary_recall_key}'))
            for class_name in classes
        }
        per_class_f1 = {
            class_name: _to_float(metrics.get(f'{class_name}_{primary_f1_key}'))
            for class_name in classes
        }

    device_cfg = model_item.get('device', bench_cfg.get('device'))
    bench_device = _device_to_mmdet_str(device_cfg)
    sample_pcd_path = _extract_sample_pcd(test_dataset_cfg)
    bench = {'fixed_input_inference_ms': None}
    if bench_cfg.get('benchmark_fixed_input', False):
        bench = benchmark_fixed_input(
            cfg_path=cfg_path,
            checkpoint_path=checkpoint,
            device=bench_device,
            sample_pcd_path=sample_pcd_path,
            iters=bench_cfg.get('benchmark_iters', 100),
            warmup=bench_cfg.get('benchmark_warmup', 20),
        )
    inf_ms = bench.get('fixed_input_inference_ms')

    summary = {
        'model': model_name,
        'train_time_sec': train_time_sec,
        'inf_time_per_frame_ms': inf_ms,
        'preprocess_time_per_frame_ms': None,
        'postprocess_time_per_frame_ms': None,
        'total_time_per_frame_ms': inf_ms,
        'checkpoint_used': str(checkpoint),
        'eval_split': _infer_split_name(test_dataset_cfg.get('ann_file')),
        'num_classes': len(classes) if classes else None,
        'bev_ap_mean': _to_float(metrics.get('bev_ap_mean')),
        '3d_ap_mean': _to_float(metrics.get('3d_ap_mean')),
        'bev_precision_mean': _to_float(metrics.get('bev_precision_mean')),
        '3d_precision_mean': _to_float(metrics.get('3d_precision_mean')),
        'bev_recall_mean': _to_float(metrics.get('bev_recall_mean')),
        '3d_recall_mean': _to_float(metrics.get('3d_recall_mean')),
        'bev_f1_mean': _to_float(metrics.get('bev_f1_mean')),
        '3d_f1_mean': _to_float(metrics.get('3d_f1_mean')),
        'bev_ap_0.25': _to_float(metrics.get('bev_ap_0.25')),
        'bev_ap_0.50': _to_float(metrics.get('bev_ap_0.50')),
        'bev_ap_0.75': _to_float(metrics.get('bev_ap_0.75')),
        'bev_precision_0.25': _to_float(metrics.get('bev_precision_0.25')),
        'bev_precision_0.50': _to_float(metrics.get('bev_precision_0.50')),
        'bev_precision_0.75': _to_float(metrics.get('bev_precision_0.75')),
        **{
            '3d_ap_0.25': _to_float(metrics.get('3d_ap_0.25')),
            '3d_ap_0.50': _to_float(metrics.get('3d_ap_0.50')),
            '3d_ap_0.75': _to_float(metrics.get('3d_ap_0.75')),
            '3d_precision_0.25': _to_float(metrics.get('3d_precision_0.25')),
            '3d_precision_0.50': _to_float(metrics.get('3d_precision_0.50')),
            '3d_precision_0.75': _to_float(metrics.get('3d_precision_0.75')),
            'bev_recall_0.25': _to_float(metrics.get('bev_recall_0.25')),
            'bev_recall_0.50': _to_float(metrics.get('bev_recall_0.50')),
            'bev_recall_0.75': _to_float(metrics.get('bev_recall_0.75')),
            '3d_recall_0.25': _to_float(metrics.get('3d_recall_0.25')),
            '3d_recall_0.50': _to_float(metrics.get('3d_recall_0.50')),
            '3d_recall_0.75': _to_float(metrics.get('3d_recall_0.75')),
            'bev_f1_0.25': _to_float(metrics.get('bev_f1_0.25')),
            'bev_f1_0.50': _to_float(metrics.get('bev_f1_0.50')),
            'bev_f1_0.75': _to_float(metrics.get('bev_f1_0.75')),
            '3d_f1_0.25': _to_float(metrics.get('3d_f1_0.25')),
            '3d_f1_0.50': _to_float(metrics.get('3d_f1_0.50')),
            '3d_f1_0.75': _to_float(metrics.get('3d_f1_0.75')),
        },
        'precision': _to_float(metrics.get(primary_precision_key)),
        'recall': _to_float(metrics.get(primary_recall_key)),
        'f1': _to_float(metrics.get(primary_f1_key)),
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'epochs': cfg.get('train_cfg', {}).get('max_epochs'),
        'batch': cfg.train_dataloader.get('batch_size'),
        'eval_batch': cfg.test_dataloader.get('batch_size'),
        'dataset': _join_dataset_path(test_dataset_cfg),
        'config_device': bench_cfg.get('device'),
        'train_results_dir': str(work_dir),
        'eval_results_dir': str(work_dir),
    }
    summary.update(bench)

    ordered = {field: summary.get(field) for field in SUMMARY_FIELDS}
    return _to_jsonable(ordered)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _to_csvable(row.get(field)) for field in SUMMARY_FIELDS})


def main() -> None:
    default_config = ROOT_3D / 'benchmarks' / 'mmdetection3d' / 'configs' / 'benchmark_mmdetection3d_agrihuman.yaml'
    default_out = (
        ROOT_3D
        / 'reports'
        / 'benchmarks'
        / 'summary'
        / 'mmdetection3d'
        / 'summary_mmdetection3d_agrihuman.csv'
    )

    parser = argparse.ArgumentParser(
        description='Train/evaluate MMDetection3D models and export summary CSV/JSON.',
    )
    parser.add_argument('--config', default=str(default_config), help='Path to benchmark YAML config.')
    parser.add_argument('--out', default=str(default_out), help='CSV output path.')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    bench_cfg = load_config(cfg_path)

    _set_cuda_device(bench_cfg.get('device'))
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BENCH_ROOT))

    rows = []
    model_items = bench_cfg.get('models', [])
    if not model_items:
        raise ValueError('No models configured. Add at least one entry under `models`.')

    for model_item in model_items:
        if isinstance(model_item, str):
            model_item = {'config': model_item}
        rows.append(train_and_eval_model(model_item, bench_cfg, REPO_ROOT))

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = (REPO_ROOT / out_csv).resolve()
    write_csv(out_csv, rows)
    out_json = out_csv.with_suffix('.json')
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} rows to {out_csv}')
    print(f'Wrote JSON to {out_json}')


if __name__ == '__main__':
    main()
