"""End-to-end PointPillars training + evaluation on KITTI (native pipeline).

Trains and evaluates using the OFFICIAL MMDetection3D KittiDataset and
KittiMetric, then saves results in the same CSV/JSON format as run_benchmark.py.

Prerequisites:
  1. Run prepare_kitti_native_dataset.py to create kitti_infos_{train,val}.pkl
     and kitti_dbinfos_train.pkl in the KITTI dataset root.

  2. mmdet3d 1.4.0, mmdet 3.3.0, mmcv 2.1.0 must be installed.

Usage:
  python src/run_kitti_native_benchmark.py \\
      --config configs/benchmark_mmdetection3d_kitti_native_pointpillars.yaml \\
      --out 3d-detection/reports/benchmarks/summary/mmdetection3d/summary_mmdetection3d_kitti_native_pointpillars.csv

  # Eval-only (no training) — supply a checkpoint in the YAML or via --checkpoint:
  python src/run_kitti_native_benchmark.py \\
      --config configs/benchmark_mmdetection3d_kitti_native_pointpillars.yaml \\
      --checkpoint /path/to/best.pth \\
      --no-train \\
      --out ...

Official KittiMetric output keys (mmdet3d 1.4.0):
  Pedestrian_3D_AP40_easy_strict / moderate_strict / hard_strict
  Pedestrian_3D_AP11_easy_strict / moderate_strict / hard_strict
  Pedestrian_BEV_AP40_easy_strict / moderate_strict / hard_strict
  (possibly prefixed with 'kitti/', 'KITTI/', or the metric default_prefix)
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_3D    = Path(__file__).resolve().parents[3]  # 3d-detection/
REPO_ROOT  = Path(__file__).resolve().parents[4]  # agri-human-dataset-benchmark/
BENCH_ROOT = Path(__file__).resolve().parents[1]  # benchmarks/mmdetection3d/

# ---------------------------------------------------------------------------
# CSV output fields (superset of AGHRI pipeline fields + native KITTI fields)
# ---------------------------------------------------------------------------

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
    # Native KITTI R40 (preferred — matches published model-zoo numbers)
    'kitti_native_3d_ap40_easy',
    'kitti_native_3d_ap40_moderate',
    'kitti_native_3d_ap40_hard',
    'kitti_native_bev_ap40_easy',
    'kitti_native_bev_ap40_moderate',
    'kitti_native_bev_ap40_hard',
    # Native KITTI R11 (older protocol, included for completeness)
    'kitti_native_3d_ap11_easy',
    'kitti_native_3d_ap11_moderate',
    'kitti_native_3d_ap11_hard',
    'kitti_native_bev_ap11_easy',
    'kitti_native_bev_ap11_moderate',
    'kitti_native_bev_ap11_hard',
    # Standard fields (will be None for native KittiMetric runs)
    'bev_ap_mean',
    '3d_ap_mean',
    'kitti_3d_ap_easy',
    'kitti_3d_ap_moderate',
    'kitti_3d_ap_hard',
    'kitti_bev_ap_easy',
    'kitti_bev_ap_moderate',
    'kitti_bev_ap_hard',
    'kitti_3d_recall_easy',
    'kitti_3d_recall_moderate',
    'kitti_3d_recall_hard',
    'bev_precision_mean',
    '3d_precision_mean',
    'bev_recall_mean',
    '3d_recall_mean',
    'bev_f1_mean',
    '3d_f1_mean',
    'bev_ap_0.25',
    'bev_ap_0.50',
    'bev_ap_0.75',
    '3d_ap_0.25',
    '3d_ap_0.50',
    '3d_ap_0.75',
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
    'all_raw_metrics',  # JSON-serialised dump of every metric the evaluator returned
]


# ---------------------------------------------------------------------------
# Metric key extraction from official KittiMetric output
# ---------------------------------------------------------------------------

# Official KittiMetric may prefix keys with its default_prefix ('kitti') or
# a user-supplied prefix, and may separate with '/' or '_'.  We try all
# combinations so the runner is robust across mmdet3d versions.
_KITTI_NATIVE_KEY_MAP = {
    # output field name : list of candidate key fragments to search for
    'kitti_native_3d_ap40_easy':      ['3D_AP40_easy',      '3d_ap40_easy',      'AP40_easy_3d',   'Pedestrian_3D_AP40_easy'],
    'kitti_native_3d_ap40_moderate':  ['3D_AP40_moderate',  '3d_ap40_moderate',  'AP40_moderate_3d','Pedestrian_3D_AP40_moderate'],
    'kitti_native_3d_ap40_hard':      ['3D_AP40_hard',      '3d_ap40_hard',      'AP40_hard_3d',   'Pedestrian_3D_AP40_hard'],
    'kitti_native_bev_ap40_easy':     ['BEV_AP40_easy',     'bev_ap40_easy',     'AP40_easy_bev',  'Pedestrian_BEV_AP40_easy'],
    'kitti_native_bev_ap40_moderate': ['BEV_AP40_moderate', 'bev_ap40_moderate', 'AP40_moderate_bev','Pedestrian_BEV_AP40_moderate'],
    'kitti_native_bev_ap40_hard':     ['BEV_AP40_hard',     'bev_ap40_hard',     'AP40_hard_bev',  'Pedestrian_BEV_AP40_hard'],
    'kitti_native_3d_ap11_easy':      ['3D_AP11_easy',      '3d_ap11_easy',      'AP11_easy_3d',   'Pedestrian_3D_AP11_easy'],
    'kitti_native_3d_ap11_moderate':  ['3D_AP11_moderate',  '3d_ap11_moderate',  'AP11_moderate_3d','Pedestrian_3D_AP11_moderate'],
    'kitti_native_3d_ap11_hard':      ['3D_AP11_hard',      '3d_ap11_hard',      'AP11_hard_3d',   'Pedestrian_3D_AP11_hard'],
    'kitti_native_bev_ap11_easy':     ['BEV_AP11_easy',     'bev_ap11_easy',     'AP11_easy_bev',  'Pedestrian_BEV_AP11_easy'],
    'kitti_native_bev_ap11_moderate': ['BEV_AP11_moderate', 'bev_ap11_moderate', 'AP11_moderate_bev','Pedestrian_BEV_AP11_moderate'],
    'kitti_native_bev_ap11_hard':     ['BEV_AP11_hard',     'bev_ap11_hard',     'AP11_hard_bev',  'Pedestrian_BEV_AP11_hard'],
}


def _extract_native_kitti_metrics(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Map official KittiMetric keys → our output field names.

    KittiMetric may return keys like:
      'kitti/Pedestrian_3D_AP40_moderate_strict'   (with prefix + /)
      'Pedestrian_3D_AP40_moderate_strict'          (no prefix)
      'pedestrian_3d_ap40_moderate_strict'          (lower-case variant)
    We search case-insensitively for each fragment from _KITTI_NATIVE_KEY_MAP.
    """
    # Build a lower-cased copy for case-insensitive matching.
    raw_lower = {k.lower(): v for k, v in raw.items()}
    # Also build a list of (original_key_lower, original_value) for prefix stripping.
    raw_items_lower = [(k.lower(), v) for k, v in raw.items()]

    result: Dict[str, Optional[float]] = {}
    for field_name, fragments in _KITTI_NATIVE_KEY_MAP.items():
        value = None
        for frag in fragments:
            frag_l = frag.lower()
            # Direct match (case-insensitive).
            for raw_key_l, raw_val in raw_items_lower:
                if frag_l in raw_key_l:
                    try:
                        value = float(raw_val)
                    except (TypeError, ValueError):
                        value = None
                    break
            if value is not None:
                break
        result[field_name] = value

    if all(v is None for v in result.values()):
        print("  WARNING: No native KittiMetric keys matched.  Raw metrics:")
        for k, v in raw.items():
            print(f"    {k!r}: {v}")
        print("  Adjust _KITTI_NATIVE_KEY_MAP in run_kitti_native_benchmark.py "
              "if the keys above use a different naming convention.")

    return result


# ---------------------------------------------------------------------------
# Utilities shared with run_benchmark.py
# ---------------------------------------------------------------------------

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(i) for i in v]
    if isinstance(v, Path):
        return str(v)
    if hasattr(v, 'item'):
        try:
            return v.item()
        except Exception:
            return v
    return v


def _to_csvable(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return str(v)
    return v


def _set_cuda_device(device_cfg: Any) -> None:
    if device_cfg is None:
        return
    if isinstance(device_cfg, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_cfg)
    elif isinstance(device_cfg, str):
        n = device_cfg.strip().lower()
        if n.startswith('cuda:'):
            os.environ['CUDA_VISIBLE_DEVICES'] = n.split(':', 1)[1]
        elif n.isdigit():
            os.environ['CUDA_VISIBLE_DEVICES'] = n
    elif isinstance(device_cfg, (list, tuple)) and device_cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in device_cfg)


def _device_str(device_cfg: Any) -> str:
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if device_cfg is None:
        return 'cuda:0' if cuda else 'cpu'
    if isinstance(device_cfg, int):
        return 'cuda:0' if cuda else 'cpu'
    if isinstance(device_cfg, str):
        v = device_cfg.strip().lower()
        if v == 'cpu':
            return 'cpu'
        if v.startswith('cuda'):
            return v if cuda else 'cpu'
        if v.isdigit():
            return 'cuda:0' if cuda else 'cpu'
    return 'cuda:0' if cuda else 'cpu'


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def _is_remote(path_str: str) -> bool:
    return urlparse(str(path_str)).scheme in {'http', 'https'}


def _install_open3d_stub() -> None:
    if 'open3d' in sys.modules:
        return
    o3d = types.ModuleType('open3d')
    for sub in ('geometry', 'utility', 'visualization'):
        mod = types.ModuleType(f'open3d.{sub}')
        setattr(o3d, sub, mod)
        sys.modules[f'open3d.{sub}'] = mod
    sys.modules['open3d'] = o3d


def _disable_visualization(cfg: Any) -> None:
    if 'default_hooks' not in cfg or cfg.default_hooks is None:
        cfg.default_hooks = {}
    cfg.default_hooks.pop('visualization', None)
    if 'custom_hooks' in cfg and isinstance(cfg.custom_hooks, list):
        cfg.custom_hooks = [
            h for h in cfg.custom_hooks
            if not (isinstance(h, dict) and 'Visualization' in str(h.get('type', '')))
        ]
    cfg['visualizer'] = None
    cfg.visualizer   = None
    if 'vis_backends' in cfg:
        cfg['vis_backends'] = []


def _latest_checkpoint(work_dir: Path) -> Optional[Path]:
    latest = work_dir / 'latest.pth'
    if latest.exists():
        return latest
    cands = sorted(work_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _best_checkpoint(work_dir: Path, metric: Optional[str]) -> Optional[Path]:
    cands = sorted(
        work_dir.glob('best*.pth'),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not cands:
        return None
    if metric:
        tok = metric.replace('/', '_')
        preferred = [p for p in cands if tok in p.name]
        if preferred:
            return preferred[0]
    return cands[0]


def _select_checkpoint(
    work_dir: Path,
    explicit: Optional[Union[str, Path]],
    prefer_best: bool,
    best_metric: Optional[str],
) -> Optional[Union[str, Path]]:
    if explicit is not None:
        return explicit
    if prefer_best:
        b = _best_checkpoint(work_dir, best_metric)
        if b:
            return b
    return _latest_checkpoint(work_dir)


def _extract_sample_pcd(test_ds_cfg: Dict[str, Any]) -> Optional[Path]:
    ann_file  = test_ds_cfg.get('ann_file')
    data_root = test_ds_cfg.get('data_root')
    if not ann_file or not data_root:
        return None
    ann_path = Path(str(ann_file))
    if not ann_path.is_absolute():
        ann_path = Path(str(data_root)) / ann_path
    if not ann_path.exists():
        return None
    with ann_path.open('rb') as fh:
        payload = pickle.load(fh)
    data_list = payload.get('data_list', [])
    if not data_list:
        return None
    first = data_list[0]
    pts_rel = first.get('lidar_points', {}).get('lidar_path')
    if not pts_rel:
        return None
    pts_prefix = test_ds_cfg.get('data_prefix', {}).get('pts', '')
    pts_path = Path(str(data_root)) / pts_prefix / pts_rel
    return pts_path.resolve()


def _count_points(bin_path: Path) -> Optional[int]:
    try:
        return int(bin_path.stat().st_size // (4 * 4))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inference benchmarking (identical to run_benchmark.py)
# ---------------------------------------------------------------------------

def benchmark_fixed_input(
    cfg_path: Path,
    checkpoint: Optional[Union[str, Path]],
    device: str,
    sample_pcd: Optional[Path],
    iters: int,
    warmup: int,
) -> Dict[str, Any]:
    if checkpoint is None or sample_pcd is None or not sample_pcd.exists():
        return {'fixed_input_inference_ms': None}
    try:
        import torch
        from mmdet3d.apis import inference_detector, init_model
    except Exception:
        return {'fixed_input_inference_ms': None}

    iters  = max(int(iters), 0)
    warmup = max(int(warmup), 0)
    if iters == 0:
        return {'fixed_input_inference_ms': None}

    model      = init_model(str(cfg_path), str(checkpoint), device=device)
    sample_str = str(sample_pcd)
    for _ in range(warmup):
        inference_detector(model, sample_str)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        inference_detector(model, sample_str)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000.0 / iters
    first_param = next(model.parameters())
    return {
        'fixed_input_inference_ms': round(elapsed_ms, 3),
        'fixed_input_sample':       sample_str,
        'fixed_input_num_points':   _count_points(sample_pcd),
        'fixed_input_dtype':        str(first_param.dtype),
        'fixed_input_device':       str(first_param.device),
        'fixed_input_iters':        iters,
        'fixed_input_warmup':       warmup,
    }


# ---------------------------------------------------------------------------
# Core training + evaluation
# ---------------------------------------------------------------------------

def _configure_checkpoint_hook(cfg: Any, bench_cfg: Dict[str, Any]) -> None:
    if 'default_hooks' not in cfg or cfg.default_hooks is None:
        cfg.default_hooks = {}
    hook = cfg.default_hooks.get('checkpoint', {}) or {}
    save_best = bench_cfg.get('save_best_metric')
    if save_best:
        hook['save_best'] = save_best
        hook['rule']      = bench_cfg.get('save_best_rule', 'greater')
    if bench_cfg.get('max_keep_ckpts') is not None:
        hook['max_keep_ckpts'] = int(bench_cfg['max_keep_ckpts'])
    cfg.default_hooks['checkpoint'] = hook


def train_and_eval(
    model_item:  Dict[str, Any],
    bench_cfg:   Dict[str, Any],
    repo_root:   Path,
    cli_checkpoint: Optional[str] = None,
    cli_no_train:   bool = False,
) -> Dict[str, Any]:
    _install_open3d_stub()

    from mmengine.config import Config
    from mmengine.runner  import Runner

    cfg_path = _resolve(model_item['config'], repo_root)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    model_name = model_item.get('name', cfg_path.stem)
    cfg = Config.fromfile(str(cfg_path))
    cfg.launcher = 'none'
    _disable_visualization(cfg)

    seed = model_item.get('seed', bench_cfg.get('seed'))
    if seed is not None:
        cfg.randomness = dict(seed=int(seed))

    work_dir = _resolve(
        model_item.get(
            'work_dir',
            str(Path(bench_cfg.get('project', '3d-detection/reports/benchmarks/mmdetection3d/runs'))
                / bench_cfg.get('name', 'kitti-native')
                / model_name),
        ),
        repo_root,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = str(work_dir)

    device_cfg  = model_item.get('device', bench_cfg.get('device'))
    bench_device = _device_str(device_cfg)
    cfg.device   = bench_device

    _configure_checkpoint_hook(cfg, bench_cfg)
    _disable_visualization(cfg)

    train_enabled = not cli_no_train and bool(model_item.get('train', True))

    # Resolve explicit checkpoint (CLI overrides YAML).
    explicit_ckpt = cli_checkpoint or model_item.get('checkpoint')
    if explicit_ckpt:
        if not _is_remote(explicit_ckpt):
            explicit_ckpt = str(_resolve(explicit_ckpt, repo_root))
        cfg.load_from = explicit_ckpt

    runner = Runner.from_cfg(cfg)

    train_time_sec = None
    if train_enabled:
        t0 = time.time()
        runner.train()
        train_time_sec = round(time.time() - t0, 2)
    elif explicit_ckpt is None and cfg.get('load_from') is None:
        raise ValueError(
            f"Model '{model_name}': train=false/--no-train but no checkpoint configured."
        )

    # Select evaluation checkpoint.
    prefer_best  = bool(bench_cfg.get('prefer_best_checkpoint', True))
    best_metric  = bench_cfg.get('save_best_metric')
    ckpt_for_eval = explicit_ckpt if not train_enabled else None

    checkpoint = _select_checkpoint(
        work_dir    = work_dir,
        explicit    = ckpt_for_eval,
        prefer_best = prefer_best,
        best_metric = best_metric,
    )
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {work_dir}.")

    cfg.load_from = str(checkpoint)
    cfg.resume    = False
    _disable_visualization(cfg)
    eval_runner = Runner.from_cfg(cfg)
    raw_metrics = _to_jsonable(eval_runner.test() or {})

    # Extract test dataset config for sample pcd + metadata.
    test_ds_cfg = cfg.test_dataloader['dataset']
    # Unwrap ConcatDataset / RepeatDataset wrappers.
    while isinstance(test_ds_cfg, dict) and 'dataset' in test_ds_cfg:
        test_ds_cfg = test_ds_cfg['dataset']

    classes = (
        test_ds_cfg.get('metainfo', {}).get('classes')
        or cfg.get('metainfo', {}).get('classes')
        or ['Pedestrian']
    )

    # ----- Native KittiMetric results -----
    native = _extract_native_kitti_metrics(raw_metrics)

    # Print a concise summary.
    print("\n  --- Native KITTI R40 Results ---")
    for diff in ('easy', 'moderate', 'hard'):
        ap3d  = native.get(f'kitti_native_3d_ap40_{diff}')
        ap_bev = native.get(f'kitti_native_bev_ap40_{diff}')
        ap3d_s  = f"{ap3d * 100:.2f}%" if ap3d  is not None else "N/A"
        ap_bevs = f"{ap_bev * 100:.2f}%" if ap_bev is not None else "N/A"
        print(f"    {diff:10s}: 3D AP={ap3d_s:10s}  BEV AP={ap_bevs}")
    print()

    # ----- Fixed-input inference benchmark -----
    sample_pcd = _extract_sample_pcd(test_ds_cfg)
    bench: Dict[str, Any] = {'fixed_input_inference_ms': None}
    if bench_cfg.get('benchmark_fixed_input', False):
        bench = benchmark_fixed_input(
            cfg_path   = cfg_path,
            checkpoint = checkpoint,
            device     = bench_device,
            sample_pcd = sample_pcd,
            iters      = bench_cfg.get('benchmark_iters', 100),
            warmup     = bench_cfg.get('benchmark_warmup', 20),
        )

    inf_ms = bench.get('fixed_input_inference_ms')

    # ----- Assemble summary row -----
    ann_file_str = test_ds_cfg.get('ann_file', '')

    def _infer_split(ann: str) -> Optional[str]:
        ann = ann.lower()
        for s in ('test', 'val', 'train'):
            if s in ann:
                return s
        return None

    def _full_ann_path(ds_cfg: Dict) -> Optional[str]:
        ann  = ds_cfg.get('ann_file')
        root = ds_cfg.get('data_root')
        if not ann:
            return None
        p = Path(str(ann))
        if p.is_absolute():
            return str(p)
        return str((Path(str(root)) / p)) if root else str(p)

    # Map native R40 values → legacy kitti_* fields for backward compatibility
    # with summary_mmdetection3d_kitti_pointpillars.csv column names.
    r40_3d_moderate = native.get('kitti_native_3d_ap40_moderate')
    r40_bev_moderate = native.get('kitti_native_bev_ap40_moderate')

    summary: Dict[str, Any] = {
        'model': model_name,
        'train_time_sec': train_time_sec,
        'inf_time_per_frame_ms': inf_ms,
        'preprocess_time_per_frame_ms': None,
        'postprocess_time_per_frame_ms': None,
        'total_time_per_frame_ms': inf_ms,
        'checkpoint_used': str(checkpoint),
        'eval_split': _infer_split(ann_file_str),
        'num_classes': len(classes),
        # Native R40
        **native,
        # Legacy aliases (for columns expected by existing analysis scripts)
        'kitti_3d_ap_easy':     native.get('kitti_native_3d_ap40_easy'),
        'kitti_3d_ap_moderate': native.get('kitti_native_3d_ap40_moderate'),
        'kitti_3d_ap_hard':     native.get('kitti_native_3d_ap40_hard'),
        'kitti_bev_ap_easy':    native.get('kitti_native_bev_ap40_easy'),
        'kitti_bev_ap_moderate':native.get('kitti_native_bev_ap40_moderate'),
        'kitti_bev_ap_hard':    native.get('kitti_native_bev_ap40_hard'),
        'kitti_3d_recall_easy':     None,
        'kitti_3d_recall_moderate': None,
        'kitti_3d_recall_hard':     None,
        # Unused fields (kept for CSV schema compatibility)
        'bev_ap_mean': None, '3d_ap_mean': None,
        'bev_precision_mean': None, '3d_precision_mean': None,
        'bev_recall_mean': None,    '3d_recall_mean': None,
        'bev_f1_mean': None,        '3d_f1_mean': None,
        'bev_ap_0.25': None, 'bev_ap_0.50': None, 'bev_ap_0.75': None,
        '3d_ap_0.25': None,  '3d_ap_0.50': None,  '3d_ap_0.75': None,
        'precision': r40_3d_moderate,
        'recall':    None,
        'f1':        None,
        'per_class_precision': {c: r40_3d_moderate for c in classes},
        'per_class_recall':    None,
        'per_class_f1':        None,
        'epochs':     cfg.get('train_cfg', {}).get('max_epochs'),
        'batch':      cfg.get('train_dataloader', {}).get('batch_size'),
        'eval_batch': cfg.get('test_dataloader',  {}).get('batch_size'),
        'dataset':    _full_ann_path(test_ds_cfg),
        'config_device': bench_cfg.get('device'),
        'train_results_dir': str(work_dir),
        'eval_results_dir':  str(work_dir),
        'all_raw_metrics': json.dumps(raw_metrics),
    }
    summary.update(bench)

    ordered = {field: summary.get(field) for field in SUMMARY_FIELDS}
    return _to_jsonable(ordered)


# ---------------------------------------------------------------------------
# CSV + JSON output
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: _to_csvable(row.get(f)) for f in SUMMARY_FIELDS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    default_cfg = (
        ROOT_3D / 'benchmarks' / 'mmdetection3d' / 'configs'
        / 'benchmark_mmdetection3d_kitti_native_pointpillars.yaml'
    )
    default_out = (
        ROOT_3D / 'reports' / 'benchmarks' / 'summary' / 'mmdetection3d'
        / 'summary_mmdetection3d_kitti_native_pointpillars.csv'
    )

    parser = argparse.ArgumentParser(
        description='Train/evaluate PointPillars on KITTI (native pipeline) '
                    'and write results to CSV + JSON.',
    )
    parser.add_argument('--config',     default=str(default_cfg), help='Benchmark YAML config path.')
    parser.add_argument('--out',        default=str(default_out), help='CSV output path.')
    parser.add_argument('--checkpoint', default=None,             help='Override checkpoint path.')
    parser.add_argument('--no-train',   action='store_true',      help='Skip training; evaluate checkpoint only.')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    with cfg_path.open('r', encoding='utf-8') as fh:
        bench_cfg = yaml.safe_load(fh)

    _set_cuda_device(bench_cfg.get('device'))

    try:
        import torch as _t
        _cuda = _t.cuda.is_available()
    except Exception:
        _cuda = False

    _wants_cuda = not (
        isinstance(bench_cfg.get('device'), str)
        and bench_cfg['device'].strip().lower() == 'cpu'
    )
    if _wants_cuda and not _cuda:
        raise RuntimeError(
            "CUDA is not available but the benchmark device is configured for GPU. "
            "Set device: cpu in the YAML or check NVIDIA driver / PyTorch installation."
        )

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(BENCH_ROOT))

    model_items = bench_cfg.get('models', [])
    if not model_items:
        raise ValueError("No models configured in the YAML file.")

    rows = []
    for item in model_items:
        if isinstance(item, str):
            item = {'config': item}
        rows.append(
            train_and_eval(
                model_item      = item,
                bench_cfg       = bench_cfg,
                repo_root       = REPO_ROOT,
                cli_checkpoint  = args.checkpoint,
                cli_no_train    = args.no_train,
            )
        )

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = (REPO_ROOT / out_csv).resolve()
    write_csv(out_csv, rows)
    out_json = out_csv.with_suffix('.json')
    out_json.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f"Wrote {len(rows)} row(s) to {out_csv}")
    print(f"Wrote JSON to {out_json}")


if __name__ == '__main__':
    main()
