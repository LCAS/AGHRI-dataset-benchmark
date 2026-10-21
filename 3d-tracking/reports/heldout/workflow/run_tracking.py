"""Tune SimpleTrack on validation, then run frozen 3D tracking evaluations."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common/mot3d'))
from evaluate_mot3d import evaluate_scene  # noqa: E402
from gt_as_detections import convert_scene_to_detections, convert_scene_to_gt_mot3d  # noqa: E402

TRACKERS = ('ab3dmot', 'centerpoint', 'simpletrack')
MODELS = ('pointpillars_scratch', 'second_scratch')
NUM_SEQUENCES = {'val': 7, 'test': 6}


def run(command: list[str]) -> None:
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def names_for(split: str) -> list[str]:
    names = [s.strip() for s in (ROOT / 'split_lists' / f'{split}.txt').read_text().splitlines() if s.strip()]
    if len(names) != NUM_SEQUENCES[split] or len(set(names)) != len(names):
        raise ValueError(f'Invalid {split} split list')
    return names


def prepare_gt(split: str, model: str, names: list[str], output: Path) -> tuple[Path, Path]:
    gt_det = output / 'gt_detections'
    gt_csv = output / 'gt_mot3d'
    for name in names:
        scene = ROOT / 'annotations' / name
        count = convert_scene_to_detections(scene, gt_det / name)
        csv_count, _ = convert_scene_to_gt_mot3d(scene, gt_csv / f'{name}.csv')
        if count != csv_count:
            raise RuntimeError(f'GT frame mismatch in {name}')
    return gt_det, gt_csv


def check_detections(split: str, model: str, names: list[str]) -> Path:
    det_root = ROOT / 'detections' / split / model
    manifest = ROOT / 'manifests' / split / f'{model}.json'
    data = json.loads(manifest.read_text())
    if data['sequences'] != names or data['model'] != model or data['split'] != split:
        raise RuntimeError(f'Export manifest mismatch: {manifest}')
    total = 0
    for name in names:
        frames = json.loads((ROOT / 'annotations' / name / 'annotations/lidar_ann.json').read_text())
        expected = {Path(f['File']).stem for f in frames if str(f.get('File', '')).endswith('.pcd')}
        actual = {p.stem for p in (det_root / name).glob('*.json')}
        if expected != actual:
            raise RuntimeError(f'Missing/extra detector frames in {split}/{model}/{name}')
        total += len(actual)
    if total != data['frames']:
        raise RuntimeError('Exported frame count differs from manifest')
    return det_root


def tracker_command(tracker: str, detections: Path, output: Path, runtime: Path, params: dict | None = None) -> list[str]:
    script = ROOT / 'benchmarks' / tracker / 'src/run_tracker.py'
    config = ROOT / 'benchmarks' / tracker / 'configs/tracking' / f'aghri_{tracker}.yaml'
    cmd = [sys.executable, str(script), '--config', str(config),
           '--detections-dir', str(detections), '--mot-output-dir', str(output),
           '--runtime-json', str(runtime)]
    if params:
        for key, value in params.items():
            cmd += ['--' + key.replace('_', '-'), str(value)]
    return cmd


def tune_simpletrack(det_root: Path, gt_root: Path, names: list[str], output: Path) -> dict:
    import motmetrics as mm
    import numpy as np
    if not hasattr(np, 'asfarray'):
        np.asfarray = lambda arr, dtype=float: np.asarray(arr, dtype=dtype)
    candidates = list(itertools.product((0.05, 0.10, 0.25), (0.25, 0.50), (1, 2, 3)))
    records = []
    best = None
    for number, (score, high, min_hits) in enumerate(candidates):
        params = {'score_threshold': score, 'high_score_thresh': high, 'min_hits': min_hits}
        trial = output / 'tuning' / f'candidate_{number:02d}'
        pred = trial / 'predictions'
        run(tracker_command('simpletrack', det_root, pred, trial / 'runtime.json', params))
        accs = [evaluate_scene(gt_root / f'{n}.csv', pred / f'{n}.csv', 0.25)[0] for n in names]
        summary = mm.metrics.create().compute_many(accs, names=names, metrics=['idf1', 'mota'])
        idf1 = float(summary['idf1'].mean())
        mota = float(summary['mota'].mean())
        row = {'candidate': number, **params, 'macro_idf1': idf1, 'macro_mota': mota}
        records.append(row)
        rank = (idf1 if math.isfinite(idf1) else -1.0,
                mota if math.isfinite(mota) else -1.0, -number)
        if best is None or rank > best[0]:
            best = (rank, row)
        print('VALIDATION TUNING', row, flush=True)
    tuning_file = output / 'simpletrack_validation_grid.csv'
    with tuning_file.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    selected = dict(best[1])
    frozen = ROOT / 'frozen' / det_root.name / 'simpletrack.json'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(json.dumps({
        'selected_on': 'AGHRI validation sequences',
        'primary_metric': 'equal-sequence mean IDF1 at BEV IoU 0.25',
        'tie_breaker': 'equal-sequence mean MOTA',
        'grid_csv': str(tuning_file),
        'parameters': {k: selected[k] for k in ('score_threshold', 'high_score_thresh', 'min_hits')},
        'selection_metrics': {'macro_idf1': selected['macro_idf1'], 'macro_mota': selected['macro_mota']},
    }, indent=2) + '\n')
    print('FROZEN SIMPLETRACK', frozen.read_text(), flush=True)
    return json.loads(frozen.read_text())['parameters']


def add_motp_iou(row: dict) -> dict:
    out = dict(row)
    value = row.get('motp')
    try:
        distance = float(value)
        out['motp_iou'] = 1.0 - distance if math.isfinite(distance) else None
    except (TypeError, ValueError):
        out['motp_iou'] = None
    return out


def evaluate_all(split: str, model: str, names: list[str], det_root: Path, gt_det: Path,
                 gt_root: Path, output: Path, simple_params: dict) -> None:
    summary_rows = []
    per_scene_rows = []
    metrics = ('mota', 'motp_iou', 'idf1', 'idp', 'idr', 'precision', 'recall', 'hota', 'deta', 'assa')
    for source, source_det in (('gt', gt_det), (model, det_root)):
        for tracker in TRACKERS:
            run_dir = output / source / tracker
            pred = run_dir / 'predictions'
            runtime = run_dir / 'runtime.json'
            params = simple_params if tracker == 'simpletrack' else None
            run(tracker_command(tracker, source_det, pred, runtime, params))
            run([sys.executable, str(ROOT / 'common/mot3d/evaluate_mot3d.py'),
                 '--gt-dir', str(gt_root), '--pred-dir', str(pred),
                 '--iou-threshold', '0.25', '--out-csv', str(run_dir / 'metrics.csv'),
                 '--out-json', str(run_dir / 'metrics.json')])
            rows = [add_motp_iou(r) for r in json.loads((run_dir / 'metrics.json').read_text())]
            scene_rows = [r for r in rows if r['index'] != 'OVERALL']
            pooled = next(r for r in rows if r['index'] == 'OVERALL')
            if {r['index'] for r in scene_rows} != set(names):
                raise RuntimeError(f'Evaluation missing sequences for {source}/{tracker}')
            for row in scene_rows:
                per_scene_rows.append({'split': split, 'detector': model, 'source': source,
                                       'tracker': tracker, 'sequence': row['index'], **row})
            runtime_data = json.loads(runtime.read_text())
            for aggregation, samples in (('macro', scene_rows), ('pooled', [pooled])):
                summary = {'split': split, 'detector': model, 'source': source,
                           'tracker': tracker, 'aggregation': aggregation,
                           'sequences': len(names), 'frames': runtime_data['total_frames'],
                           'tracker_fps': runtime_data['tracking_fps'],
                           'num_switches': sum(int(r['num_switches']) for r in scene_rows)}
                for key in metrics:
                    if aggregation == 'pooled' and key in ('hota', 'deta', 'assa'):
                        summary[key] = None  # Existing evaluator only supplies macro HOTA.
                    else:
                        values = [float(r[key]) for r in samples if r.get(key) is not None and math.isfinite(float(r[key]))]
                        summary[key] = sum(values) / len(values) if values else None
                summary_rows.append(summary)
    for filename, rows in (('per_sequence.csv', per_scene_rows), ('summary.csv', summary_rows)):
        with (output / filename).open('w', newline='') as stream:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary_rows, indent=2, allow_nan=False) + '\n')
    print('COMPLETE', split, model, 'runs:', len(per_scene_rows), 'summary:', output / 'summary.csv')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=NUM_SEQUENCES, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    names = names_for(args.split)
    det_root = check_detections(args.split, args.model, names)
    output = ROOT / 'outputs' / args.split / args.model
    output.mkdir(parents=True, exist_ok=True)
    gt_det, gt_root = prepare_gt(args.split, args.model, names, output)
    if args.split == 'val':
        params = tune_simpletrack(det_root, gt_root, names, output)
    else:
        frozen = ROOT / 'frozen' / args.model / 'simpletrack.json'
        params = json.loads(frozen.read_text())['parameters']
    evaluate_all(args.split, args.model, names, det_root, gt_det, gt_root, output, params)


if __name__ == '__main__':
    main()
