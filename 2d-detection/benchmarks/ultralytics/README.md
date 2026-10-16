# Ultralytics Benchmark

This module benchmarks Ultralytics YOLO models for person detection and also exports detections for the downstream tracking pipeline.

## What It Covers

- benchmark runs for in-domain training and evaluation
- zero-shot or pretrained evaluation without fine-tuning
- cross-domain transfer evaluation
- export of frame-wise detections to the repository JSON format used by `2d-tracking`

## Structure

- `src/run_benchmark.py`: benchmark runner with an explicit YAML schema
- `src/dump_detections.py`: export YOLO predictions to tracking-ready JSON
- `configs/benchmark_*.yaml`: benchmark definitions for train, eval, and transfer settings
- `configs/predict_*.yaml`: detection-export configs for tracking inputs
- `scripts/run_ultralytics_benchmark.sh`: local launcher
- `scripts/run_ultralytics_benchmark.sbatch`: cluster launcher
- `scripts/dump_detections.sbatch`: cluster launcher for detection export
- `requirements.txt`: minimal Python dependencies for this module

## Environment

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For AGHRI `train_eval` runs, the dataset YAML `val` entry is used for checkpoint selection during `model.train()`. The benchmark configs set `eval_split: test` for the final `model.val()` call. Keep the YAML `val` and `test` entries distinct.

## Benchmark Modes

`run_benchmark.py` accepts three benchmark modes through YAML:

- `train_eval`: train on `train_dataset`, then evaluate on `eval_dataset`
- `transfer_eval`: evaluate existing checkpoints on a different target dataset
- `eval_only`: evaluate pretrained or previously trained checkpoints without training

## Benchmark Config Schema

Typical fields:

- `mode`
- `train_dataset`
- `eval_dataset`
- `imgsz`
- `epochs`
- `batch`
- `device`
- `eval_split`
- `project`
- `name`
- `models`

Each item in `models` must define:

- `name`
- `source`

`source` can point to:

- an Ultralytics model identifier such as `yolov8s.pt`
- a local checkpoint file
- a previous run directory containing `weights/best.pt`
- a remote HTTP or HTTPS checkpoint reference

## Run A Benchmark

```bash
python src/run_benchmark.py \
  --config configs/benchmark_ultralytics_zedrgb.yaml \
  --out ../../reports/benchmarks/summary/ultralytics/summary_ultralytics_zedrgb.csv
```

The runner writes:

- a summary CSV
- a JSON version of the same rows
- native Ultralytics run outputs under the configured `project` directory

## Export Detections For Tracking

```bash
python src/dump_detections.py \
  --config configs/predict_zedrgb.yaml
```

This generates one JSON file per model, with records shaped like:

```json
[
  {
    "File": "frame_000001.png",
    "Labels": [
      {
        "Class": "person",
        "BoundingBoxes": [100.0, 50.0, 80.0, 180.0]
      }
    ]
  }
]
```

That format is consumed directly by the tracking benchmarks.

## Path Notes

- benchmark `project` paths are resolved relative to `2d-detection` unless absolute
- model `source` values are resolved from the benchmark directory or `2d-detection`
- prediction configs usually point directly to dataset and checkpoint locations

## Outputs

- benchmark summaries: `2d-detection/reports/benchmarks/summary/ultralytics/`
- native Ultralytics runs: `2d-detection/reports/benchmarks/ultralytics/runs/`
- tracking-ready detections: typically `2d-detection/reports/detections/`

## When To Use This Module

Use this module when you want fast detector iteration, direct YOLO comparisons, or a clean path from detector checkpoints to tracking-ready detection exports.
