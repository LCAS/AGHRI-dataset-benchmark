# MMDetection Benchmark

This module benchmarks MMDetection models, mainly Faster R-CNN variants, for agricultural person detection experiments.

## What It Covers

- benchmark runs defined by lightweight YAML files
- training and evaluation through local MMDetection configs
- checkpoint-based transfer evaluation
- class-wise AP extraction and fixed-input latency reporting

## Structure

- `src/run_benchmark.py`: benchmark runner
- `configs/benchmark_*.yaml`: benchmark definitions selecting models and run settings
- `configs/models/`: local MMDetection model configs used by the benchmark
- `configs/datasets/`: dataset definitions for local agricultural and reference datasets
- `requirements.txt`: pinned Python package requirements for this benchmark
- `scripts/run_mmdetection_benchmark.sbatch`: cluster launcher and reference environment setup

AGHRI training selects the best checkpoint on `instances_val.json`; final evaluation uses `instances_test.json`. The two splits must remain separate in the dataset export.

## Environment

This module uses MMDetection as an external Python dependency. The current benchmark configuration is pinned to the standard `mmdet==3.3.0` package and compatible MMEngine/MMCV releases; there is no project-specific MMDetection fork required for the code currently in this repository.

Install MMCV for your platform first, then install the benchmark requirements:

```bash
pip install 'mmcv==2.1.0' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r requirements.txt
```

The canonical environment recipe in this repository is the cluster script:

- `scripts/run_mmdetection_benchmark.sbatch`

That script shows the expected package versions and installation order. If you are setting up a local environment, use it as the reference rather than inventing a different stack.

If future benchmark work requires framework-level MMDetection changes, keep those changes in an external fork of `open-mmlab/mmdetection` and replace the `mmdet==...` entry in `requirements.txt` with a git reference to that fork. Do not copy the full framework source back into this repository.

## Benchmark Config Schema

Typical top-level fields:

- `device`
- `seed`
- `benchmark_fixed_input`
- `benchmark_warmup`
- `benchmark_iters`
- `prefer_best_checkpoint`
- `save_best_metric`
- `save_best_rule`
- `max_keep_ckpts`
- `project`
- `name`
- `models`

Each entry in `models` can define:

- `name`
- `config`
- `checkpoint`
- `train`
- `cfg_options`
- `device`
- `seed`

## How It Works

`run_benchmark.py` does the following for each configured model:

1. load the MMDetection config
2. merge benchmark-level and model-level overrides
3. optionally train the model
4. select the best or latest checkpoint
5. run evaluation through MMDetection
6. export one summary row to CSV and JSON

The runner also enables class-wise COCO metrics when possible and can benchmark fixed-input inference latency on a synthetic image.

## Run A Benchmark

```bash
python src/run_benchmark.py \
  --config configs/benchmark_mmdetection_fisheye.yaml \
  --out ../../reports/benchmarks/summary/mmdetection/summary_mmdetection_fisheye.csv
```

## Path Notes

- benchmark YAML paths are typically resolved from the repository root
- model config paths in benchmark YAML should point to local files under `2d-detection/benchmarks/mmdetection/configs/`
- local model configs inherit upstream defaults via `mmdet::...` package config imports
- checkpoints may be local files or remote HTTP or HTTPS references
- report outputs are usually written under `2d-detection/reports/benchmarks/`

## Outputs

- benchmark summaries: `2d-detection/reports/benchmarks/summary/mmdetection/`
- training and evaluation work dirs: `2d-detection/reports/benchmarks/mmdetection/runs/`

## When To Use This Module

Use this module when you need more configurable detector experiments than the Ultralytics runner provides, especially for controlled Faster R-CNN style comparisons and benchmark reproducibility through explicit MMDetection configs.
