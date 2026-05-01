# MMDetection3D Benchmark

This module extends the repository benchmark pattern to LiDAR-based 3D person detection with MMDetection3D.

## What It Covers

- direct training and evaluation on a processed MMDetection3D-ready dataset
- training and evaluation through local MMDetection3D configs
- benchmark summaries exported to CSV and JSON
- cluster-ready SLURM entrypoints

## Structure

- `src/prepare_aghri_lidar_dataset.py`: optional raw dataset to MMDetection3D conversion
- `src/run_benchmark.py`: train/evaluate configured models and export summaries
- `configs/datasets/`: dataset and dataloader config
- `configs/models/`: PointPillars and SECOND configs
- `configs/benchmark_mmdetection3d_aghri.yaml`: benchmark manifest
- `aghri3d/`: custom dataset and metric
- `scripts/*.sbatch`: cluster launchers

## Supported Models

- `pointpillars_aghri.py`
- `second_aghri.py`

## Dataset Expectations

The normal benchmark path assumes you already have a processed MMDetection3D-style dataset root such as:

- `D:\AOC\datasets\agri-human-sensing\mmdet3d_person`
- `/workspace/datasets/agri-human-sensing/mmdet3d_person`

The expected processed layout is:

- `points/*.bin`
- `ImageSets/train.txt`, `val.txt`, `test.txt`
- `infos/agri_person_infos_train.pkl`
- `infos/agri_person_infos_val.pkl`
- `infos/agri_person_infos_test.pkl`

The raw conversion script remains available, but it is no longer the default workflow.

## Environment

This benchmark expects MMDetection3D as an external dependency. The cluster scripts install:

- PyTorch 2.1 / CUDA 11.8
- `mmcv==2.1.0`
- `mmdet==3.3.0`
- `mmdet3d==1.4.0`
- `spconv-cu118`

Install order and exact commands are documented in:

- `scripts/prepare_aghri_lidar_dataset.sbatch`
- `scripts/run_mmdetection3d_benchmark.sbatch`

## Typical Workflow

1. Export the processed dataset root:

```bash
export AGHRI_3D_DATA_ROOT=/workspace/datasets/agri-human-sensing/mmdet3d_person
```

2. Run the benchmark:

```bash
python src/run_benchmark.py \
  --config configs/benchmark_mmdetection3d_aghri.yaml \
  --out ../../reports/benchmarks/summary/mmdetection3d/summary_mmdetection3d_aghri.csv
```

3. Optional: if you need to regenerate the processed dataset from raw labelled scenes, use:

```bash
python src/prepare_aghri_lidar_dataset.py \
  --raw-root /workspace/datasets/agri-human-sensing/labelled_dataset \
  --out-root /workspace/datasets/agri-human-sensing/mmdet3d_person
```

## Outputs

- benchmark summaries: `3d-detection/reports/benchmarks/summary/mmdetection3d/`
- per-model work dirs: `3d-detection/reports/benchmarks/mmdetection3d/runs/`

## Notes

- The benchmark assumes a single processed detection class: `person`.
- Scene-level splits are generated deterministically to avoid frame leakage across train, val, and test.
- The converter decodes the PCD `rgb` field into a normalized intensity channel for MMDetection3D.
