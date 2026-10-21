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
- `configs/benchmark_mmdetection3d_aghri_heldout_test.yaml`: frozen four-model test manifest
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

## AGHRI train/validation/test protocol

The released sequence lists contain 52 training, seven validation, and six
held-out test sequences. Training uses only the training split. The four
AGHRI training manifests set `evaluation_split: val`, so checkpoint selection
and the summary emitted by a training run use validation annotations. The
checkpoint metric is `aghri/bev_ap_mean`, averaged across BEV IoU thresholds
0.25, 0.50, and 0.75. The dataset config also points its validation
loader/evaluator to `agri_person_infos_val.pkl`.

After selection, `configs/benchmark_mmdetection3d_aghri_heldout_test.yaml`
loads the four frozen checkpoints with `train: false` and
`evaluation_split: test`. `run_benchmark.py` sets both the test loader and
metric annotation file to `agri_person_infos_test.pkl` and checks the split
recorded in each summary. Set `AGHRI_3D_DATA_ROOT` to a processed dataset
root containing the requested split before running. The saved cluster test
export was intentionally separate from the train/validation export. The exact
cluster evaluation launcher is retained in `scripts/aghri_3d_test_eval.slurm`.

Saved validation-selection summaries are in
[`reports/benchmarks/summary/mmdetection3d/aghri_validation_selection/`](../../reports/benchmarks/summary/mmdetection3d/aghri_validation_selection/).
Copies of the completed cluster training configs are in `configs/completed_training/`.
The completed held-out results and checkpoint hashes are in
[`reports/benchmarks/summary/mmdetection3d/aghri_heldout_test/`](../../reports/benchmarks/summary/mmdetection3d/aghri_heldout_test/).

## Outputs

- benchmark summaries: `3d-detection/reports/benchmarks/summary/mmdetection3d/`
- per-model work dirs: `3d-detection/reports/benchmarks/mmdetection3d/runs/`

## Notes

- The benchmark assumes a single processed detection class: `person`.
- The completed runs use the released sequence-level split lists so that no released sequence crosses train, validation, and test. The older optional raw conversion script expects frame-level split files; the completed cluster exports used a sequence-aware converter.
- The converter maps the PCD `rgb` field to a normalized RGB-derived scalar feature for MMDetection3D; this is not native LiDAR intensity.
