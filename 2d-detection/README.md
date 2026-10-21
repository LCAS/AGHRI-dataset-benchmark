# 2D Detection Workspace

This workspace contains the repository's implemented 2D person-detection benchmarks.

Its job is to make detector comparisons repeatable across agricultural and reference datasets, especially when comparing in-domain performance against transfer performance across sensing setups such as standard RGB and fisheye imagery.

## What This Workspace Is For

Use `2d-detection` to:

- train and evaluate detector baselines
- compare architectures and training strategies
- measure cross-domain transfer between datasets
- export detections for the tracking workspace
- analyse benchmark results through committed summaries and notebooks

The current configs already cover image domains such as `zedrgb`, `fisheye`, `fieldsafepedestrian`, `kitti_filtered`, and `coco2017_filtered`.

## Structure

- `benchmarks/ultralytics`: YOLO-based benchmark runner and detection export tooling
- `benchmarks/mmdetection`: MMDetection-based benchmark runner using local config files plus an external `mmdet` install
- `reports/benchmarks`: committed benchmark summaries and selected run artifacts
- `notebooks`: analysis notebooks for comparing experiment outputs

## AGHRI split protocol

For AGHRI, fine-tuned and scratch models use the validation split to select checkpoints and the held-out test split for final evaluation. Direct pretrained models are evaluated on the test split without training. MMDetection uses `instances_val.json` for validation and `instances_test.json` for testing. Ultralytics uses the dataset YAML `val` entry during training and `eval_split: test` for final evaluation. Benchmark results are in `reports/benchmarks/summary/`, and the paper figures are in `notebooks/figures/`.

## Detection Workflow

1. Choose a benchmark framework.
2. Edit one benchmark YAML to point at the right dataset or checkpoint paths.
3. Run training, evaluation, or transfer evaluation.
4. Collect CSV and JSON summaries under `reports/benchmarks/summary/`.
5. If tracking is needed, export detections in the project JSON format and hand them to `2d-tracking`.

## Frameworks

### Ultralytics

Best fit for YOLO benchmarking and easy detector export.

- local docs: [`benchmarks/ultralytics/README.md`](benchmarks/ultralytics/README.md)
- key entrypoints:
  - `benchmarks/ultralytics/src/run_benchmark.py`
  - `benchmarks/ultralytics/src/dump_detections.py`

### MMDetection

Best fit for config-driven detector benchmarking and richer experiment control.

- local docs: [`benchmarks/mmdetection/README.md`](benchmarks/mmdetection/README.md)
- key entrypoint:
  - `benchmarks/mmdetection/src/run_benchmark.py`

## Reports And Analysis

- benchmark summaries live under `reports/benchmarks/summary/`
- native framework run artifacts live under framework-specific subdirectories in `reports/benchmarks/`
- the comparison notebook lives under `notebooks/` and reads the committed summary CSV files
- the four paper figures live under `notebooks/figures/`

## Cluster Entrypoints

- `2d-detection/benchmarks/ultralytics/scripts/run_ultralytics_benchmark.sbatch`
- `2d-detection/benchmarks/ultralytics/scripts/dump_detections.sbatch`
- `2d-detection/benchmarks/mmdetection/scripts/run_mmdetection_benchmark.sbatch`

## Path Conventions

Be careful with path resolution:

- Ultralytics report paths are resolved from `2d-detection`
- MMDetection benchmark configs usually refer to files from the repository root
- most sample configs use site-specific dataset paths under `/workspace/data/...`

Adapt those paths to your local machine, server, or cluster mount.

## Relationship To The Wider Benchmark

This workspace is the implemented 2D detection layer of the larger agricultural safety benchmark vision. It is where detector-level comparisons are established before extending the project toward future 3D and multimodal perception tasks.
