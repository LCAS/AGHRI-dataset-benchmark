# Agri-Human Dataset Benchmark

This repository is a benchmark workspace for agricultural human detection and tracking, built to support safer human-robot interaction in field robotics.

The core idea is simple: if agricultural robots are expected to work around people, they need perception systems that can detect, track, and eventually fuse observations of humans reliably across difficult real-world conditions such as occlusion, clutter, motion, wide-angle optics, and changing light. This repository exists to make those comparisons repeatable.

Today, the implemented code covers 2D detection, 2D tracking, and LiDAR-based 3D detection. The broader benchmark direction is multimodal: RGB, fisheye, LiDAR point clouds, rosbag-derived streams, and future 3D or fusion pipelines under the same benchmark philosophy.

## Why This Repo Exists

Agricultural robotics operates in environments where standard benchmark assumptions often break down:

- people move unpredictably around machines
- sensing conditions vary across rows, canopies, footpaths, and work zones
- camera geometry may be conventional RGB, stereo, or fisheye
- safety depends on robust perception, not just high scores on generic urban datasets

This repository is intended to help answer questions such as:

- How well do person detectors trained on one agricultural domain transfer to another?
- How much performance changes between RGB and fisheye imagery?
- Which trackers are most stable when detections are noisy or sparse?
- How should future 3D and multimodal benchmarks be organised so they remain comparable?

## What Is Implemented Now

### `2d-detection`

Training, evaluation, transfer evaluation, and result summarisation for 2D person detection benchmarks.

- `benchmarks/ultralytics`: YOLO-based benchmark runner and detection export pipeline
- `benchmarks/mmdetection`: MMDetection-based benchmark runner for Faster R-CNN style experiments
- `reports/benchmarks`: committed benchmark summaries and selected run artifacts
- `notebooks`: analysis notebooks

### `2d-tracking`

Tracker benchmarking and MOT-style evaluation for detections exported from the 2D detection workspace.

- `benchmarks/boxmot`: BoxMOT wrappers for ByteTrack, BoTSORT, StrongSORT, OCSORT, DeepOCSORT, and BoostTrack
- `benchmarks/deepsort`: local DeepSORT implementation with frozen-graph ReID support
- `benchmarks/norfair`: Norfair-based tracking benchmark
- `common/mot`: shared ground-truth conversion and MOT evaluation utilities
- `reports`: per-run tracking outputs and summary metrics

### `3d-detection`

LiDAR-based 3D human detection benchmarks for the agricultural sensing dataset.

- `benchmarks/mmdetection3d`: MMDetection3D-based data preparation, training, and evaluation
- `reports/benchmarks`: 3D benchmark summaries and run artifacts

## Benchmark Scope

The repository is designed around benchmark comparison, not around a single model implementation.

Current benchmark patterns in the codebase include:

- in-domain train and evaluate
- zero-shot or pretrained evaluation without fine-tuning
- cross-domain transfer evaluation
- export of detector outputs into a common JSON format for downstream tracking
- MOT-format evaluation of tracker outputs
- LiDAR point-cloud conversion and 3D detector benchmarking

The current configs already reflect several image domains, including `zedrgb`, `fisheye`, `fieldsafepedestrian`, `kitti_filtered`, and `coco2017_filtered`.

## High-Level Workflow

1. Train or evaluate 2D detectors in [`2d-detection`](2d-detection/README.md).
2. Export frame-wise detections to the repository JSON format.
3. Run tracker benchmarks in [`2d-tracking`](2d-tracking/README.md).
4. Convert project annotations to MOT ground truth and evaluate tracker outputs.
5. Train or evaluate LiDAR 3D detectors in [`3d-detection`](3d-detection/README.md).
6. Compare reports across datasets, models, and sensing conditions.

## Repository Layout

```text
.
|-- 2d-detection/
|   |-- benchmarks/
|   |-- notebooks/
|   `-- reports/
|-- 2d-tracking/
|   |-- benchmarks/
|   |-- common/
|   `-- reports/
|-- 3d-detection/
|   |-- benchmarks/
|   `-- reports/
`-- LICENSE
```

## Data And Paths

The repository does not package the full datasets. Most configs point at local or cluster-mounted dataset paths such as `/workspace/data/...`, which you should adapt to your own environment.

Path conventions differ slightly by subsystem:

- Ultralytics benchmark report paths are resolved from `2d-detection`
- MMDetection benchmark YAML paths are typically resolved from the repository root
- tracking configs resolve relative paths from `2d-tracking`

Each workspace README explains its own path rules in more detail.

## MMDetection Dependency

The MMDetection benchmark code in this repository is designed to use the external `mmdet` Python package rather than a vendored framework copy. Local benchmark logic and dataset-specific configs stay under `2d-detection/benchmarks/mmdetection/`.

If framework-level MMDetection changes are ever required, they should live in an external fork of `open-mmlab/mmdetection` and be referenced as a dependency from environment setup documentation or `requirements.txt`, not copied into this repository.

## Intended Direction

This repository should be read as both:

- a working benchmark implementation for current 2D perception experiments
- a foundation for publishing and extending an agricultural human-safety dataset benchmark across more sensing modalities

That future expansion can include:

- 3D tracking
- multimodal fusion
- rosbag-to-benchmark conversion utilities
- cross-modal safety analysis for human-robot interaction

## License

The repository's benchmark code and documentation are licensed under Apache License 2.0. See [`LICENSE`](LICENSE).

Additional attribution information is recorded in [`NOTICE`](NOTICE).

If third-party code is added to the repository in the future, its original license and notices should be preserved.
