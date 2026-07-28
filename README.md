# AGHRI Dataset Benchmark

This repository is a benchmark workspace for agricultural human detection, tracking, multimodal fusion, person re-identification, and target following, built to support safer human-robot interaction in field robotics. It provides benchmark code, evaluation tools, reports, and reproducibility assets for human perception using the **AGHRI dataset**.

AGHRI is a multimodal dataset for robot perception of humans in agricultural and off-road environments, with synchronised RGB, depth, fisheye, and 3D LiDAR data collected in strawberry polytunnels, vineyards, footpaths, and farmsides. It supports 2D and 3D human detection, multi-object tracking, multimodal fusion, and person re-identification using identity-consistent annotations.

**Dataset:** [AGHRI: A dataset for multimodal robot perception of humans in agricultural and off-road environments](https://doi.org/10.24385/lincoln.32982638)

The dataset files are not distributed through this Git repository. Download AGHRI separately from the DOI above and configure the task-specific paths described in each component README.

## Repository Purpose

Agricultural robots increasingly operate near people in cluttered and dynamic environments where humans may be partially or fully occluded. Reliable human perception in these settings requires more than evaluation on conventional urban or indoor datasets.

This repository brings together independent but related AGHRI benchmark workspaces for:

- image-based 2D human detection
- image-based 2D multi-object tracking
- LiDAR-based 3D human detection
- LiDAR-based 3D multi-object tracking
- camera-guided LiDAR early fusion
- camera-LiDAR late fusion with DeepFusionMOT tracking
- person re-identification and target following

## Implemented Benchmark Areas

### `2d-detection`

Training, evaluation, transfer evaluation, and result summarisation for 2D person detection benchmarks.

- `benchmarks/ultralytics`: YOLO-based benchmark runner and detection export pipeline
- `benchmarks/mmdetection`: MMDetection-based benchmark runner for Faster R-CNN style experiments
- `reports/benchmarks`: committed benchmark summaries and selected run artifacts
- `notebooks`: analysis notebooks

See [`2d-detection/README.md`](2d-detection/README.md).

### `2d-tracking`

Tracker benchmarking and MOT-style evaluation for detections exported from the 2D detection workspace.

- `benchmarks/boxmot`: BoxMOT wrappers for ByteTrack, BoTSORT, StrongSORT, OCSORT, DeepOCSORT, and BoostTrack
- `benchmarks/deepsort`: local DeepSORT implementation with frozen-graph ReID support
- `benchmarks/norfair`: Norfair-based tracking benchmark
- `common/mot`: shared ground-truth conversion and MOT evaluation utilities
- `reports`: per-run tracking outputs and summary metrics

See [`2d-tracking/README.md`](2d-tracking/README.md).

### `3d-detection`

LiDAR-based 3D human detection benchmarks for the agricultural sensing dataset.

- `benchmarks/mmdetection3d`: MMDetection3D-based data preparation, training, and evaluation
- `reports/benchmarks`: 3D benchmark summaries and run artifacts

See [`3d-detection/README.md`](3d-detection/README.md).

### `3d-tracking`

LiDAR-based 3D multi-object tracking using frame-wise 3D person detections.

- `benchmarks/ab3dmot`: local AB3DMOT implementation
- `benchmarks/centerpoint`: local CenterPoint implementation
- `benchmarks/simpletrack`: local SimpleTrack implementation
- `common/mot3d`: shared tools for preparing 3D tracking data, evaluating the tracks, and summarising the results
- `reports`: tracker summaries and runtime results
- `notebooks`: 3D tracking analysis notebook

See [`3d-tracking/README.md`](3d-tracking/README.md).

### `early_fusion/camera_lidar_humble`

Camera-guided LiDAR early fusion for AGHRI ZED RGB and Livox data.

- `camera_lidar_fusion/lidar_fusion.py`: main ROS 2 camera-LiDAR fusion node
- `training`: ZED RGB YOLO dataset preparation and fine-tuning tools
- `tools`: offline evaluation and ROS bag conversion
- `launch` and `rviz`: ROS 2 playback and visualisation files
- `notebooks`: early fusion analysis notebook

See [`early_fusion/camera_lidar_humble/README.md`](early_fusion/camera_lidar_humble/README.md).

### `late_fusion/deepfusionmot`

Camera-LiDAR late fusion and DeepFusionMOT tracking.

- `late_fusion_pkg`: ROS 2 detector, fusion, visualisation, and tracking nodes
- `config`: detector, fusion, tracking, and visualisation configurations
- `launch`: live and cached AGHRI playback launch files
- `tools`: evaluation and reporting utilities
- `notebooks`: late fusion analysis notebook

See [`late_fusion/deepfusionmot/README.md`](late_fusion/deepfusionmot/README.md).

### `person-reid/OCLReID`

Person re-identification and target-following evaluation using ZED RGB and fisheye image sequences.

- `aghri_video_preparation`: AGHRI camera-video and frame-manifest preparation
- `scripts`: experiment execution, result evaluation, and report generation
- `methods`: method-specific AGHRI test launchers
- `notebooks`: person ReID and target-following analysis notebook

See [`person-reid/OCLReID/README.md`](person-reid/OCLReID/README.md).

## Benchmark Scope

The repository is designed around benchmark comparison, not around a single model implementation.

Current benchmark patterns in the codebase include:

- in-domain training and evaluation
- zero-shot or pretrained evaluation without fine-tuning
- cross-domain transfer evaluation
- export of detector outputs into common formats for downstream tracking
- MOT-format evaluation of 2D tracker outputs
- LiDAR point-cloud conversion and 3D detector benchmarking
- LiDAR/BEV evaluation of 3D tracker outputs
- offline evaluation and ROS 2 camera-LiDAR fusion workflows
- person re-identification and target-following evaluation

The current configs reflect several image domains, including `zedrgb`, `fisheye`, `fieldsafepedestrian`, `kitti_filtered`, and `coco2017_filtered`.

The task folders are related, but they are not one mandatory end-to-end pipeline. Each workspace has its own environment, data preparation, metrics, and reports.

## High-Level Workflow

1. Train or evaluate 2D detectors in [`2d-detection`](2d-detection/README.md).
2. Export frame-wise detections to the repository JSON format.
3. Run tracker benchmarks in [`2d-tracking`](2d-tracking/README.md).
4. Convert image annotations to MOT ground truth and evaluate 2D tracker outputs.
5. Train or evaluate LiDAR 3D detectors in [`3d-detection`](3d-detection/README.md).
6. Export frame-wise 3D detections for [`3d-tracking`](3d-tracking/README.md).
7. Convert AGHRI LiDAR annotations to MOT3D ground truth and evaluate 3D tracks using BEV IoU.
8. Run [`early_fusion/camera_lidar_humble`](early_fusion/camera_lidar_humble/README.md) for camera-guided LiDAR early fusion evaluation and ROS 2 visualisation.
9. Run [`late_fusion/deepfusionmot`](late_fusion/deepfusionmot/README.md) for camera-LiDAR late fusion, DeepFusionMOT tracking, and ROS 2 visualisation.
10. Run [`person-reid/OCLReID`](person-reid/OCLReID/README.md) for person re-identification and target-following evaluation.
11. Compare task-specific reports across datasets, models, and sensing conditions.

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
|   |-- notebooks/
|   `-- reports/
|-- 3d-detection/
|   |-- benchmarks/
|   |-- notebooks/
|   `-- reports/
|-- 3d-tracking/
|   |-- benchmarks/
|   |-- common/
|   |-- notebooks/
|   |-- reports/
|   `-- tools/
|-- early_fusion/
|   `-- camera_lidar_humble/
|-- late_fusion/
|   `-- deepfusionmot/
|-- person-reid/
|   `-- OCLReID/
|-- LICENSE
|-- NOTICE
`-- README.md
```

## Data And Paths

The repository does not package the full AGHRI dataset. Download it separately from:

```text
https://doi.org/10.24385/lincoln.32982638
```

Most configs point at local or cluster-mounted dataset paths such as `/workspace/data/...`, which should be adapted to your own environment.

Path conventions differ slightly by subsystem:

- Ultralytics benchmark report paths are resolved from `2d-detection`
- MMDetection benchmark YAML paths are typically resolved from the repository root
- 2D tracking configs resolve relative paths from `2d-tracking`
- 3D detection and 3D tracking use task-specific LiDAR dataset and output paths
- early-fusion and late-fusion workflows use their own ROS 2 workspace, dataset, calibration, checkpoint, and bag paths
- person ReID uses its own environment, dataset, checkpoint, and output paths

Each workspace README explains its own path rules in more detail.

## Environment Requirements

There is no single installation command for the full repository because the benchmark areas use different software stacks.

Examples include:

- Ultralytics or MMDetection environments for 2D detection
- tracker-specific BoxMOT, DeepSORT, or Norfair environments for 2D tracking
- PyTorch, CUDA, MMDetection3D, MMCV, and spconv for 3D detection
- a lightweight NumPy, SciPy, PyYAML, and motmetrics stack for 3D tracking
- ROS 2 Humble for early and late fusion
- a project-specific Conda environment for OCLReID

Choose one task and follow its README and requirements files.

## MMDetection Dependency

The MMDetection benchmark code in this repository is designed to use the external `mmdet` Python package rather than a vendored framework copy. Local benchmark logic and dataset-specific configs stay under `2d-detection/benchmarks/mmdetection/`.

If framework-level MMDetection changes are required, they should live in an external fork of `open-mmlab/mmdetection` and be referenced as a dependency from environment setup documentation or `requirements.txt`, not copied into this repository.

The 3D detection workspace similarly uses the external MMDetection3D software stack.

## Intended Direction

This repository should be read as both:

- a working benchmark implementation for AGHRI human detection, tracking, fusion, and person re-identification
- a foundation for extending agricultural human-safety perception benchmarks across additional methods and sensing conditions

Future work can include:

- additional 2D and 3D detection or tracking methods
- stronger cross-domain and cross-modal comparisons
- improved checkpoint and environment distribution
- additional multimodal fusion approaches

## Dataset Citation

When using AGHRI, cite the official dataset record:

```text
AGHRI: A dataset for multimodal robot perception of humans in
agricultural and off-road environments.
https://doi.org/10.24385/lincoln.32982638
```

Use the official dataset page for the current author list, version information, and citation format.

## License

The repository's benchmark code and documentation are licensed under Apache License 2.0. See [`LICENSE`](LICENSE).

The AGHRI dataset is distributed separately under CC BY 4.0. The dataset licence is independent of the benchmark code licence.

Additional attribution information is recorded in [`NOTICE`](NOTICE) and in the licence or citation files inside the component folders.

If third-party code is added to or redistributed with the repository, its original licence, notices, and citations should be preserved.
