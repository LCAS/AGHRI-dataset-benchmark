# 3D Detection Workspace

This workspace adds LiDAR-based 3D human detection to the benchmark repository.

Its structure:

- `benchmarks/mmdetection3d`: MMDetection3D-based training, evaluation, and data preparation
- `reports/benchmarks`: run artifacts and benchmark summaries

The current implementation targets the agri-human LiDAR dataset and provides two benchmarked models:

- PointPillars
- SECOND

See [`benchmarks/mmdetection3d/README.md`](benchmarks/mmdetection3d/README.md) for the full workflow.
