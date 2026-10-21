# 3D Detection Workspace

This workspace adds LiDAR-based 3D human detection to the benchmark repository.

Its structure:

- `benchmarks/mmdetection3d`: MMDetection3D-based training, evaluation, and data preparation
- `reports/benchmarks`: run artifacts and benchmark summaries

The current implementation targets the AGHRI LiDAR dataset and provides two benchmarked models:

- PointPillars
- SECOND

See [`benchmarks/mmdetection3d/README.md`](benchmarks/mmdetection3d/README.md) for the full workflow.

The current AGHRI protocol selects checkpoints on validation and evaluates the frozen
checkpoints on the held-out test sequences. See the [3D test results](reports/benchmarks/summary/mmdetection3d/aghri_heldout_test/README.md).
