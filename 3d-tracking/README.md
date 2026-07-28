# 3D Tracking Workspace

This workspace benchmarks LiDAR-based 3D multi-object tracking on top of the 3D person detections produced by `3d-detection`.

Its purpose is to evaluate whether a tracker can keep a stable person identity over time using frame-wise 3D bounding boxes in the AGHRI LiDAR coordinate frame.

## What This Workspace Is For

Use `3d-tracking` to:

- run multiple 3D tracker families on a shared detection format
- compare ground-truth detections with detector-generated detections
- generate per-scene MOT3D tracking files
- evaluate 3D tracks using common tracking metrics
- visualise ground-truth and predicted tracks over LiDAR point clouds

## Structure

- `benchmarks/ab3dmot`: local AB3DMOT implementation
- `benchmarks/centerpoint`: local CenterPoint implementation
- `benchmarks/simpletrack`: local SimpleTrack implementation
- `common/mot3d`: shared tools for preparing 3D tracking data, evaluating the tracks, and summarising the results
- `reports`: tracker summaries and runtime results
- `notebooks`: 3D tracking analysis notebook

## Tracking Workflow

1. Start from AGHRI LiDAR ground-truth boxes or exported 3D detections.
2. Run AB3DMOT, the CenterPoint, or SimpleTrack.
3. Save the predicted tracks in MOT3D CSV format.
4. Convert AGHRI 3D annotations into MOT3D ground truth.
5. Match predictions and ground truth using oriented bird's-eye-view IoU.
6. Compare tracking and identity metrics across trackers.

## Common Detection Format

Detector outputs are stored by scene and frame:

```text
<detections-dir>/<scene-name>/000000.json
<detections-dir>/<scene-name>/000001.json
```

Each JSON file contains zero or more detections:

```json
[
  {
    "x": 5.2,
    "y": 0.4,
    "z": 0.8,
    "l": 0.6,
    "w": 0.7,
    "h": 1.7,
    "yaw": 0.0,
    "score": 0.85
  }
]
```

The box representation is:

```text
[x, y, z, length, width, height, yaw]
```

The 3D detections can be exported from:

```text
../3d-detection/benchmarks/mmdetection3d/src/dump_detections_3d.py
```

## Trackers

### AB3DMOT

Uses Kalman prediction, BEV-IoU association, Hungarian assignment, and track lifecycle management.

### CenterPoint Tracker

Uses velocity-based centre prediction and XY centre-distance association. It is independent of the CenterPoint detector and can use any compatible 3D detections.

### SimpleTrack

Uses two-stage high- and low-confidence association with tentative, confirmed, and lost track states.

## Shared MOT3D Utilities

The main shared scripts are:

- `common/mot3d/convert_gt_to_mot3d.py`: converts AGHRI LiDAR annotations to MOT3D ground truth
- `common/mot3d/run_tracking_suite.py`: runs the tracker comparison
- `common/mot3d/evaluate_mot3d.py`: evaluates predictions
- `common/mot3d/bev_iou.py`: computes oriented BEV IoU
- `common/mot3d/hota.py`: computes HOTA-family metrics

## Reports

- `notebooks/3d_tracking_benchmark.ipynb`: analysis notebook

The committed comparison includes:

- AB3DMOT
- CenterPoint tracking
- SimpleTrack
- ground-truth boxes as oracle detections
- AGHRI PointPillars detections
- three representative AGHRI scenes

## Visualisation

- `tools/vis_annotations_pcd.py`: displays AGHRI 3D annotations over LiDAR point clouds
- `tools/vis_tracks_pcd.py`: displays predicted tracks over LiDAR point clouds

These tools are for qualitative inspection and do not calculate benchmark metrics.

## Relationship To The Wider Benchmark

This workspace is the temporal tracking layer for the LiDAR-based 3D benchmark.

- [`../3d-detection`](../3d-detection/README.md) supplies compatible 3D detection outputs.
