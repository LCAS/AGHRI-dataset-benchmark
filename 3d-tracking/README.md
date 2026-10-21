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
- `reports/heldout`: validation selection records, six-sequence test results, and tracker predictions
- `notebooks/3d_tracking_paper_results.ipynb`: six-sequence paper tables and the five revised figures in manuscript order

## Tracking Workflow

1. Use the seven AGHRI validation sequences to select each detector checkpoint and tune SimpleTrack separately for each detector. AB3DMOT and CenterPoint use fixed tracker settings.
2. Freeze those choices, then export scored detections for all six held-out test sequences. The detector export retains scores down to 0.05; each tracker applies its configured input threshold.
3. Run AB3DMOT, CenterPoint, and SimpleTrack on both detector-based and ground-truth-based boxes. Ground-truth boxes have score 1.0 and their participant IDs are withheld from the trackers.
4. Save predicted tracks in MOT3D CSV format and convert AGHRI annotations to identity-preserving MOT3D ground truth.
5. Evaluate all six sequences using oriented bird's-eye-view IoU matching at 0.25 and calculate sequence, macro, and pooled results. Detector-based and ground-truth-based results remain separate.

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

The current held-out comparison covers six AGHRI test sequences (1,308 LiDAR frames). PointPillars scratch, selected at epoch 108 on AGHRI validation, is the detector used for the manuscript's main 3D tracking comparison. SECOND scratch, selected at epoch 36, is retained as a separate detector comparison. The held-out test split was not used to choose a checkpoint or tune SimpleTrack.

- `reports/heldout/test/<model>/summary.csv`: six tracker/input combinations, each with `macro` and `pooled` rows.
- `reports/heldout/test/<model>/per_sequence.csv`: results for all six sequences, three trackers, and two input settings.
- `reports/heldout/test/<model>/<input>/<tracker>/`: per-sequence metrics, runtime, and MOT3D prediction CSVs. `<input>` is `gt` or the detector model name.
- `reports/heldout/val/<model>/`: seven-sequence validation summaries and the 18-candidate SimpleTrack tuning grid.
- `reports/heldout/frozen/<model>/simpletrack.json`: validation-selected SimpleTrack thresholds and confirmation count. PointPillars uses score threshold 0.10, high-score threshold 0.50, and three hits; SECOND uses 0.05, 0.50, and one hit.
- `reports/heldout/manifests/`: detector checkpoint hashes and sequence membership for validation and test.
- `reports/heldout/split_lists/`: the exact validation and test sequence lists used by this evaluation.
- `reports/heldout/workflow/`: archived cluster export and evaluation scripts used to produce these files. They expect the corresponding cluster-stage directory layout and dependencies.

The `macro` row averages each sequence's MOTA, MOTP, IDF1, recall, HOTA, DetA, and AssA equally. The `pooled` row calculates CLEAR and identity metrics by combining the six sequence accumulators; it weights observations according to their contribution to the combined evaluation. HOTA, DetA, and AssA are only reported as equal-sequence means. Identity switches are summed over sequences for both summaries. The paper uses the `macro` rows. Tracker-update throughput is total processed frames divided by tracker-side update time and excludes detector inference.

The evaluator computes MOTP as mean matched BEV IoU distance (`motp` in the per-sequence files). The `motp_iou` column is `1 - motp`, the mean matched BEV IoU; higher values indicate better localisation. The selected tracker predictions and ground truth are matched at oriented BEV IoU 0.25. HOTA-family values average thresholds from 0.05 to 0.95.

Use `notebooks/3d_tracking_paper_results.ipynb` for the six-sequence paper tables and figures; its images are in `notebooks/figures/`.

## Visualisation

- `tools/vis_annotations_pcd.py`: displays AGHRI 3D annotations over LiDAR point clouds
- `tools/vis_tracks_pcd.py`: displays predicted tracks over LiDAR point clouds

These tools are for qualitative inspection and do not calculate benchmark metrics.

## Relationship To The Wider Benchmark

This workspace is the temporal tracking layer for the LiDAR-based 3D benchmark.

- [`../3d-detection`](../3d-detection/README.md) supplies compatible 3D detection outputs.
