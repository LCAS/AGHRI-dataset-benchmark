# 2D Tracking Workspace

This workspace benchmarks multi-object tracking on top of the detector outputs produced by `2d-detection`.

Its role in the repository is to answer a different safety question from detection alone: not only can a robot see a person, but can it keep a stable identity over time well enough to support motion reasoning, warning logic, and later fusion with other sensors.

## What This Workspace Is For

Use `2d-tracking` to:

- run multiple tracker families on a shared detection export format
- compare tracker behaviour under different detector quality levels
- generate MOTChallenge-format tracking outputs
- evaluate trackers with the same MOT metrics across frameworks

The current AGHRI evaluation covers six test recordings in each of the ZED RGB and front-fisheye streams. Each camera uses its own validation-selected YOLOv8s and YOLO11s checkpoints. Ground-truth boxes provide a tracker-only control.

## Structure

- `common/mot`: shared MOT conversion, suite-running, and evaluation utilities
- `benchmarks/boxmot`: BoxMOT-based tracking wrappers
- `benchmarks/deepsort`: local DeepSORT implementation with frozen-graph ReID
- `benchmarks/norfair`: Norfair-based tracking wrapper
- `configs/aghri_test.yaml`: AGHRI test protocol and detector-family selection
- `reports/aghri_test`: current evaluation results, timing, and checkpoint records
- `reports/runs` and `reports/summary`: output locations for new local runs
- `notebooks`: analysis notebooks

## Tracking Workflow

1. Start from a frame-ordered detections JSON file exported by `2d-detection`. The current AGHRI exports retain detector scores and use a 0.05 score floor.
2. Run one tracker framework to produce a MOT-format prediction file. Norfair applies a 0.25 input confidence threshold; the other trackers use their own configured thresholds.
3. Convert project ground truth to MOT format with `common/mot/convert_gt_to_mot.py`.
4. Evaluate predictions using image IoU matching at 0.50. The current AGHRI reports use TrackEval for HOTA, CLEAR MOTA, IDF1, and MOTP.
5. Compare per-sequence and per-source results across trackers and detector inputs.

For the paper's 2D tracking comparison, the selected detector family is YOLOv8s for both cameras, based on validation detection results. The repository retains the YOLO11s comparisons as well. A combined two-camera result must be calculated from the sequence results, not by averaging the two camera summary rows.

## Common Input Format

All implemented trackers read the same repository JSON shape:

```json
[
  {
    "File": "frame_000001.png",
    "Labels": [
      {
        "Class": "person",
        "BoundingBoxes": [100.0, 50.0, 80.0, 180.0],
        "Confidence": 0.93
      }
    ]
  }
]
```

`File` is the frame name, `BoundingBoxes` are in `x, y, w, h` form, and `Confidence` is the detector score. Ground-truth-box controls use confidence 1.0.

## Frameworks

### BoxMOT

Supports ByteTrack, OC-SORT, StrongSORT, BoT-SORT, Deep OC-SORT, and BoostTrack behind one wrapper.

- local docs: [`benchmarks/boxmot/README.md`](benchmarks/boxmot/README.md)

### DeepSORT

Implements a dedicated DeepSORT pipeline with TensorFlow frozen-graph appearance embeddings.

- local docs: [`benchmarks/deepsort/README.md`](benchmarks/deepsort/README.md)

### Norfair

Provides a lightweight tracker benchmark with scored detections and optional rendering.

- local docs: [`benchmarks/norfair/README.md`](benchmarks/norfair/README.md)

### Shared MOT Utilities

- local docs: [`common/mot/README.md`](common/mot/README.md)

## Reports

The current AGHRI results are in `reports/aghri_test`:

- [How to read HOTA, equal-sequence, pooled, and throughput results](reports/aghri_test/README.md)
- [TrackEval results by detector source and tracker](reports/aghri_test/trackeval_summary_by_source.csv)
- [Per-sequence TrackEval results](reports/aghri_test/trackeval_per_sequence.csv)
- [Detailed TrackEval tables](reports/aghri_test/trackeval_detailed/)
- [Tracker-only throughput by source](reports/aghri_test/tracker_throughput_by_source.csv)
- [Per-run tracker timing](reports/aghri_test/tracker_throughput_per_run.csv)
- [Detector checkpoint hashes and selection split](reports/aghri_test/detector_checkpoints.csv)
- [Paper result CSVs and figures](reports/aghri_test/paper_results/)
- [Paper-results notebook](notebooks/2d_tracking_paper_results.ipynb)

HOTA is available in the detailed TrackEval tables, the per-sequence extract, and the source summary. The paper notebook calculates equal-run means from per-sequence TrackEval values; the source summary separately retains macro and pooled/global results. The report guide explains the fields, scales, and throughput aggregation.

The result tables give the evaluated frame counts. TrackEval MOTP is the mean IoU of matched boxes, so higher is better. Tracker throughput covers tracking computation only; it excludes detector inference, image I/O, rendering, and evaluation. `reports/runs` and `reports/summary` remain output locations for future runs.

## Path Conventions

Tracking configs resolve relative paths from `2d-tracking` unless the path is already absolute. That applies to:

- detections JSON inputs
- frame directories
- MOT outputs
- rendered videos and frame dumps
- evaluation CSV and JSON summaries

The tracker-family configs are reusable templates. The completed AGHRI protocol is recorded in [`configs/aghri_test.yaml`](configs/aghri_test.yaml).

## Relationship To The Wider Benchmark

This workspace is the temporal evaluation layer of the AGHRI benchmark. Its camera-based tracking results complement the repository's 2D detection, 3D tracking, and multimodal fusion experiments.
