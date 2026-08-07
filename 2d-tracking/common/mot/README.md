# Shared MOT Utilities

This directory contains tracker-agnostic utilities shared by the tracking benchmarks.

## What It Does

- convert project annotation JSON into MOTChallenge ground-truth text
- evaluate predicted MOT files against ground truth using `motmetrics`

These utilities keep benchmark logic consistent across BoxMOT, DeepSORT, and Norfair.

## Files

- `convert_gt_to_mot.py`: convert project annotations to MOT text
- `convert_all_gt_to_mot.py`: bulk-convert every scenario under `reports/GTs`
- `run_tracking_suite.py`: batch-run supported trackers and aggregate MOT metrics
- `evaluate_mot.py`: evaluate one predicted MOT file against one GT file
- `__init__.py`: package marker

## Annotation Input Format

The converter expects the same frame-wise JSON style used elsewhere in the repository:

```json
[
  {
    "File": "frame_000001.png",
    "Labels": [
      {
        "Class": "07",
        "BoundingBoxes": [100.0, 50.0, 80.0, 180.0]
      }
    ]
  }
]
```

`convert_gt_to_mot.py` will:

- read frame records in order
- extract `Class` and `BoundingBoxes`
- convert each box to MOT `frame,id,x,y,w,h,...`
- require `Class` to be a positive digit-only AGHRI person identity
- convert the identity to an integer MOT track ID (`"07"` becomes `7`)

Non-numeric, empty, and zero-valued identities are rejected rather than mapped
to fallback track IDs.

## Convert Ground Truth

```bash
python common/mot/convert_gt_to_mot.py \
  --ground-truth-json /path/to/annotations.json \
  --out reports/runs/example/ground_truth.txt
```

To bulk-convert all scenario annotations under `reports/GTs` into
`reports/runs/GTs_MOT`:

```bash
python common/mot/convert_all_gt_to_mot.py --skip-existing
```

## Evaluate A Prediction

Use one of the benchmark-specific wrappers so the correct default config is supplied:

```bash
python benchmarks/boxmot/src/evaluate_mot.py \
  --config benchmarks/boxmot/configs/evaluation/default.yaml
```

Or with direct overrides:

```bash
python benchmarks/boxmot/src/evaluate_mot.py \
  --config benchmarks/boxmot/configs/evaluation/default.yaml \
  --gt reports/runs/example/ground_truth.txt \
  --pred reports/runs/example/tracks.txt \
  --out-csv reports/summary/example_metrics.csv
```

## Batch Tracking Suite

To run the locally supported trackers on available detector exports and GT-based
perfect-detection inputs, then save one aggregate summary for downstream
analysis:

```bash
python common/mot/run_tracking_suite.py --skip-existing
```

By default this writes:

- GT-normalized tracker inputs under `reports/runs/tracker_suite/_inputs/`
- tracker MOT outputs under `reports/runs/tracker_suite/`
- per-run metrics under `reports/summary/tracker_suite/`
- aggregate CSV and JSON summaries under `reports/summary/tracker_suite/`

To run only the GT-as-detections case:

```bash
python common/mot/run_tracking_suite.py --sources gt --skip-existing
```

## Evaluation Metrics

The default metric list includes:

- `idf1`
- `idp`
- `idr`
- `precision`
- `recall`
- `mota`
- `motp`
- `num_switches`
- `num_false_positives`
- `num_misses`
- `mostly_tracked`
- `mostly_lost`

## Path Convention

Relative paths resolve from `2d-tracking`, which keeps configs portable across benchmark wrappers.
