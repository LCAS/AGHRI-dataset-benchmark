# R1-Q3-13 Existing-Output Early-Fusion Evaluation

This report was generated from saved AGHRI early-fusion outputs. It does not
retrain YOLO11s, rerun detection, alter fusion thresholds, or modify the
original result files.

## Current Matching Algorithm

The frozen evaluator first computes a fused output for each YOLO person
detection. Ground truth is used only after fusion. The primary match is in 2D
image space: the YOLO box is compared with ZED RGB ground-truth boxes using IoU,
with a match threshold of `IoU >= 0.10`. If the top two 2D GT candidates differ
by less than `0.05` IoU, the output is marked ambiguous. If no 2D identity match
is available, the evaluator falls back to nearest 3D LiDAR GT centre matching.
The 3D fallback is marked ambiguous if the second-nearest GT is within `0.50 m`
of the nearest GT.

The original conditional localisation errors are computed only for valid fused
outputs that receive a GT match. Missed GT persons, invalid fusion outputs, and
false/duplicate fused outputs are reported separately here rather than being
assigned artificial large errors.

## End-to-End Definitions

For GT-centric localisation, a ground-truth person can contribute at most once
per frame. If multiple valid detections match the same GT identity in the same
frame, that GT contributes one true localisation and the extra valid fused
outputs count against one-to-one precision as false localisations. Ambiguous
valid outputs are not counted as unique GT localisations.

- GT-person localisation recall = uniquely localised GT persons / total GT persons.
- Miss rate = missed GT persons / total GT persons.
- One-to-one localisation precision = uniquely localised GT persons / valid fused outputs.
- False-localisation rate = false localisations / valid fused outputs.
- Invalid-output rate = invalid fusion outputs / all YOLO person detections.

## Overall Results

| model | total_gt_persons | uniquely_localised_gt_persons | gt_person_localisation_recall | miss_rate | total_valid_fused_outputs | one_to_one_localisation_precision | false_localisation_rate | invalid_output_rate | conditional_mean_3d_error_m | conditional_median_3d_error_m | conditional_p95_3d_error_m | conditional_error_gt_1m_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generic YOLO11s | 3155 | 2306 | 0.7309 | 0.2691 | 2569 | 0.8976 | 0.1024 | 0.0782 | 0.4765 | 0.2038 | 1.8703 | 0.0859 |
| AGHRI fine-tuned YOLO11s | 3155 | 2355 | 0.7464 | 0.2536 | 2453 | 0.9600 | 0.0400 | 0.0320 | 0.3926 | 0.1911 | 1.3595 | 0.0748 |

## Uncertainty

Confidence intervals use recording-level bootstrap over the six held-out test
recordings. Each bootstrap sample resamples six recordings with replacement and
recomputes metrics from summed numerators and denominators. Bootstrap samples:
`10000`. Fixed seed: `20261006`.

| model | recall_ci95 | precision_ci95 | miss_rate_ci95 | false_rate_ci95 | invalid_rate_ci95 |
| --- | --- | --- | --- | --- | --- |
| Generic YOLO11s | [0.5858, 0.8718] | [0.8675, 0.9522] | [0.1282, 0.4142] | [0.0478, 0.1325] | [0.0499, 0.0910] |
| AGHRI fine-tuned YOLO11s | [0.5927, 0.8920] | [0.9453, 0.9841] | [0.1080, 0.4073] | [0.0159, 0.0547] | [0.0186, 0.0460] |

## Stratification Notes

Distance-stratified GT recall is reconstructed from the AGHRI 3D annotation
centres transformed into the ZED RGB camera frame, using the same `<2 m`,
`2-5 m`, and `>=5 m` bands used by the evaluator. Motion is recording-level.
LiDAR point-density bins are detection-side bins from the saved evaluator
outputs, so they support invalid-output and conditional-error analysis but not a
GT-person denominator for missed people. Occlusion is included only as the
existing ZED RGB event/frame-level category, not as a new per-person occlusion
label.

## Limitations

The saved evaluator performs per-detection matching rather than a global
Hungarian assignment. This script therefore uses the saved per-frame unique
matched GT identities for GT-centric recall and precision accounting, while
preserving the original conditional error distribution for valid matched
detections. Distance/occlusion strata should be read with the denominator notes
above.

## Outputs

- `overall_r1_q3_13.csv`
- `stratified_distance.csv`
- `stratified_motion.csv`
- `stratified_point_density.csv`
- `stratified_occlusion_frame_event.csv`
- `recording_level_counts.csv`
- `bootstrap_ci.json`

## Command

```bash
/usr/bin/python3 tools/analyze_aghri_r1_q3_13_existing_outputs.py \
  --source-root /home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2 \
  --manifest /home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2/manifest/test_manifest.csv \
  --output-dir /home/prabuddhi/AGHRI-dataset-benchmark/early_fusion/camera_lidar_humble/results/r1_q3_13_existing_output_eval \
  --config /home/prabuddhi/AGHRI-dataset-benchmark/early_fusion/camera_lidar_humble/config/aghri_zed_livox.yaml \
  --bootstrap-samples 10000 \
  --seed 20261006
```
