# AGHRI YOLO11s Fusion 2x2 Benchmark

This benchmark compares generic YOLO11s and AGHRI ZED RGB fine-tuned YOLO11s on the official held-out AGHRI ZED RGB test split, with the frozen `legacy_box` and `hybrid_depth_legacy` camera-LiDAR association methods.

## Purpose

The primary scientific comparison is:

```text
Generic YOLO11s + legacy_box
versus
AGHRI-fine-tuned YOLO11s + legacy_box
```

Hybrid results are reported as a secondary association-method analysis. The benchmark does not tune detector confidence, association parameters, calibration, or matching rules using test results.

## Checkpoints

Generic checkpoint:

```text
/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/yolo11s.pt
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
```

Fine-tuned checkpoint:

```text
/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt
bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110
```

The fine-tuned model was initialized from generic YOLO11s and trained on AGHRI ZED RGB train/validation data only. The official test split was not used for training, validation, or checkpoint selection.

## Data And Sensors

Only `cam_zed_rgb` images are used because the detector was trained for ZED RGB and the fusion stack projects Livox points into that camera. The benchmark uses the original PNG images, Livox PCD files, AGHRI calibration ZIP, 2D annotations, 3D annotations, and deterministic timestamp matching.

## Frozen Fusion Settings

Legacy:

```yaml
association_method: legacy_box
centre_statistic: mean
box_bottom_trim_ratio: 0.0
```

Hybrid:

```yaml
association_method: hybrid_depth_legacy
centre_statistic: mean
box_bottom_trim_ratio: 0.0
depth_cluster_gap_m: 0.35
depth_cluster_min_points: 5
hybrid_min_cluster_points: 5
hybrid_min_support_ratio: 0.20
hybrid_max_depth_spread_m: 0.45
hybrid_min_cluster_separation_m: 0.15
hybrid_fallback_on_single_cluster: false
hybrid_fallback_on_sparse_candidates: true
```

Detector settings are fixed to person class `0`, confidence threshold `0.1`, and image size `640`.

## Detection Caches

The benchmark writes separate caches:

```text
/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2/caches/generic_yolo11s
/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2/caches/finetuned_yolo11s
```

Within each detector, legacy and hybrid consume the same cache. Generic and fine-tuned detections are never mixed because cache metadata includes checkpoint path, checkpoint SHA256, image SHA256, threshold, class filter, image size, and Ultralytics version.

## Run

Use the local system Python environment with the installed ROS/Ultralytics dependencies:

```bash
cd /home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble
/usr/bin/python3 tools/run_aghri_detector_fusion_2x2.py
```

Outputs:

```text
/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2
```

Regenerate only tables and reports from existing raw runs:

```bash
/usr/bin/python3 tools/run_aghri_detector_fusion_2x2.py --postprocess-only
```

## Artifacts

Important outputs include:

```text
final_report.md
manifest/test_manifest.csv
manifest/test_manifest.json
metadata/reproducibility_manifest.json
metadata/generic_result_reuse_check.json
tables/overall_2x2.csv
tables/per_recording_2x2.csv
tables/detector_comparison_legacy.csv
tables/detector_comparison_hybrid.csv
tables/fusion_comparison_generic.csv
tables/fusion_comparison_finetuned.csv
tables/condition_breakdowns.csv
tables/failure_taxonomy.csv
comparisons/pairwise_comparisons.json
comparisons/primary_checkpoint_comparison.json
comparisons/bootstrap_results.json
visuals/visual_manifest.csv
```

## Interpretation

Read the detector effect from the GT-centric legacy comparison first. Hybrid can change fusion coverage and tail errors, so it should be interpreted as a secondary association result, not as a replacement for the pre-declared primary comparison.
