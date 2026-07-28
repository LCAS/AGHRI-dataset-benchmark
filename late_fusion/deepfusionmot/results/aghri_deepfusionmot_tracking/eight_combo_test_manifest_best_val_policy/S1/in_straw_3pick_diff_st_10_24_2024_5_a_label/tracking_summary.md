# AGHRI DeepFusionMOT-Style Tracking Experiment

- Combination ID: `S1`
- Combination: G YOLO + G SECOND
- Recording used: `in_straw_3pick_diff_st_10_24_2024_5_a_label`
- Frames processed: 122
- Camera model: `generic`
- LiDAR detector: `second`
- LiDAR model: `generic`
- Camera cache: `/home/prabuddhi/Desktop/late_fusion/deepfusionmot/results/aghri_pointpillars_detector_comparison/cache/yolo_generic/detections.csv`
- LiDAR cache: `/home/prabuddhi/Desktop/late_fusion/deepfusionmot/results/aghri_finetuned_detector_comparison/cache/second_generic/detections.csv`
- Camera score threshold: 0.100
- LiDAR score threshold: 0.200
- LiDAR NMS IoU threshold: 0.000
- LiDAR-only initiation score threshold: 0.000
- Calibration source: `/media/prabuddhi/Backup2/Updated Dataset_PW/calibration`
- Tracker preset: `deepfusionmot_strict`
- Tracker output mode: `deepfusionmot`
- Ego motion mode: `aghri_odom`
- Ego motion source: `/media/prabuddhi/Backup2/Updated Dataset_PW/dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label/metadata/odom_global.jsonl`
- Ego motion applied frames: 121
- Camera detections: 239
- LiDAR detections: 42
- Score-filtered LiDAR detections before NMS: 42
- NMS-suppressed LiDAR detections: 0
- Fused detections: 30
- Tracks created: 11
- Confirmed tracks: 3
- Mean confirmed-track length: 15.67
- Maximum track length: 37
- Prediction-only updates: 28
- Deleted tracks: 10
- Tracker-only FPS: 2648.05
- Cached-pipeline FPS: 980.86
- Invalid detections rejected: 0
- Annotated visual frames: 0
- Visual MP4 created: False
- Visual MP4/status: `no visual frames were generated`
- Visual ID continuity appears reasonable: True
- Recording selection reason: Selected from AGHRI cache/manifest evidence because it has consecutive timestamped frames, valid ZED RGB image paths, FT YOLO detections, FT PointPillars detections, and valid AGHRI calibration.

## Interface Inspection

- `Detection2D`: `bbox_xyxy`, `score`, `label_id`, `label_name`, `detection_id`, `sample_id`, `timestamp`, `metadata`.
- `Detection3D`: `center_xyz`, `size_lwh`, `yaw`, `score`, `label_id`, `label_name`, `detection_id`, `sample_id`, `timestamp`, `frame_id`, `box_origin`, `metadata`.
- `FusionResult`: `matches`, `matched_3d`, `unmatched_lidar`, `unmatched_camera`, `final_3d`, `projected_lidar`, `iou_matrix`, `confidence_policy`, `output_policy`.
- AGHRI 3D boxes are consumed in `front_lidar_link`; dimensions are `(length, width, height)` and yaw is LiDAR-frame z-yaw.
- PointPillars cache boxes are loaded through the detector profile box-origin setting and normalized by the cache loader before tracking.
- Cache lookup is timestamp-based with the configured tolerance; recording boundaries are explicit `recording`/`recording_id` values.
- The tracker is a temporal layer after existing fusion; matched, unmatched-camera, unmatched-LiDAR, and ALL3D detection outputs are not replaced.
- With `--deepfusionmot-strict`, the tracker uses the DeepFusionMOT Pedestrian-style settings: `min_hits=3`, `max_age_frames=25`, 3D IoU gate `0.01`, 2D IoU gate `0.50`, even 3D IDs, odd 2D IDs, frame-count lifecycle, and confirmed-or-early-frame 3D output.
- AGHRI calibration is loaded from `intrinsics.json` and `extrinsics.json`; `ego_motion_mode=aghri_odom` uses per-recording `metadata/odom_global.jsonl` rather than KITTI OXTS files.

## Visual Validation

- Visual frames directory: `results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy/S1/in_straw_3pick_diff_st_10_24_2024_5_a_label/visual_frames`
- Visual frames generated: 0
- Visual MP4/status: `no visual frames were generated`
- The visual overlays show projected tracked 3D boxes, persistent track IDs, stable per-ID colours, and track state/update source labels.
- Visual continuity is treated as a smoke-test proxy only, not as HOTA/IDF1/MOTA identity accuracy.

## Known Limitations

- This is a smoke test, not the eight-combination AGHRI benchmark.
- `ego_motion_mode=aghri_odom` uses AGHRI per-recording odometry; `none` leaves tracks in the previous detector frame.
- HOTA, IDF1, IDSW, and MOTA are produced by `tools/evaluate_aghri_deepfusionmot_metrics.py` when AGHRI annotation identities are available.
- Track IDs are algorithmic IDs; metric evaluation compares them against AGHRI annotation identities.
