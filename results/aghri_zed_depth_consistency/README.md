# AGHRI ZED Depth Consistency Check

This is not an absolute 3D localization benchmark. ZED depth maps provide per-pixel surface depth, not annotated 3D person centres. The evaluation compares the fused LiDAR camera-frame depth `Zc` with the median valid ZED depth inside a central region of each fine-tuned YOLO person box.

Summary CSV: `/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/results/aghri_zed_depth_consistency/finetuned_zed_depth_consistency_summary.csv`

Per-detection CSV: `/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/results/aghri_zed_depth_consistency/finetuned_zed_depth_consistency_per_detection.csv`
