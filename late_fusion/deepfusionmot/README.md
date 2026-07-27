# Late Camera-LiDAR Human Detection Fusion and Tracking Workflow

This is the AGHRI workflow for late camera-LiDAR human detection fusion and multi-object tracking. It covers detector fine-tuning, eight-combination offline evaluation and ROS 2 replay.

Numerical detection and tracking results come from the offline evaluators. ROS 2 bag replay is used for qualitative inspection.

This repository builds on three earlier implementations:

- The original [`wangxiyang2022/DeepFusionMOT`](https://github.com/wangxiyang2022/DeepFusionMOT) repository implements 3D multi-object tracking with camera-LiDAR deep association, Kalman filtering, track management, and separate 2D and 3D tracking branches.
- The earlier [`Prabuddhi-05/deepfusion`](https://github.com/Prabuddhi-05/deepfusion) ROS 2 package performs synchronized real-time fusion of 2D and 3D detections and publishes fused, camera-only, and LiDAR-only outputs.
- The [`LCAS/late_fusion`](https://github.com/LCAS/late_fusion) ROS 2 package provides a configurable real-time late-fusion node.

The AGHRI version keeps the detection-level late-fusion principle and adds tracking for people using ZED RGB images and Livox point clouds. The TrackEval metrics required for AGHRI tracking evaluation are vendored under `late_fusion_pkg/vendor/trackeval`.

## 1. Overview

The final benchmark evaluates eight detector combinations. `G` means generic pretrained detector and `FT` means AGHRI fine-tuned detector.

| ID | Camera detector | LiDAR detector |
| --- | --- | --- |
| S1 | G YOLO11s | G SECOND |
| S2 | FT YOLO11s | G SECOND |
| S3 | G YOLO11s | FT SECOND |
| S4 | FT YOLO11s | FT SECOND |
| P1 | G YOLO11s | G PointPillars |
| P2 | FT YOLO11s | G PointPillars |
| P3 | G YOLO11s | FT PointPillars |
| P4 | FT YOLO11s | FT PointPillars |

All combinations use the same fixed projection-based late-fusion procedure and the same DeepFusionMOT-style tracking framework. The trainable components are the three detectors:

- YOLO11s for camera-based 2D person detection
- SECOND for LiDAR-based 3D person detection
- PointPillars for LiDAR-based 3D person detection

The geometric fusion, Kalman filtering, association cascade, odometry compensation, and track lifecycle logic are not neural networks and are not trained.

For every synchronized frame, the workflow produces:

- matched camera-supported 3D detections
- unmatched LiDAR 3D detections
- unmatched camera 2D detections
- complete `ALL3D` output containing matched and unmatched LiDAR boxes
- persistent 3D and 2D tracks after DeepFusionMOT processing

## 2. Path Placeholders

Set these paths for your local machine before running the commands below:

```bash
export LATE_FUSION_WS="/home/prabuddhi/Desktop/late_fusion_ws"
export LATE_FUSION_REPO="/home/prabuddhi/Desktop/late_fusion/deepfusionmot"
export DATASET_ROOT="<path-to-AGHRI-dataset-root>"
export ROSBAG_ROOT="<path-to-AGHRI-test-rosbags>"
export TEST_MANIFEST="<path-to-AGHRI-held-out-test-manifest.csv>"
export MMDET3D_PYTHON="<path-to-aghri-mmdet3d-environment>/bin/python"
export TRACKING_OUTPUT_ROOT="${LATE_FUSION_REPO}/results/aghri_deepfusionmot_tracking"
```

## 3. Environment Requirements

Use ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

Build in a clean ROS 2 workspace:

```bash
mkdir -p "${LATE_FUSION_WS}"
cd "${LATE_FUSION_WS}"

source /opt/ros/humble/setup.bash

colcon build \
  --base-paths "${LATE_FUSION_REPO}" \
  --packages-select late_fusion_pkg \
  --symlink-install

source "${LATE_FUSION_WS}/install/setup.bash"
```

## 4. Why ZED RGB and Livox Are Used

The camera branch is trained and evaluated on `cam_zed_rgb`. YOLO11s produces 2D person boxes in the ZED RGB image.

The LiDAR branch uses the Livox point cloud in `front_lidar_link`. SECOND or PointPillars produces a complete detector-generated person cuboid:

```text
[x, y, z, length, width, height, yaw]
```

The fusion stage constructs the cuboid corners, transforms them from the LiDAR frame into the ZED camera frame, projects them into the image, and forms a projected 2D envelope. Image-space intersection over union (IoU) and Hungarian assignment are then used to associate the projected LiDAR boxes with YOLO detections.

Unlike the early-fusion workflow, which estimates a 3D centre from raw projected LiDAR points inside a camera box, this workflow preserves the complete SECOND or PointPillars cuboid and its confidence.

## 5. Dataset and Evaluation Split

The workflow uses the official AGHRI train, validation, and held-out test recordings.

- Training recordings are used to fine-tune YOLO11s, SECOND, and PointPillars.
- Validation recordings are used for checkpoint selection and tracking-parameter selection.
- Held-out test recordings are excluded from training and model selection.
- The final eight-combination comparison is evaluated on the six held-out AGHRI test recordings.

Ground truth is used only during offline evaluation. It is not used by the live fusion or tracking pipeline.

## 6. Detector Fine-Tuning

### 6.1 YOLO11s

The camera branch uses the same AGHRI ZED RGB YOLO11s checkpoints as the early-fusion workflow.

```text
Architecture: YOLO11s
Initial checkpoint: generic yolo11s.pt
Image size: 640
Batch: 16
Epochs: 50
Seed: 42
Best epoch: 46
Class mapping: {0: person}
```

### 6.2 SECOND

```text
Architecture: SECOND
Initial checkpoint: standard pretrained generic SECOND
Dataset: AGHRI Livox train/validation export
Class mapping: {0: person}
Training scenes: 52
Training frames: 11,411
Validation scenes: 7
Validation frames: 1,201
Epochs: 80
Validation interval: every 2 epochs
Selection metric: AGHRI bird's-eye-view average precision mean
Seed: 42
Best epoch: 2
Validation bird's-eye-view average precision mean: 0.245199
Validation 3D average precision mean: 0.204003
```

### 6.3 PointPillars

```text
Architecture: PointPillars
Initial checkpoint: standard pretrained generic PointPillars
Dataset: AGHRI Livox train/validation export
Class mapping: {0: person}
Training scenes: 52
Training frames: 11,411
Validation scenes: 7
Validation frames: 1,201
Epochs: 160
Selection metric: AGHRI bird's-eye-view average precision mean
Seed: 42
Best epoch: 126
Validation bird's-eye-view average precision mean: 0.273129
Validation 3D average precision mean: 0.237417
```

The fusion and tracking stages do not contain additional trainable weights.

## 7. Late-Fusion and Tracking Workflow

For every frame, the pipeline performs the following steps:

1. YOLO11s produces camera-based 2D person detections.
2. SECOND or PointPillars produces LiDAR-based 3D person cuboids.
3. Detector outputs are approximately synchronized by timestamp.
4. LiDAR cuboids are projected into the ZED RGB image using CameraInfo and transform (TF) calibration.
5. Projected LiDAR envelopes and YOLO boxes are associated using image-space IoU and Hungarian assignment.
6. The association produces matched 3D, LiDAR-only 3D, and camera-only 2D outputs.
7. Existing tracks are predicted with 2D and 3D Kalman filters.
8. AGHRI odometry transforms previous 3D tracks into the current LiDAR frame.
9. The four-level association cascade processes the strongest evidence first:
   - Level 1: matched 3D detections to active 3D tracks
   - Level 2: LiDAR-only detections to remaining 3D tracks
   - Level 3: camera-only detections to active 2D tracks
   - Level 4: 2D tracks support compatible projected unmatched 3D tracks
10. Matched tracks are corrected, unmatched detections create tentative tracks, and missed tracks follow the lifecycle policy.
11. Confirmed tracks are published with persistent IDs.

The final selected AGHRI configuration was chosen using validation Higher Order Tracking Accuracy (HOTA). The exact resolved detector thresholds, association thresholds, lifecycle settings, and checkpoint paths should be taken from the configuration files stored with the completed result batch.

## 8. Offline Eight-Combination Evaluation

The offline workflow uses native AGHRI files and frozen detector outputs. It is separate from ROS 2 playback so the quantitative comparison is reproducible and is not affected by ROS scheduling or visualisation.

The main tools are:

```text
tools/run_aghri_deepfusionmot_batch.py
tools/evaluate_aghri_deepfusionmot_metrics.py
```

Inspect the available arguments in the active checkout:

```bash
cd "${LATE_FUSION_REPO}"

PYTHONPATH=. /usr/bin/python3 tools/run_aghri_deepfusionmot_batch.py --help
PYTHONPATH=. /usr/bin/python3 tools/evaluate_aghri_deepfusionmot_metrics.py --help
```

The completed final metric table is stored at:

```text
results/aghri_deepfusionmot_tracking/
  eight_combo_test_manifest_best_val_policy_metrics/
    deepfusionmot_metrics.md
```

TrackEval computes HOTA, CLEAR multi-object tracking, and identity metrics from the AGHRI evaluation adapter. In this workflow, tracked 3D cuboids are projected into the ZED RGB image and compared with the ZED RGB ground-truth boxes and person identities. These are therefore image-space tracking metrics for projected 3D tracks, not direct LiDAR-frame 3D tracking metrics.

## 9. Final Numerical Results
| ID | Combination | HOTA | DetA | AssA | IDSW | MOTP | MOTA | IDF1 | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | G YOLO + G SECOND | 0.1524 | 0.2129 | 0.1107 | 40 | 0.6291 | 0.0065 | 0.1664 | 12.73 |
| S2 | FT YOLO + G SECOND | 0.1576 | 0.2244 | 0.1127 | 43 | 0.6278 | -0.0306 | 0.1706 | 12.27 |
| S3 | G YOLO + FT SECOND | 0.2317 | 0.3335 | 0.1631 | 53 | **0.6559** | 0.1473 | 0.2581 | 12.69 |
| S4 | FT YOLO + FT SECOND | 0.2425 | 0.3511 | 0.1697 | 59 | 0.6536 | **0.1571** | 0.2727 | 12.43 |
| P1 | G YOLO + G PointPillars | 0.1098 | 0.1515 | 0.0824 | 37 | 0.6207 | -0.1061 | 0.1271 | **13.57** |
| P2 | FT YOLO + G PointPillars | 0.1161 | 0.1533 | 0.0903 | 36 | 0.6195 | -0.1053 | 0.1285 | 13.38 |
| P3 | G YOLO + FT PointPillars | 0.3539 | 0.4102 | 0.3089 | **29** | 0.6527 | -0.0020 | **0.3538** | 12.15 |
| P4 | FT YOLO + FT PointPillars | **0.3553** | **0.4116** | **0.3102** | **29** | 0.6526 | -0.0053 | 0.3533 | 12.14 |

The best HOTA was obtained by:

```text
P4: AGHRI fine-tuned YOLO11s + AGHRI fine-tuned PointPillars
```

## 10. ROS 2 Test Bags

The online workflow uses the six held-out AGHRI ROS 2 test bags. The bags contain the recorded ZED RGB image, Livox point cloud, CameraInfo, dynamic TF, static TF, and ROS clock topics required by the live detector, fusion, tracking, and visualisation nodes.

The original sensor timestamps are preserved. ZED RGB and Livox messages are not forced to have identical timestamps; the online workflow uses approximate synchronization.

Set the bag root with:

```bash
export ROSBAG_ROOT="<path-to-AGHRI-test-rosbags>"
```

## 11. Build and Source

```bash
source /opt/ros/humble/setup.bash
cd "${LATE_FUSION_WS}"

colcon build \
  --base-paths "${LATE_FUSION_REPO}" \
  --packages-select late_fusion_pkg \
  --symlink-install

source "${LATE_FUSION_WS}/install/setup.bash"
```

## 12. Live ROS 2 Playback

The following example runs live fine-tuned YOLO11s, live fine-tuned PointPillars, projection-based late fusion, DeepFusionMOT tracking, and RViz visualisation:

```bash
PYTHONNOUSERSITE=1 /opt/ros/humble/bin/ros2 launch late_fusion_pkg aghri_online_late_fusion.launch.py \
  inference_mode:=live \
  camera_model:=finetuned \
  lidar_detector:=pointpillars \
  lidar_model:=finetuned \
  bag_path:="${ROSBAG_ROOT}/<test-recording>" \
  enable_tracking:=true \
  enable_visualisation:=true \
  show_original_lidar_markers:=true \
  show_unmatched_lidar_markers:=true \
  show_tracked_markers:=true \
  launch_rviz:=true \
  playback_rate:=1.0
```

To evaluate another detector combination, change:

```text
camera_model:=generic | finetuned
lidar_detector:=second | pointpillars
lidar_model:=generic | finetuned
```

For a headless run, disable RViz and visualisation:

```text
enable_visualisation:=false
launch_rviz:=false
```

## 13. RViz2 Visualisation

The AGHRI visualisation distinguishes the processing stages using consistent colours:

```text
Blue    = YOLO 2D detections
Yellow  = original SECOND or PointPillars 3D detections
Green   = matched camera-supported 3D detections
Red     = unmatched LiDAR 3D detections
Cyan    = complete / canonical ALL3D output
Purple  = DeepFusionMOT tracked 3D boxes with persistent IDs
```

The main fixed frame is:

```text
front_lidar_link
```

## 14. Topic Reference

### Sensor and calibration inputs

| ROS 2 topic | Purpose |
| --- | --- |
| `/dataset/cam_zed_rgb/image` | ZED RGB image input |
| `/dataset/cam_zed_rgb/camera_info` | ZED projection matrix and camera metadata |
| `/dataset/lidar/points` | Livox point-cloud input |
| `/tf` | Dynamic `map -> base_link` motion |
| `/tf_static` | Static transforms between the robot sensors |
| `/clock` | ROS time during bag playback |

### Detector outputs

| ROS 2 topic | Purpose |
| --- | --- |
| `/detector/camera/detections` | Live YOLO11s 2D person detections |
| `/detector/lidar/detections` | Live SECOND or PointPillars 3D person detections |

### Late-fusion outputs

| ROS 2 topic | Purpose |
| --- | --- |
| `/late_fusion/matched_3d` | Camera-supported LiDAR 3D detections |
| `/late_fusion/unmatched_lidar` | LiDAR detections without a camera match |
| `/late_fusion/unmatched_camera` | Camera detections without a LiDAR match |
| `/late_fusion/fused_3d` | Complete canonical 3D output used by the framework |

### Tracking outputs

| ROS 2 topic | Purpose |
| --- | --- |
| `/deepfusionmot/tracked_3d` | Final tracked 3D person cuboids with persistent IDs |
| `/deepfusionmot/tracked_2d` | Auxiliary camera-space tracks |

Ground-truth annotations are used by the offline evaluators only and are not required by the live fusion and tracking nodes.

## 15. Tracking Metric Meanings

Abbreviations used in the equations: `GT` is the number of ground-truth objects; `TP`, `FP`, and `FN` are true positives, false positives, and false negatives; `IDTP`, `IDFP`, and `IDFN` are identity true positives, identity false positives, and identity false negatives.

| Metric | Meaning | Equation or interpretation |
| --- | --- | --- |
| HOTA (Higher Order Tracking Accuracy) | Overall tracking quality | Approximately balances detection and association: `HOTA ~= sqrt(DetA x AssA)` |
| DetA (Detection Accuracy) | Detection accuracy within tracking | `DetA = TP / (TP + FP + FN)` |
| AssA (Association Accuracy) | Association accuracy | Measures whether detections retain the correct identity over time |
| IDSW (Identity Switches) | Identity switches | Number of times one ground-truth person changes predicted track ID |
| MOTP (Multi-Object Tracking Precision) | Localisation precision | `MOTP = sum of matched IoU values / number of matches` |
| MOTA (Multi-Object Tracking Accuracy) | CLEAR tracking accuracy | `MOTA = 1 - (FN + FP + IDSW) / GT` |
| IDF1 (Identity F1 score) | Identity-tracking F1 score | `IDF1 = 2 x IDTP / (2 x IDTP + IDFP + IDFN)` |
| FPS (Frames Per Second) | ROS 2 live playback throughput | `FPS = tracked output frames / measured playback time` |

## 16. Repository Layout

```text
late_fusion/
  late_fusion_pkg/       ROS 2 nodes, fusion, tracking, adapters, and vendored TrackEval
  launch/                AGHRI live and replay launch files
  config/                Detector, fusion, tracking, and visualisation configurations
  rviz/                  AGHRI RViz configurations
  tools/                 Dataset preparation, evaluation, tracking, and reporting tools
  scripts/               Environment and cluster helper scripts
  checkpoints/           AGHRI checkpoints and checkpoint manifests
  results/               Final tracking metrics, notebook assets, visual examples, and ROS 2 FPS results
  data_manifests/        AGHRI train, validation, and test manifests
  notebooks/             Paper figures and result summaries
```

## Results Notebook

The late-fusion paper-assets notebook is stored under:

```text
notebooks/
```

It contains the method summary, final eight-combination table, figures, qualitative examples, and interpretation of the AGHRI results.
