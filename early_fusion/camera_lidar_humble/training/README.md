# AGHRI ZED RGB YOLO11s Fine-Tuning

This directory contains the reproducible local workflow for fine-tuning YOLO11s on AGHRI ZED RGB person detections. It keeps generated images, labels, checkpoints, and logs outside this ROS package.

ZED RGB is used because the camera-LiDAR fusion benchmark consumes the ZED RGB stream. Fisheye images, ZED depth, and LiDAR are deliberately excluded from detector training.

The source dataset is read from:

```bash
/media/prabuddhi/Backup2/Updated Dataset_PW
```

Official recording-level splits are read from:

```bash
/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists
```

The official test recordings are written only to `excluded_test.csv`; they are not used for training, validation, checkpoint selection, threshold selection, or this validation-only report.

## Environment

Use system Python 3.10 with Conda deactivated. Do not install packages for this workflow.

```bash
conda deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
cd /home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble
```

Record environment and GPU metadata:

```bash
bash training/run_aghri_yolo11s_finetune.sh env
```

## Prepare Dataset

The generated YOLO dataset is created outside the repository:

```bash
python3 training/prepare_aghri_zedrgb_yolo.py \
  --dataset-root "/media/prabuddhi/Backup2/Updated Dataset_PW" \
  --split-lists-dir "/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists" \
  --output-root "/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset" \
  --mode symlink \
  --overwrite
```

Dry-run without writing:

```bash
python3 training/prepare_aghri_zedrgb_yolo.py \
  --dataset-root "/media/prabuddhi/Backup2/Updated Dataset_PW" \
  --split-lists-dir "/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists" \
  --output-root "/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset" \
  --dry-run
```

Verify an existing prepared dataset:

```bash
python3 training/verify_aghri_yolo_dataset.py \
  --output-root "/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset" \
  --split-lists-dir "/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists"
```

Expected generated structure:

```text
/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/
  images/train/<recording>/*.png
  images/val/<recording>/*.png
  labels/train/<recording>/*.txt
  labels/val/<recording>/*.txt
  manifests/train.csv
  manifests/val.csv
  manifests/excluded_test.csv
  manifests/preparation_summary.json
  data.yaml
```

Boxes are converted from AGHRI pixel `x,y,width,height` annotations. Partially visible boxes are clipped to image bounds before YOLO normalization. Invalid boxes with non-positive source or clipped extents are rejected and reported. All AGHRI person identities are mapped to YOLO class `0` named `person`.

Visual audit images are written to:

```bash
/home/prabuddhi/Desktop/Early_Fus/training_outputs/aghri_yolo11s_finetune/dataset_audit
```

## Preflight And Smoke Test

Run a one-batch validation preflight on the validation split:

```bash
bash training/run_aghri_yolo11s_finetune.sh preflight
```

Run a one-epoch smoke train/eval using the real train and validation splits:

```bash
bash training/run_aghri_yolo11s_finetune.sh smoke
```

Smoke-test weights are not the final checkpoint.

## Full Fine-Tuning

The full config is:

```bash
/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/training/configs/aghri_zedrgb_yolo11s_finetune.yaml
```

It starts from the verified generic YOLO11s checkpoint:

```bash
/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/yolo11s.pt
```

Run full training:

```bash
bash training/run_aghri_yolo11s_finetune.sh full
```

The shell wrapper uses `training/run_ultralytics_config.py`, which preserves the benchmark config shape but writes outputs outside the read-only benchmark repository. This local runner is used because the mounted benchmark runner expects `eval_results.save_dir`, which is not exposed by the installed Ultralytics 8.3.154 `DetMetrics` object.

Expected final checkpoint:

```bash
/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt
```

Hash it:

```bash
sha256sum /home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt
```

Write checkpoint provenance:

```bash
python3 training/write_checkpoint_manifest.py \
  --best-checkpoint "/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt" \
  --last-checkpoint "/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/last.pt"
```

Resume is intentionally conservative: only resume manually after confirming the run directory metadata still matches the checkpoint, dataset YAML, split manifests, image size, batch, epochs, seed, and Ultralytics version.

## Validation-Only Evaluation

Generic YOLO11s validation context:

```bash
bash training/run_aghri_yolo11s_finetune.sh eval-generic
```

Fine-tuned validation is run by the benchmark runner during full training on `eval_split: val`. Do not evaluate on the official test split in this task.

## Fusion Configuration

The fine-tuned fusion config is:

```bash
/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/config/aghri_zed_livox_yolo11s_finetuned.yaml
```

It must differ from `config/aghri_zed_livox.yaml` only in `YOLO_model`.

Fusion launch example:

```bash
source /home/prabuddhi/Desktop/Early_Fus/install/setup.bash
ros2 launch camera_lidar_fusion launch.py config:=/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/config/aghri_zed_livox_yolo11s_finetuned.yaml
```

The future generic-versus-fine-tuned comparison should keep association method, mean center statistic, bottom trim, confidence threshold, person class, calibration, synchronization, LiDAR processing, official six-recording test split, matching, and failure definitions fixed. Only the detector checkpoint should change.

## Outputs

Dataset:

```bash
/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset
```

Checkpoints:

```bash
/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune
```

Logs and reports:

```bash
/home/prabuddhi/Desktop/Early_Fus/training_outputs/aghri_yolo11s_finetune
```

## Known Limitations

The training scripts rely on the existing local Ultralytics installation and GPU environment. Batch size should remain `16` unless a real CUDA out-of-memory failure requires reducing only the batch size. The official test split is intentionally untouched until the separate final fusion comparison task.

## Cluster Checkpoint Used For Final Fusion Benchmark

The accepted cluster checkpoint is copied locally to:

```bash
/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt
```

Expected SHA256:

```text
bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110
```

Cluster source path:

```text
/work/users/pwariyapperuma/aghri_yolo11s_finetune/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt
```

Local mounted source path:

```text
/home/prabuddhi/cluster/aghri_yolo11s_finetune/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt
```

Verification:

```bash
sha256sum /home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt
```

Fine-tuned fusion configs:

```text
/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/config/aghri_zed_livox_yolo11s_finetuned.yaml
/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble/config/aghri_zed_livox_yolo11s_finetuned_hybrid.yaml
```

Final offline 2x2 benchmark:

```bash
cd /home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble
/usr/bin/python3 tools/run_aghri_detector_fusion_2x2.py
```

Outputs are written outside the Git repository:

```text
/home/prabuddhi/Desktop/Early_Fus/results/aghri_detector_fusion_2x2
```
