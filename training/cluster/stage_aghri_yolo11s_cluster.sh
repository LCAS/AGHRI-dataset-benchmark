#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble"
LOCAL_DATASET="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset"
LOCAL_GENERIC_CKPT="/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/yolo11s.pt"
CLUSTER_ROOT="${1:-/home/prabuddhi/cluster/aghri_yolo11s_finetune}"

echo "[INFO] Staging AGHRI YOLO11s payload to ${CLUSTER_ROOT}"
echo "[INFO] Dataset copy dereferences symlinked images; this is intentionally larger but cluster-portable."

mkdir -p \
  "${CLUSTER_ROOT}/camera_lidar_humble/training" \
  "${CLUSTER_ROOT}/aghri_zedrgb_yolo_dataset" \
  "${CLUSTER_ROOT}/checkpoints/generic" \
  "${CLUSTER_ROOT}/logs" \
  "${CLUSTER_ROOT}/training_outputs/aghri_yolo11s_finetune"

cp -R "${REPO_ROOT}/training/." "${CLUSTER_ROOT}/camera_lidar_humble/training/"
cp -f "${REPO_ROOT}/training/cluster/aghri_yolo11s_train.slurm" "${CLUSTER_ROOT}/aghri_yolo11s_train.slurm"
cp -f "${LOCAL_GENERIC_CKPT}" "${CLUSTER_ROOT}/checkpoints/generic/yolo11s.pt"

cp -RLn "${LOCAL_DATASET}/." "${CLUSTER_ROOT}/aghri_zedrgb_yolo_dataset/"

cat > "${CLUSTER_ROOT}/aghri_zedrgb_yolo_dataset/data.yaml" <<EOF
path: /work/users/pwariyapperuma/aghri_yolo11s_finetune/aghri_zedrgb_yolo_dataset
train: images/train
val: images/val
names:
  0: person
EOF

cat > "${CLUSTER_ROOT}/STAGED_FROM.txt" <<EOF
Created: $(date -Is)
Local repo: ${REPO_ROOT}
Local dataset: ${LOCAL_DATASET}
Local generic checkpoint: ${LOCAL_GENERIC_CKPT}
Cluster local mount: ${CLUSTER_ROOT}
Slurm cluster path: /work/users/pwariyapperuma/aghri_yolo11s_finetune
Dataset copy mode: cp -RLn, symlinks dereferenced and existing files left untouched
EOF

echo "[INFO] Staged payload. Submit from the cluster with:"
echo "  cd /work/users/pwariyapperuma/aghri_yolo11s_finetune"
echo "  sbatch aghri_yolo11s_train.slurm"
