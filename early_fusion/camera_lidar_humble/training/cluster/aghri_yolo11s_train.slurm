#!/bin/bash
#SBATCH --job-name=aghri_yolo11s
#SBATCH --output=/work/users/pwariyapperuma/aghri_yolo11s_finetune/logs/aghri_yolo11s_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_a100:1
#SBATCH --constraint=nvidia_a100
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/work/users/pwariyapperuma/aghri_yolo11s_finetune/camera_lidar_humble

set -euo pipefail

CLUSTER_ROOT=/work/users/pwariyapperuma/aghri_yolo11s_finetune
REPO_ROOT=${CLUSTER_ROOT}/camera_lidar_humble
DATASET_ROOT=${CLUSTER_ROOT}/aghri_zedrgb_yolo_dataset
GENERIC_CKPT=${CLUSTER_ROOT}/checkpoints/generic/yolo11s.pt
CHECKPOINT_ROOT=${CLUSTER_ROOT}/checkpoints/aghri_yolo11s_finetune
OUTPUT_ROOT=${CLUSTER_ROOT}/training_outputs/aghri_yolo11s_finetune
CONFIG_PATH=${OUTPUT_ROOT}/configs/aghri_zedrgb_yolo11s_finetune.cluster.yaml
CONTAINER_IMAGE=/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs

mkdir -p "${CLUSTER_ROOT}/logs" "${CHECKPOINT_ROOT}" "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/summaries"

module load Pyxis Enroot >/dev/null 2>&1 || echo "[WARN] Pyxis/Enroot modulefiles not found; continuing because Slurm container options may already be enabled."

echo "[HOST] $(hostname)"
echo "[HOST] Cluster root: ${CLUSTER_ROOT}"
echo "[HOST] Container: ${CONTAINER_IMAGE}"

srun --ntasks=1 --gpus=1 --gpu-bind=closest --mpi=none \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/work/users:/work/users,/dev/shm:/dev/shm \
  --container-workdir="${REPO_ROOT}" \
  bash -lc '
    set -euo pipefail

    CLUSTER_ROOT=/work/users/pwariyapperuma/aghri_yolo11s_finetune
    REPO_ROOT=${CLUSTER_ROOT}/camera_lidar_humble
    DATASET_ROOT=${CLUSTER_ROOT}/aghri_zedrgb_yolo_dataset
    GENERIC_CKPT=${CLUSTER_ROOT}/checkpoints/generic/yolo11s.pt
    CHECKPOINT_ROOT=${CLUSTER_ROOT}/checkpoints/aghri_yolo11s_finetune
    OUTPUT_ROOT=${CLUSTER_ROOT}/training_outputs/aghri_yolo11s_finetune
    CONFIG_PATH=${OUTPUT_ROOT}/configs/aghri_zedrgb_yolo11s_finetune.cluster.yaml
    VENV_DIR=${CLUSTER_ROOT}/venvs/yolo11s_ultralytics

    cd "${REPO_ROOT}"
    mkdir -p "${OUTPUT_ROOT}/configs" "${OUTPUT_ROOT}/summaries" "${CHECKPOINT_ROOT}" "${CLUSTER_ROOT}/venvs"

    echo "[IN-CTR] Host: $(hostname)"
    echo "[IN-CTR] PWD:  $(pwd)"
    echo "[IN-CTR] Python before venv: $(python -V 2>&1 || true)"
    nvidia-smi || true

    cat > "${DATASET_ROOT}/data.yaml" <<YAML
path: ${DATASET_ROOT}
train: images/train
val: images/val
names:
  0: person
YAML

    if [ ! -f "${GENERIC_CKPT}" ]; then
      echo "[ERROR] Missing generic YOLO11s checkpoint: ${GENERIC_CKPT}" >&2
      exit 2
    fi
    if [ ! -d "${DATASET_ROOT}/images/train" ] || [ ! -d "${DATASET_ROOT}/images/val" ]; then
      echo "[ERROR] Missing staged train/val image directories under ${DATASET_ROOT}" >&2
      exit 2
    fi

    PYTHON_BIN=$(command -v python3 || command -v python)
    REBUILD_VENV=0
    if [ ! -x "${VENV_DIR}/bin/python" ]; then
      REBUILD_VENV=1
    elif grep -q "include-system-site-packages = true" "${VENV_DIR}/pyvenv.cfg"; then
      echo "[IN-CTR] Existing venv inherits old container packages; rebuilding it."
      REBUILD_VENV=1
    fi
    if [ "${REBUILD_VENV}" = "1" ]; then
      rm -rf "${VENV_DIR}"
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    source "${VENV_DIR}/bin/activate"
    export PIP_CACHE_DIR="${CLUSTER_ROOT}/.cache/pip"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    mkdir -p "${PIP_CACHE_DIR}"
    python -m pip install --upgrade pip setuptools wheel
    python - <<PY || {
import torch, ultralytics, yaml
from packaging.version import Version
assert Version(str(torch.__version__).split("+")[0]) >= Version("2.0.0"), torch.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY
      python -m pip install --upgrade --extra-index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.4.1+cu121" "torchvision==0.19.1+cu121"
      python -m pip install --upgrade "ultralytics==8.3.154" pyyaml packaging
    }
    python - <<PY
import torch, ultralytics
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY

    export HOME=${CLUSTER_ROOT}
    export YOLO_CONFIG_DIR=${CLUSTER_ROOT}/.config/Ultralytics
    export OMP_NUM_THREADS=8
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    cat > "${CONFIG_PATH}" <<YAML
mode: train_eval
train_dataset: ${DATASET_ROOT}/data.yaml
eval_dataset: ${DATASET_ROOT}/data.yaml
imgsz: 640
epochs: 50
batch: 16
device: 0
seed: 42
eval_split: val
rect: false
half: false
dnn: false
workers: 8
project: ${CHECKPOINT_ROOT}
name: yolo11s_aghri_zedrgb_seed42
models:
- name: yolo11s
  source: ${GENERIC_CKPT}
notes: AGHRI ZED RGB YOLO11s cluster fine-tuning; official train and validation recordings only.
YAML

    python training/run_ultralytics_config.py \
      --config "${CONFIG_PATH}" \
      --out "${OUTPUT_ROOT}/summaries/full_summary.csv"

    BEST=${CHECKPOINT_ROOT}/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt
    LAST=${CHECKPOINT_ROOT}/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/last.pt
    python training/write_checkpoint_manifest.py \
      --best-checkpoint "${BEST}" \
      --last-checkpoint "${LAST}" \
      --generic-checkpoint "${GENERIC_CKPT}" \
      --dataset-yaml "${DATASET_ROOT}/data.yaml" \
      --train-manifest "${DATASET_ROOT}/manifests/train.csv" \
      --val-manifest "${DATASET_ROOT}/manifests/val.csv" \
      --output "${CHECKPOINT_ROOT}/checkpoint_manifest.json" \
      --notes "Cluster Slurm fine-tune on AGHRI ZED RGB train/val only."
  '

echo "[HOST] AGHRI YOLO11s Slurm job finished."
