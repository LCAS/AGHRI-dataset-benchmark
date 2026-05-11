#!/bin/bash
#SBATCH --job-name=aghri3d_prep
#SBATCH --output=/home/users/%u/logs/aghri3d_prep_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=short
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --container-image=/home/users/hvu/mmdet_frcnn_dcn/mmdet_pytorch_cu118.sqfs
#SBATCH --container-mounts=/home/users/hvu:/workspace
#SBATCH --container-workdir=/workspace/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d

module purge

export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export REPO_ROOT=/workspace/agri-human-dataset-benchmark
export BENCH_ROOT=$REPO_ROOT/3d-detection/benchmarks/mmdetection3d
export VENV_PATH=$BENCH_ROOT/.venv
export CACHE_ROOT=$BENCH_ROOT/.cache
export RAW_DATA_ROOT=${RAW_DATA_ROOT:-/workspace/datasets/agri-human-sensing/labelled_dataset}
export DATA_ROOT=${DATA_ROOT:-/workspace/datasets/agri-human-sensing/mmdet3d_lidar_aghri}

srun bash -c "
    set -e

    mkdir -p \$CACHE_ROOT/pip \$CACHE_ROOT/torch \$CACHE_ROOT/mim \$BENCH_ROOT/.config
    export HOME=/workspace
    export XDG_CACHE_HOME=\$CACHE_ROOT
    export MIM_CACHE_DIR=\$CACHE_ROOT/mim
    export TORCH_HOME=\$CACHE_ROOT/torch

    if [ ! -d \$VENV_PATH ]; then
        python -m venv \$VENV_PATH
        source \$VENV_PATH/bin/activate
        pip install -U pip setuptools wheel
        pip install 'numpy<2' 'opencv-python-headless==4.11.0.86'
        pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
        pip install -U openmim
        mim install mmengine
        pip install 'mmcv==2.1.0' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
        pip install cumm-cu118 spconv-cu118
        pip install -r \$BENCH_ROOT/requirements.txt
    else
        source \$VENV_PATH/bin/activate
    fi

    export AGHRI_3D_DATA_ROOT=\$DATA_ROOT

    python src/prepare_aghri_lidar_dataset.py \
        --raw-root \$RAW_DATA_ROOT \
        --out-root \$DATA_ROOT \
        --workers ${SLURM_CPUS_PER_TASK:-8}
"
