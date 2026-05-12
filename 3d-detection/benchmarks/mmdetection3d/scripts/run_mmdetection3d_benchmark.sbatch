#!/bin/bash
#SBATCH --job-name=mmdet3d_bench
#SBATCH --output=/home/users/%u/logs/bench_mmdetection3d_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --container-image=/home/users/hvu/mmdet_frcnn_dcn/mmdet_pytorch_cu118.sqfs
#SBATCH --container-mounts=/home/users/hvu:/workspace
#SBATCH --container-workdir=/workspace/agri-human-dataset-benchmark/3d-detection/benchmarks/mmdetection3d

module purge

export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29521
export REPO_ROOT=/workspace/agri-human-dataset-benchmark
export BENCH_ROOT=$REPO_ROOT/3d-detection/benchmarks/mmdetection3d
export VENV_PATH=$BENCH_ROOT/.venv
export CACHE_ROOT=$BENCH_ROOT/.cache
export OUT_DIR=$REPO_ROOT/3d-detection/reports/benchmarks/summary/mmdetection3d
export RAW_DATA_ROOT=${RAW_DATA_ROOT:-/workspace/datasets/agri-human-sensing/labelled_dataset}
export DATA_ROOT=${DATA_ROOT:-/workspace/data/mmdet3d_lidar_aghri} # TODO: Change default data root as needed
export PREPARE_DATASET=${PREPARE_DATASET:-0}
# TODO: Change default config and output file names as needed
export CONFIG_FILE=${CONFIG_FILE:-benchmark_mmdetection3d_kitti_pretrained_aghri_second.yaml}
export OUT_FILE=${OUT_FILE:-summary_mmdetection3d_kitti_pretrained_aghri_second.csv}

srun bash -c "
    set -e

    mkdir -p \$OUT_DIR \$CACHE_ROOT/pip \$CACHE_ROOT/torch \$CACHE_ROOT/mim \$BENCH_ROOT/.config
    export HOME=/workspace
    export XDG_CACHE_HOME=\$CACHE_ROOT
    export MIM_CACHE_DIR=\$CACHE_ROOT/mim
    export TORCH_HOME=\$CACHE_ROOT/torch

    echo '--- Virtual Environment Management ---'
    if [ ! -d \$VENV_PATH ]; then
        python -m venv \$VENV_PATH
        source \$VENV_PATH/bin/activate
        pip install -U pip setuptools wheel
        pip install 'numpy<2' 'opencv-python-headless==4.11.0.86'
        pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
        pip install -U openmim
        pip install 'mmcv==2.1.0' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
        pip install -r \$BENCH_ROOT/requirements.txt
        pip install certifi
    else
        source \$VENV_PATH/bin/activate
    fi

    # Check numpy version and fix if needed
    _NP_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'missing')
    if [[ \"\$_NP_VER\" == 2.* ]] || [[ \"\$_NP_VER\" == \"missing\" ]]; then
        echo \"Fixing/Installing numpy - current: \$_NP_VER\"
        pip install 'numpy<2'
    fi

    # Check opencv and fix if needed
    if ! python -c 'import cv2' 2>/dev/null; then
        echo \"Installing opencv-python-headless...\"
        # Uninstall both to prevent conflicts between 'headless' and standard 'opencv-python'
        pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
        pip install \"opencv-python-headless==4.11.0.86\"
    fi

    echo '--- Verification ---'
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
    
    python -c 'import torch, mmdet; print(f\"PyTorch: {torch.__version__} | MMDetection: {mmdet.__version__}\")'

    export AGHRI_3D_DATA_ROOT=\$DATA_ROOT

    if [ \"\$PREPARE_DATASET\" = \"1\" ]; then
        python src/prepare_aghri_lidar_dataset.py \
            --raw-root \$RAW_DATA_ROOT \
            --out-root \$DATA_ROOT \
            --workers ${SLURM_CPUS_PER_TASK:-8}
    fi

    python src/run_benchmark.py \
        --config \$BENCH_ROOT/configs/\$CONFIG_FILE \
        --out \$OUT_DIR/\$OUT_FILE
"
