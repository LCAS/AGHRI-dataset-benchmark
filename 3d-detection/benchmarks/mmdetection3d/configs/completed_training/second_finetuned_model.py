"""AGHRI SECOND fine-tuning config template.

This file is copied to the cluster run directory by scripts/stage_aghri_second_cluster.sh.
Paths are controlled by environment variables so the same file works from the
local mounted cluster view and inside the container on /work/users.
"""

import os

_base_ = [
    os.environ.get(
        "AGHRI_SECOND_BASE_CONFIG",
        "/work/users/pwariyapperuma/aghri_second_finetune/code/mmdetection3d/configs/models/second_aghri.py",
    )
]

data_root = os.environ.get(
    "AGHRI_SECOND_DATA_ROOT",
    "/work/users/pwariyapperuma/aghri_second_finetune/data/aghri_mmdet3d",
)
load_from = os.environ.get(
    "AGHRI_SECOND_GENERIC_CHECKPOINT",
    "/work/users/pwariyapperuma/aghri_second_finetune/checkpoints/generic/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth",
)
work_dir = os.environ.get(
    "AGHRI_SECOND_WORK_DIR",
    "/work/users/pwariyapperuma/aghri_second_finetune/outputs/second_aghri_seed42",
)

randomness = dict(seed=42, deterministic=False)
env_cfg = dict(cudnn_benchmark=True)

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
val_evaluator = dict(ann_file=data_root + "/infos/agri_person_infos_val.pkl")
test_evaluator = dict(ann_file=data_root + "/infos/agri_person_infos_val.pkl")

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=2,
        max_keep_ckpts=3,
        save_best="aghri/bev_ap_mean",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50),
)
