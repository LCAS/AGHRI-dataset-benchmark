"""AGHRI PointPillars fine-tuning config template.

This file mirrors the SECOND fine-tune scaffold but keeps PointPillars in its
own detector family. It is copied to the cluster workspace before training.
"""

import os

_base_ = [
    os.environ.get(
        "AGHRI_POINTPILLARS_BASE_CONFIG",
        "/work/users/pwariyapperuma/aghri_pointpillars_finetune/code/mmdetection3d/configs/models/pointpillars_aghri.py",
    )
]

data_root = os.environ.get(
    "AGHRI_POINTPILLARS_DATA_ROOT",
    "/work/users/pwariyapperuma/aghri_second_finetune/data/aghri_mmdet3d",
)
load_from = os.environ.get(
    "AGHRI_POINTPILLARS_GENERIC_CHECKPOINT",
    "/work/users/pwariyapperuma/aghri_pointpillars_finetune/checkpoints/generic/"
    "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth",
)
work_dir = os.environ.get(
    "AGHRI_POINTPILLARS_WORK_DIR",
    "/work/users/pwariyapperuma/aghri_pointpillars_finetune/outputs/pointpillars_aghri_seed42",
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
