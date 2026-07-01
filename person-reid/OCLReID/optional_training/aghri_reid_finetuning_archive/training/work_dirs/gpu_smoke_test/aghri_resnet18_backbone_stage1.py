optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=3.0),
            layer3=dict(lr_mult=1.0),
            layer4=dict(lr_mult=1.0),
            layer1=dict(lr_mult=0.0, decay_mult=0.0),
            layer2=dict(lr_mult=0.0, decay_mult=0.0),
            conv1=dict(lr_mult=0.0, decay_mult=0.0),
            bn1=dict(lr_mult=0.0, decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
TRAIN_REID = True
dataset_type = 'ReIDDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqResize',
        img_scale=(192, 256),
        share_params=False,
        keep_ratio=False,
        bbox_clip_border=False,
        override=False),
    dict(
        type='SeqRandomFlip',
        share_params=False,
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='VideoCollect', keys=['img', 'gt_label']),
    dict(type='ReIDFormatBundle')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(192, 256), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=[])
]
model = dict(
    type='BaseReID',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        frozen_stages=2,
        norm_eval=True),
    neck=dict(type='GlobalAveragePooling', kernel_size=(8, 6), stride=1),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=512,
        fc_channels=256,
        out_channels=128,
        num_classes=3,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_pairwise=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    init_cfg=dict(
        type='Pretrained', checkpoint='checkpoints/reid/resnet18.pth'))
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ReIDDataset',
        triplet_sampler=dict(num_ids=3, ins_per_id=4),
        data_prefix='',
        ann_file='aghri_reid_stage1/manifests/train_reid.txt',
        pipeline=[
            dict(type='LoadMultiImagesFromFile', to_float32=True),
            dict(
                type='SeqResize',
                img_scale=(192, 256),
                share_params=False,
                keep_ratio=False,
                bbox_clip_border=False,
                override=False),
            dict(
                type='SeqRandomFlip',
                share_params=False,
                flip_ratio=0.5,
                direction='horizontal'),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='VideoCollect', keys=['img', 'gt_label']),
            dict(type='ReIDFormatBundle')
        ]),
    val=dict(
        type='ReIDDataset',
        triplet_sampler=None,
        data_prefix='',
        ann_file='aghri_reid_stage1/manifests/val_reid.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(192, 256), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ]),
    test=dict(
        type='ReIDDataset',
        triplet_sampler=None,
        data_prefix='',
        ann_file='aghri_reid_stage1/manifests/val_reid.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(192, 256), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=[])
        ]))
evaluation = dict(interval=1, metric=['mAP', 'CMC'])
lr_config = dict(policy='step', step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=1)
total_epochs = 1
work_dir = 'aghri_reid_stage1/training/work_dirs/gpu_smoke_test'
seed = 42
gpu_ids = [0]
device = 'cuda'
