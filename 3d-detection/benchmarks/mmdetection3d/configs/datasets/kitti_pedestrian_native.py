# Native KITTI Pedestrian dataset configuration.
#
# Uses the OFFICIAL MMDetection3D KittiDataset and KittiMetric instead of the
# custom AghriLidarDataset + Kitti3DMetric used for the AGHRI pipeline.
#
# Prerequisites:
#   Run prepare_kitti_native_dataset.py to create the official pkl files:
#     kitti_infos_train.pkl, kitti_infos_val.pkl, kitti_dbinfos_train.pkl
#
# The official KittiDataset:
#   - Stores 3D boxes in CAMERA frame (not LiDAR) in the pkl.
#   - Converts camera → LiDAR internally with correct z_bottom convention.
#   - Avoids the z_center/z_bottom mismatch that collapsed performance.
#
# The official KittiMetric:
#   - Uses KITTI R40 (40-point interpolation) and R11 evaluation.
#   - Returns difficulty-based AP: easy / moderate / hard.
#   - IoU threshold = 0.5 for Pedestrian (KITTI standard).

dataset_type = 'KittiDataset'
data_root    = '/workspace/data/kitti-3d/'
class_names  = ['Pedestrian']
metainfo     = dict(classes=class_names)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# Standard KITTI forward-facing LiDAR range.
point_cloud_range = [0, -39.68, -1, 69.12, 39.68, 3]

# Ground-truth sampling augmentation — requires kitti_dbinfos_train.pkl.
# Paste 15 Pedestrian instances per frame to balance the rare class.
db_sampler = dict(
    type='DataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Pedestrian=5),
    ),
    classes=class_names,
    sample_groups=dict(Pedestrian=15),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    backend_args=backend_args,
)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',  point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter',   classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
    ),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=(
            'lidar_path', 'sample_idx', 'box_mode_3d', 'box_type_3d',
            'num_pts_feats', 'ori_shape',
        ),
    ),
]

eval_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_train.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        filter_empty_gt=True,   # discard frames with no pedestrians
        box_type_3d='LiDAR',
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_infos_val.pkl',
        data_prefix=dict(pts='training/velodyne'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

# Official KITTI evaluator — returns difficulty-based AP (easy/moderate/hard)
# using R40 (40-point) interpolation with IoU threshold = 0.5 for Pedestrian.
val_evaluator = dict(
    type='KittiMetric',
    ann_file=data_root + 'kitti_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args,
)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
