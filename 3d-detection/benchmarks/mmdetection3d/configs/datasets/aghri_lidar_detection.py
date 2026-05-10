custom_imports = dict(imports=['aghri3d'], allow_failed_imports=False)

dataset_type = 'AghriLidarDataset'
data_root = 'D:/AOC/datasets/agri-human-sensing/mmdet3d_lidar_aghri'
class_names = ['person']
metainfo = dict(classes=class_names)
point_cloud_range = [-10.24, -10.24, -2.0, 25.6, 10.24, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.39269908, 0.39269908],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.0, 0.0, 0.0],
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('lidar_path', 'sample_idx', 'box_type_3d', 'box_mode_3d'),
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
        meta_keys=('lidar_path', 'sample_idx', 'box_type_3d', 'box_mode_3d'),
    ),
]

eval_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/agri_person_infos_train.pkl',
        data_prefix=dict(pts=''),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
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
        ann_file='infos/agri_person_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='infos/agri_person_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type='Aghri3DMetric',
    ann_file=data_root + '/infos/agri_person_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
)

test_evaluator = dict(
    type='Aghri3DMetric',
    ann_file=data_root + '/infos/agri_person_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
