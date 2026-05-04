_base_ = ['./pointpillars_kitti.py']

# Zero-shot cross-dataset evaluation: KITTI-trained PointPillars on AGHRI data.
# AGHRI scans are filtered to the KITTI spatial range [0, 69.12] x [-39.68, 39.68]
# (front hemisphere only) so the source checkpoint loads without any shape mismatch.

_aghri_root = '/workspace/data/aghri-3d'
_kitti_range = [0, -39.68, -1, 69.12, 39.68, 3]

_test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=_kitti_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points'],
        meta_keys=('lidar_path', 'sample_idx', 'box_type_3d', 'box_mode_3d'),
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AghriLidarDataset',
        data_root=_aghri_root,
        ann_file='infos/agri_person_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=_test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['person']),
        box_type_3d='LiDAR',
        backend_args=None,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AghriLidarDataset',
        data_root=_aghri_root,
        ann_file='infos/agri_person_infos_test.pkl',
        data_prefix=dict(pts=''),
        pipeline=_test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['person']),
        box_type_3d='LiDAR',
        backend_args=None,
    ),
)

val_evaluator = dict(
    type='Aghri3DMetric',
    ann_file=_aghri_root + '/infos/agri_person_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
)

test_evaluator = dict(
    type='Aghri3DMetric',
    ann_file=_aghri_root + '/infos/agri_person_infos_test.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
)
