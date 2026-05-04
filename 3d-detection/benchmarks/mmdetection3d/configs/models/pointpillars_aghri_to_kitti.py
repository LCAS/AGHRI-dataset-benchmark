_base_ = ['./pointpillars_aghri.py']

# Zero-shot cross-dataset evaluation: AGHRI-trained PointPillars on KITTI data.
# KITTI scans are filtered to the AGHRI spatial range [-24, 30.4] x [-28, 28].
# Since KITTI is forward-facing (x >= 0), the effective front range is [0, 30.4] x [-28, 28].
# KITTI test split lacks ground-truth labels, so val is used for both val and test.

_kitti_root = '/workspace/data/kitti-3d'
_aghri_range = [-24.0, -28.0, -1.0, 30.4, 28.0, 15.0]

_test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=_aghri_range),
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
        data_root=_kitti_root,
        ann_file='infos/kitti_person_infos_val.pkl',
        data_prefix=dict(pts=''),
        pipeline=_test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        test_mode=True,
        metainfo=dict(classes=['person']),
        box_type_3d='LiDAR',
        backend_args=None,
    ),
)

# KITTI test split has no annotations — val is used for final evaluation.
test_dataloader = val_dataloader

val_evaluator = dict(
    type='Kitti3DMetric',
    ann_file=_kitti_root + '/infos/kitti_person_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
)

test_evaluator = val_evaluator
