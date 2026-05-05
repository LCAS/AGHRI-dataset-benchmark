_base_ = [
    'mmdet3d::_base_/models/second_hv_secfpn_kitti.py',
    'mmdet3d::_base_/default_runtime.py',
]

custom_imports = dict(imports=['aghri3d'], allow_failed_imports=False)

# Zero-shot evaluation: public KITTI 3-class SECOND pretrained weights on AGHRI.
# No model overrides — architecture must match the pretrained checkpoint exactly so all
# weights (SparseEncoder backbone, SECONDFPN neck, anchor head) load without any shape
# mismatch. AGHRI scans are filtered to the KITTI spatial range so the voxelizer is
# correct. Pedestrian is class index 1 in the pretrained model's ['Car', 'Pedestrian',
# 'Cyclist'].

_aghri_root = '/workspace/data/aghri-3d'
_kitti_range = [0, -39.68, -3, 69.12, 39.68, 1]

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
    pred_class_id=1,
)

test_evaluator = dict(
    type='Aghri3DMetric',
    ann_file=_aghri_root + '/infos/agri_person_infos_test.pkl',
    iou_thresholds=[0.25, 0.5, 0.75],
    score_thr=0.0,
    pred_class_id=1,
)

val_cfg = dict()
test_cfg = dict()

load_from = None
