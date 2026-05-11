# PointPillars for KITTI Pedestrian — official MMDetection3D configuration.
#
# Targets the published mmdet3d result for the Pedestrian class:
#   3D AP R40:  Easy ≈ 32.10 %  |  Moderate ≈ 24.79 %  |  Hard ≈ 22.73 %
#   BEV AP R40: Easy ≈ 52.06 %  |  Moderate ≈ 46.54 %  |  Hard ≈ 44.11 %
#
# Key differences from the current pointpillars_kitti.py (AGHRI pipeline):
#   • Uses KittiDataset (official) instead of AghriLidarDataset (custom).
#   • Anchor size matches mean Pedestrian dimensions: [l=0.84, w=0.69, h=1.77].
#   • Anchor bottom height -0.6 m (unchanged, matches KITTI ground clearance).
#   • Training includes GT-sampling augmentation (ObjectSample) via the
#     kitti_pedestrian_native.py dataset config.
#   • 160 epochs with the standard 1-cycle cosine LR schedule.
#
# NOTE on batch size:
#   The published model was trained with 8 GPUs × 6 = 48 total batch.
#   Single-GPU training with batch_size=6 (from the dataset config) and the
#   same absolute LR gives equivalent per-sample gradient scaling and should
#   reproduce the results to within a few AP points.  Reduce batch_size in
#   the dataset config if GPU memory is tight; adjust lr proportionally.

_base_ = [
    'mmdet3d::_base_/models/pointpillars_hv_secfpn_kitti.py',
    '../datasets/kitti_pedestrian_native.py',
    'mmdet3d::_base_/default_runtime.py',
]

point_cloud_range = [0, -39.68, -1, 69.12, 39.68, 3]
voxel_size        = [0.16, 0.16, 4.0]

# Anchor size [l, w, h] and bottom-centre height for KITTI Pedestrian.
# Values match the mean bounding-box dimensions of the Pedestrian class
# in the KITTI training set as used by the official mmdet3d configs.
anchor_size          = [0.84, 0.69, 1.77]
anchor_bottom_height = -0.6

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
        ),
    ),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[496, 432],
    ),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[
                point_cloud_range[0], point_cloud_range[1], anchor_bottom_height,
                point_cloud_range[3], point_cloud_range[4], anchor_bottom_height,
            ]],
            sizes=[anchor_size],
            rotations=[0, 1.5707963],
            reshape_out=False,
        ),
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=100,
    ),
)

# ---------------------------------------------------------------------------
# Training schedule — 160 epochs, 1-cycle cosine LR
# ---------------------------------------------------------------------------

lr        = 0.001
epoch_num = 160

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg   = dict()
test_cfg  = dict()

load_from = None
