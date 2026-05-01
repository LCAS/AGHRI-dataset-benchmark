_base_ = [
    'mmdet3d::pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py',
    '../datasets/agrihuman_lidar_detection.py',
]

point_cloud_range = [-24.0, -28.0, -1.0, 30.0, 28.0, 15.0]
voxel_size = [0.2, 0.2, 16.0]
anchor_bottom_height = 0.34
anchor_size = [0.46, 0.57, 1.50]

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(12000, 16000),
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
        output_shape=[280, 270],
    ),
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[
                point_cloud_range[0],
                point_cloud_range[1],
                anchor_bottom_height,
                point_cloud_range[3],
                point_cloud_range[4],
                anchor_bottom_height,
            ]],
            sizes=[anchor_size],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
    ),
    train_cfg=dict(
        _delete_=True,
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
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
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=100,
    ),
)

load_from = None
