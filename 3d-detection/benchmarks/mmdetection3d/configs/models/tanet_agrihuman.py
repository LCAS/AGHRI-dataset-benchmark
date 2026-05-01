_base_ = ['./pointpillars_agrihuman.py']

custom_imports = dict(imports=['agrihuman3d'], allow_failed_imports=False)

model = dict(
    voxel_encoder=dict(
        type='TripleAttentionPillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[0.2, 0.2, 16.0],
        point_cloud_range=[-24.0, -28.0, -1.0, 30.0, 28.0, 15.0],
        max_num_points=32,
        attention_channels=64,
        reduction_ratio=8,
        use_paca_weight=True,
    ),
)

load_from = None
