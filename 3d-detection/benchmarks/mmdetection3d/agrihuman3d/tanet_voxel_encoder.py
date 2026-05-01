from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from mmdet3d.models.voxel_encoders.utils import PFNLayer, get_paddings_indicator
from mmdet3d.registry import MODELS


class _PointAttention(nn.Module):
    def __init__(self, max_num_points: int, reduction_ratio: int) -> None:
        super().__init__()
        hidden = max(1, max_num_points // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(max_num_points, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, max_num_points),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.max(x, dim=2)[0]
        return self.fc(pooled).unsqueeze(-1)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.max(x, dim=1)[0]
        return self.fc(pooled).unsqueeze(1)


class _PointChannelAttention(nn.Module):
    def __init__(self, channels: int, max_num_points: int, reduction_ratio: int) -> None:
        super().__init__()
        self.point_attention = _PointAttention(max_num_points, reduction_ratio)
        self.channel_attention = _ChannelAttention(channels, reduction_ratio)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        weights = self.activation(self.point_attention(x) * self.channel_attention(x))
        return x * weights, weights


class _VoxelAttention(nn.Module):
    def __init__(self, channels: int, max_num_points: int) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(channels + 3, 1),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(max_num_points, 1),
            nn.ReLU(inplace=True),
        )
        self.activation = nn.Sigmoid()

    def forward(self, voxel_center: Tensor, features: Tensor) -> Tensor:
        repeated_center = voxel_center.repeat(1, features.shape[1], 1)
        merged = torch.cat([features, repeated_center], dim=-1)
        voxel_scores = self.fc1(merged).permute(0, 2, 1).contiguous()
        return self.activation(self.fc2(voxel_scores))


class _TripleAttentionBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        attention_channels: int,
        max_num_points: int,
        reduction_ratio: int,
        use_paca_weight: bool,
    ) -> None:
        super().__init__()
        self.paca1 = _PointChannelAttention(input_channels, max_num_points, reduction_ratio)
        self.paca2 = _PointChannelAttention(attention_channels, max_num_points, reduction_ratio)
        self.voxel_attention1 = _VoxelAttention(input_channels, max_num_points)
        self.voxel_attention2 = _VoxelAttention(attention_channels, max_num_points)
        self.use_paca_weight = use_paca_weight
        self.fc1 = nn.Sequential(
            nn.Linear(input_channels * 2, attention_channels),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(attention_channels, attention_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, voxel_center: Tensor, features: Tensor) -> Tensor:
        paca1, paca_weight1 = self.paca1(features)
        voxel_weight1 = self.voxel_attention1(voxel_center, paca1)
        if self.use_paca_weight:
            paca1 = paca1 * paca_weight1
        stage1 = torch.cat([voxel_weight1 * paca1, features], dim=-1)
        stage1 = self.fc1(stage1)

        paca2, paca_weight2 = self.paca2(stage1)
        voxel_weight2 = self.voxel_attention2(voxel_center, paca2)
        if self.use_paca_weight:
            paca2 = paca2 * paca_weight2
        return self.fc2(stage1 + voxel_weight2 * paca2)


@MODELS.register_module()
class TripleAttentionPillarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = 4,
        feat_channels: Optional[tuple] = (64, ),
        with_distance: Optional[bool] = False,
        with_cluster_center: Optional[bool] = True,
        with_voxel_center: Optional[bool] = True,
        voxel_size: Optional[Tuple[float, float, float]] = (0.2, 0.2, 4),
        point_cloud_range: Optional[Tuple[float, float, float, float, float, float]] = (
            0,
            -40,
            -3,
            70.4,
            40,
            1,
        ),
        norm_cfg: Optional[dict] = dict(type='BN1d', eps=1e-3, momentum=0.01),
        mode: Optional[str] = 'max',
        legacy: Optional[bool] = True,
        max_num_points: int = 32,
        attention_channels: int = 64,
        reduction_ratio: int = 8,
        use_paca_weight: bool = True,
    ) -> None:
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center

        decorated_channels = in_channels
        if with_cluster_center:
            decorated_channels += 3
        if with_voxel_center:
            decorated_channels += 3
        if with_distance:
            decorated_channels += 1

        self.triple_attention = _TripleAttentionBlock(
            input_channels=decorated_channels,
            attention_channels=attention_channels,
            max_num_points=max_num_points,
            reduction_ratio=reduction_ratio,
            use_paca_weight=use_paca_weight,
        )

        pfn_channels = [attention_channels] + list(feat_channels)
        self.pfn_layers = nn.ModuleList()
        for index in range(len(pfn_channels) - 1):
            self.pfn_layers.append(
                PFNLayer(
                    pfn_channels[index],
                    pfn_channels[index + 1],
                    norm_cfg=norm_cfg,
                    last_layer=index == len(pfn_channels) - 2,
                    mode=mode,
                ))

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

    def forward(self, features: Tensor, num_points: Tensor, coors: Tensor, *args, **kwargs) -> Tensor:
        features_ls = [features]

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(
            -1, 1, 1)
        if self._with_cluster_center:
            features_ls.append(features[:, :, :3] - points_mean)

        if self._with_voxel_center:
            dtype = features.dtype
            f_center = torch.zeros_like(features[:, :, :3])
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 1].to(dtype).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = mask.unsqueeze(-1).type_as(features)
        features = features * mask

        features = self.triple_attention(points_mean, features) * mask
        for pfn in self.pfn_layers:
            features = pfn(features, num_points)
        return features.squeeze(1)
