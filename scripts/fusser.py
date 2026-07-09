# Copyright 2025 LCAS GROUP, University of Lincoln
#
# Licensed under the GLWTS  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://github.com/me-shaon/GLWTPL/tree/master
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fusser: Camera-LiDAR Early Fusion Module

This module implements the early fusion of 2D object detections (YOLO)
with 3D LiDAR points, allowing the assignment of 3D coordinates to detected objects.

Author: LCAS GROUP, University of Lincoln
License: GLWTS
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


ASSOCIATION_METHODS = {
    'legacy_box',
    'depth_cluster_nearest',
    'depth_cluster_densest',
    'hybrid_depth_legacy',
}
CENTRE_STATISTICS = {'mean', 'median'}


def validate_association_config(
    association_method,
    centre_statistic,
    depth_cluster_gap_m,
    depth_cluster_min_points,
    box_bottom_trim_ratio,
    hybrid_min_cluster_points=5,
    hybrid_min_support_ratio=0.35,
    hybrid_max_depth_spread_m=0.35,
    hybrid_min_cluster_separation_m=0.20,
    hybrid_fallback_on_single_cluster=True,
    hybrid_fallback_on_sparse_candidates=True,
):
    """Validate association parameters shared by runtime and offline tools."""
    if association_method not in ASSOCIATION_METHODS:
        raise ValueError(f"Unsupported association_method: {association_method}")
    if centre_statistic not in CENTRE_STATISTICS:
        raise ValueError(f"Unsupported centre_statistic: {centre_statistic}")

    depth_cluster_gap_m = float(depth_cluster_gap_m)
    depth_cluster_min_points = int(depth_cluster_min_points)
    box_bottom_trim_ratio = float(box_bottom_trim_ratio)
    hybrid_min_cluster_points = int(hybrid_min_cluster_points)
    hybrid_min_support_ratio = float(hybrid_min_support_ratio)
    hybrid_max_depth_spread_m = float(hybrid_max_depth_spread_m)
    hybrid_min_cluster_separation_m = float(hybrid_min_cluster_separation_m)
    hybrid_fallback_on_single_cluster = bool(hybrid_fallback_on_single_cluster)
    hybrid_fallback_on_sparse_candidates = bool(
        hybrid_fallback_on_sparse_candidates,
    )

    if depth_cluster_gap_m <= 0.0:
        raise ValueError('depth_cluster_gap_m must be positive')
    if depth_cluster_min_points <= 0:
        raise ValueError('depth_cluster_min_points must be positive')
    if not 0.0 <= box_bottom_trim_ratio <= 0.5:
        raise ValueError('box_bottom_trim_ratio must be in [0.0, 0.5]')
    if hybrid_min_cluster_points <= 0:
        raise ValueError('hybrid_min_cluster_points must be positive')
    if not 0.0 <= hybrid_min_support_ratio <= 1.0:
        raise ValueError('hybrid_min_support_ratio must be in [0.0, 1.0]')
    if hybrid_max_depth_spread_m <= 0.0:
        raise ValueError('hybrid_max_depth_spread_m must be positive')
    if hybrid_min_cluster_separation_m < 0.0:
        raise ValueError('hybrid_min_cluster_separation_m must be non-negative')

    return (
        association_method,
        centre_statistic,
        depth_cluster_gap_m,
        depth_cluster_min_points,
        box_bottom_trim_ratio,
        hybrid_min_cluster_points,
        hybrid_min_support_ratio,
        hybrid_max_depth_spread_m,
        hybrid_min_cluster_separation_m,
        hybrid_fallback_on_single_cluster,
        hybrid_fallback_on_sparse_candidates,
    )


def trim_box_bottom(box, trim_ratio):
    """Return a copy of an xyxy box with the bottom removed by trim_ratio."""
    box_arr = np.asarray(box, dtype=np.float32).copy()
    if trim_ratio == 0.0:
        return box_arr
    height = box_arr[3] - box_arr[1]
    box_arr[3] = box_arr[3] - height * float(trim_ratio)
    return box_arr


def points_in_box_mask(points_2d, box):
    """Return a boolean mask for projected points inside an xyxy box."""
    points_arr = np.asarray(points_2d, dtype=np.float32)
    box_arr = np.asarray(box, dtype=np.float32)
    if points_arr.size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (points_arr[:, 0] >= box_arr[0]) &
        (points_arr[:, 0] <= box_arr[2]) &
        (points_arr[:, 1] >= box_arr[1]) &
        (points_arr[:, 1] <= box_arr[3])
    )


def depth_cluster_indices(depths, gap_m, min_points):
    """Split positive camera depths into deterministic gap-based clusters."""
    clusters = depth_clusters_all(depths, gap_m)
    return [cluster for cluster in clusters if len(cluster) >= min_points]


def depth_clusters_all(depths, gap_m):
    """Split positive camera depths into all deterministic gap-based clusters."""
    depths = np.asarray(depths, dtype=np.float32)
    if depths.size == 0:
        return []

    valid_mask = np.isfinite(depths) & (depths > 0.0)
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return []

    order = valid_indices[np.argsort(depths[valid_indices], kind='mergesort')]
    clusters = []
    start = 0
    sorted_depths = depths[order]
    for pos in range(1, len(order)):
        if sorted_depths[pos] - sorted_depths[pos - 1] > gap_m:
            cluster = order[start:pos]
            clusters.append(cluster)
            start = pos
    cluster = order[start:]
    clusters.append(cluster)
    return clusters


def choose_depth_cluster(depths, method, gap_m, min_points):
    """Choose a depth cluster by nearest-depth or densest-support policy."""
    clusters = depth_cluster_indices(depths, gap_m, min_points)
    if not clusters:
        return np.empty((0,), dtype=np.int64)

    if method == 'depth_cluster_nearest':
        return min(
            clusters,
            key=lambda idx: (
                float(np.min(depths[idx])),
                -len(idx),
                int(np.min(idx)),
            ),
        )

    if method == 'depth_cluster_densest':
        return min(
            clusters,
            key=lambda idx: (
                -len(idx),
                float(np.mean(depths[idx])),
                int(np.min(idx)),
            ),
        )

    raise ValueError(f"Unsupported association_method: {method}")


def depth_cluster_metadata(depths, gap_m, min_points, method):
    """Return deterministic cluster diagnostics for a candidate depth set."""
    depths = np.asarray(depths, dtype=np.float32)
    all_clusters = depth_clusters_all(depths, gap_m)
    valid_clusters = [
        cluster for cluster in all_clusters
        if len(cluster) >= int(min_points)
    ]
    if not valid_clusters:
        return {
            'all_clusters': all_clusters,
            'valid_clusters': valid_clusters,
            'selected_cluster': np.empty((0,), dtype=np.int64),
            'selected_cluster_size': 0,
            'selected_cluster_support_ratio': None,
            'selected_cluster_depth_spread': None,
            'selected_cluster_separation': None,
        }

    if method == 'depth_cluster_nearest':
        selected = min(
            valid_clusters,
            key=lambda idx: (
                float(np.min(depths[idx])),
                -len(idx),
                int(np.min(idx)),
            ),
        )
    elif method == 'depth_cluster_densest':
        selected = min(
            valid_clusters,
            key=lambda idx: (
                -len(idx),
                float(np.mean(depths[idx])),
                int(np.min(idx)),
            ),
        )
    else:
        raise ValueError(f"Unsupported association_method: {method}")

    selected_min = float(np.min(depths[selected]))
    selected_max = float(np.max(depths[selected]))
    separations = []
    for cluster in valid_clusters:
        if np.array_equal(cluster, selected):
            continue
        cluster_min = float(np.min(depths[cluster]))
        cluster_max = float(np.max(depths[cluster]))
        if cluster_min > selected_max:
            separations.append(cluster_min - selected_max)
        elif selected_min > cluster_max:
            separations.append(selected_min - cluster_max)
        else:
            separations.append(0.0)

    return {
        'all_clusters': all_clusters,
        'valid_clusters': valid_clusters,
        'selected_cluster': selected,
        'selected_cluster_size': int(len(selected)),
        'selected_cluster_support_ratio': float(len(selected) / len(depths))
        if len(depths) else None,
        'selected_cluster_depth_spread': float(selected_max - selected_min),
        'selected_cluster_separation': float(min(separations))
        if separations else None,
    }


def centre_from_points(points, statistic):
    """Calculate a 3D centre with the requested statistic."""
    if points.numel() == 0:
        return None
    if statistic == 'mean':
        return points.mean(dim=0)
    if statistic == 'median':
        return points.median(dim=0).values
    raise ValueError(f"Unsupported centre_statistic: {statistic}")


class Fusser:
    """Early fusion between camera detections and LiDAR point cloud data."""

    def __init__(
        self,
        P,
        R0,
        V2C,
        model,
        RF,
        classes,
        thresh,
        points,
        bboxes,
        write,
        projection_matrix=None,
        lidar_to_camera_matrix=None,
        device='auto',
        point_depth_filter_mode='legacy_lidar_x',
        minimum_forward_distance=2.0,
        output_coordinate_mode='legacy_kitti',
        association_method='legacy_box',
        centre_statistic='mean',
        depth_cluster_gap_m=0.35,
        depth_cluster_min_points=5,
        box_bottom_trim_ratio=0.0,
        hybrid_min_cluster_points=5,
        hybrid_min_support_ratio=0.35,
        hybrid_max_depth_spread_m=0.35,
        hybrid_min_cluster_separation_m=0.20,
        hybrid_fallback_on_single_cluster=True,
        hybrid_fallback_on_sparse_candidates=True,
        coordinate_label_mode='xyz',
    ):
        """
        Initialize the Fusser.

        Args:
            P (np.ndarray): Camera projection matrix (3x4).
            R0 (np.ndarray): Rectification rotation matrix (3x3).
            V2C (np.ndarray): Velodyne-to-Camera transformation matrix (3x4).
            model (str): Path to YOLO model.
            RF (float): Reduction factor for bounding boxes.
            classes (list): Classes to detect.
            thresh (float): Confidence threshold.
            points (bool): Whether to draw LiDAR points.
            bboxes (bool): Whether to draw bounding boxes.
            write (bool): Whether to write coordinates on image.
        """
        self.device = self._select_device(device)
        self.point_depth_filter_mode = point_depth_filter_mode
        self.minimum_forward_distance = float(minimum_forward_distance)
        self.output_coordinate_mode = output_coordinate_mode
        self.coordinate_label_mode = coordinate_label_mode
        association_config = validate_association_config(
            association_method,
            centre_statistic,
            depth_cluster_gap_m,
            depth_cluster_min_points,
            box_bottom_trim_ratio,
            hybrid_min_cluster_points,
            hybrid_min_support_ratio,
            hybrid_max_depth_spread_m,
            hybrid_min_cluster_separation_m,
            hybrid_fallback_on_single_cluster,
            hybrid_fallback_on_sparse_candidates,
        )
        (
            self.association_method,
            self.centre_statistic,
            self.depth_cluster_gap_m,
            self.depth_cluster_min_points,
            self.box_bottom_trim_ratio,
            self.hybrid_min_cluster_points,
            self.hybrid_min_support_ratio,
            self.hybrid_max_depth_spread_m,
            self.hybrid_min_cluster_separation_m,
            self.hybrid_fallback_on_single_cluster,
            self.hybrid_fallback_on_sparse_candidates,
        ) = association_config

        if self.point_depth_filter_mode not in {
            'legacy_lidar_x',
            'positive_camera_depth',
        }:
            raise ValueError(
                f"Unsupported point_depth_filter_mode: "
                f"{self.point_depth_filter_mode}"
            )
        if self.output_coordinate_mode not in {
            'legacy_kitti',
            'lidar_xyz',
            'camera_xyz',
        }:
            raise ValueError(
                f"Unsupported output_coordinate_mode: {self.output_coordinate_mode}"
            )
        if self.coordinate_label_mode not in {'xyz', 'depth'}:
            raise ValueError(
                f"Unsupported coordinate_label_mode: {self.coordinate_label_mode}"
            )

        if projection_matrix is None:
            R0_padded = np.column_stack([
                np.vstack([R0, [0, 0, 0]]),
                [0, 0, 0, 1],
            ])
            V2C_padded = np.vstack((V2C, [0, 0, 0, 1]))
            projection_matrix = np.dot(P, np.dot(R0_padded, V2C_padded))
            lidar_to_camera_matrix = np.dot(R0_padded, V2C_padded)
        elif lidar_to_camera_matrix is None:
            raise ValueError(
                'lidar_to_camera_matrix is required with projection_matrix'
            )

        self.trafo_matrix = torch.tensor(
            projection_matrix,
            dtype=torch.float32,
            device=self.device,
        )
        self.lidar_to_camera_matrix = torch.tensor(
            lidar_to_camera_matrix,
            dtype=torch.float32,
            device=self.device,
        )

        self.RF = RF
        self.classes = classes
        self.thresh = thresh
        self.points = points
        self.bboxes = bboxes
        self.write = write

        self.model = YOLO(model).to(self.device)

        cmap = plt.cm.get_cmap("hsv", 256)
        self.cmap = cmap(np.arange(256))[:, :3] * 255
        self.last_association_metadata = []

    def _select_device(self, requested_device):
        """Select the torch device used by YOLO and projection tensors."""
        if requested_device == 'auto':
            selected = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif requested_device in {'cuda', 'cpu'}:
            selected = requested_device
        else:
            raise ValueError(f"Unsupported device: {requested_device}")

        if selected == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")

        return torch.device(selected)

    def draw_bboxes(self, img, bbox):
        """Draws bounding boxes on the image."""
        x1, y1, x2, y2 = map(int, bbox[:4].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def draw_points(self, img, original_indices):
        """Draws the LiDAR points associated with a detection on the image."""
        selected_points = self.points_2d[original_indices]
        if self.point_depth_filter_mode == 'positive_camera_depth':
            point_depths = self.pc_camera[original_indices, 2]
        else:
            point_depths = self.pc_velo[original_indices, 0]

        color_indices = torch.floor(
            510.0 / torch.clamp(point_depths, min=1e-6)
        ).to(torch.int32)
        color_indices = torch.clamp(color_indices, min=0, max=255)

        pts_2d = selected_points.cpu().numpy().astype(int)
        color_indices = color_indices.cpu().numpy()

        for (x, y), color_idx in zip(pts_2d, color_indices):
            cv2.circle(img, (x, y), 1, tuple(self.cmap[color_idx]), -1)

    def write_coordinates(self, img, coordinates, box):
        """Writes class label and 3D coordinates above the detected object."""
        x1, y1 = int(box[0]), int(box[1])
        x, y, z = map(float, coordinates)

        class_ = 'car' if int(box[5]) == 2 else 'person'
        if self.coordinate_label_mode == 'depth':
            depth = x if self.output_coordinate_mode == 'lidar_xyz' else z
            label = f"{class_} : {depth:.2f}m"
        else:
            label = f"{class_} : {x:.2f},{y:.2f},{z:.2f}m"
        cv2.putText(
            img,
            label,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    def outlier_mask(self, points: torch.Tensor) -> torch.Tensor:
        """
        Return a Modified Z-Score inlier mask for a set of points.

        Args:
            points (torch.Tensor): Input points (n, 3).

        Returns:
            torch.Tensor: Boolean inlier mask.
        """
        median = points.median(dim=0).values
        mad = (points - median).abs().median(dim=0).values

        mad_safe = torch.where(mad == 0, torch.ones_like(mad) * 1e-6, mad)
        modified_z_scores = 0.6745 * (points - median) / mad_safe
        return (modified_z_scores.abs() < 3.5).all(dim=1)

    def filter_outliers(self, points: torch.Tensor) -> torch.Tensor:
        """
        Removes outliers from a set of points using the Modified Z-Score method.

        Args:
            points (torch.Tensor): Input points (n, 3).

        Returns:
            torch.Tensor: Filtered points.
        """
        return points[self.outlier_mask(points)]

    def get_coordinates(self, points, camera_points=None):
        """
        Calculates the mean 3D coordinate of a set of points.

        Args:
            points (torch.Tensor): Input points (n, 3).

        Returns:
            torch.Tensor: Mean coordinate (3D).
        """
        if points.numel() != 0:
            if self.output_coordinate_mode == 'camera_xyz':
                if camera_points is None or camera_points.numel() == 0:
                    return None
                return centre_from_points(camera_points, self.centre_statistic)

            coordinates = centre_from_points(points, self.centre_statistic)
            if coordinates is None:
                return None
            if self.output_coordinate_mode == 'lidar_xyz':
                return coordinates
            return torch.stack([-coordinates[1], coordinates[2], coordinates[0]])

    def run_obstacle_detection(self, img):
        """
        Runs YOLO model on the input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            torch.Tensor: Bounding boxes (n, 6).
        """
        predictions = self.model(img, conf=self.thresh, classes=self.classes, verbose=False)
        pred_bboxes = (predictions[0].boxes.data).detach()
        return pred_bboxes

    def get_lidar_on_image_fov(self, pc_velo, img):
        """
        Projects LiDAR points onto the image plane and selects points within FOV.

        Args:
            pc_velo (np.ndarray): Point cloud (n, 3).
            img (np.ndarray): Input image.
        """
        pc_velo_torch = torch.tensor(
            pc_velo,
            dtype=torch.float32,
            device=self.device,
        )
        if len(pc_velo_torch) == 0:
            self.points_2d = torch.empty((0, 2), dtype=torch.float32, device=self.device)
            self.pc_velo = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            self.pc_camera = torch.empty((0, 3), dtype=torch.float32, device=self.device)
            return

        ones = torch.ones(
            len(pc_velo_torch),
            1,
            dtype=torch.float32,
            device=self.device,
        )
        pts_3d_homo = torch.cat([pc_velo_torch, ones], dim=1)

        pts_2d_homo = torch.matmul(self.trafo_matrix, pts_3d_homo.T)
        pts_2d = (pts_2d_homo[:2] / pts_2d_homo[2]).T
        pts_camera_homo = torch.matmul(self.lidar_to_camera_matrix, pts_3d_homo.T)
        pts_camera = pts_camera_homo[:3].T

        if self.point_depth_filter_mode == 'positive_camera_depth':
            depth_mask = pts_camera[:, 2] > self.minimum_forward_distance
        else:
            depth_mask = pc_velo_torch[:, 0] > self.minimum_forward_distance

        fov_mask = (
            (pts_2d[:, 0] < img.shape[1]) &
            (pts_2d[:, 0] >= 0) &
            (pts_2d[:, 1] < img.shape[0]) &
            (pts_2d[:, 1] >= 0) &
            depth_mask
        )

        self.points_2d = pts_2d[fov_mask]
        self.pc_velo = pc_velo_torch[fov_mask]
        self.pc_camera = pts_camera[fov_mask]

    def reduce_bbox(self, box):
        """
        Shrinks a bounding box based on the reduction factor (RF).

        Args:
            box (torch.Tensor): Bounding box (4D).

        Returns:
            torch.Tensor: Reduced bounding box.
        """
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        reduced_width = box_width * self.RF
        reduced_height = box_height * self.RF

        return torch.stack([
            box_center_x - reduced_width / 2,
            box_center_y - reduced_height / 2,
            box_center_x + reduced_width / 2,
            box_center_y + reduced_height / 2
        ])

    def association_bbox(self, box):
        """Return the reduced, optionally bottom-trimmed association box."""
        reduced_bbox = self.reduce_bbox(box)
        if self.box_bottom_trim_ratio == 0.0:
            return reduced_bbox

        trimmed_bbox = reduced_bbox.clone()
        height = trimmed_bbox[3] - trimmed_bbox[1]
        trimmed_bbox[3] = trimmed_bbox[3] - height * self.box_bottom_trim_ratio
        return trimmed_bbox

    def legacy_association_indices(self, available_points, association_bbox):
        """Select all available projected LiDAR indices inside a box."""
        available_points_2d = self.points_2d[available_points]

        x_check = (
            (available_points_2d[:, 0] >= association_bbox[0]) &
            (available_points_2d[:, 0] <= association_bbox[2])
        )
        y_check = (
            (available_points_2d[:, 1] >= association_bbox[1]) &
            (available_points_2d[:, 1] <= association_bbox[3])
        )
        box_mask = x_check & y_check

        if not torch.any(box_mask):
            return torch.empty((0,), dtype=torch.long, device=self.device)

        return torch.nonzero(
            available_points,
            as_tuple=False,
        ).flatten()[box_mask]

    def _base_association_metadata(self, association_bbox, candidate_count):
        """Create a common diagnostics payload for association decisions."""
        return {
            'association_method_requested': self.association_method,
            'association_path': self.association_method,
            'association_reason': None,
            'fallback_occurred': False,
            'fallback_reason': None,
            'candidate_point_count': int(candidate_count),
            'cluster_count': 0,
            'valid_cluster_count': 0,
            'selected_cluster_size': 0,
            'selected_cluster_support_ratio': None,
            'selected_cluster_depth_spread': None,
            'selected_cluster_separation': None,
            'selected_point_count': 0,
            'association_box': association_bbox.detach().cpu().numpy()
            .astype(float).tolist(),
        }

    def _select_hybrid_association_indices(self, original_indices, metadata):
        """Accept a reliable nearest depth cluster or fall back to legacy."""
        candidate_count = int(len(original_indices))
        if candidate_count == 0:
            metadata['association_reason'] = 'no_points'
            metadata['association_path'] = 'none'
            return original_indices, metadata

        if (
            self.hybrid_fallback_on_sparse_candidates and
            candidate_count < self.hybrid_min_cluster_points
        ):
            metadata['association_path'] = 'legacy_box'
            metadata['fallback_occurred'] = True
            metadata['fallback_reason'] = 'sparse_candidates'
            metadata['association_reason'] = 'sparse_candidates'
            metadata['selected_point_count'] = candidate_count
            return original_indices, metadata

        depths = self.pc_camera[original_indices, 2].detach().cpu().numpy()
        cluster_meta = depth_cluster_metadata(
            depths,
            self.depth_cluster_gap_m,
            self.depth_cluster_min_points,
            'depth_cluster_nearest',
        )
        metadata['cluster_count'] = int(len(cluster_meta['all_clusters']))
        metadata['valid_cluster_count'] = int(len(cluster_meta['valid_clusters']))
        metadata['selected_cluster_size'] = cluster_meta['selected_cluster_size']
        metadata['selected_cluster_support_ratio'] = cluster_meta[
            'selected_cluster_support_ratio'
        ]
        metadata['selected_cluster_depth_spread'] = cluster_meta[
            'selected_cluster_depth_spread'
        ]
        metadata['selected_cluster_separation'] = cluster_meta[
            'selected_cluster_separation'
        ]

        fallback_reason = None
        if metadata['valid_cluster_count'] == 0:
            fallback_reason = 'no_valid_cluster'
        elif metadata['selected_cluster_size'] < self.hybrid_min_cluster_points:
            fallback_reason = 'too_few_cluster_points'
        elif (
            metadata['selected_cluster_support_ratio'] is not None and
            metadata['selected_cluster_support_ratio'] <
            self.hybrid_min_support_ratio
        ):
            fallback_reason = 'low_support_ratio'
        elif (
            metadata['selected_cluster_depth_spread'] is not None and
            metadata['selected_cluster_depth_spread'] >
            self.hybrid_max_depth_spread_m
        ):
            fallback_reason = 'excessive_depth_spread'
        elif (
            self.hybrid_fallback_on_single_cluster and
            metadata['valid_cluster_count'] < 2
        ):
            fallback_reason = 'insufficient_cluster_separation'
        elif (
            metadata['selected_cluster_separation'] is not None and
            metadata['selected_cluster_separation'] <
            self.hybrid_min_cluster_separation_m
        ):
            fallback_reason = 'insufficient_cluster_separation'

        if fallback_reason is not None:
            metadata['association_path'] = 'legacy_box'
            metadata['fallback_occurred'] = True
            metadata['fallback_reason'] = fallback_reason
            metadata['association_reason'] = fallback_reason
            metadata['selected_point_count'] = candidate_count
            return original_indices, metadata

        local_cluster_indices = torch.as_tensor(
            cluster_meta['selected_cluster'],
            dtype=torch.long,
            device=self.device,
        )
        selected = original_indices[local_cluster_indices]
        metadata['association_path'] = 'depth_cluster_nearest'
        metadata['association_reason'] = 'accepted_depth_cluster'
        metadata['selected_point_count'] = int(len(selected))
        return selected, metadata

    def select_association_indices_with_metadata(
        self,
        available_points,
        association_bbox,
    ):
        """Select projected LiDAR indices and report deterministic diagnostics."""
        original_indices = self.legacy_association_indices(
            available_points,
            association_bbox,
        )
        metadata = self._base_association_metadata(
            association_bbox,
            len(original_indices),
        )

        if self.association_method == 'legacy_box':
            metadata['association_reason'] = 'legacy_box'
            metadata['selected_point_count'] = int(len(original_indices))
            return original_indices, metadata

        if self.association_method == 'hybrid_depth_legacy':
            return self._select_hybrid_association_indices(
                original_indices,
                metadata,
            )

        if len(original_indices) == 0:
            metadata['association_reason'] = 'no_points'
            metadata['association_path'] = 'none'
            return original_indices, metadata

        depths = self.pc_camera[original_indices, 2].detach().cpu().numpy()
        cluster_meta = depth_cluster_metadata(
            depths,
            self.depth_cluster_gap_m,
            self.depth_cluster_min_points,
            self.association_method,
        )
        metadata['cluster_count'] = int(len(cluster_meta['all_clusters']))
        metadata['valid_cluster_count'] = int(len(cluster_meta['valid_clusters']))
        metadata['selected_cluster_size'] = cluster_meta['selected_cluster_size']
        metadata['selected_cluster_support_ratio'] = cluster_meta[
            'selected_cluster_support_ratio'
        ]
        metadata['selected_cluster_depth_spread'] = cluster_meta[
            'selected_cluster_depth_spread'
        ]
        metadata['selected_cluster_separation'] = cluster_meta[
            'selected_cluster_separation'
        ]
        local_cluster_indices = choose_depth_cluster(
            depths,
            self.association_method,
            self.depth_cluster_gap_m,
            self.depth_cluster_min_points,
        )
        if len(local_cluster_indices) == 0:
            metadata['association_reason'] = 'no_valid_cluster'
            metadata['association_path'] = 'none'
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                metadata,
            )

        local_cluster_indices = torch.as_tensor(
            local_cluster_indices,
            dtype=torch.long,
            device=self.device,
        )
        selected = original_indices[local_cluster_indices]
        metadata['association_reason'] = 'accepted_depth_cluster'
        metadata['selected_point_count'] = int(len(selected))
        return selected, metadata

    def select_association_indices(self, available_points, association_bbox):
        """Select available projected LiDAR indices for the configured method."""
        selected, _metadata = self.select_association_indices_with_metadata(
            available_points,
            association_bbox,
        )
        return selected

    def lidar_camera_fusion(self, pred_bboxes, img):
        """
        Associates 2D detections with 3D LiDAR points.

        Args:
            pred_bboxes (torch.Tensor): 2D detections.
            img (np.ndarray): Input image.

        Returns:
            torch.Tensor: Updated detections with 3D coordinates.
        """
        global_mask = torch.zeros(
            len(self.points_2d),
            dtype=torch.bool,
            device=self.device,
        )
        zeros = torch.zeros_like(pred_bboxes[:, :3])
        pred_bboxes = torch.cat([pred_bboxes, zeros], dim=1)
        self.last_association_metadata = []

        for box in pred_bboxes:
            available_points = ~global_mask
            association_bbox = self.association_bbox(box)
            original_indices, metadata = self.select_association_indices_with_metadata(
                available_points,
                association_bbox,
            )
            self.last_association_metadata.append(metadata)
            if len(original_indices) == 0:
                continue
            global_mask[original_indices] = True

            masked_points = self.pc_velo[original_indices, :]
            masked_camera_points = self.pc_camera[original_indices, :]

            if self.points:
                self.draw_points(img, original_indices)
            if self.bboxes:
                self.draw_bboxes(img, box)

            if len(masked_points) > 2:
                inlier_mask = self.outlier_mask(masked_points)
                filtered_points = masked_points[inlier_mask]
                filtered_camera_points = masked_camera_points[inlier_mask]
                coordinates = self.get_coordinates(
                    filtered_points,
                    filtered_camera_points,
                )
                if coordinates is None:
                    continue

                if self.write:
                    self.write_coordinates(img, coordinates, box)

                box[-3:] = coordinates

        return pred_bboxes

    def pipeline(self, image, point_cloud):
        """
        Full processing pipeline: detection, fusion, and output formatting.

        Args:
            image (np.ndarray): Input image.
            point_cloud (np.ndarray): Input point cloud.

        Returns:
            np.ndarray: Detections (n, 9) [x1, y1, x2, y2, confidence, class, x, y, z].
            np.ndarray: Annotated image.
        """
        self.get_lidar_on_image_fov(point_cloud[:, :3], image)
        pred_bboxes = self.run_obstacle_detection(image)

        if pred_bboxes.any():
            pred_bboxes = self.lidar_camera_fusion(pred_bboxes, image)

        return pred_bboxes.cpu().numpy(), image
