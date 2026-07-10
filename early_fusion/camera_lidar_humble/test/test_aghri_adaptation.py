# Copyright 2026 LCAS GROUP, University of Lincoln
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

"""Tests for AGHRI compatibility helpers."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
import yaml
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from camera_lidar_fusion.lidar_fusion import (
    load_aghri_projection,
    pointcloud2_to_xyz_array,
    transform_between_frames,
    validate_sync_config,
)
from scripts.fusser import (
    Fusser,
    centre_from_points,
    choose_depth_cluster,
    depth_cluster_indices,
    points_in_box_mask,
    trim_box_bottom,
    validate_association_config,
)
from tools.evaluate_aghri_association import (
    MATCH_IOU_THRESHOLD,
    build_official_manifest,
    cache_key,
    match_ground_truth,
    paired_comparison,
    read_manifest,
    write_manifest_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _cloud(points):
    """Build a PointCloud2 message from XYZ points."""
    header = Header()
    header.frame_id = 'lidar'
    return point_cloud2.create_cloud_xyz32(header, points)


def test_unorganised_xyz_keeps_valid_points_without_ring_reshape():
    """Unorganised mode keeps all finite XYZ points and does not reshape."""
    points = [(float(i), float(i + 1), float(i + 2)) for i in range(65)]
    cloud = _cloud(points)

    out, raw_count, valid_count = pointcloud2_to_xyz_array(
        cloud,
        'unorganised_xyz',
    )

    assert raw_count == 65
    assert valid_count == 65
    assert out.shape == (65, 3)
    np.testing.assert_allclose(out[-1], [64.0, 65.0, 66.0])


def test_unorganised_xyz_filters_nan_and_empty_clouds():
    """Unorganised mode removes NaNs and safely handles an empty cloud."""
    cloud = _cloud([(1.0, 2.0, 3.0), (float('nan'), 2.0, 3.0)])

    out, raw_count, valid_count = pointcloud2_to_xyz_array(
        cloud,
        'unorganised_xyz',
    )

    assert raw_count == 2
    assert valid_count == 1
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])

    empty, raw_count, valid_count = pointcloud2_to_xyz_array(
        _cloud([]),
        'unorganised_xyz',
    )
    assert raw_count == 0
    assert valid_count == 0
    assert empty.shape == (0, 3)


def test_organised_ring_mode_remains_available():
    """Organised-ring mode preserves the legacy ring subsampling behaviour."""
    points = [(float(i), 0.0, 0.0) for i in range(130)]
    cloud = _cloud(points)

    out, raw_count, valid_count = pointcloud2_to_xyz_array(
        cloud,
        'organised_rings',
        total_rings=64,
        rings_to_use=8,
    )

    assert raw_count == 130
    assert valid_count == 130
    assert out.shape == (16, 3)
    np.testing.assert_allclose(
        out[:, 0],
        [0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113],
    )


def _write_calibration_zip(path):
    """Create a minimal AGHRI-like calibration ZIP fixture."""
    intrinsics = {
        'cam_zed_rgb': {
            'header': {'frame_id': 'camera_optical'},
            'p': [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        },
    }
    extrinsics = {
        'transforms': [
            {
                'header': {'frame_id': 'base_link'},
                'child_frame_id': 'lidar',
                'transform': {
                    'translation': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
            {
                'header': {'frame_id': 'base_link'},
                'child_frame_id': 'camera_optical',
                'transform': {
                    'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
        ],
    }

    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('calibration/intrinsics.json', json.dumps(intrinsics))
        archive.writestr('calibration/extrinsics.json', json.dumps(extrinsics))


def test_aghri_calibration_zip_projection_and_transform_chain(tmp_path):
    """AGHRI calibration is derived from JSON graph data, not hard-coded."""
    zip_path = tmp_path / 'calibration.zip'
    _write_calibration_zip(zip_path)

    projection, camera_from_lidar, path, frame = load_aghri_projection(
        str(zip_path),
        'calibration/intrinsics.json',
        'calibration/extrinsics.json',
        'cam_zed_rgb',
        'lidar',
        'camera_optical',
    )

    assert path == ['lidar', 'base_link', 'camera_optical']
    assert frame == 'camera_optical'
    np.testing.assert_allclose(camera_from_lidar[:3, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(projection[:, 3], [1.0, 0.0, 0.0])


def test_missing_calibration_frame_raises_clear_error(tmp_path):
    """A missing frame in the transform graph is reported as an error."""
    zip_path = tmp_path / 'calibration.zip'
    _write_calibration_zip(zip_path)

    with pytest.raises(KeyError, match='not found'):
        load_aghri_projection(
            str(zip_path),
            'calibration/intrinsics.json',
            'calibration/extrinsics.json',
            'cam_zed_rgb',
            'missing_lidar',
            'camera_optical',
        )


def test_transform_chain_composition():
    """Transform-chain composition follows ROS TransformStamped semantics."""
    extrinsics = {
        'transforms': [
            {
                'header': {'frame_id': 'a'},
                'child_frame_id': 'b',
                'transform': {
                    'translation': {'x': 1.0, 'y': 0.0, 'z': 0.0},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
            {
                'header': {'frame_id': 'b'},
                'child_frame_id': 'c',
                'transform': {
                    'translation': {'x': 0.0, 'y': 2.0, 'z': 0.0},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
        ],
    }

    transform, path = transform_between_frames(extrinsics, 'a', 'c')

    assert path == ['a', 'b', 'c']
    np.testing.assert_allclose(transform[:3, 3], [-1.0, -2.0, 0.0])


def test_ros_transform_direction_maps_child_points_to_parent():
    """A stored parent->child edge maps child-frame points to the parent frame."""
    extrinsics = {
        'transforms': [
            {
                'header': {'frame_id': 'base_link'},
                'child_frame_id': 'front_lidar_link',
                'transform': {
                    'translation': {'x': 0.615, 'y': 0.28, 'z': 0.33},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
            {
                'header': {'frame_id': 'base_link'},
                'child_frame_id': 'front_camera_link',
                'transform': {
                    'translation': {'x': 0.55, 'y': 0.0, 'z': 0.61},
                    'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
                },
            },
            {
                'header': {'frame_id': 'front_camera_link'},
                'child_frame_id': 'front_left_camera_optical_frame',
                'transform': {
                    'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'rotation': {
                        'x': 0.5,
                        'y': -0.4999999999999999,
                        'z': 0.5,
                        'w': -0.5000000000000001,
                    },
                },
            },
        ],
    }

    transform, path = transform_between_frames(
        extrinsics,
        'front_lidar_link',
        'front_left_camera_optical_frame',
    )

    point_camera = transform @ np.array([5.0, 0.0, 0.0, 1.0])
    farther_camera = transform @ np.array([6.0, 0.0, 0.0, 1.0])
    lateral_camera = transform @ np.array([5.0, 1.0, 0.0, 1.0])
    vertical_camera = transform @ np.array([5.0, 0.0, 1.0, 1.0])
    assert path == [
        'front_lidar_link',
        'base_link',
        'front_camera_link',
        'front_left_camera_optical_frame',
    ]
    np.testing.assert_allclose(point_camera[:3], [-0.28, 0.28, 5.065])
    assert farther_camera[2] > point_camera[2]
    assert lateral_camera[0] != point_camera[0]
    assert vertical_camera[1] != point_camera[1]


def test_sync_config_validation():
    """Exact and approximate sync settings validate with clear failures."""
    assert validate_sync_config('exact', 10, 0.1) == ('exact', 10, 0.1)
    assert validate_sync_config('approximate', 30, 0.1) == (
        'approximate',
        30,
        0.1,
    )

    with pytest.raises(ValueError, match='Unsupported sync_mode'):
        validate_sync_config('nearest', 10, 0.1)


def test_original_and_aghri_configs_load():
    """The original and AGHRI parameter files are valid YAML."""
    for path in (
        'config/config.yaml',
        'config/aghri_zed_livox.yaml',
        'config/aghri_zed_livox_hybrid.yaml',
    ):
        with open(REPO_ROOT / path, 'r', encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
        assert 'lidar_fusion' in data
        assert 'ros__parameters' in data['lidar_fusion']


def test_draw_points_clamps_camera_depth_colour_indices():
    """AGHRI camera-depth colouring does not index outside the palette."""
    fusser = Fusser.__new__(Fusser)
    fusser.device = torch.device('cpu')
    fusser.point_depth_filter_mode = 'positive_camera_depth'
    fusser.points_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    fusser.pc_velo = torch.tensor([[-0.5, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    fusser.pc_camera = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 20.0]])
    fusser.cmap = np.zeros((256, 3), dtype=np.float64)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    fusser.draw_points(image, torch.tensor([0, 1]))


def _association_fusser(method='legacy_box', statistic='mean', trim=0.0):
    fusser = Fusser.__new__(Fusser)
    fusser.device = torch.device('cpu')
    fusser.association_method = method
    fusser.centre_statistic = statistic
    fusser.depth_cluster_gap_m = 0.35
    fusser.depth_cluster_min_points = 2
    fusser.box_bottom_trim_ratio = trim
    fusser.hybrid_min_cluster_points = 2
    fusser.hybrid_min_support_ratio = 0.20
    fusser.hybrid_max_depth_spread_m = 0.35
    fusser.hybrid_min_cluster_separation_m = 0.20
    fusser.hybrid_fallback_on_single_cluster = True
    fusser.hybrid_fallback_on_sparse_candidates = True
    fusser.points_2d = torch.tensor([
        [10.0, 10.0],
        [11.0, 10.0],
        [12.0, 10.0],
        [20.0, 10.0],
        [21.0, 10.0],
        [22.0, 10.0],
    ])
    fusser.pc_camera = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, 1.2],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, 3.1],
        [0.0, 0.0, 3.2],
    ])
    return fusser


def test_legacy_association_selection_matches_box_mask():
    """Legacy association returns all available points inside the box."""
    fusser = _association_fusser()
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([9.0, 9.0, 21.0, 11.0])

    selected = fusser.select_association_indices(available, box)

    assert selected.tolist() == [0, 1, 2, 3, 4]


def test_depth_cluster_nearest_selection():
    """Nearest cluster chooses the closest valid depth group."""
    fusser = _association_fusser(method='depth_cluster_nearest')
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    selected = fusser.select_association_indices(available, box)

    assert selected.tolist() == [0, 1, 2]


def test_depth_cluster_densest_selection_and_tie_breaking():
    """Densest cluster uses support, then nearest mean depth as tie-break."""
    depths = np.array([4.0, 4.1, 1.0, 1.1], dtype=np.float32)

    selected = choose_depth_cluster(
        depths,
        'depth_cluster_densest',
        gap_m=0.35,
        min_points=2,
    )

    assert selected.tolist() == [2, 3]


def test_depth_cluster_empty_behind_and_minimum_size():
    """Empty, behind-camera, and undersized clusters are rejected."""
    assert depth_cluster_indices(np.array([]), 0.35, 2) == []
    assert depth_cluster_indices(np.array([-1.0, 0.0, -0.2]), 0.35, 2) == []
    assert depth_cluster_indices(np.array([1.0, 2.0]), 0.35, 2) == []


def test_mean_and_median_centres():
    """Centre statistics are deterministic."""
    points = torch.tensor([[0.0, 0.0, 0.0], [10.0, 2.0, 4.0], [2.0, 4.0, 8.0]])

    np.testing.assert_allclose(centre_from_points(points, 'mean'), [4.0, 2.0, 4.0])
    np.testing.assert_allclose(centre_from_points(points, 'median'), [2.0, 2.0, 4.0])


def test_bottom_trim_and_point_mask_helpers():
    """A zero trim preserves the box and positive trim removes the bottom."""
    box = np.array([0.0, 0.0, 10.0, 20.0], dtype=np.float32)
    points = np.array([[1.0, 1.0], [9.0, 19.0], [11.0, 1.0]], dtype=np.float32)

    np.testing.assert_allclose(trim_box_bottom(box, 0.0), box)
    np.testing.assert_allclose(trim_box_bottom(box, 0.10), [0.0, 0.0, 10.0, 18.0])
    assert points_in_box_mask(points, box).tolist() == [True, True, False]


def test_invalid_association_parameters_raise():
    """Invalid association settings fail clearly."""
    with pytest.raises(ValueError, match='association_method'):
        validate_association_config('bad', 'mean', 0.35, 2, 0.0)
    with pytest.raises(ValueError, match='centre_statistic'):
        validate_association_config('legacy_box', 'mode', 0.35, 2, 0.0)
    with pytest.raises(ValueError, match='box_bottom_trim_ratio'):
        validate_association_config('legacy_box', 'mean', 0.35, 2, 0.9)
    with pytest.raises(ValueError, match='hybrid_min_support_ratio'):
        validate_association_config(
            'hybrid_depth_legacy',
            'mean',
            0.35,
            2,
            0.0,
            hybrid_min_support_ratio=1.2,
        )


def test_association_repeated_output_is_deterministic():
    """Repeated depth clustering returns identical indices."""
    fusser = _association_fusser(method='depth_cluster_densest')
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    first = fusser.select_association_indices(available, box)
    second = fusser.select_association_indices(available, box)

    assert first.tolist() == second.tolist()


def test_hybrid_accepts_reliable_nearest_depth_cluster():
    """Hybrid accepts a compact, supported, separated nearest cluster."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == [0, 1, 2]
    assert metadata['association_path'] == 'depth_cluster_nearest'
    assert metadata['association_reason'] == 'accepted_depth_cluster'
    assert metadata['fallback_occurred'] is False
    assert metadata['candidate_point_count'] == 6
    assert metadata['cluster_count'] == 2
    assert metadata['selected_cluster_size'] == 3
    assert metadata['selected_cluster_support_ratio'] == pytest.approx(0.5)


def test_hybrid_sparse_candidates_fall_back_to_exact_legacy_box():
    """Sparse candidate sets return the legacy box selection unchanged."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    fusser.hybrid_min_cluster_points = 4
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([9.0, 9.0, 12.0, 11.0])

    legacy = _association_fusser().select_association_indices(available, box)
    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == legacy.tolist()
    assert metadata['fallback_reason'] == 'sparse_candidates'
    assert metadata['association_path'] == 'legacy_box'


def test_hybrid_low_support_falls_back_to_exact_legacy_box():
    """Low selected-cluster support falls back instead of overcommitting."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    fusser.hybrid_min_support_ratio = 0.75
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    legacy = _association_fusser().select_association_indices(available, box)
    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == legacy.tolist()
    assert metadata['fallback_reason'] == 'low_support_ratio'


def test_hybrid_excessive_spread_falls_back_to_exact_legacy_box():
    """A selected cluster that is too deep falls back to legacy."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    fusser.hybrid_max_depth_spread_m = 0.05
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    legacy = _association_fusser().select_association_indices(available, box)
    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == legacy.tolist()
    assert metadata['fallback_reason'] == 'excessive_depth_spread'


def test_hybrid_poor_separation_falls_back_to_exact_legacy_box():
    """Weak separation between valid clusters triggers legacy fallback."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    fusser.depth_cluster_gap_m = 0.12
    fusser.hybrid_min_cluster_separation_m = 0.20
    fusser.pc_camera = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.1],
        [0.0, 0.0, 1.2],
        [0.0, 0.0, 1.35],
        [0.0, 0.0, 1.45],
        [0.0, 0.0, 1.55],
    ])
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    legacy = _association_fusser().select_association_indices(available, box)
    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == legacy.tolist()
    assert metadata['fallback_reason'] == 'insufficient_cluster_separation'


def test_hybrid_no_valid_cluster_falls_back_to_exact_legacy_box():
    """If depth clustering has no valid cluster, hybrid returns legacy points."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    fusser.depth_cluster_min_points = 4
    fusser.hybrid_fallback_on_sparse_candidates = False
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([0.0, 0.0, 30.0, 20.0])

    legacy = _association_fusser().select_association_indices(available, box)
    selected, metadata = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert selected.tolist() == legacy.tolist()
    assert metadata['fallback_reason'] == 'no_valid_cluster'


def test_hybrid_empty_selection_is_safe_and_deterministic():
    """Empty boxes return no points and stable metadata."""
    fusser = _association_fusser(method='hybrid_depth_legacy')
    available = torch.ones(6, dtype=torch.bool)
    box = torch.tensor([100.0, 100.0, 110.0, 110.0])

    first, first_meta = fusser.select_association_indices_with_metadata(
        available,
        box,
    )
    second, second_meta = fusser.select_association_indices_with_metadata(
        available,
        box,
    )

    assert first.tolist() == []
    assert second.tolist() == []
    assert first_meta['association_reason'] == 'no_points'
    assert first_meta == second_meta


def _make_manifest_dataset(tmp_path):
    """Create a tiny AGHRI-like split fixture for manifest tests."""
    root = tmp_path / 'dataset'
    split_dir = root / 'split_lists'
    split_dir.mkdir(parents=True)
    (split_dir / 'train.txt').write_text('train_rec_label\n', encoding='utf-8')
    (split_dir / 'val.txt').write_text('val_rec_label\n', encoding='utf-8')
    (split_dir / 'test.txt').write_text(
        'test_rec_label\nmissing_rec_label\n',
        encoding='utf-8',
    )
    rec = root / 'dataset_part1' / 'test_rec_label'
    (rec / 'sensor_data/cam_zed_rgb').mkdir(parents=True)
    (rec / 'sensor_data/lidar').mkdir(parents=True)
    (rec / 'annotations').mkdir()
    (rec / 'sensor_data/cam_zed_rgb' / '1.png').write_bytes(b'png')
    (rec / 'sensor_data/lidar' / '1.pcd').write_bytes(b'pcd')
    (rec / 'annotations/cam_zed_rgb_ann.json').write_text(json.dumps([{
        'Timestamp': 1.0,
        'File': '1.png',
        'Labels': [{'Class': '01', 'BoundingBoxes': [0, 0, 10, 10]}],
    }]), encoding='utf-8')
    (rec / 'annotations/lidar_ann.json').write_text(json.dumps([{
        'Timestamp': 1.0,
        'File': '1.pcd',
        'Labels': [{
            'Class': '01',
            'BoundingBoxes': [1, 2, 3, 0.5, 0.5, 1.0, 0, 0, 0],
        }],
    }]), encoding='utf-8')
    return root


def test_official_manifest_lists_supported_and_missing_recordings(tmp_path):
    """Manifest construction keeps every split entry and marks support status."""
    root = _make_manifest_dataset(tmp_path)

    manifest = build_official_manifest(root, split='test', image_slop=0.1)

    assert [row['recording_name'] for row in manifest] == [
        'test_rec_label',
        'missing_rec_label',
    ]
    assert manifest[0]['supported'] is True
    assert manifest[0]['expected_synchronised_frame_count'] == 1
    assert manifest[0]['train_val_leakage'] is False
    assert manifest[1]['supported'] is False
    assert manifest[1]['skip_reason'] == 'recording_directory_missing'


def test_manifest_read_filters_unsupported_rows_without_duplicates(tmp_path):
    """Evaluator manifests skip unsupported rows and remain deterministic."""
    manifest = [{
        'split': 'test',
        'recording_name': 'a',
        'recording_path': str(tmp_path / 'a'),
        'dataset_part': 'dataset_part1',
        'scene': 'footpath',
        'motion': 'moving',
        'zed_rgb_available': True,
        'lidar_available': True,
        'annotation_3d_available': True,
        'occlusion_log_available': False,
        'rgb_image_count': 1,
        'lidar_point_cloud_count': 1,
        'lidar_annotation_frame_count': 1,
        'expected_synchronised_frame_count': 1,
        'supported': True,
        'skip_reason': None,
        'train_val_leakage': False,
    }, {
        'split': 'test',
        'recording_name': 'b',
        'recording_path': str(tmp_path / 'b'),
        'dataset_part': 'dataset_part1',
        'scene': 'footpath',
        'motion': 'moving',
        'zed_rgb_available': False,
        'lidar_available': True,
        'annotation_3d_available': True,
        'occlusion_log_available': False,
        'rgb_image_count': 0,
        'lidar_point_cloud_count': 1,
        'lidar_annotation_frame_count': 1,
        'expected_synchronised_frame_count': 0,
        'supported': False,
        'skip_reason': 'zed_rgb',
        'train_val_leakage': False,
    }]
    csv_path = tmp_path / 'manifest.csv'
    json_path = tmp_path / 'manifest.json'

    write_manifest_outputs(csv_path, json_path, manifest)
    write_manifest_outputs(csv_path, json_path, manifest)

    assert read_manifest(csv_path) == [(tmp_path / 'a', 'test')]


def test_cache_identity_key_changes_for_detection_inputs():
    """YOLO cache identity changes when important inputs change."""
    base = {
        'image_path': '/a.png',
        'image_timestamp': 1.0,
        'image_sha256': 'image',
        'checkpoint_sha256': 'model',
        'confidence_threshold': 0.1,
        'classes': [0],
        'image_size_hw': [376, 672],
        'ultralytics_version': '8',
    }
    changed = dict(base)
    changed['checkpoint_sha256'] = 'other'

    assert cache_key(base) != cache_key(changed)


def test_object_id_matching_uses_camera_identity_before_distance():
    """A 2D identity match selects the matching 3D identity."""
    gt3d = [
        {'id': '01', 'class': '01', 'center_lidar': np.array([5, 0, 0], dtype=np.float32)},
        {'id': '02', 'class': '02', 'center_lidar': np.array([0.2, 0, 0], dtype=np.float32)},
    ]
    gt2d = [{'id': '01', 'box': np.array([0, 0, 10, 10], dtype=np.float32)}]

    match = match_ground_truth(
        np.array([0, 0, 10, 10], dtype=np.float32),
        np.array([0.2, 0, 0], dtype=np.float32),
        np.array([0, 0, 5], dtype=np.float32),
        np.eye(4, dtype=np.float32),
        gt3d,
        gt2d,
    )

    assert match['match_type'] == 'object_id_from_2d_gt'
    assert match['gt_identity'] == '01'


def test_ambiguous_2d_identity_is_not_forced():
    """Near-tied camera GT overlaps are marked ambiguous."""
    gt3d = [
        {'id': '01', 'class': '01', 'center_lidar': np.array([1, 0, 0], dtype=np.float32)},
        {'id': '02', 'class': '02', 'center_lidar': np.array([2, 0, 0], dtype=np.float32)},
    ]
    gt2d = [
        {'id': '01', 'box': np.array([0, 0, 10, 10], dtype=np.float32)},
        {'id': '02', 'box': np.array([0, 0, 10, 10], dtype=np.float32)},
    ]

    match = match_ground_truth(
        np.array([0, 0, 10, 10], dtype=np.float32),
        np.array([1, 0, 0], dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
        np.eye(4, dtype=np.float32),
        gt3d,
        gt2d,
    )

    assert match['ambiguous'] is True
    assert match['gt_match'] is None
    assert match['match_type'] == 'ambiguous_2d_identity'


def test_nearest_lidar_matching_fallback_requires_unambiguous_distance():
    """Nearest-centre fallback works only when the distance gap is clear."""
    gt3d = [
        {'id': 'near', 'center_lidar': np.array([1.0, 0, 0], dtype=np.float32)},
        {'id': 'far', 'center_lidar': np.array([3.0, 0, 0], dtype=np.float32)},
    ]

    match = match_ground_truth(
        np.array([50, 50, 60, 60], dtype=np.float32),
        np.array([1.1, 0, 0], dtype=np.float32),
        np.array([0, 0, 1.1], dtype=np.float32),
        np.eye(4, dtype=np.float32),
        gt3d,
        [],
    )

    assert match['match_type'] == 'nearest_lidar_center'
    assert match['gt_identity'] == 'near'


def test_paired_comparison_is_deterministic_and_aggregates():
    """Paired metrics compare the same detection keys."""
    legacy = [
        {'recording': 'r', 'frame_number': 0, 'detection_index': 0, 'errors': {'lidar_center_error': 1.0}},
        {'recording': 'r', 'frame_number': 1, 'detection_index': 0, 'errors': {'lidar_center_error': 1.0}},
    ]
    hybrid = [
        {'recording': 'r', 'frame_number': 0, 'detection_index': 0, 'errors': {'lidar_center_error': 0.5}},
        {'recording': 'r', 'frame_number': 1, 'detection_index': 0, 'errors': {'lidar_center_error': 1.5}},
    ]

    first = paired_comparison(legacy, hybrid, bootstrap_samples=25)
    second = paired_comparison(legacy, hybrid, bootstrap_samples=25)

    assert first['jointly_valid_pairs'] == 2
    assert first['hybrid_improves_count'] == 1
    assert first['legacy_improves_count'] == 1
    assert first['mean_difference_bootstrap_ci95'] == second['mean_difference_bootstrap_ci95']


def test_frozen_hybrid_config_values_load_unchanged():
    """The promoted hybrid config contains the frozen validation values."""
    with open(REPO_ROOT / 'config/aghri_zed_livox_hybrid.yaml', 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)['lidar_fusion']['ros__parameters']

    assert params['association_method'] == 'hybrid_depth_legacy'
    assert params['centre_statistic'] == 'mean'
    assert params['box_bottom_trim_ratio'] == 0.0
    assert params['hybrid_min_support_ratio'] == pytest.approx(0.20)
    assert params['hybrid_fallback_on_single_cluster'] is False


def test_launch_file_declares_params_file_argument():
    """The launch file keeps its default config and accepts params_file."""
    text = (REPO_ROOT / 'launch/launch.py').read_text(encoding='utf-8')

    assert 'DeclareLaunchArgument' in text
    assert "'params_file'" in text
    assert 'LaunchConfiguration' in text
