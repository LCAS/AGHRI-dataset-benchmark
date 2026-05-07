"""Convert raw KITTI label_2 + calib files to AGHRI-format pkl.

Reads directly from training/label_2/ and training/calib/ — no need to run
create_data.py first.  Keeps only Pedestrian instances as a single 'person'
class (label 0).  All other classes are discarded.

Output pkl format (compatible with AghriLidarDataset and Kitti3DMetric):
  {
    'metainfo': {'classes': ('person',)},
    'data_list': [
      {
        'sample_idx': '000000',
        'lidar_points': {
            'lidar_path': 'training/velodyne/000000.bin',
            'num_pts_feats': 4,
        },
        'instances': [
          {'bbox_3d': [x, y, z, l, w, h, yaw],   # LiDAR frame, bottom-center
           'bbox_label_3d': 0},
          ...
        ],
      },
      ...
    ],
  }

USAGE:
  python src/prepare_kitti_lidar_dataset.py \
  --kitti-root /data/kitti \
  --out-root /data/kitti \
  --splits train val
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

PERSON_CLASSES = {'Pedestrian'}


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _parse_calib(calib_path: Path) -> dict:
    data = {}
    with calib_path.open('r') as fh:
        for line in fh:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.array([float(v) for v in vals.split()])
    return data


def _rect_to_lidar_mat(calib: dict) -> np.ndarray:
    """Return 4×4 matrix that maps rectified-camera coords to LiDAR coords."""
    R0 = calib['R0_rect'].reshape(3, 3)
    Tr = calib['Tr_velo_to_cam'].reshape(3, 4)
    Tr4 = np.eye(4)
    Tr4[:3, :] = Tr
    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0
    return np.linalg.inv(Tr4) @ np.linalg.inv(R0_4)


# ---------------------------------------------------------------------------
# Box conversion: KITTI camera → AGHRI LiDAR
# ---------------------------------------------------------------------------

def _cam_box_to_lidar(h: float, w: float, l: float,
                      x_c: float, y_c: float, z_c: float,
                      ry: float,
                      mat: np.ndarray) -> list:
    """Convert a KITTI camera-frame 3D box to LiDAR-frame box.

    KITTI label columns: h w l x_c y_c z_c ry
      (x_c, y_c, z_c) = bottom-center in rectified camera frame
      h = height (camera y), w = width (camera x), l = length (camera z)
      ry = rotation around camera y-axis

    Output: [x_l, y_l, z_bottom, l, w, h, yaw] (AGHRI LiDAR format)
      origin is bottom-center, z_bottom is ground level in LiDAR frame
    """
    # Geometric center in camera frame (y_c is bottom; y_cam points down)
    center_cam = np.array([x_c, y_c - h / 2.0, z_c, 1.0])
    cx, cy, cz = (mat @ center_cam)[:3]
    z_bottom = cz - h / 2.0

    # Standard yaw conversion: camera ry → LiDAR yaw
    yaw = float(-ry - np.pi / 2.0)
    yaw = float((yaw + np.pi) % (2.0 * np.pi) - np.pi)  # normalize to [-π, π]

    return [float(cx), float(cy), float(z_bottom), float(l), float(w), float(h), yaw]


# ---------------------------------------------------------------------------
# Label parser
# ---------------------------------------------------------------------------

def _parse_labels(label_path: Path, mat: np.ndarray) -> list:
    """Return list of AGHRI-format instance dicts for person-class objects."""
    instances = []
    with label_path.open('r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in PERSON_CLASSES:
                continue
            truncation = float(parts[1])
            occlusion = int(parts[2])
            # 2D bbox in image pixels: left, top, right, bottom
            height_in_image = float(parts[7]) - float(parts[5])
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x_c, y_c, z_c = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            bbox_lidar = _cam_box_to_lidar(h, w, l, x_c, y_c, z_c, ry, mat)
            instances.append({
                'bbox_3d': bbox_lidar,
                'bbox_label_3d': 0,
                'truncation': truncation,
                'occlusion': occlusion,
                'height_in_image': height_in_image,
            })
    return instances


# ---------------------------------------------------------------------------
# Split converter
# ---------------------------------------------------------------------------

def convert_split(kitti_root: Path, out_pkl: Path, split: str) -> None:
    imageset_file = kitti_root / 'ImageSets' / f'{split}.txt'
    if not imageset_file.exists():
        raise FileNotFoundError(f'ImageSets file not found: {imageset_file}')

    sample_ids = [line.strip() for line in imageset_file.read_text().splitlines() if line.strip()]

    label_dir = kitti_root / 'training' / 'label_2'
    calib_dir = kitti_root / 'training' / 'calib'
    velodyne_dir = kitti_root / 'training' / 'velodyne'

    # Test split has no labels — produce samples with empty instances
    has_labels = label_dir.exists()

    data_list = []
    skipped = 0
    for sid in sample_ids:
        bin_rel = f'training/velodyne/{sid}.bin'
        bin_abs = velodyne_dir / f'{sid}.bin'
        if not bin_abs.exists():
            skipped += 1
            continue

        entry = {
            'sample_idx': sid,
            'lidar_points': {
                'lidar_path': bin_rel,
                'num_pts_feats': 4,
            },
            'instances': [],
        }

        if has_labels:
            calib_path = calib_dir / f'{sid}.txt'
            label_path = label_dir / f'{sid}.txt'
            if calib_path.exists() and label_path.exists():
                calib = _parse_calib(calib_path)
                mat = _rect_to_lidar_mat(calib)
                entry['instances'] = _parse_labels(label_path, mat)

        data_list.append(entry)

    if skipped:
        print(f'  Warning: skipped {skipped} missing .bin files.')

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open('wb') as fh:
        pickle.dump({'metainfo': {'classes': ('person',)}, 'data_list': data_list}, fh)

    total_inst = sum(len(d['instances']) for d in data_list)
    print(f'  [{split:10s}] {len(data_list):5d} samples | {total_inst:6d} person instances -> {out_pkl}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Prepare KITTI data for AGHRI-format benchmarking. '
                    'No need to run create_data.py first.'
    )
    ap.add_argument('--kitti-root', required=True,
                    help='KITTI root directory (contains ImageSets/, training/, testing/).')
    ap.add_argument('--out-root', required=True,
                    help='Output root; pkls are written to <out-root>/infos/.')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'],
                    help='Which ImageSets splits to process (default: train val).')
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root)
    out_root = Path(args.out_root)

    print(f'KITTI root : {kitti_root}')
    print(f'Output root: {out_root}')
    print(f'Person classes merged: {sorted(PERSON_CLASSES)}')
    print()

    for split in args.splits:
        out_pkl = out_root / 'infos' / f'kitti_person_infos_{split}.pkl'
        convert_split(kitti_root, out_pkl, split)


if __name__ == '__main__':
    main()
