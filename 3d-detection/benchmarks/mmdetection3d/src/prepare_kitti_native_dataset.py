"""Prepare KITTI dataset for the native MMDetection3D pipeline.

Generates (without requiring mmdet3d's tools/create_data.py):
  kitti_infos_train.pkl, kitti_infos_val.pkl   — official KittiDataset format
  kitti_dbinfos_train.pkl                       — GT-sampling database for ObjectSample
  kitti_gt_database/                            — per-instance point-cloud .bin files

WHY THE CURRENT PREPARE_KITTI_LIDAR_DATASET.PY APPROACH GIVES LOW PERFORMANCE:
================================================================================

  Bug 1 — Training (z-convention):
    prepare_kitti_lidar_dataset.py stores z_CENTER in each bbox_3d.
    AghriLidarDataset wraps those arrays with origin=(0.5, 0.5, 0.0),
    which tells mmdet3d "z is the BOTTOM of the box."
    mmdet3d therefore shifts every GT box UP by h/2 (≈ 0.87 m for a 1.73 m
    pedestrian).  The model learns wrong z regression targets.

  Bug 2 — Evaluation (3D IoU):
    metric.py _iou3d computes:
        z1_min = box[2]          ← treated as z_bottom
        z1_max = z1_min + box[5] ← + full height
    GT stores z_center → effective GT height window is [z_center, z_center+h]
    instead of [z_center-h/2, z_center+h/2].  Height overlap is ~50 % too
    low → 3D IoU is deflated → 3D AP collapses 3–5×.

  Bug 3 — AP protocol:
    Custom _compute_ap uses all-points VOC interpolation; published KITTI
    numbers use R40 (40-point uniform), implemented by the official KittiMetric.

CORRECT APPROACH (this script):
  Store 3D boxes in CAMERA frame (bottom-center, as per KITTI label format).
  KittiDataset converts them to LiDAR internally with the correct z_bottom
  convention.  KittiMetric then uses official KITTI R40 evaluation.

USAGE:
  # Prepare data (run once):
  python src/prepare_kitti_native_dataset.py --kitti-root D:/AOC/datasets/kitti-3d

  # Show z-convention bugs in the existing custom pkl:
  python src/prepare_kitti_native_dataset.py \\
      --kitti-root D:/AOC/datasets/kitti-3d --diagnose --skip-create

  # Skip data prep if pkls already exist:
  python src/prepare_kitti_native_dataset.py \\
      --kitti-root D:/AOC/datasets/kitti-3d --skip-create
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# KITTI class definitions (full three-class set — dataset filters per config)
# ---------------------------------------------------------------------------

KITTI_CLASSES = ('Car', 'Pedestrian', 'Cyclist')
CLASS_TO_LABEL = {c: i for i, c in enumerate(KITTI_CLASSES)}

# Classes included in the GT database (only classes we train on)
DB_CLASSES = {'Pedestrian'}

# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _parse_calib(calib_path: Path) -> dict:
    raw = {}
    with calib_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            raw[key.strip()] = np.array([float(v) for v in vals.split()])
    return {
        'P2':            raw['P2'].reshape(3, 4).astype(np.float64),
        'R0_rect':       raw['R0_rect'].reshape(3, 3).astype(np.float64),
        'Tr_velo_to_cam':raw['Tr_velo_to_cam'].reshape(3, 4).astype(np.float64),
    }


def _cam_to_lidar_mat4(calib: dict) -> np.ndarray:
    """4×4 matrix: rectified-camera coords → LiDAR coords."""
    R0  = calib['R0_rect']          # 3×3
    Tr  = calib['Tr_velo_to_cam']   # 3×4
    Tr4 = np.eye(4, dtype=np.float64)
    Tr4[:3, :] = Tr
    R04 = np.eye(4, dtype=np.float64)
    R04[:3, :3] = R0
    return np.linalg.inv(Tr4) @ np.linalg.inv(R04)


def _cam_bottom_to_lidar(x: float, y: float, z: float,
                          mat: np.ndarray) -> np.ndarray:
    """Convert camera-frame BOTTOM center → LiDAR-frame BOTTOM center."""
    pt_cam = np.array([x, y, z, 1.0], dtype=np.float64)
    return (mat @ pt_cam)[:3]


def _cam_ry_to_lidar_yaw(ry: float) -> float:
    """Convert camera rotation_y → LiDAR yaw (both in [-π, π])."""
    yaw = float(-ry - np.pi / 2.0)
    return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------

def _difficulty(height_px: float, occluded: int, truncated: float) -> int:
    if height_px >= 40 and occluded <= 0 and truncated <= 0.15:
        return 0
    if height_px >= 25 and occluded <= 1 and truncated <= 0.30:
        return 1
    if height_px >= 25 and occluded <= 2 and truncated <= 0.50:
        return 2
    return -1


# ---------------------------------------------------------------------------
# Label parser
# ---------------------------------------------------------------------------

def _parse_label(label_path: Path, calib: dict) -> tuple:
    """Parse a KITTI label_2 file.

    Returns:
        annos   : dict in KITTI evaluation format (all objects incl. DontCare)
        instances : list of dicts in mmdet3d modern format (target classes only)
        lidar_boxes : list of [x,y,z_bottom,l,w,h,yaw] in LiDAR for dbinfos
    """
    mat = _cam_to_lidar_mat4(calib)

    names_, trunc_, occ_, alpha_ = [], [], [], []
    bbox2d_, dims_, locs_, rots_ = [], [], [], []
    diffs_, grpids_ = [], []

    instances = []
    lidar_boxes_by_cls = {cls: [] for cls in KITTI_CLASSES}

    group_counter = 0
    with label_path.open() as fh:
        for raw_line in fh:
            parts = raw_line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]

            trunc = float(parts[1])
            occ   = int(parts[2])
            alpha = float(parts[3])
            x1, y1, x2, y2 = (float(p) for p in parts[4:8])
            h, w, l        = (float(p) for p in parts[8:11])
            x_c, y_c, z_c  = (float(p) for p in parts[11:14])
            ry             = float(parts[14])

            height_px = y2 - y1
            diff = _difficulty(height_px, occ, trunc) if cls != 'DontCare' else -1

            names_.append(cls)
            trunc_.append(trunc)
            occ_.append(occ)
            alpha_.append(alpha)
            bbox2d_.append([x1, y1, x2, y2])
            dims_.append([h, w, l])       # [height, width, length] — KITTI order
            locs_.append([x_c, y_c, z_c])  # camera bottom-center
            rots_.append(ry)
            diffs_.append(diff)
            grpids_.append(group_counter)
            group_counter += 1

            if cls in KITTI_CLASSES:
                label_id = CLASS_TO_LABEL[cls]
                # bbox_3d: [x, y, z, l, w, h, ry] — camera BOTTOM center
                bbox_3d = [x_c, y_c, z_c, l, w, h, ry]
                instances.append({
                    'bbox':          [x1, y1, x2, y2],
                    'bbox_label':    label_id,
                    'bbox_3d':       bbox_3d,
                    'bbox_label_3d': label_id,
                    'bbox_3d_isvalid': diff >= 0,
                    'truncated':     trunc,
                    'occluded':      occ,
                    'alpha':         alpha,
                })

                # LiDAR-frame box for the GT database (z is BOTTOM, not center)
                lidar_bot = _cam_bottom_to_lidar(x_c, y_c, z_c, mat)
                yaw_lidar = _cam_ry_to_lidar_yaw(ry)
                lidar_boxes_by_cls[cls].append(
                    (label_id, diff, grpids_[-1],
                     [float(lidar_bot[0]), float(lidar_bot[1]), float(lidar_bot[2]),
                      l, w, h, yaw_lidar])
                )

    n = len(names_)
    if n > 0:
        annos = {
            'name':       np.array(names_),
            'truncated':  np.array(trunc_,  dtype=np.float32),
            'occluded':   np.array(occ_,    dtype=np.int32),
            'alpha':      np.array(alpha_,  dtype=np.float32),
            'bbox':       np.array(bbox2d_, dtype=np.float32),
            'dimensions': np.array(dims_,   dtype=np.float32),  # (N, 3) [h, w, l]
            'location':   np.array(locs_,   dtype=np.float32),  # (N, 3) cam bottom-center
            'rotation_y': np.array(rots_,   dtype=np.float32),
            'difficulty': np.array(diffs_,  dtype=np.int32),
            'index':      np.arange(n,       dtype=np.int32),
            'group_ids':  np.array(grpids_, dtype=np.int32),
            'camera_id':  np.zeros(n,        dtype=np.int32),
        }
    else:
        annos = {
            'name':       np.array([]),
            'truncated':  np.array([], dtype=np.float32),
            'occluded':   np.array([], dtype=np.int32),
            'alpha':      np.array([], dtype=np.float32),
            'bbox':       np.zeros((0, 4), dtype=np.float32),
            'dimensions': np.zeros((0, 3), dtype=np.float32),
            'location':   np.zeros((0, 3), dtype=np.float32),
            'rotation_y': np.array([], dtype=np.float32),
            'difficulty': np.array([], dtype=np.int32),
            'index':      np.array([], dtype=np.int32),
            'group_ids':  np.array([], dtype=np.int32),
            'camera_id':  np.array([], dtype=np.int32),
        }

    return annos, instances, lidar_boxes_by_cls


# ---------------------------------------------------------------------------
# Points-in-box helper
# ---------------------------------------------------------------------------

def _points_in_box(pts_xyz: np.ndarray,
                   bx: float, by: float, z_bottom: float,
                   l: float, w: float, h: float, yaw: float) -> np.ndarray:
    """Boolean mask: which rows of pts_xyz fall inside the 3D box."""
    z_center = z_bottom + h / 2.0
    dx = pts_xyz[:, 0] - bx
    dy = pts_xyz[:, 1] - by
    dz = pts_xyz[:, 2] - z_center
    c, s = float(np.cos(-yaw)), float(np.sin(-yaw))
    xr   =  dx * c - dy * s
    yr   =  dx * s + dy * c
    eps  = 1e-3
    return (np.abs(xr) <= l / 2.0 + eps) & (np.abs(yr) <= w / 2.0 + eps) & (np.abs(dz) <= h / 2.0 + eps)


# ---------------------------------------------------------------------------
# Info creation for a single split
# ---------------------------------------------------------------------------

def create_kitti_infos(kitti_root: Path, out_dir: Path, split: str) -> None:
    """Write kitti_infos_{split}.pkl to out_dir."""
    imageset = kitti_root / 'ImageSets' / f'{split}.txt'
    if not imageset.exists():
        print(f"  Skip {split}: {imageset} not found.")
        return

    sample_ids = [l.strip() for l in imageset.read_text().splitlines() if l.strip()]
    label_dir  = kitti_root / 'training' / 'label_2'
    calib_dir  = kitti_root / 'training' / 'calib'

    has_labels = label_dir.exists()
    data_list  = []

    for sid in sample_ids:
        sid_int = int(sid)
        info: dict = {
            'sample_idx': sid_int,
            'images': {
                'CAM2': {
                    'img_path': f'training/image_2/{sid}.png',
                    # cam2img populated below if calib available
                }
            },
            'lidar_points': {
                'num_pts_feats': 4,
                'lidar_path': f'{sid}.bin',  # relative to data_prefix['pts']
            },
            'instances': [],
        }

        calib_path = calib_dir / f'{sid}.txt'
        if calib_path.exists():
            calib = _parse_calib(calib_path)

            # lidar2cam = R0_rect @ [Tr_velo_to_cam; 0 0 0 1]  (4×4)
            # This transforms LiDAR points to rectified-camera frame.
            # KittiDataset.parse_ann_info uses it to convert camera-frame GT
            # boxes to LiDAR frame during dataset loading.
            Tr4 = np.eye(4, dtype=np.float64)
            Tr4[:3, :] = calib['Tr_velo_to_cam']
            R04 = np.eye(4, dtype=np.float64)
            R04[:3, :3] = calib['R0_rect']
            lidar2cam = (R04 @ Tr4).tolist()

            info['calib'] = {
                'P2':             calib['P2'].tolist(),
                'R0_rect':        calib['R0_rect'].tolist(),
                'Tr_velo_to_cam': calib['Tr_velo_to_cam'].tolist(),
            }
            info['images']['CAM2']['cam2img']   = calib['P2'].tolist()
            info['images']['CAM2']['lidar2cam'] = lidar2cam
        else:
            calib = None

        if has_labels:
            label_path = label_dir / f'{sid}.txt'
            if label_path.exists() and calib is not None:
                annos, instances, _ = _parse_label(label_path, calib)
                info['annos']     = annos
                info['instances'] = instances

        data_list.append(info)

    out_pkl = out_dir / f'kitti_infos_{split}.pkl'
    payload = {
        'metainfo':  {'dataset': 'kitti', 'classes': KITTI_CLASSES},
        'data_list': data_list,
    }
    with out_pkl.open('wb') as fh:
        pickle.dump(payload, fh)

    n_inst = sum(len(d['instances']) for d in data_list)
    print(f"  [{split:10s}] {len(data_list):5d} samples | {n_inst:6d} total KITTI-class instances"
          f"  →  {out_pkl}")


# ---------------------------------------------------------------------------
# GT database (for ObjectSample augmentation)
# ---------------------------------------------------------------------------

def create_gt_database(kitti_root: Path, out_dir: Path) -> None:
    """Create kitti_dbinfos_train.pkl and kitti_gt_database/ sub-directory."""
    imageset = kitti_root / 'ImageSets' / 'train.txt'
    if not imageset.exists():
        print("  Skipping GT database: ImageSets/train.txt not found.")
        return

    sample_ids = [l.strip() for l in imageset.read_text().splitlines() if l.strip()]
    label_dir  = kitti_root / 'training' / 'label_2'
    calib_dir  = kitti_root / 'training' / 'calib'
    velo_dir   = kitti_root / 'training' / 'velodyne'

    db_dir = out_dir / 'kitti_gt_database'
    db_dir.mkdir(parents=True, exist_ok=True)

    dbinfos: dict = {cls: [] for cls in KITTI_CLASSES}
    total_saved = 0

    for sid in sample_ids:
        label_path = label_dir / f'{sid}.txt'
        calib_path = calib_dir / f'{sid}.txt'
        velo_path  = velo_dir  / f'{sid}.bin'

        if not (label_path.exists() and calib_path.exists() and velo_path.exists()):
            continue

        calib = _parse_calib(calib_path)
        _, _, lidar_boxes_by_cls = _parse_label(label_path, calib)

        # Load point cloud
        pts = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)
        pts_xyz = pts[:, :3]

        for cls, box_entries in lidar_boxes_by_cls.items():
            if cls not in DB_CLASSES:
                continue
            for gt_idx, (_, diff, grp_id, box7) in enumerate(box_entries):
                bx, by, z_bot, l, w, h, yaw = box7
                mask   = _points_in_box(pts_xyz, bx, by, z_bot, l, w, h, yaw)
                in_pts = pts[mask]
                if len(in_pts) == 0:
                    continue

                rel_path = f'kitti_gt_database/{sid}_{cls}_{gt_idx}.bin'
                abs_path = out_dir / rel_path
                in_pts.tofile(str(abs_path))

                dbinfos[cls].append({
                    'name':            cls,
                    'path':            rel_path,
                    'image_idx':       int(sid),
                    'gt_idx':          gt_idx,
                    'box3d_lidar':     np.array(box7, dtype=np.float32),
                    'num_points_in_gt':len(in_pts),
                    'difficulty':      diff,
                    'group_id':        grp_id,
                })
                total_saved += 1

    out_pkl = out_dir / 'kitti_dbinfos_train.pkl'
    with out_pkl.open('wb') as fh:
        pickle.dump(dbinfos, fh)

    summary = {cls: len(v) for cls, v in dbinfos.items() if v}
    print(f"  GT database: {total_saved} instances saved  →  {out_pkl}")
    print(f"  Per-class:   {summary}")


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------

REQUIRED_TRAIN_DIRS = ['training/velodyne', 'training/label_2', 'training/calib']
REQUIRED_IMAGESETS  = ['train.txt', 'val.txt']


def verify_dataset(kitti_root: Path) -> None:
    print("Verifying KITTI dataset structure ...")
    ok = True

    for rel in REQUIRED_TRAIN_DIRS:
        d = kitti_root / rel
        if not d.exists():
            print(f"  MISSING : {d}")
            ok = False
        else:
            n = sum(1 for _ in d.iterdir())
            print(f"  OK ({n:6d} files) : {d}")

    imageset_dir = kitti_root / 'ImageSets'
    for fname in REQUIRED_IMAGESETS:
        f = imageset_dir / fname
        if not f.exists():
            print(f"  MISSING : {f}")
            ok = False
        else:
            ids = [l.strip() for l in f.read_text().splitlines() if l.strip()]
            print(f"  OK ({len(ids):6d} ids   ) : {f}")

    # trainval.txt is optional — create if missing
    trainval = imageset_dir / 'trainval.txt'
    if not trainval.exists():
        train_set = set((imageset_dir / 'train.txt').read_text().splitlines())
        val_set   = set((imageset_dir / 'val.txt').read_text().splitlines())
        combined  = sorted(train_set | val_set)
        trainval.write_text('\n'.join(combined) + '\n')
        print(f"  Created trainval.txt ({len(combined)} ids)")

    if not ok:
        raise FileNotFoundError(
            "Dataset verification failed. Fix the missing directories above."
        )
    print("Dataset structure OK.\n")


# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------

def verify_output(out_dir: Path) -> None:
    print("Checking output pkl files ...")
    for name in ['kitti_infos_train.pkl', 'kitti_infos_val.pkl']:
        pkl = out_dir / name
        if not pkl.exists():
            print(f"  MISSING : {pkl}")
            continue
        with pkl.open('rb') as fh:
            data = pickle.load(fh)
        n_s   = len(data.get('data_list', []))
        n_i   = sum(len(d.get('instances', [])) for d in data.get('data_list', []))
        ped_i = sum(
            sum(1 for inst in d.get('instances', []) if inst.get('bbox_label') == 1)
            for d in data.get('data_list', [])
        )
        print(f"  OK : {pkl.name}  |  {n_s} samples, {n_i} KITTI-class instances"
              f"  (Pedestrian: {ped_i})")

    dbinfo = out_dir / 'kitti_dbinfos_train.pkl'
    if dbinfo.exists():
        with dbinfo.open('rb') as fh:
            db = pickle.load(fh)
        s = {cls: len(items) for cls, items in db.items() if items}
        print(f"  OK : kitti_dbinfos_train.pkl  |  {s}")
    else:
        print(f"  MISSING (optional, needed for ObjectSample) : {dbinfo}")
    print()


# ---------------------------------------------------------------------------
# Diagnostic: expose z-convention bugs in the existing custom pkl
# ---------------------------------------------------------------------------

def diagnose_custom_pkl(kitti_root: Path) -> None:
    custom_pkl = kitti_root / 'infos' / 'kitti_person_infos_val.pkl'
    if not custom_pkl.exists():
        print(f"Custom pkl not found at {custom_pkl}; skipping.\n")
        return

    with custom_pkl.open('rb') as fh:
        data = pickle.load(fh)
    data_list = data.get('data_list', [])

    n_s = len(data_list)
    n_i = sum(len(d['instances']) for d in data_list)

    print("=" * 72)
    print("DIAGNOSTIC: Current custom AGHRI-format pkl (kitti_person_infos_val.pkl)")
    print("=" * 72)
    print(f"  Samples: {n_s}  |  Pedestrian instances: {n_i}\n")
    print(f"  {'sid':>10}  {'z_stored':>10}  {'h':>6}  "
          f"{'correct_z_range':>25}  {'buggy_z_range_used_by_iou3d':>30}")
    print("  " + "-" * 90)

    shown = 0
    for entry in data_list:
        if shown >= 5:
            break
        for inst in entry['instances'][:2]:
            x, y, z, l, w, h, yaw = inst['bbox_3d']
            correct = f"[{z-h/2:.2f}, {z+h/2:.2f}]"
            buggy   = f"[{z:.2f}, {z+h:.2f}]  (+{h/2:.2f}m shift)"
            sid = entry.get('sample_idx', '?')
            print(f"  {str(sid):>10}  {z:>10.3f}  {h:>6.3f}  {correct:>25}  {buggy}")
            shown += 1

    print()
    print("  BUG 1 (training):")
    print("    AghriLidarDataset loads z_center with origin=(0.5,0.5,0.0).")
    print("    mmdet3d adds h/2 to all GT boxes → they float h/2 above reality")
    print("    during training → model learns wrong z regression targets.")
    print()
    print("  BUG 2 (evaluation):")
    print("    _iou3d in metric.py uses z_stored as z_bottom → uses 'buggy_z_range'")
    print("    above instead of 'correct_z_range' → height overlap ~50% too low")
    print("    → 3D IoU is deflated by ~3-5x → 3D AP collapses.")
    print()
    print("  BUG 3 (AP protocol):")
    print("    Custom _compute_ap ≠ KITTI R40 (40-point) used for published numbers.")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--kitti-root', required=True,
        help='Root of the KITTI dataset (contains ImageSets/, training/, testing/).',
    )
    parser.add_argument(
        '--out-dir', default=None,
        help='Directory for output pkl files (default: same as --kitti-root).',
    )
    parser.add_argument(
        '--diagnose', action='store_true',
        help='Show z-convention diagnostics for the existing custom pkl.',
    )
    parser.add_argument(
        '--skip-create', action='store_true',
        help='Skip info generation (just verify existing files).',
    )
    parser.add_argument(
        '--no-db', action='store_true',
        help='Skip GT database creation (ObjectSample augmentation will be unavailable).',
    )
    args = parser.parse_args()

    kitti_root = Path(args.kitti_root).resolve()
    out_dir    = Path(args.out_dir).resolve() if args.out_dir else kitti_root

    if args.diagnose:
        diagnose_custom_pkl(kitti_root)

    verify_dataset(kitti_root)

    if not args.skip_create:
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Creating KITTI info pkl files ...")
        for split in ('train', 'val', 'test'):
            create_kitti_infos(kitti_root, out_dir, split)
        print()

        if not args.no_db:
            print("Creating GT database (ObjectSample) ...")
            create_gt_database(kitti_root, out_dir)
            print()
    else:
        print("Skipping info creation (--skip-create).\n")

    verify_output(out_dir)

    print("Done.  Next steps:")
    print("  python src/run_kitti_native_benchmark.py \\")
    print("      --config configs/benchmark_mmdetection3d_kitti_native_pointpillars.yaml \\")
    print("      --out 3d-detection/reports/benchmarks/summary/mmdetection3d/"
          "summary_mmdetection3d_kitti_native_pointpillars.csv")


if __name__ == '__main__':
    main()
