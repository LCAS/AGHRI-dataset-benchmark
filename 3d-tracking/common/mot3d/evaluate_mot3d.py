"""
Evaluate 3D tracking predictions against ground truth using BEV IoU distance.

Reads per-scene MOT3D CSV files (format: frame_id,track_id,x,y,z,l,w,h,yaw,score)
and computes:
  - CLEAR MOT:  MOTA, MOTP, num_switches, precision, recall, num_fp, num_fn
  - ID metrics: IDF1, IDP, IDR
  - Track-level: mostly_tracked, mostly_lost, num_fragmentations
  - HOTA:       HOTA, DetA, AssA  (averaged over alpha thresholds 0.05..0.95)

Usage:
    python evaluate_mot3d.py \\
        --gt-dir  /path/to/gt_mot3d \\
        --pred-dir /path/to/tracks/ab3dmot \\
        --iou-threshold 0.25 \\
        --out-csv reports/ab3dmot_metrics.csv \\
        --out-json reports/ab3dmot_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

COMMON_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(COMMON_ROOT))
from bev_iou import iou_matrix
from mot3d_writer import load_mot3d
from hota import compute_hota

DEFAULT_METRICS = [
    "num_frames",
    "idf1",
    "idp",
    "idr",
    "precision",
    "recall",
    "mota",
    "motp",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
    "num_fragmentations",
]


def _boxes_from_frame(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a frame's (N, 9) rows into track_ids (N,) and boxes (N, 7)."""
    if rows is None or len(rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 7), dtype=np.float64)
    ids = rows[:, 0].astype(np.int64)
    boxes = rows[:, 1:8]  # x,y,z,l,w,h,yaw
    return ids, boxes


def evaluate_scene(
    gt_file: Path,
    pred_file: Path,
    iou_threshold: float,
) -> Tuple:
    """
    Evaluate one scene.

    Returns (motmetrics_accumulator, gt_data, pred_data) so that HOTA can be
    computed from the same loaded data without re-reading files.
    """
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise RuntimeError(
            "motmetrics is not installed. Install it with: pip install motmetrics"
        ) from exc

    gt_data = load_mot3d(gt_file)
    pred_data = load_mot3d(pred_file) if pred_file.exists() else {}

    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))
    acc = mm.MOTAccumulator(auto_id=False)

    for fid in all_frames:
        gt_ids, gt_boxes = _boxes_from_frame(gt_data.get(fid))
        pred_ids, pred_boxes = _boxes_from_frame(pred_data.get(fid))

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            dist = np.full((len(gt_ids), len(pred_ids)), np.nan)
        else:
            iou = iou_matrix(gt_boxes, pred_boxes)
            dist = np.where(iou >= iou_threshold, 1.0 - iou, np.nan)

        acc.update(gt_ids.tolist(), pred_ids.tolist(), dist, frameid=fid)

    return acc, gt_data, pred_data


def evaluate_all(
    gt_dir: Path,
    pred_dir: Path,
    iou_threshold: float = 0.25,
    output_csv: Optional[Path] = None,
    output_json: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
) -> None:
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise RuntimeError(
            "motmetrics is not installed. Install it with: pip install motmetrics"
        ) from exc

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda arr, dtype=float: np.asarray(arr, dtype=dtype)

    metrics = metrics or DEFAULT_METRICS

    gt_files = sorted(gt_dir.glob("*.csv"))
    if not gt_files:
        raise FileNotFoundError(f"No GT CSV files found in {gt_dir}")

    accs = []
    names = []
    hota_per_scene: Dict[str, Dict[str, float]] = {}

    for gt_file in gt_files:
        pred_file = pred_dir / gt_file.name
        acc, gt_data, pred_data = evaluate_scene(gt_file, pred_file, iou_threshold)
        accs.append(acc)
        names.append(gt_file.stem)

        hota_per_scene[gt_file.stem] = compute_hota(gt_data, pred_data, iou_matrix)
        print(f"  Evaluated: {gt_file.stem}  "
              f"HOTA={hota_per_scene[gt_file.stem]['hota']:.3f}")

    # --- motmetrics (CLEAR MOT + ID metrics) ---
    metrics_handler = mm.metrics.create()
    summary = metrics_handler.compute_many(
        accs,
        metrics=metrics,
        names=names,
        generate_overall=True,
    )

    print("\n" + mm.io.render_summary(
        summary,
        formatters=metrics_handler.formatters,
        namemap=mm.io.motchallenge_metric_names,
    ))

    # Overall HOTA = mean of per-scene scores
    overall_hota = {
        k: float(np.mean([v[k] for v in hota_per_scene.values()]))
        for k in ("hota", "deta", "assa")
    }
    print(f"\n  HOTA (overall): "
          f"HOTA={overall_hota['hota']:.3f}  "
          f"DetA={overall_hota['deta']:.3f}  "
          f"AssA={overall_hota['assa']:.3f}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        # Augment summary with HOTA columns before saving
        for col, key in [("hota", "hota"), ("deta", "deta"), ("assa", "assa")]:
            values = [
                hota_per_scene.get(name, {}).get(key, float("nan"))
                for name in names
            ] + [overall_hota[key]]
            summary[col] = values
        summary.to_csv(output_csv)
        print(f"CSV written: {output_csv}")

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        rows = summary.reset_index().to_dict(orient="records")
        # Inject HOTA into each row (summary may not have the columns yet if csv not requested)
        for row in rows:
            scene = row.get("index", "OVERALL")
            src = hota_per_scene.get(scene, overall_hota)
            row["hota"] = round(src.get("hota", float("nan")), 6)
            row["deta"] = round(src.get("deta", float("nan")), 6)
            row["assa"] = round(src.get("assa", float("nan")), 6)
        output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"JSON written: {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D tracking predictions using BEV IoU distance.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--gt-dir", required=True, type=Path,
                        help="Directory with per-scene GT MOT3D CSV files.")
    parser.add_argument("--pred-dir", required=True, type=Path,
                        help="Directory with per-scene predicted MOT3D CSV files.")
    parser.add_argument("--iou-threshold", type=float, default=0.25,
                        help="Minimum BEV IoU to count as a match (default: 0.25).")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    evaluate_all(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        iou_threshold=args.iou_threshold,
        output_csv=args.out_csv,
        output_json=args.out_json,
    )


if __name__ == "__main__":
    main()
