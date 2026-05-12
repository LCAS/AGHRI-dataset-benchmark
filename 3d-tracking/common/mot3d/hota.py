"""
HOTA (Higher Order Tracking Accuracy) for 3D MOT evaluation.

Reference: Luiten et al., "HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking", IJCV 2021.

HOTA(alpha) = sqrt( DetA(alpha) * AssA(alpha) )
HOTA        = mean over alpha in {0.05, 0.10, ..., 0.95}

For each localization threshold alpha:
  - DetA = TP / (TP + FP + FN)               — detection accuracy
  - AssA = (1/TP) * sum_c [TPA(c) / (TPA(c) + FPA(c) + FNA(c))]  — association accuracy

where for each TP detection c matched as (gt_id, pred_id):
  TPA(c) = frames where gt_id and pred_id are TP-matched together
  FNA(c) = frames where gt_id is TP but matched to a different predicted track
  FPA(c) = frames where pred_id is TP but matched to a different GT track

Matching uses greedy assignment (highest BEV IoU first), which is consistent
with the original HOTA paper.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np

HOTA_ALPHAS: np.ndarray = np.round(np.arange(0.05, 0.96, 0.05), 2)   # 19 thresholds


def compute_hota(
    gt_data: Dict[int, np.ndarray],
    pred_data: Dict[int, np.ndarray],
    iou_matrix_fn: Callable,
) -> Dict[str, float]:
    """
    Compute HOTA, DetA, and AssA averaged over all alpha thresholds.

    Args:
        gt_data:       frame_id → (N, 9) array [track_id, x,y,z,l,w,h,yaw,score]
        pred_data:     frame_id → (M, 9) array (same layout)
        iou_matrix_fn: function(boxes1, boxes2) → (N, M) BEV IoU matrix

    Returns:
        dict with keys "hota", "deta", "assa" (all floats in [0, 1])
    """
    all_frames = sorted(set(gt_data) | set(pred_data))

    # Pre-compute IoU and ID arrays once (reused across all alpha thresholds).
    iou_cache:      Dict[int, np.ndarray] = {}
    gt_ids_cache:   Dict[int, np.ndarray] = {}
    pred_ids_cache: Dict[int, np.ndarray] = {}

    for fid in all_frames:
        gt_rows   = gt_data.get(fid)
        pred_rows = pred_data.get(fid)

        gt_ids   = gt_rows[:, 0].astype(np.int64)   if gt_rows   is not None and len(gt_rows)   else np.empty(0, np.int64)
        pred_ids = pred_rows[:, 0].astype(np.int64) if pred_rows is not None and len(pred_rows) else np.empty(0, np.int64)
        gt_boxes   = gt_rows[:, 1:8]   if len(gt_ids)   else np.empty((0, 7))
        pred_boxes = pred_rows[:, 1:8] if len(pred_ids) else np.empty((0, 7))

        gt_ids_cache[fid]   = gt_ids
        pred_ids_cache[fid] = pred_ids
        iou_cache[fid] = (
            iou_matrix_fn(gt_boxes, pred_boxes)
            if len(gt_ids) > 0 and len(pred_ids) > 0
            else np.empty((len(gt_ids), len(pred_ids)), dtype=np.float64)
        )

    hota_scores: List[float] = []
    deta_scores: List[float] = []
    assa_scores: List[float] = []

    for alpha in HOTA_ALPHAS:
        tp_total = 0
        fp_total = 0
        fn_total = 0

        # gt_to_pred[gt_id][frame] = pred_id  (for each TP match)
        # pred_to_gt[pred_id][frame] = gt_id
        gt_to_pred: Dict[int, Dict[int, int]] = defaultdict(dict)
        pred_to_gt: Dict[int, Dict[int, int]] = defaultdict(dict)

        for fid in all_frames:
            gt_ids   = gt_ids_cache[fid]
            pred_ids = pred_ids_cache[fid]
            iou      = iou_cache[fid]
            n_gt, n_pred = len(gt_ids), len(pred_ids)

            if n_gt == 0 and n_pred == 0:
                continue
            if n_gt == 0:
                fp_total += n_pred
                continue
            if n_pred == 0:
                fn_total += n_gt
                continue

            # Greedy match: sort all (i, j) pairs by IoU descending.
            flat = iou.ravel()
            order = np.argsort(-flat)
            used_gt:   set = set()
            used_pred: set = set()

            for idx in order:
                if flat[idx] < alpha:
                    break
                i, j = divmod(int(idx), n_pred)
                if i in used_gt or j in used_pred:
                    continue
                used_gt.add(i)
                used_pred.add(j)
                gid, pid = int(gt_ids[i]), int(pred_ids[j])
                gt_to_pred[gid][fid] = pid
                pred_to_gt[pid][fid] = gid

            tp_this = len(used_gt)
            tp_total  += tp_this
            fp_total  += n_pred - len(used_pred)
            fn_total  += n_gt   - tp_this

        # DetA
        denom_det = tp_total + fp_total + fn_total
        deta = tp_total / denom_det if denom_det > 0 else 0.0

        # AssA
        if tp_total == 0:
            assa = 0.0
        else:
            assa_sum = 0.0

            for gid, frame_pred in gt_to_pred.items():
                for fid, pid in frame_pred.items():
                    # TPA: frames where this (gid, pid) pair is TP-matched
                    all_gid_frames = gt_to_pred[gid]          # {frame: pred_id}
                    all_pid_frames = pred_to_gt[pid]           # {frame: gt_id}

                    tpa = sum(1 for p in all_gid_frames.values() if p == pid)
                    fna = sum(1 for p in all_gid_frames.values() if p != pid)
                    fpa = sum(1 for g in all_pid_frames.values() if g != gid)

                    d = tpa + fna + fpa
                    assa_sum += tpa / d if d > 0 else 0.0

            assa = assa_sum / tp_total

        hota_scores.append(float(np.sqrt(deta * assa)))
        deta_scores.append(deta)
        assa_scores.append(assa)

    return {
        "hota": float(np.mean(hota_scores)),
        "deta": float(np.mean(deta_scores)),
        "assa": float(np.mean(assa_scores)),
    }
