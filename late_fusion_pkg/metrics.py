"""Small deterministic metrics used by smoke tests and offline reports."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from late_fusion_pkg.detection_types import Detection3D
from late_fusion_pkg.fusion_core import iou_2d


@dataclass(frozen=True)
class PRMetrics:
    ap: float
    precision: float
    recall: float
    f1: float
    predictions: int
    ground_truth: int


def average_precision(
    predictions: Sequence[tuple[float, object]],
    ground_truth: Sequence[object],
    overlap_fn: Callable[[object, object], float],
    threshold: float,
) -> PRMetrics:
    ordered = sorted(predictions, key=lambda item: float(item[0]), reverse=True)
    matched = [False] * len(ground_truth)
    tp = np.zeros((len(ordered),), dtype=float)
    fp = np.zeros((len(ordered),), dtype=float)
    for pred_idx, (_score, pred) in enumerate(ordered):
        best_idx = -1
        best_overlap = 0.0
        for gt_idx, gt in enumerate(ground_truth):
            if matched[gt_idx]:
                continue
            overlap = overlap_fn(pred, gt)
            if overlap > best_overlap:
                best_idx = gt_idx
                best_overlap = overlap
        if best_idx >= 0 and best_overlap >= threshold:
            matched[best_idx] = True
            tp[pred_idx] = 1.0
        else:
            fp[pred_idx] = 1.0

    if len(ground_truth) == 0:
        precision = 0.0
        recall = 0.0
        ap = 0.0
    else:
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall_curve = cum_tp / max(len(ground_truth), 1)
        precision_curve = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        precision = float(precision_curve[-1]) if precision_curve.size else 0.0
        recall = float(recall_curve[-1]) if recall_curve.size else 0.0
        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], precision_curve, [0.0]))
        for idx in range(mpre.size - 1, 0, -1):
            mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
        changed = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return PRMetrics(ap=ap, precision=precision, recall=recall, f1=f1, predictions=len(ordered), ground_truth=len(ground_truth))


def box2d_ap(pred_boxes: Sequence[tuple[float, tuple[float, float, float, float]]], gt_boxes: Sequence[tuple[float, float, float, float]], threshold: float = 0.5) -> PRMetrics:
    return average_precision(pred_boxes, gt_boxes, iou_2d, threshold)


def center_distance(a: Detection3D, b: Detection3D) -> float:
    return float(np.linalg.norm(np.asarray(a.center_xyz, dtype=float) - np.asarray(b.center_xyz, dtype=float)))


def range_error(a: Detection3D, b: Detection3D) -> float:
    ar = math.hypot(float(a.center_xyz[0]), float(a.center_xyz[1]))
    br = math.hypot(float(b.center_xyz[0]), float(b.center_xyz[1]))
    return float(abs(ar - br))

