import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric
from shapely.geometry import Polygon

from mmdet3d.registry import METRICS


def _get_field(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def _normalize_sample_id(value) -> str:
    if value is None:
        return ''
    text = str(value).replace('\\', '/').strip()
    if not text:
        return ''

    suffix = Path(text).suffix.lower()
    if suffix in {'.bin', '.pcd', '.npy'}:
        parts = [part for part in text.split('/') if part]
        if 'points' in parts:
            points_idx = parts.index('points')
            rel_parts = parts[points_idx + 1:]
            if len(rel_parts) >= 2:
                rel_parts[-1] = Path(rel_parts[-1]).stem
                return '/'.join(rel_parts)
            if len(rel_parts) == 1:
                return Path(rel_parts[0]).stem
        return Path(text).stem

    return text


def _extract_sample_id(data_sample) -> str:
    metainfo = _get_field(data_sample, 'metainfo', {}) or {}
    # lidar_path is always an absolute file path set by LoadPointsFromFile and
    # normalizes unambiguously via _normalize_sample_id. sample_idx may be an
    # integer index or a differently formatted string depending on the PKL/MMDet3D
    # version, so we only fall back to it when lidar_path is absent.
    lidar_path = metainfo.get('lidar_path') or _get_field(data_sample, 'lidar_path')
    if lidar_path:
        return _normalize_sample_id(lidar_path)
    sample_idx = metainfo.get('sample_idx') or _get_field(data_sample, 'sample_idx')
    return _normalize_sample_id(sample_idx)


def _box_to_bev_polygon(box: np.ndarray) -> Polygon:
    x, y, z, length, width, height, yaw = [float(v) for v in box[:7]]
    half_l = length / 2.0
    half_w = width / 2.0
    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ], dtype=np.float32)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    rotated = corners @ rotation.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    return Polygon(rotated)


def _bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    poly1 = _box_to_bev_polygon(box1)
    poly2 = _box_to_bev_polygon(box2)
    inter_area = poly1.intersection(poly2).area
    if inter_area <= 0:
        return 0.0
    union_area = poly1.area + poly2.area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def _iou3d(box1: np.ndarray, box2: np.ndarray) -> float:
    inter_area = _box_to_bev_polygon(box1).intersection(_box_to_bev_polygon(box2)).area
    if inter_area <= 0:
        return 0.0

    z1_min = float(box1[2])
    z1_max = z1_min + float(box1[5])
    z2_min = float(box2[2])
    z2_max = z2_min + float(box2[5])
    inter_height = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    if inter_height <= 0:
        return 0.0

    inter_volume = inter_area * inter_height
    vol1 = float(box1[3] * box1[4] * box1[5])
    vol2 = float(box2[3] * box2[4] * box2[5])
    union_volume = vol1 + vol2 - inter_volume
    if union_volume <= 0:
        return 0.0
    return float(inter_volume / union_volume)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _evaluate_predictions(
    predictions: List[dict],
    ground_truth: Dict[str, dict],
    iou_threshold: float,
    mode: str,
    class_id: int,
) -> Dict[str, float]:
    assert mode in {'bev', '3d'}
    iou_fn = _bev_iou if mode == 'bev' else _iou3d

    gt_by_sample: Dict[str, List[np.ndarray]] = {}
    total_gt = 0
    for sample_idx, sample in ground_truth.items():
        boxes = [
            np.asarray(instance['bbox_3d'], dtype=np.float32)
            for instance in sample['instances']
            if int(instance['bbox_label_3d']) == class_id
        ]
        gt_by_sample[sample_idx] = boxes
        total_gt += len(boxes)

    if total_gt == 0:
        return dict(ap=0.0, precision=0.0, recall=0.0, f1=0.0)

    filtered_predictions = [p for p in predictions if int(p['label']) == class_id]
    filtered_predictions.sort(key=lambda item: float(item['score']), reverse=True)

    matched = {
        sample_idx: np.zeros(len(boxes), dtype=bool)
        for sample_idx, boxes in gt_by_sample.items()
    }

    tp = np.zeros(len(filtered_predictions), dtype=np.float32)
    fp = np.zeros(len(filtered_predictions), dtype=np.float32)

    for pred_idx, prediction in enumerate(filtered_predictions):
        sample_idx = prediction['sample_idx']
        pred_box = prediction['bbox']
        gt_boxes = gt_by_sample.get(sample_idx, [])
        if not gt_boxes:
            fp[pred_idx] = 1.0
            continue

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if matched[sample_idx][gt_idx]:
                continue
            overlap = iou_fn(pred_box, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched[sample_idx][best_gt_idx] = True
            tp[pred_idx] = 1.0
        else:
            fp[pred_idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
    ap = _compute_ap(recalls, precisions)
    recall = float(recalls[-1]) if recalls.size else 0.0
    precision = float(precisions[-1]) if precisions.size else 0.0
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))
    return dict(ap=ap, precision=precision, recall=recall, f1=f1)


@METRICS.register_module()
class Aghri3DMetric(BaseMetric):
    default_prefix = 'aghri'

    def __init__(
        self,
        ann_file: str,
        iou_thresholds: Iterable[float] = (0.25, 0.5),
        score_thr: float = 0.0,
        prefix: str = None,
        collect_device: str = 'cpu',
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = str(ann_file)
        self.iou_thresholds = tuple(float(v) for v in iou_thresholds)
        self.score_thr = float(score_thr)

    def process(self, data_batch, data_samples) -> None:
        batch_data_samples = _get_field(data_batch, 'data_samples', []) or []

        for idx, data_sample in enumerate(data_samples):
            sample_idx = ''
            if idx < len(batch_data_samples):
                sample_idx = _extract_sample_id(batch_data_samples[idx])
            if not sample_idx:
                sample_idx = _extract_sample_id(data_sample)
            pred_instances = _get_field(data_sample, 'pred_instances_3d', {}) or {}
            boxes = _to_numpy(_get_field(pred_instances, 'bboxes_3d'))
            scores = _to_numpy(_get_field(pred_instances, 'scores_3d'))
            labels = _to_numpy(_get_field(pred_instances, 'labels_3d'))

            if boxes is None or scores is None or labels is None:
                continue

            for box, score, label in zip(boxes, scores, labels):
                if float(score) < self.score_thr:
                    continue
                self.results.append(
                    dict(
                        sample_idx=sample_idx,
                        bbox=np.asarray(box, dtype=np.float32),
                        score=float(score),
                        label=int(label),
                    ))

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        ann_path = Path(self.ann_file)
        with ann_path.open('rb') as handle:
            raw_infos = pickle.load(handle)

        data_list = raw_infos['data_list']
        ground_truth = {
            _normalize_sample_id(
                item.get('lidar_points', {}).get('lidar_path') or item.get('sample_idx')
            ): item
            for item in data_list
        }

        classes = tuple(self.dataset_meta.get('classes', ('person', )))
        metrics: Dict[str, float] = {}
        per_class_summary = defaultdict(dict)
        gt_sample_ids = set(ground_truth.keys())
        pred_sample_ids = {
            _normalize_sample_id(item.get('sample_idx'))
            for item in results
            if _normalize_sample_id(item.get('sample_idx'))
        }
        metrics['num_predictions'] = len(results)
        metrics['num_pred_samples'] = len(pred_sample_ids)
        metrics['num_gt_samples'] = len(gt_sample_ids)
        metrics['num_pred_sample_matches'] = len(pred_sample_ids & gt_sample_ids)
        metrics['num_gt_instances'] = int(sum(len(item['instances']) for item in data_list))

        for class_id, class_name in enumerate(classes):
            for threshold in self.iou_thresholds:
                threshold_key = f'{threshold:.2f}'
                bev_metrics = _evaluate_predictions(
                    results, ground_truth, threshold, 'bev', class_id)
                metrics_3d = _evaluate_predictions(
                    results, ground_truth, threshold, '3d', class_id)

                metrics[f'{class_name}_bev_ap_{threshold_key}'] = round(bev_metrics['ap'], 6)
                metrics[f'{class_name}_bev_precision_{threshold_key}'] = round(
                    bev_metrics['precision'], 6)
                metrics[f'{class_name}_bev_recall_{threshold_key}'] = round(
                    bev_metrics['recall'], 6)
                metrics[f'{class_name}_bev_f1_{threshold_key}'] = round(bev_metrics['f1'], 6)

                metrics[f'{class_name}_3d_ap_{threshold_key}'] = round(metrics_3d['ap'], 6)
                metrics[f'{class_name}_3d_precision_{threshold_key}'] = round(
                    metrics_3d['precision'], 6)
                metrics[f'{class_name}_3d_recall_{threshold_key}'] = round(
                    metrics_3d['recall'], 6)
                metrics[f'{class_name}_3d_f1_{threshold_key}'] = round(metrics_3d['f1'], 6)

                for metric_name in ('ap', 'precision', 'recall', 'f1'):
                    per_class_summary[f'bev_{metric_name}_{threshold_key}'][class_name] = bev_metrics[
                        metric_name]
                    per_class_summary[f'3d_{metric_name}_{threshold_key}'][class_name] = metrics_3d[
                        metric_name]

        for threshold in self.iou_thresholds:
            threshold_key = f'{threshold:.2f}'
            for metric_name in (
                'bev_ap',
                'bev_precision',
                'bev_recall',
                'bev_f1',
                '3d_ap',
                '3d_precision',
                '3d_recall',
                '3d_f1',
            ):
                class_metric_key = f'{metric_name}_{threshold_key}'
                values = list(per_class_summary[class_metric_key].values())
                metrics[class_metric_key] = round(float(np.mean(values)) if values else 0.0, 6)

        for metric_name in (
            'bev_ap',
            'bev_precision',
            'bev_recall',
            'bev_f1',
            '3d_ap',
            '3d_precision',
            '3d_recall',
            '3d_f1',
        ):
            threshold_values = [
                float(metrics.get(f'{metric_name}_{threshold:.2f}', 0.0))
                for threshold in self.iou_thresholds
            ]
            metrics[f'{metric_name}_mean'] = round(
                float(np.mean(threshold_values)) if threshold_values else 0.0, 6)

        return metrics
