"""Deterministic reliability fusion helpers for part-OCLReID."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


VALID_MODES = {"baseline", "global_fallback", "dual_orientation", "combined"}


def bounded_score(value: float, reference_scores: Iterable[float]) -> float:
    scores = [float(score) for score in reference_scores if score is not None]
    if not scores:
        return float(value)
    low = min(scores)
    high = max(scores)
    if low == high:
        return float(value)
    return float(max(low, min(high, value)))


def pose_reliability(
    mean_confidence: float | None,
    visible_part_count: int | None,
    pose_confidence_threshold: float,
    minimum_visible_parts: int,
    minimum_reliability: float,
) -> float:
    if mean_confidence is None:
        confidence_term = minimum_reliability
    elif pose_confidence_threshold <= 0:
        confidence_term = 1.0
    else:
        confidence_term = min(1.0, max(minimum_reliability, mean_confidence / pose_confidence_threshold))

    if visible_part_count is None:
        visibility_term = minimum_reliability
    elif minimum_visible_parts <= 0:
        visibility_term = 1.0
    else:
        visibility_term = min(1.0, max(minimum_reliability, visible_part_count / float(minimum_visible_parts)))

    return float(min(confidence_term, visibility_term))


def blend_global_fallback(
    part_average_score: float,
    global_score: float | None,
    reliability: float,
    fallback_weight: float,
) -> Tuple[float, float]:
    if global_score is None:
        return float(part_average_score), 0.0
    fallback_weight = max(0.0, min(1.0, float(fallback_weight)))
    alpha = fallback_weight * max(0.0, min(1.0, 1.0 - float(reliability)))
    score = (1.0 - alpha) * float(part_average_score) + alpha * float(global_score)
    return float(score), float(alpha)


def aggregate_dual_orientation(
    hard_score: float,
    front_score: float | None,
    back_score: float | None,
    orientation_stability: float,
    aggregation: str,
) -> Tuple[float, str]:
    valid = [score for score in (front_score, back_score) if score is not None]
    if len(valid) == 0:
        return float(hard_score), "hard_orientation_no_alternate_evidence"
    if len(valid) == 1:
        return float(valid[0]), "single_valid_bank"
    if orientation_stability >= 1.0:
        return float(hard_score), "hard_orientation_stable"
    if aggregation == "max":
        return float(max(valid)), "dual_orientation_max"
    if aggregation == "mean":
        return float(sum(valid) / len(valid)), "dual_orientation_mean"
    return float(min(valid)), "dual_orientation_min"


def global_slot(start_idx: int, seg_idx: int) -> int:
    return int(start_idx + seg_idx - 1)


def active_counts(part_scores: Dict[int, float], front_end: int, part_nums: int) -> Dict[str, int]:
    front = sum(1 for index in part_scores if index < front_end)
    back = sum(1 for index in part_scores if front_end <= index < part_nums)
    return {
        "active_fitted_classifier_slots": len(part_scores),
        "active_front_bank_classifier_count": front,
        "active_back_bank_classifier_count": back,
    }
