#!/usr/bin/env python3
"""Evaluate OCLReID predictions using AGHRI frame manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from statistics import mean
from typing import Any

from scripts.aghri_alignment import (
    collapse_duplicate_boxes,
    load_annotations,
    load_frame_manifest,
    validate_manifest_against_sources,
)


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def valid_box(box: Any) -> bool:
    if box is None or len(box) != 4:
        return False
    x1, y1, x2, y2 = map(float, box)
    return x2 > x1 and y2 > y1


def calculate_iou(box_a: Any, box_b: Any) -> float:
    if not valid_box(box_a) or not valid_box(box_b):
        return 0.0
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def calculate_center_error(box_a: Any, box_b: Any) -> float | None:
    if not valid_box(box_a) or not valid_box(box_b):
        return None
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    return math.hypot(((ax1 + ax2) / 2.0) - ((bx1 + bx2) / 2.0), ((ay1 + ay2) / 2.0) - ((by1 + by2) / 2.0))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def extract_prediction(predictions: dict[str, Any], frame_index: int) -> dict[str, Any]:
    for key in (f"{frame_index:06d}.jpg", f"{frame_index:06d}"):
        frame_result = predictions.get(key)
        if isinstance(frame_result, dict):
            target_info = frame_result.get("target_info", [])
            if isinstance(target_info, list) and len(target_info) >= 6:
                try:
                    bbox = [float(target_info[1]), float(target_info[2]), float(target_info[3]), float(target_info[4])]
                    tracker_id = int(target_info[0])
                    confidence = float(target_info[5])
                except (TypeError, ValueError):
                    break
                present = tracker_id >= 0 and valid_box(bbox)
                return {
                    "present": present,
                    "tracker_id": tracker_id,
                    "bbox": bbox if present else None,
                    "confidence": confidence,
                }
    return {"present": False, "tracker_id": -1, "bbox": None, "confidence": -1.0}


def prediction_indices(predictions: dict[str, Any]) -> set[int]:
    indices = set()
    for key in predictions:
        stem = Path(str(key)).stem
        if stem.isdigit():
            indices.add(int(stem))
    return indices


def build_row(
    *,
    annotation_record_index: int,
    video_frame_index: int,
    previous_annotated_video_frame: int | None,
    annotation_record: dict[str, Any],
    target_class: str,
    prediction: dict[str, Any],
    iou_threshold: float,
    center_threshold: float,
    paper_center_threshold: float,
) -> dict[str, Any]:
    labels = annotation_record.get("Labels", [])
    if not isinstance(labels, list):
        labels = []
    gt_status, target_xywh, other_xywh_boxes = collapse_duplicate_boxes(labels, target_class)

    annotation_gap = (
        None
        if previous_annotated_video_frame is None
        else video_frame_index - previous_annotated_video_frame
    )
    contiguous = annotation_gap == 1
    valid_for_metrics = gt_status in {"visible", "absent"}

    target_gt_box = xywh_to_xyxy(target_xywh) if gt_status == "visible" else None
    other_gt_boxes = [
        {"class": item["class"], "bbox": xywh_to_xyxy(item["bbox_xywh"])}
        for item in other_xywh_boxes
    ]

    prediction_present = prediction["present"]
    gt_visible = gt_status == "visible"
    gt_absent = gt_status == "absent"

    iou = calculate_iou(prediction["bbox"], target_gt_box) if gt_visible and prediction_present else 0.0
    center_error = calculate_center_error(prediction["bbox"], target_gt_box) if gt_visible and prediction_present else None

    highest_other_iou = 0.0
    highest_other_class = ""
    if gt_visible and prediction_present:
        for other in other_gt_boxes:
            other_iou = calculate_iou(prediction["bbox"], other["bbox"])
            if other_iou > highest_other_iou:
                highest_other_iou = other_iou
                highest_other_class = other["class"]

    wrong_person = (
        gt_visible
        and prediction_present
        and highest_other_iou >= iou_threshold
        and highest_other_iou > iou
    )
    correctly_localized = (
        gt_visible
        and prediction_present
        and iou >= iou_threshold
        and not wrong_person
    )
    false_positive_when_absent = gt_absent and prediction_present
    correct_rejection_when_absent = gt_absent and not prediction_present
    target_state_correct = (
        correctly_localized if gt_visible else correct_rejection_when_absent if gt_absent else False
    )

    return {
        "annotation_record_index": annotation_record_index,
        "video_frame_index": video_frame_index,
        "frame_index": video_frame_index,
        "file": annotation_record.get("File", ""),
        "timestamp": annotation_record.get("Timestamp", ""),
        "previous_annotated_video_frame": previous_annotated_video_frame if previous_annotated_video_frame is not None else "",
        "annotation_gap_frames": annotation_gap if annotation_gap is not None else "",
        "temporally_contiguous_with_previous_annotation": contiguous,
        "target_class": target_class,
        "gt_status": gt_status,
        "valid_for_metrics": valid_for_metrics,
        "gt_visible": gt_visible,
        "gt_x1": target_gt_box[0] if gt_visible else "",
        "gt_y1": target_gt_box[1] if gt_visible else "",
        "gt_x2": target_gt_box[2] if gt_visible else "",
        "gt_y2": target_gt_box[3] if gt_visible else "",
        "prediction_present": prediction_present,
        "tracker_id": prediction["tracker_id"],
        "pred_x1": prediction["bbox"][0] if prediction_present else "",
        "pred_y1": prediction["bbox"][1] if prediction_present else "",
        "pred_x2": prediction["bbox"][2] if prediction_present else "",
        "pred_y2": prediction["bbox"][3] if prediction_present else "",
        "target_confidence": prediction["confidence"],
        "iou": iou,
        "center_error_pixels": center_error if center_error is not None else "",
        "correctly_localized": correctly_localized,
        "center_success_strict": gt_visible and prediction_present and center_error is not None and center_error <= center_threshold,
        "center_success_paper_style": gt_visible and prediction_present and center_error is not None and center_error <= paper_center_threshold,
        "wrong_person": wrong_person,
        "highest_other_iou": highest_other_iou,
        "highest_other_class": highest_other_class,
        "false_positive_when_target_absent": false_positive_when_absent,
        "correct_rejection_when_target_absent": correct_rejection_when_absent,
        "target_state_correct": target_state_correct,
    }


def find_reappearance_events(rows: list[dict[str, Any]], fps: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact = []
    uncertain = []
    previous_row = None
    index = 0

    while index < len(rows):
        row = rows[index]

        if previous_row is None:
            previous_row = row
            index += 1
            continue

        previous_status = previous_row.get("gt_status")
        current_status = row.get("gt_status")
        current_gap = int(row["video_frame_index"]) - int(previous_row["video_frame_index"])

        if previous_status == "ambiguous_duplicate_target":
            previous_row = row
            index += 1
            continue

        absent_to_visible = previous_status == "absent" and current_status == "visible"
        if absent_to_visible:
            event = {
                "previous_absent_video_frame": previous_row["video_frame_index"],
                "reappearance_frame": row["video_frame_index"],
                "annotation_gap_frames": current_gap,
                "reacquired_frame": None,
                "success": False,
                "delay_frames": None,
                "delay_seconds": None,
                "ended_by": None,
            }

            if current_gap != 1:
                event["ended_by"] = "annotation_gap"
                uncertain.append(event)
                previous_row = row
                index += 1
                continue

            scan_index = index
            last_scan_row = previous_row

            while scan_index < len(rows):
                scan_row = rows[scan_index]
                scan_gap = int(scan_row["video_frame_index"]) - int(last_scan_row["video_frame_index"])

                if scan_gap != 1:
                    event["ended_by"] = "annotation_gap"
                    uncertain.append(event)
                    break

                scan_status = scan_row.get("gt_status")

                if scan_status == "ambiguous_duplicate_target":
                    event["ended_by"] = "ambiguous_ground_truth"
                    uncertain.append(event)
                    break

                if scan_status == "absent":
                    event["ended_by"] = "target_absent_again"
                    exact.append(event)
                    break

                if scan_status == "visible" and scan_row["correctly_localized"]:
                    reacquired_frame = int(scan_row["video_frame_index"])
                    reappearance_frame = int(event["reappearance_frame"])
                    delay_frames = reacquired_frame - reappearance_frame
                    event.update(
                        {
                            "reacquired_frame": reacquired_frame,
                            "success": True,
                            "delay_frames": delay_frames,
                            "delay_seconds": delay_frames / fps,
                            "ended_by": "correct_localization",
                        }
                    )
                    exact.append(event)
                    break

                last_scan_row = scan_row
                scan_index += 1
            else:
                event["ended_by"] = "sequence_end"
                exact.append(event)

        previous_row = row
        index += 1

    return exact, uncertain


def find_tracker_recovery_events(rows: list[dict[str, Any]], fps: float) -> tuple[list[dict[str, Any]], int]:
    events = []
    uncertain_count = 0
    start = None
    last_failed = None
    gap_uncertain = False
    previous_visible_frame = None

    for row in rows:
        if row["gt_status"] != "visible":
            if start is not None:
                events.append(
                    {
                        "loss_start_frame": start,
                        "last_failed_frame": last_failed,
                        "recovery_frame": None,
                        "success": False,
                        "delay_frames": None,
                        "delay_seconds": None,
                        "uncertain_due_to_annotation_gap": gap_uncertain,
                        "ended_by": "target_not_valid_visible",
                    }
                )
                if gap_uncertain:
                    uncertain_count += 1
                start = None
                last_failed = None
                gap_uncertain = False
            previous_visible_frame = None
            continue

        if previous_visible_frame is not None and row["video_frame_index"] - previous_visible_frame > 1:
            gap_uncertain = True
        previous_visible_frame = row["video_frame_index"]

        if row["correctly_localized"]:
            if start is not None:
                delay = row["video_frame_index"] - start
                events.append(
                    {
                        "loss_start_frame": start,
                        "last_failed_frame": last_failed,
                        "recovery_frame": row["video_frame_index"],
                        "success": True,
                        "delay_frames": delay,
                        "delay_seconds": delay / fps,
                        "uncertain_due_to_annotation_gap": gap_uncertain,
                        "ended_by": "correct_localization",
                    }
                )
                if gap_uncertain:
                    uncertain_count += 1
                start = None
                last_failed = None
                gap_uncertain = False
            continue

        if start is None:
            start = row["video_frame_index"]
        last_failed = row["video_frame_index"]

    if start is not None:
        events.append(
            {
                "loss_start_frame": start,
                "last_failed_frame": last_failed,
                "recovery_frame": None,
                "success": False,
                "delay_frames": None,
                "delay_seconds": None,
                "uncertain_due_to_annotation_gap": gap_uncertain,
                "ended_by": "end_of_sequence",
            }
        )
        if gap_uncertain:
            uncertain_count += 1
    return events, uncertain_count


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate(args: argparse.Namespace) -> None:
    prediction_path = Path(args.predictions)
    annotation_path = Path(args.annotations)
    manifest_path = Path(args.frame_manifest)
    output_dir = Path(args.output_dir)
    if args.fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    if args.initialization_video_frame < 0:
        raise ValueError("initialization_video_frame must be non-negative.")

    predictions = read_json(prediction_path)
    if not isinstance(predictions, dict):
        raise ValueError("The predictions JSON must contain a dictionary indexed by frame.")
    annotations = load_annotations(annotation_path)
    manifest = load_frame_manifest(manifest_path)
    if abs(float(manifest.get("fps", args.fps)) - args.fps) > 0.1:
        raise ValueError(f"Manifest FPS {manifest.get('fps')} does not match evaluator FPS {args.fps}.")

    file_to_video_frame = manifest["file_to_video_frame"]
    rows = []
    mapping_rows = []
    previous_annotated_video_frame = None
    prediction_frame_indices = prediction_indices(predictions)
    max_manifest_frame = len(manifest["frames"]) - 1
    inconsistent_prediction_indices = sorted(index for index in prediction_frame_indices if index > max_manifest_frame)
    if inconsistent_prediction_indices:
        raise ValueError(
            f"Predictions contain frame indices outside manifest range: {inconsistent_prediction_indices[:10]}"
        )

    for annotation_record_index, annotation_record in enumerate(annotations):
        filename = annotation_record.get("File")
        if filename not in file_to_video_frame:
            raise ValueError(f"Annotation file {filename!r} missing from manifest.")
        video_frame_index = int(file_to_video_frame[filename])
        mapping_rows.append(
            {
                "annotation_record_index": annotation_record_index,
                "file": filename,
                "video_frame_index": video_frame_index,
            }
        )
        if video_frame_index <= args.initialization_video_frame:
            previous_annotated_video_frame = video_frame_index
            continue
        prediction = extract_prediction(predictions, video_frame_index)
        rows.append(
            build_row(
                annotation_record_index=annotation_record_index,
                video_frame_index=video_frame_index,
                previous_annotated_video_frame=previous_annotated_video_frame,
                annotation_record=annotation_record,
                target_class=args.target_class,
                prediction=prediction,
                iou_threshold=args.iou_threshold,
                center_threshold=args.center_threshold,
                paper_center_threshold=args.paper_center_threshold,
            )
        )
        previous_annotated_video_frame = video_frame_index

    valid_rows = [row for row in rows if row["valid_for_metrics"]]
    visible_rows = [row for row in valid_rows if row["gt_status"] == "visible"]
    absent_rows = [row for row in valid_rows if row["gt_status"] == "absent"]
    ambiguous_rows = [row for row in rows if row["gt_status"] == "ambiguous_duplicate_target"]
    predicted_visible_rows = [row for row in visible_rows if row["prediction_present"]]
    correct_rows = [row for row in visible_rows if row["correctly_localized"]]
    wrong_person_rows = [row for row in visible_rows if row["wrong_person"]]
    false_positive_rows = [row for row in absent_rows if row["false_positive_when_target_absent"]]
    correct_absent_rejection_rows = [row for row in absent_rows if row["correct_rejection_when_target_absent"]]
    centre_success_rows = [row for row in visible_rows if row["center_success_strict"]]
    paper_centre_success_rows = [row for row in visible_rows if row["center_success_paper_style"]]
    target_state_correct_rows = [row for row in valid_rows if row["target_state_correct"]]

    valid_center_errors = [row["center_error_pixels"] for row in predicted_visible_rows if row["center_error_pixels"] != ""]
    exact_reappearance_events, uncertain_reappearance_events = find_reappearance_events(rows, args.fps)
    successful_exact = [event for event in exact_reappearance_events if event["success"]]
    recovery_events, uncertain_recovery_count = find_tracker_recovery_events(valid_rows, args.fps)
    successful_recovery_events = [event for event in recovery_events if event["success"]]
    unrecovered_recovery_events = [event for event in recovery_events if not event["success"]]

    summary = {
        "predictions_file": str(prediction_path),
        "annotations_file": str(annotation_path),
        "frame_manifest": str(manifest_path),
        "target_class": args.target_class,
        "initialization_video_frame": args.initialization_video_frame,
        "start_frame": args.initialization_video_frame + 1,
        "fps": args.fps,
        "iou_threshold": args.iou_threshold,
        "center_threshold_pixels": args.center_threshold,
        "paper_style_center_threshold_pixels": args.paper_center_threshold,
        "total_evaluated_frames": len(rows),
        "valid_annotated_frames": len(valid_rows),
        "ambiguous_duplicate_target_frames": len(ambiguous_rows),
        "ground_truth_visible_frames": len(visible_rows),
        "ground_truth_absent_frames": len(absent_rows),
        "prediction_present_on_visible_frames": len(predicted_visible_rows),
        "correctly_localized_frames": len(correct_rows),
        "missed_or_incorrect_visible_frames": len(visible_rows) - len(correct_rows),
        "mean_iou_on_visible_frames": mean(row["iou"] for row in visible_rows) if visible_rows else 0.0,
        "mean_iou_when_prediction_present": mean(row["iou"] for row in predicted_visible_rows) if predicted_visible_rows else 0.0,
        "success_rate_iou_threshold": rate(len(correct_rows), len(visible_rows)) or 0.0,
        "prediction_availability_on_visible_frames": rate(len(predicted_visible_rows), len(visible_rows)) or 0.0,
        "precision_center_threshold": rate(len(centre_success_rows), len(visible_rows)) or 0.0,
        "center_success_rate_50px": rate(len(paper_centre_success_rows), len(visible_rows)) or 0.0,
        "paper_style_center_success_rate": rate(len(paper_centre_success_rows), len(visible_rows)) or 0.0,
        "mean_center_error_pixels_when_prediction_present": mean(valid_center_errors) if valid_center_errors else None,
        "mean_center_error_pixels": mean(valid_center_errors) if valid_center_errors else None,
        "wrong_person_frames": len(wrong_person_rows),
        "wrong_person_rate_on_predicted_visible_frames": rate(len(wrong_person_rows), len(predicted_visible_rows)) or 0.0,
        "wrong_person_rate_on_all_visible_frames": rate(len(wrong_person_rows), len(visible_rows)) or 0.0,
        "wrong_person_rate": rate(len(wrong_person_rows), len(predicted_visible_rows)) or 0.0,
        "false_positive_frames_when_target_absent": len(false_positive_rows),
        "false_positive_rate_when_target_absent": rate(len(false_positive_rows), len(absent_rows)) or 0.0,
        "correct_rejection_frames_when_target_absent": len(correct_absent_rejection_rows),
        "correct_rejection_rate_when_target_absent": rate(len(correct_absent_rejection_rows), len(absent_rows)),
        "annotated_frame_target_state_correct_frames": len(target_state_correct_rows),
        "annotated_frame_target_state_accuracy": rate(len(target_state_correct_rows), len(valid_rows)) or 0.0,
        "full_sequence_target_state_accuracy_deprecated": rate(len(target_state_correct_rows), len(valid_rows)) or 0.0,
        "full_sequence_target_state_accuracy": rate(len(target_state_correct_rows), len(valid_rows)) or 0.0,
        "full_sequence_target_state_accuracy_note": "Deprecated alias; use annotated_frame_target_state_accuracy.",
        "exact_reappearance_events": exact_reappearance_events,
        "uncertain_reappearance_events": uncertain_reappearance_events,
        "exact_reappearance_count": len(exact_reappearance_events),
        "uncertain_reappearance_count": len(uncertain_reappearance_events),
        "successful_exact_reacquisitions": len(successful_exact),
        "exact_reacquisition_rate": rate(len(successful_exact), len(exact_reappearance_events)),
        "average_exact_reacquisition_delay_frames": mean([event["delay_frames"] for event in successful_exact]) if successful_exact else None,
        "average_exact_reacquisition_delay_seconds": mean([event["delay_seconds"] for event in successful_exact]) if successful_exact else None,
        "ground_truth_reappearance_events": exact_reappearance_events,
        "ground_truth_reappearance_count": len(exact_reappearance_events),
        "successful_reacquisitions_after_reappearance": len(successful_exact),
        "reacquisition_rate_after_reappearance": rate(len(successful_exact), len(exact_reappearance_events)),
        "visible_target_recovery_events": recovery_events,
        "tracker_recovery_events": recovery_events,
        "visible_target_loss_episode_count": len(recovery_events),
        "successful_visible_target_recoveries": len(successful_recovery_events),
        "unrecovered_visible_target_loss_episodes": len(unrecovered_recovery_events),
        "visible_target_recovery_rate": rate(len(successful_recovery_events), len(recovery_events)),
        "average_visible_target_recovery_delay_frames": mean([event["delay_frames"] for event in successful_recovery_events]) if successful_recovery_events else None,
        "average_visible_target_recovery_delay_seconds": mean([event["delay_seconds"] for event in successful_recovery_events]) if successful_recovery_events else None,
        "average_tracker_recovery_delay_frames": mean([event["delay_frames"] for event in successful_recovery_events]) if successful_recovery_events else None,
        "average_tracker_recovery_delay_seconds": mean([event["delay_seconds"] for event in successful_recovery_events]) if successful_recovery_events else None,
        "uncertain_recovery_event_count": uncertain_recovery_count,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_csv_path = output_dir / "per_frame_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "annotation_record_index",
        "video_frame_index",
        "file",
        "gt_status",
        "valid_for_metrics",
    ]
    with frame_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(output_dir / "summary_metrics.json", summary)
    write_json(
        output_dir / "evaluation_mapping_report.json",
        {
            "frame_manifest": str(manifest_path),
            "annotation_records": len(annotations),
            "evaluated_annotation_records": len(rows),
            "initialization_video_frame": args.initialization_video_frame,
            "prediction_frame_count": len(prediction_frame_indices),
            "mapping": mapping_rows,
        },
    )

    print("=" * 70)
    print("OCLReID EVALUATION RESULTS")
    print("=" * 70)
    print(f"Evaluated annotated frames: {summary['total_evaluated_frames']}")
    print(f"Valid annotated frames:     {summary['valid_annotated_frames']}")
    print(f"Ambiguous target frames:    {summary['ambiguous_duplicate_target_frames']}")
    print(f"GT-visible frames:          {summary['ground_truth_visible_frames']}")
    print(f"Mean IoU:                   {summary['mean_iou_on_visible_frames']:.4f}")
    print(f"Success@IoU >= {args.iou_threshold}:      {summary['success_rate_iou_threshold']:.4f}")
    print(f"Centre Precision@{args.center_threshold:g}px: {summary['precision_center_threshold']:.4f}")
    print(f"Annotated state accuracy:   {summary['annotated_frame_target_state_accuracy']:.4f}")
    print(f"Exact reappearances:        {summary['exact_reappearance_count']}")
    print(f"Uncertain reappearances:    {summary['uncertain_reappearance_count']}")
    print(f"\nPer-frame results:\n{frame_csv_path}")
    print(f"\nSummary:\n{output_dir / 'summary_metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--frame_manifest", required=True)
    parser.add_argument("--target_class", required=True)
    parser.add_argument("--start_frame", type=int, default=None, help="Deprecated; use --initialization_video_frame.")
    parser.add_argument("--initialization_video_frame", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--center_threshold", type=float, default=20.0)
    parser.add_argument("--paper_center_threshold", type=float, default=50.0)
    parser.add_argument("--min_reappearance_absent_frames", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
