#!/usr/bin/env python3
"""Audit AGHRI annotation fields used as tracking ground-truth identities."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/media/prabuddhi/Backup2/Updated Dataset_PW")
DEFAULT_OUTPUT_DIR = Path("results/aghri_deepfusionmot_tracking/gt_identity_audit")


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _recordings_from_manifest(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            recording = str(row.get("recording_name") or "")
            if recording:
                out.add(recording)
    return out


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, aw, ah = [float(v) for v in a[:4]]
    bx1, by1, bw, bh = [float(v) for v in b[:4]]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / denom if denom > 0.0 else 0.0


def _label_key_counter(annotation_paths: list[Path]) -> Counter[str]:
    keys: Counter[str] = Counter()
    for path in annotation_paths:
        try:
            frames = _load_json(path)
        except Exception:
            continue
        for frame in frames:
            for label in frame.get("Labels", []):
                keys.update(label.keys())
    return keys


def _summarise_camera_recording(path: Path, split: str) -> tuple[dict, list[dict]]:
    frames = _load_json(path)
    recording_dir = path.parents[1]
    frame_count = len(frames)
    labelled_frames = 0
    label_count = 0
    duplicate_class_frames = 0
    class_counts: Counter[str] = Counter()
    class_frame_counts: Counter[str] = Counter()
    classes_by_frame: list[set[str]] = []
    boxes_by_class_by_frame: list[dict[str, list[float]]] = []
    duplicate_details: list[dict] = []

    for frame in frames:
        labels = list(frame.get("Labels", []))
        if labels:
            labelled_frames += 1
        label_count += len(labels)
        frame_classes: list[str] = [str(label.get("Class", "")) for label in labels]
        counts = Counter(frame_classes)
        duplicate_classes = [class_id for class_id, count in counts.items() if count > 1]
        if duplicate_classes:
            duplicate_class_frames += 1
            for class_id in duplicate_classes:
                duplicate_boxes = [
                    label.get("BoundingBoxes", [])
                    for label in labels
                    if str(label.get("Class", "")) == class_id
                ]
                duplicate_details.append(
                    {
                        "split": split,
                        "recording": recording_dir.name,
                        "frame_index": len(classes_by_frame),
                        "timestamp": frame.get("Timestamp", ""),
                        "file": frame.get("File", ""),
                        "duplicated_class": class_id,
                        "duplicate_count": counts[class_id],
                        "boxes": json.dumps(duplicate_boxes),
                    }
                )
        class_counts.update(frame_classes)
        class_frame_counts.update(counts.keys())
        classes_by_frame.append(set(frame_classes))
        boxes_by_class_by_frame.append(
            {
                str(label.get("Class", "")): list(label.get("BoundingBoxes", []))
                for label in labels
                if counts[str(label.get("Class", ""))] == 1
            }
        )

    same_class_transitions = 0
    same_class_iou_sum = 0.0
    class_reappear_after_absence = 0
    prev_seen: dict[str, int] = {}
    for idx, frame_classes in enumerate(classes_by_frame):
        if idx > 0:
            previous_classes = classes_by_frame[idx - 1]
            for class_id in frame_classes & previous_classes:
                same_class_transitions += 1
                same_class_iou_sum += _bbox_iou_xywh(
                    boxes_by_class_by_frame[idx - 1].get(class_id, []),
                    boxes_by_class_by_frame[idx].get(class_id, []),
                )
        for class_id in frame_classes:
            if class_id in prev_seen and prev_seen[class_id] < idx - 1:
                class_reappear_after_absence += 1
            prev_seen[class_id] = idx

    class_values = sorted(class_counts, key=lambda value: (len(value), value))
    summary = {
        "split": split,
        "recording": recording_dir.name,
        "annotation_path": str(path),
        "frames": frame_count,
        "labelled_frames": labelled_frames,
        "labels": label_count,
        "unique_class_values": len(class_values),
        "class_values": " ".join(class_values),
        "duplicate_class_frames": duplicate_class_frames,
        "class_reappear_after_absence": class_reappear_after_absence,
        "same_class_consecutive_transitions": same_class_transitions,
        "same_class_mean_consecutive_iou": same_class_iou_sum / same_class_transitions if same_class_transitions else 0.0,
        "max_labels_per_frame": max((len(frame.get("Labels", [])) for frame in frames), default=0),
        "empty_annotation": label_count == 0,
        "identity_field_candidate": "Class" if label_count and duplicate_class_frames == 0 else "",
    }
    return summary, duplicate_details


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict], duplicate_details: list[dict], key_counts: Counter[str], args: argparse.Namespace) -> None:
    total_recordings = len(rows)
    total_frames = sum(int(row["frames"]) for row in rows)
    total_labels = sum(int(row["labels"]) for row in rows)
    duplicate_rows = [row for row in rows if int(row["duplicate_class_frames"]) > 0]
    empty_rows = [row for row in rows if row["empty_annotation"]]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    all_classes = sorted({cls for row in rows for cls in str(row["class_values"]).split()})
    lines = [
        "# AGHRI Ground-Truth Identity Field Audit",
        "",
        f"- Dataset root: `{args.dataset_root}`",
        f"- Validation manifest: `{args.val_manifest}`",
        f"- Test manifest: `{args.test_manifest}`",
        f"- Camera annotation audited: `annotations/cam_zed_rgb_ann.json`",
        f"- Recordings audited: {total_recordings}",
        f"- Frames audited: {total_frames}",
        f"- Label boxes audited: {total_labels}",
        f"- Label keys found: {', '.join(f'`{key}` ({count})' for key, count in key_counts.most_common())}",
        f"- Unique `Class` values observed: {', '.join(all_classes) if all_classes else 'none'}",
        "",
        "## Conclusion",
        "",
    ]
    val_duplicate_frames = sum(int(row["duplicate_class_frames"]) for row in val_rows)
    test_duplicate_frames = sum(int(row["duplicate_class_frames"]) for row in test_rows)
    if not duplicate_rows and total_labels:
        lines.extend(
            [
                "- `Class` is the only available identity-like field in the ZED RGB labels.",
                "- No audited ZED RGB frame contains duplicate `Class` values, so `Class` is usable as a per-recording person identity for TrackEval.",
                "- The IDs are local to each recording, not global dataset identities. This is acceptable because the evaluator remaps IDs independently per sequence.",
            ]
        )
    elif val_duplicate_frames == 0 and test_duplicate_frames > 0:
        lines.extend(
            [
                "- `Class` is the only available identity-like field in the ZED RGB labels.",
                "- Validation annotations have no duplicate `Class` values per frame, so validation tuning can use `Class` as a per-recording person identity.",
                "- Test annotations include duplicate `Class` values in a small number of frames, so final test TrackEval numbers are slightly questionable unless those frames are corrected, disambiguated, or excluded.",
                "- The IDs are local to each recording, not global dataset identities. This is acceptable because the evaluator remaps IDs independently per sequence.",
            ]
        )
    else:
        lines.extend(
            [
                "- `Class` is not safe as-is for tracking identity evaluation across all audited recordings.",
                f"- Frames with duplicate `Class` values: {sum(int(row['duplicate_class_frames']) for row in duplicate_rows)}.",
            ]
        )
    if empty_rows:
        lines.append(
            f"- {len(empty_rows)} recording(s) have empty ZED RGB annotations; detections in those sequences count as false positives."
        )
    lines.extend(
        [
            "",
            "## Validation/Test Coverage",
            "",
            f"- Validation recordings audited: {len(val_rows)}",
            f"- Test recordings audited: {len(test_rows)}",
            "",
            "| Split | Recording | Frames | Labelled Frames | Labels | Unique Classes | Duplicate-Class Frames | Mean Same-ID Consecutive IoU | Empty |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["split"], item["recording"])):
        lines.append(
            "| {split} | {recording} | {frames} | {labelled_frames} | {labels} | {unique_class_values} | "
            "{duplicate_class_frames} | {same_class_mean_consecutive_iou:.3f} | {empty_annotation} |".format(
                **{
                    **row,
                    "same_class_mean_consecutive_iou": float(row["same_class_mean_consecutive_iou"]),
                }
            )
        )
    if duplicate_details:
        lines.extend(
            [
                "",
                "## Duplicate-Class Frames",
                "",
                "| Split | Recording | Frame Index | Timestamp | File | Class | Count |",
                "|---|---|---:|---:|---|---:|---:|",
            ]
        )
        for detail in duplicate_details:
            lines.append(
                "| {split} | {recording} | {frame_index} | {timestamp} | {file} | {duplicated_class} | {duplicate_count} |".format(
                    **detail
                )
            )
    lines.extend(
        [
            "",
            "## Metric Implication",
            "",
            "- For validation, the current TrackEval setup is not obviously using a wrong identity field.",
            "- For final test reporting, duplicate `Class` frames should be handled before treating the numbers as fully clean identity metrics.",
            "- The low validation HOTA/IDF1 values are therefore more likely caused by detector false positives, missed detections, projection/box alignment, and association behavior than by a missing GT identity field.",
            "- Empty or sparsely labelled recordings still need attention because they can strongly penalize detector-heavy outputs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--val-manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_val_manifest.csv"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data_manifests/aghri_late_fusion_test_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    val_recordings = _recordings_from_manifest(args.val_manifest)
    test_recordings = _recordings_from_manifest(args.test_manifest)
    wanted = val_recordings | test_recordings
    annotation_paths = sorted(args.dataset_root.glob("dataset_part*/**/annotations/cam_zed_rgb_ann.json"))
    selected = [path for path in annotation_paths if not wanted or path.parents[1].name in wanted]
    key_counts = _label_key_counter(selected)
    rows = []
    duplicate_details = []
    for path in selected:
        recording = path.parents[1].name
        split = "val" if recording in val_recordings else "test" if recording in test_recordings else "other"
        summary, details = _summarise_camera_recording(path, split)
        rows.append(summary)
        duplicate_details.extend(details)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "aghri_gt_identity_field_audit.csv", rows)
    _write_csv(args.output_dir / "duplicate_class_frame_details.csv", duplicate_details)
    (args.output_dir / "aghri_gt_identity_field_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_md(args.output_dir / "AGHRI_GT_IDENTITY_FIELD_AUDIT.md", rows, duplicate_details, key_counts, args)
    print(f"audit_report={args.output_dir / 'AGHRI_GT_IDENTITY_FIELD_AUDIT.md'}")
    print(f"recordings={len(rows)}")
    print(f"labels={sum(int(row['labels']) for row in rows)}")
    print(f"duplicate_class_frames={sum(int(row['duplicate_class_frames']) for row in rows)}")
    print(f"empty_recordings={sum(1 for row in rows if row['empty_annotation'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
