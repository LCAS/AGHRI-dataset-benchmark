"""
Convert a ground-truth annotations JSON file into MOTChallenge format.
Relative CLI paths resolve from the 2d-tracking root.

Examples:
    python common/mot/convert_gt_to_mot.py --ground-truth-json /path/to/annotations.json --out reports/runs/example/ground_truth.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

TRACKING_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    """Resolve repo-relative CLI paths from the 2d-tracking root."""

    if path.is_absolute():
        return path
    return (TRACKING_ROOT / path).resolve()


def labels_iter(record: Dict[str, Any]) -> Iterable[Tuple[object, Tuple[float, float, float, float]]]:
    """Yield `(class_name, xywh_box)` pairs from one annotation record."""

    # Ground-truth records follow the same File/Labels pattern as detector exports.
    for label in record.get("Labels", []):
        if not isinstance(label, dict):
            continue
        class_name = label.get("Class") or label.get("class") or label.get("label")
        bbox = label.get("BoundingBoxes") or label.get("bbox") or label.get("box")
        if class_name is None or bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = map(float, bbox)
        yield class_name, (x, y, w, h)


def class_to_track_id(class_name: object) -> int:
    """Convert a positive digit-only AGHRI identity into a MOT track ID."""

    if not isinstance(class_name, str):
        raise ValueError(
            f"Invalid AGHRI person identity {class_name!r}; expected a string such as '01'."
        )
    identity = class_name.strip()
    if not identity.isdigit() or int(identity) <= 0:
        raise ValueError(
            f"Invalid AGHRI person identity {class_name!r}; expected a positive "
            "digit-only Class value such as '01' or '10'."
        )
    return int(identity)


def convert_gt_to_mot(annotations_json: Path, output_path: Path) -> None:
    """Write one project annotation JSON file to MOTChallenge text format."""

    records = json.loads(annotations_json.read_text(encoding="utf-8"))
    lines = []

    for frame_idx, record in enumerate(records, start=1):
        for class_name, (x, y, w, h) in labels_iter(record):
            track_id = class_to_track_id(class_name)
            lines.append(f"{frame_idx},{track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")

    print(f"GT MOT file written: {output_path}")
    print(f"Total frames: {len(records)}, Total lines: {len(lines)}")


def parse_args() -> argparse.Namespace:
    """Define the shared CLI for GT-to-MOT conversion."""

    parser = argparse.ArgumentParser(
        description="Convert annotations JSON into MOT format.",
        epilog=(
            "Example:\n"
            "  python common/mot/convert_gt_to_mot.py --ground-truth-json /path/to/annotations.json "
            "--out reports/runs/example/ground_truth.txt"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ground-truth-json", required=True, type=Path, help="Ground-truth annotations JSON file.")
    parser.add_argument("--out", required=True, type=Path, help="Output MOT txt file.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the shared conversion utility."""

    args = parse_args()
    convert_gt_to_mot(resolve_path(args.ground_truth_json), resolve_path(args.out))


if __name__ == "__main__":
    main()
