"""
Run tracker predictions and MOT evaluation on one MOTChallenge-style dataset root.
Relative CLI paths resolve from the 2d-tracking root.

This script is intentionally separate from `run_tracking_suite.py` so the agri-human
tracking suite can stay focused on the repository's native inputs.

Supported input modes:
- public detector files under `<sequence>/det/det.txt`
- GT files under `<sequence>/gt/gt.txt`, converted into detector-style JSON for
  oracle tracking runs

Example:
    python common/mot/run_motchallenge_suite.py --skip-existing
    python common/mot/run_motchallenge_suite.py --motchallenge-inputs det gt --skip-existing
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from run_tracking_suite import (
    TRACKER_SPECS,
    build_evaluation_command,
    build_tracking_command,
    load_runtime_json,
    load_summary_json,
    module_available,
    resolve_path,
    run_command,
)


@dataclass(frozen=True)
class MotChallengeSequence:
    name: str
    root_dir: Path
    frames_dir: Path
    gt_mot: Path
    det_mot: Optional[Path]
    seq_length: int
    image_ext: str
    dataset_name: str
    dataset_split: str


@dataclass(frozen=True)
class MotChallengeInput:
    input_json: Path
    gt_mot: Path
    input_kind: str
    input_name: str
    run_name: str
    source_path: Path
    sequence_name: str
    frames_dir: Path
    dataset_name: str
    dataset_split: str


def include_matches(includes: Optional[set[str]], *names: str) -> bool:
    """Return whether any provided basename matches the include filter."""

    if not includes:
        return True
    return any(name in includes for name in names if name)


def write_json(path: Path, payload: object) -> None:
    """Write one JSON payload with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_aggregate_summary(results: List[Dict[str, object]], summary_json_path: Path, summary_csv_path: Path) -> None:
    """Persist the current aggregate summary so partial runs are still inspectable."""

    write_json(summary_json_path, results)

    fieldnames: List[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def empty_detection_records(seq_length: int, image_ext: str) -> List[Dict[str, object]]:
    """Create one empty detector-style record per frame in the sequence."""

    if seq_length <= 0:
        return []
    return [
        {
            "File": f"{frame_idx:06d}{image_ext}",
            "Labels": [],
        }
        for frame_idx in range(1, seq_length + 1)
    ]


def load_sequence(sequence_dir: Path, dataset_name: str, dataset_split: str) -> Optional[MotChallengeSequence]:
    """Read one MOTChallenge sequence directory into normalized metadata."""

    seqinfo_path = sequence_dir / "seqinfo.ini"
    if not seqinfo_path.is_file():
        return None

    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    if not parser.has_section("Sequence"):
        return None

    sequence_name = parser.get("Sequence", "name", fallback=sequence_dir.name).strip() or sequence_dir.name
    image_dir_name = parser.get("Sequence", "imDir", fallback="img1").strip() or "img1"
    image_ext = parser.get("Sequence", "imExt", fallback=".jpg").strip() or ".jpg"
    seq_length = parser.getint("Sequence", "seqLength", fallback=0)
    det_mot = sequence_dir / "det" / "det.txt"

    return MotChallengeSequence(
        name=sequence_name,
        root_dir=sequence_dir,
        frames_dir=sequence_dir / image_dir_name,
        gt_mot=sequence_dir / "gt" / "gt.txt",
        det_mot=det_mot if det_mot.is_file() else None,
        seq_length=seq_length,
        image_ext=image_ext,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )


def convert_det_to_json(det_mot: Path, output_json: Path, seq_length: int, image_ext: str) -> None:
    """Convert one MOTChallenge detector file into the shared tracker JSON shape."""

    records = empty_detection_records(seq_length, image_ext)
    with det_mot.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 6:
                continue
            frame_idx = int(float(row[0]))
            if frame_idx < 1 or frame_idx > len(records):
                continue

            confidence = 1.0
            if len(row) > 6 and row[6] not in ("", "-1"):
                confidence = float(row[6])

            records[frame_idx - 1]["Labels"].append(
                {
                    "Class": "person",
                    "BoundingBoxes": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                    "Confidence": confidence,
                }
            )

    write_json(output_json, records)


def convert_gt_to_json(gt_mot: Path, output_json: Path, seq_length: int, image_ext: str) -> None:
    """Convert one MOTChallenge GT file into detector-style JSON for oracle tracking."""

    records = empty_detection_records(seq_length, image_ext)
    with gt_mot.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 6:
                continue
            frame_idx = int(float(row[0]))
            if frame_idx < 1 or frame_idx > len(records):
                continue

            if len(row) > 6 and float(row[6]) <= 0:
                continue
            if len(row) > 7 and int(float(row[7])) != 1:
                continue

            records[frame_idx - 1]["Labels"].append(
                {
                    "Class": "person",
                    "BoundingBoxes": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                    "Confidence": 1.0,
                }
            )

    write_json(output_json, records)


def iter_motchallenge_inputs(
    motchallenge_root: Path,
    generated_inputs_dir: Path,
    includes: Optional[set[str]],
    motchallenge_inputs: set[str],
) -> Iterable[MotChallengeInput]:
    """Yield tracking-ready inputs for one MOTChallenge-style dataset root."""

    dataset_name = motchallenge_root.parent.name or "MOTChallenge"
    dataset_split = motchallenge_root.name
    mot_inputs_dir = generated_inputs_dir / "motchallenge" / dataset_split

    for sequence_dir in sorted(path for path in motchallenge_root.iterdir() if path.is_dir()):
        sequence = load_sequence(sequence_dir, dataset_name, dataset_split)
        if sequence is None or not sequence.gt_mot.is_file() or not sequence.frames_dir.is_dir():
            continue

        if "det" in motchallenge_inputs and sequence.det_mot is not None:
            run_name = f"{sequence.name}_motchallenge_det"
            generated_json = mot_inputs_dir / f"{run_name}.json"
            if include_matches(includes, sequence.name, run_name, generated_json.name, generated_json.stem):
                convert_det_to_json(sequence.det_mot, generated_json, sequence.seq_length, sequence.image_ext)
                yield MotChallengeInput(
                    input_json=generated_json,
                    gt_mot=sequence.gt_mot,
                    input_kind="motchallenge_det",
                    input_name="motchallenge_public_det",
                    run_name=run_name,
                    source_path=sequence.det_mot,
                    sequence_name=sequence.name,
                    frames_dir=sequence.frames_dir,
                    dataset_name=sequence.dataset_name,
                    dataset_split=sequence.dataset_split,
                )

        if "gt" in motchallenge_inputs:
            run_name = f"{sequence.name}_motchallenge_gt"
            generated_json = mot_inputs_dir / f"{run_name}.json"
            if include_matches(includes, sequence.name, run_name, generated_json.name, generated_json.stem):
                convert_gt_to_json(sequence.gt_mot, generated_json, sequence.seq_length, sequence.image_ext)
                yield MotChallengeInput(
                    input_json=generated_json,
                    gt_mot=sequence.gt_mot,
                    input_kind="motchallenge_gt",
                    input_name="gt_as_detections",
                    run_name=run_name,
                    source_path=sequence.gt_mot,
                    sequence_name=sequence.name,
                    frames_dir=sequence.frames_dir,
                    dataset_name=sequence.dataset_name,
                    dataset_split=sequence.dataset_split,
                )


def parse_args() -> argparse.Namespace:
    """Define the CLI for the MOTChallenge suite."""

    parser = argparse.ArgumentParser(
        description="Run supported trackers on MOTChallenge-style sequences such as MOT20.",
    )
    parser.add_argument(
        "--motchallenge-root",
        type=Path,
        default=Path(r"D:/AOC/datasets/MOT20/train"),
        help="MOTChallenge sequence root, for example D:/AOC/datasets/MOT20/train.",
    )
    parser.add_argument(
        "--motchallenge-inputs",
        nargs="+",
        choices=("det", "gt"),
        default=("det",),
        help="Which MOTChallenge inputs to benchmark: public detections, GT-as-detections, or both.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("reports/runs/tracker_suite"),
        help="Directory for tracker prediction artifacts.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports/summary/tracker_suite"),
        help="Directory for per-run metrics and aggregate summaries.",
    )
    parser.add_argument(
        "--summary-prefix",
        type=str,
        default="motchallenge_tracking_suite_summary",
        help="Basename for the aggregate MOTChallenge summary files.",
    )
    parser.add_argument(
        "--include-inputs",
        nargs="+",
        help="Optional list of sequence names or generated input basenames to process.",
    )
    parser.add_argument(
        "--trackers",
        nargs="+",
        choices=sorted(TRACKER_SPECS),
        default=sorted(TRACKER_SPECS),
        help="Tracker suite to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose prediction MOT and metrics JSON already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the MOTChallenge tracking benchmark suite."""

    args = parse_args()
    motchallenge_root = resolve_path(args.motchallenge_root)
    runs_dir = resolve_path(args.runs_dir)
    summary_dir = resolve_path(args.summary_dir)
    generated_inputs_dir = runs_dir / "_inputs"
    includes = set(args.include_inputs) if args.include_inputs else None
    motchallenge_inputs = set(args.motchallenge_inputs)

    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    generated_inputs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "_weights").mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    summary_json_path = summary_dir / f"{args.summary_prefix}.json"
    summary_csv_path = summary_dir / f"{args.summary_prefix}.csv"

    for input_record in iter_motchallenge_inputs(
        motchallenge_root,
        generated_inputs_dir,
        includes,
        motchallenge_inputs,
    ):
        sequence_dir = runs_dir / input_record.run_name
        sequence_summary_dir = summary_dir / input_record.run_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        sequence_summary_dir.mkdir(parents=True, exist_ok=True)

        for tracker_key in args.trackers:
            spec = TRACKER_SPECS[tracker_key]
            pred_mot = sequence_dir / f"{tracker_key}_tracks.txt"
            runtime_json = sequence_summary_dir / f"{tracker_key}_runtime.json"
            metrics_csv = sequence_summary_dir / f"{tracker_key}_metrics.csv"
            metrics_json = sequence_summary_dir / f"{tracker_key}_metrics.json"

            base_record: Dict[str, object] = {
                "framework": spec.framework,
                "tracker": tracker_key,
                "status": "pending",
                "input_kind": input_record.input_kind,
                "input_name": input_record.input_name,
                "input_json": str(input_record.input_json),
                "source_path": str(input_record.source_path),
                "gt_mot": str(input_record.gt_mot),
                "frames_dir": str(input_record.frames_dir),
                "model_path": str(spec.model_path) if spec.model_path is not None else "",
                "pred_mot": str(pred_mot),
                "runtime_json": str(runtime_json),
                "metrics_csv": str(metrics_csv),
                "metrics_json": str(metrics_json),
                "sequence_name": input_record.sequence_name,
                "dataset_name": input_record.dataset_name,
                "dataset_split": input_record.dataset_split,
            }

            if not module_available(spec.import_name):
                results.append(
                    {
                        **base_record,
                        "status": "skipped_missing_dependency",
                        "error": f"Python package {spec.import_name!r} is not importable.",
                    }
                )
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                continue

            if spec.model_path is not None and not resolve_path(spec.model_path).is_file():
                results.append(
                    {
                        **base_record,
                        "status": "skipped_missing_model",
                        "error": f"Required model file not found: {resolve_path(spec.model_path)}",
                    }
                )
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                continue

            if args.skip_existing and pred_mot.exists() and metrics_json.exists() and runtime_json.exists():
                metrics_record = load_summary_json(metrics_json) if metrics_json.exists() else {}
                runtime_record = load_runtime_json(runtime_json)
                results.append({**base_record, **runtime_record, **metrics_record, "status": "completed_cached"})
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                print(f"Skipping existing run: {tracker_key} on {input_record.run_name}")
                continue

            track_result = run_command(
                build_tracking_command(spec, input_record.input_json, pred_mot, runtime_json, input_record.frames_dir)
            )
            if track_result.returncode != 0:
                results.append(
                    {
                        **base_record,
                        "status": "tracking_failed",
                        "error": track_result.stderr.strip() or track_result.stdout.strip(),
                    }
                )
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                print(f"Tracking failed: {tracker_key} on {input_record.run_name}")
                continue

            if not runtime_json.exists():
                results.append(
                    {
                        **base_record,
                        "status": "tracking_failed",
                        "error": f"Tracker run completed but runtime summary was not created: {runtime_json}",
                    }
                )
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                print(f"Tracking failed: {tracker_key} on {input_record.run_name}")
                continue

            runtime_record = load_runtime_json(runtime_json)

            eval_result = run_command(build_evaluation_command(spec, input_record.gt_mot, pred_mot, metrics_csv, metrics_json))
            if eval_result.returncode != 0:
                results.append(
                    {
                        **base_record,
                        **runtime_record,
                        "status": "evaluation_failed",
                        "error": eval_result.stderr.strip() or eval_result.stdout.strip(),
                    }
                )
                write_aggregate_summary(results, summary_json_path, summary_csv_path)
                print(f"Evaluation failed: {tracker_key} on {input_record.run_name}")
                continue

            metrics_record = load_summary_json(metrics_json)
            results.append({**base_record, **runtime_record, **metrics_record, "status": "completed"})
            write_aggregate_summary(results, summary_json_path, summary_csv_path)
            print(f"Completed: {tracker_key} on {input_record.run_name}")

    write_aggregate_summary(results, summary_json_path, summary_csv_path)

    completed = sum(1 for row in results if str(row.get("status", "")).startswith("completed"))
    failed = sum(1 for row in results if str(row.get("status", "")).endswith("failed"))
    skipped = sum(1 for row in results if str(row.get("status", "")).startswith("skipped"))
    print(f"MOTChallenge suite finished. Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
    print(f"Aggregate summary CSV: {summary_csv_path}")
    print(f"Aggregate summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
