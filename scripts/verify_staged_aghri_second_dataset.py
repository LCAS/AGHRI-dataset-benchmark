#!/usr/bin/env python3
"""Verify a staged AGHRI MMDetection3D export for SECOND fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List


def load_infos(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def verify_dataset(data_root: Path, allow_test: bool = False) -> dict:
    infos_dir = data_root / "infos"
    required = {
        "train": infos_dir / "agri_person_infos_train.pkl",
        "val": infos_dir / "agri_person_infos_val.pkl",
    }
    errors: List[str] = []
    warnings: List[str] = []
    split_samples: Dict[str, set] = {}
    summary = {"data_root": str(data_root), "splits": {}, "allow_test": allow_test}

    if not data_root.exists():
        errors.append(f"Data root does not exist: {data_root}")
    for split, path in required.items():
        if not path.exists():
            errors.append(f"Missing {split} info file: {path}")
            continue
        payload = load_infos(path)
        data_list = payload.get("data_list", [])
        split_samples[split] = {str(item.get("sample_idx", "")) for item in data_list}
        missing_points = []
        instance_count = 0
        empty_frame_count = 0
        for item in data_list:
            rel = item.get("lidar_points", {}).get("lidar_path", "")
            if not rel or not (data_root / rel).exists():
                missing_points.append(str(item.get("sample_idx", rel)))
            instances = item.get("instances", [])
            instance_count += len(instances)
            if not instances:
                empty_frame_count += 1
        if missing_points:
            errors.append(f"{split} has {len(missing_points)} info rows with missing point files")
        if empty_frame_count:
            warnings.append(f"{split} has {empty_frame_count} frames without 3D boxes")
        summary["splits"][split] = {
            "info_file": str(path),
            "frames": len(data_list),
            "instances": instance_count,
            "empty_frames": empty_frame_count,
            "missing_point_files": len(missing_points),
        }

    shared = sorted(split_samples.get("train", set()) & split_samples.get("val", set()))
    if shared:
        errors.append(f"Train/val sample leakage: {len(shared)} shared samples")

    test_paths = [
        infos_dir / "agri_person_infos_test.pkl",
        data_root / "agri_person_infos_test.pkl",
        data_root / "ImageSets" / "test.txt",
    ]
    present_test = [str(path) for path in test_paths if path.exists()]
    summary["test_artifacts_present"] = present_test
    if present_test and not allow_test:
        errors.append("Held-out test split artifacts are present in active training root")

    manifest_path = data_root / "aghri_second_finetune_verification_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "frames", "instances", "empty_frames", "missing_point_files"])
        writer.writeheader()
        for split, values in summary["splits"].items():
            writer.writerow({"split": split, **{k: values[k] for k in writer.fieldnames if k != "split"}})
    summary["manifest_csv"] = str(manifest_path)
    summary["errors"] = errors
    summary["warnings"] = warnings
    summary["ok"] = not errors
    return summary


def write_report(summary: dict, path: Path) -> None:
    lines = [
        "# AGHRI SECOND staged dataset verification",
        "",
        f"- Data root: `{summary['data_root']}`",
        f"- Status: {'PASS' if summary['ok'] else 'FAIL'}",
        f"- Test artifacts present: {len(summary['test_artifacts_present'])}",
        "",
        "## Splits",
        "",
        "| split | frames | instances | empty frames | missing point files |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, values in summary["splits"].items():
        lines.append(
            f"| {split} | {values['frames']} | {values['instances']} | "
            f"{values['empty_frames']} | {values['missing_point_files']} |"
        )
    if summary["errors"]:
        lines += ["", "## Errors", ""]
        lines += [f"- {item}" for item in summary["errors"]]
    if summary["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {item}" for item in summary["warnings"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--json-out")
    parser.add_argument("--md-out")
    args = parser.parse_args()

    summary = verify_dataset(Path(args.data_root).resolve(), allow_test=args.allow_test)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.md_out:
        write_report(summary, Path(args.md_out))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["ok"] else 2)


if __name__ == "__main__":
    main()
