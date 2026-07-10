#!/usr/bin/env python3
"""Verify a prepared AGHRI ZED RGB YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_aghri_zedrgb_yolo import verify_dataset


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset",
        help="Prepared YOLO dataset root.",
    )
    parser.add_argument(
        "--split-lists-dir",
        default="/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists",
        help="Official split-lists directory.",
    )
    args = parser.parse_args()
    summary = verify_dataset(
        Path(args.output_root).resolve(),
        Path(args.split_lists_dir).resolve(),
        write_summary=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
