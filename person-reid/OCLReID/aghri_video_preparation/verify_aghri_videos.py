#!/usr/bin/env python3
"""Verify one AGHRI MP4 against one frame manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

def frame_rows(manifest: dict) -> list[dict]:
    for key in ("frames", "rows", "frame_manifest"):
        value = manifest.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("No frame list found in manifest.")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = frame_rows(manifest)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")
    report = {
        "video": str(args.video),
        "manifest": str(args.manifest),
        "video_opens": True,
        "video_frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "manifest_row_count": len(rows),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "first_manifest_row": rows[0] if rows else {},
        "last_manifest_row": rows[-1] if rows else {},
        "duplicate_manifest_entries": len(rows) - len({json.dumps(row, sort_keys=True) for row in rows}),
    }
    capture.release()
    report["frame_count_matches_manifest"] = report["video_frame_count"] == report["manifest_row_count"]
    print(json.dumps(report, indent=2))
    return 0 if report["frame_count_matches_manifest"] and report["duplicate_manifest_entries"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
