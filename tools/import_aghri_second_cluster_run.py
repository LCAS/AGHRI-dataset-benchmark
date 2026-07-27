#!/usr/bin/env python3
"""Import a completed AGHRI SECOND fine-tune run from the cluster mount."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aghri_second_finetune_common import sha256_file


ARTIFACT_DIRS = ("configs", "manifests", "environment", "summaries", "logs")


def _is_smoke(path: Path) -> bool:
    return "smoke" in {part.lower() for part in path.parts}


def _full_checkpoint_candidates(source_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in source_root.glob(pattern) if not _is_smoke(path))


def find_best_checkpoint(source_root: Path) -> Optional[Path]:
    candidates = _full_checkpoint_candidates(source_root, "outputs/**/best_aghri_bev_ap_mean*.pth")
    if candidates:
        return candidates[-1]
    candidates = _full_checkpoint_candidates(source_root, "outputs/**/best_*.pth")
    return candidates[-1] if candidates else None


def find_last_checkpoint(source_root: Path) -> Optional[Path]:
    epoch_candidates = _full_checkpoint_candidates(source_root, "outputs/**/epoch_*.pth")
    if not epoch_candidates:
        return None

    def epoch_number(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return sorted(epoch_candidates, key=lambda path: (epoch_number(path), path.stat().st_mtime))[-1]


def copy_with_sha(source: Path, dest: Path, dry_run: bool = False, overwrite: bool = False) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Refusing to overwrite existing file: {dest}")
    before = sha256_file(source)
    result = {"source": str(source), "dest": str(dest), "source_sha256": before, "sha256": before, "dry_run": dry_run}
    if dry_run:
        return result
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    after = sha256_file(dest)
    if after != before:
        raise RuntimeError(f"SHA mismatch after copy: {before} != {after}")
    result["destination_sha256"] = after
    result["copied"] = True
    return result


def _include_artifact_file(base: Path, path: Path) -> bool:
    rel = path.relative_to(base)
    parts = {part.lower() for part in rel.parts}
    if any(part.startswith("venv") for part in parts):
        return False
    if "__pycache__" in parts:
        return False
    return True


def copy_tree_files(source: Path, dest: Path, dry_run: bool, overwrite: bool) -> list[dict]:
    records = []
    if not source.exists():
        return records
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(source):
        dirnames[:] = [
            dirname for dirname in dirnames
            if not dirname.lower().startswith("venv") and dirname != "__pycache__"
        ]
        root_path = Path(root)
        for filename in filenames:
            path = root_path / filename
            if _include_artifact_file(source, path):
                files.append(path)
    for path in sorted(files):
        target = dest / path.relative_to(source)
        records.append(copy_with_sha(path, target, dry_run=dry_run, overwrite=overwrite))
    return records


def write_sha256s(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS.txt"):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", "--source-root", dest="source_root", default="/home/prabuddhi/cluster/aghri_second_finetune")
    parser.add_argument("--dest-root", "--destination-root", dest="destination_root", default="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/checkpoints/aghri_second_finetune/cluster_run")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path. Defaults to newest outputs/**/best_*.pth")
    parser.add_argument("--last-checkpoint", help="Explicit last checkpoint path. Defaults to highest full-run epoch_*.pth")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Perform the import and verify copied hashes.")
    parser.add_argument("--verify-only", action="store_true", help="Inspect selected artifacts without copying.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    best_checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else find_best_checkpoint(source_root)
    last_checkpoint = Path(args.last_checkpoint).resolve() if args.last_checkpoint else find_last_checkpoint(source_root)
    if best_checkpoint is None:
        raise SystemExit(f"No full-run best checkpoint found under {source_root}/outputs")
    if last_checkpoint is None:
        raise SystemExit(f"No full-run epoch checkpoint found under {source_root}/outputs")
    if _is_smoke(best_checkpoint) or _is_smoke(last_checkpoint):
        raise SystemExit("Refusing to import a smoke checkpoint")

    dest_root = Path(args.destination_root).resolve()
    dry_run = args.dry_run or args.verify_only
    records = {
        "source_root": str(source_root),
        "destination_root": str(dest_root),
        "dry_run": dry_run,
        "best_checkpoint_original": str(best_checkpoint),
        "last_checkpoint_original": str(last_checkpoint),
        "weights": [
            copy_with_sha(best_checkpoint, dest_root / "weights" / "best.pth", dry_run=dry_run, overwrite=args.overwrite),
            copy_with_sha(last_checkpoint, dest_root / "weights" / "last.pth", dry_run=dry_run, overwrite=args.overwrite),
        ],
        "artifact_dirs": {},
    }

    for dirname in ARTIFACT_DIRS:
        records["artifact_dirs"][dirname] = copy_tree_files(
            source_root / dirname,
            dest_root / dirname,
            dry_run=dry_run,
            overwrite=args.overwrite,
        )

    if not dry_run:
        (dest_root / "import_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        write_sha256s(dest_root)

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
