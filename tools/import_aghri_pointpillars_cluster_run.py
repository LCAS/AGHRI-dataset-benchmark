#!/usr/bin/env python3
"""Import a completed AGHRI PointPillars fine-tune run from the cluster mount."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ARTIFACT_DIRS = ("configs", "manifests", "environment", "summaries", "logs")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_smoke(path: Path) -> bool:
    return "smoke" in {part.lower() for part in path.parts} or "smoke" in path.name.lower()


def full_checkpoint_candidates(source_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in source_root.glob(pattern) if not is_smoke(path))


def find_best_checkpoint(source_root: Path) -> Path | None:
    candidates = full_checkpoint_candidates(source_root, "outputs/**/best_aghri_bev_ap_mean*.pth")
    if candidates:
        return candidates[-1]
    candidates = full_checkpoint_candidates(source_root, "outputs/**/best_*.pth")
    return candidates[-1] if candidates else None


def find_last_checkpoint(source_root: Path) -> Path | None:
    candidates = full_checkpoint_candidates(source_root, "outputs/**/epoch_*.pth")
    if not candidates:
        candidates = full_checkpoint_candidates(source_root, "outputs/**/last*.pth")
    if not candidates:
        return None

    def epoch_number(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    return sorted(candidates, key=lambda path: (epoch_number(path), path.stat().st_mtime))[-1]


def copy_with_sha(source: Path, dest: Path, *, dry_run: bool, overwrite: bool) -> dict:
    if not source.exists():
        raise FileNotFoundError(source)
    if is_smoke(source):
        raise SystemExit(f"Refusing to import smoke checkpoint/artifact: {source}")
    source_sha = sha256_file(source)
    record = {"source": str(source), "dest": str(dest), "sha256": source_sha, "dry_run": dry_run}
    if dry_run:
        return record
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    copied_sha = sha256_file(dest)
    if copied_sha != source_sha:
        raise RuntimeError(f"SHA mismatch after copy: {source_sha} != {copied_sha}")
    record["copied"] = True
    return record


def copy_tree(source: Path, dest: Path, *, dry_run: bool, overwrite: bool) -> list[dict]:
    records = []
    if not source.exists():
        return records
    for root, dirnames, filenames in os.walk(source):
        dirnames[:] = [name for name in dirnames if name != "__pycache__" and not name.lower().startswith("venv")]
        for filename in filenames:
            path = Path(root) / filename
            if "__pycache__" in path.parts or is_smoke(path):
                continue
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
    parser.add_argument("--source-root", default="/home/prabuddhi/cluster/aghri_pointpillars_finetune")
    parser.add_argument(
        "--dest-root",
        "--destination-root",
        dest="dest_root",
        default="/home/prabuddhi/Desktop/late_fusion/deepfusionmot/checkpoints/aghri_pointpillars_finetune/cluster_run",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--last-checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Preview the import. This is the default unless --import-run is used.")
    parser.add_argument("--import-run", action="store_true", help="Actually copy files. Default is dry-run only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    dest_root = Path(args.dest_root).resolve()
    best = Path(args.checkpoint).resolve() if args.checkpoint else find_best_checkpoint(source_root)
    last = Path(args.last_checkpoint).resolve() if args.last_checkpoint else find_last_checkpoint(source_root)
    if best is None:
        raise SystemExit(f"No full-run best checkpoint found under {source_root}/outputs")
    if last is None:
        raise SystemExit(f"No full-run last/epoch checkpoint found under {source_root}/outputs")
    if is_smoke(best) or is_smoke(last):
        raise SystemExit("Refusing to import a smoke checkpoint")

    dry_run = args.dry_run or not args.import_run
    records = {
        "source_root": str(source_root),
        "destination_root": str(dest_root),
        "dry_run": dry_run,
        "weights": [
            copy_with_sha(best, dest_root / "weights" / "best.pth", dry_run=dry_run, overwrite=args.overwrite),
            copy_with_sha(last, dest_root / "weights" / "last.pth", dry_run=dry_run, overwrite=args.overwrite),
        ],
        "artifact_dirs": {},
    }
    for dirname in ARTIFACT_DIRS:
        records["artifact_dirs"][dirname] = copy_tree(source_root / dirname, dest_root / dirname, dry_run=dry_run, overwrite=args.overwrite)
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
        (dest_root / "import_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        write_sha256s(dest_root)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
