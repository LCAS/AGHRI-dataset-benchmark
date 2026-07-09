#!/usr/bin/env python3
"""Run AGHRI Ultralytics train/eval from the local benchmark-shaped config."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config."""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_rows(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write summary rows as CSV and JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    out_path.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def eval_results_dir(model: Any) -> Optional[str]:
    """Return the validation save directory across Ultralytics versions."""
    for owner in (getattr(model, "validator", None), getattr(model, "trainer", None)):
        save_dir = getattr(owner, "save_dir", None)
        if save_dir:
            return str(save_dir)
    return None


def trained_checkpoint(model: Any, train_results: Any) -> Optional[Path]:
    """Find the best trained checkpoint."""
    candidates = [
        getattr(train_results, "save_dir", None),
        getattr(getattr(model, "trainer", None), "save_dir", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        run_dir = Path(candidate)
        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            return best
        last = run_dir / "weights" / "last.pt"
        if last.is_file():
            return last
    return None


def collect_metrics(
    cfg: Dict[str, Any],
    model_name: str,
    model_source: str,
    checkpoint: Optional[Path],
    train_time_sec: Optional[float],
    train_results_dir: Optional[str],
    eval_results: Any,
    eval_dir: Optional[str],
) -> Dict[str, Any]:
    """Collect benchmark-like metrics from Ultralytics DetMetrics."""
    box = eval_results.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    speed = getattr(eval_results, "speed", None) or {}
    return {
        "model": model_name,
        "model_source": model_source,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "run_type": cfg["mode"],
        "train_enabled": cfg["mode"] == "train_eval",
        "train_time_sec": train_time_sec,
        "eval_split": cfg.get("eval_split", "val"),
        "num_classes": len(getattr(eval_results, "names", {}) or {}),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inf_time_per_frame_ms": speed.get("inference"),
        "preprocess_time_per_frame_ms": speed.get("preprocess"),
        "postprocess_time_per_frame_ms": speed.get("postprocess"),
        "epochs": cfg.get("epochs"),
        "imgsz": cfg.get("imgsz"),
        "batch": cfg.get("batch"),
        "dataset": cfg["eval_dataset"],
        "train_dataset": cfg.get("train_dataset"),
        "eval_dataset": cfg["eval_dataset"],
        "config_device": cfg.get("device"),
        "train_results_dir": train_results_dir,
        "eval_results_dir": eval_dir,
    }


def run_config(config_path: Path, out_path: Path) -> List[Dict[str, Any]]:
    """Run train/eval for each configured model."""
    from ultralytics import YOLO

    cfg = load_yaml(config_path)
    mode = cfg.get("mode")
    if mode != "train_eval":
        raise ValueError("This local runner currently supports mode: train_eval")
    rows: List[Dict[str, Any]] = []
    project = str(Path(cfg["project"]).resolve())
    base_name = str(cfg.get("name", "agri-human-detection"))
    for model_spec in cfg["models"]:
        model_name = str(model_spec["name"])
        source = str(Path(model_spec["source"]).resolve())
        model = YOLO(source)
        train_name = f"{base_name}/{model_name}"
        started = time.time()
        train_results = model.train(
            data=str(Path(cfg["train_dataset"]).resolve()),
            imgsz=cfg["imgsz"],
            epochs=int(cfg["epochs"]),
            batch=int(cfg["batch"]),
            device=cfg.get("device"),
            seed=cfg.get("seed"),
            project=project,
            name=train_name,
            verbose=True,
        )
        train_time = round(time.time() - started, 2)
        checkpoint = trained_checkpoint(model, train_results)
        train_dir = str(checkpoint.parent.parent) if checkpoint else None
        eval_model = YOLO(str(checkpoint)) if checkpoint else model
        eval_results = eval_model.val(
            data=str(Path(cfg["eval_dataset"]).resolve()),
            imgsz=cfg["imgsz"],
            batch=int(cfg["batch"]),
            device=cfg.get("device"),
            split=cfg.get("eval_split", "val"),
            project=project,
            name=f"{base_name}/{model_name}/{cfg.get('eval_split', 'val')}",
            verbose=True,
            rect=bool(cfg.get("rect", False)),
            half=bool(cfg.get("half", False)),
            dnn=bool(cfg.get("dnn", False)),
            workers=int(cfg.get("workers", 8)),
        )
        rows.append(
            collect_metrics(
                cfg=cfg,
                model_name=model_name,
                model_source=source,
                checkpoint=checkpoint,
                train_time_sec=train_time,
                train_results_dir=train_dir,
                eval_results=eval_results,
                eval_dir=eval_results_dir(eval_model),
            )
        )
    write_rows(rows, out_path)
    return rows


def main() -> None:
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = run_config(Path(args.config).resolve(), Path(args.out).resolve())
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
