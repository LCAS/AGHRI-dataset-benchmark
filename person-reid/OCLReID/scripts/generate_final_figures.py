#!/usr/bin/env python3
"""Regenerate compact final comparison plots from active CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/final_comparison"
FIG_DIR = RESULT_DIR / "figures"

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def bar_plot(path: Path, x_key: str, value_key: str, title: str, output: str) -> None:
    rows = read_rows(path)
    labels = [row[x_key] for row in rows]
    values = [float(row[value_key]) for row in rows]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=["#3a6ea5", "#4f8f58", "#c46a3a"][: len(labels)])
    plt.ylabel(value_key.replace("_", " "))
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / output, dpi=220)
    plt.close()

def main() -> None:
    bar_plot(RESULT_DIR / "overall_method_comparison.csv", "method", "success_iou_0_5", "Overall Success@0.5", "regenerated_overall_success.png")
    print(f"Regenerated figures in {FIG_DIR}")

if __name__ == "__main__":
    main()
