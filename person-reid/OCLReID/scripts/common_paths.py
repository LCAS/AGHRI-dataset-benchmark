#!/usr/bin/env python3
"""Shared active paths for OCLReID AGHRI workflows."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(os.environ.get("AGHRI_DATASET_ROOT", "/media/prabuddhi/Backup2/Updated Dataset_PW"))
DEFAULT_VIDEO_ROOT = Path(os.environ.get("AGHRI_VIDEO_ROOT", "/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos"))
DEFAULT_RESULTS_ROOT = Path(os.environ.get("OCLREID_RESULTS_ROOT", PROJECT_ROOT / "results/reproduced_runs"))
RELEASED_REID_CHECKPOINT = Path(os.environ.get("REID_CHECKPOINT", PROJECT_ROOT / "checkpoints/reid/resnet18.pth"))
