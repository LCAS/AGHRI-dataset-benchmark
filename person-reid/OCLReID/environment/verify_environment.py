#!/usr/bin/env python3
"""Verify the active OCLReID runtime environment."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CHECKPOINT = Path(os.environ.get("REID_CHECKPOINT", ROOT / "checkpoints/reid/resnet18.pth"))

def check_import(name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(name)
        return {"name": name, "status": "ok", "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        return {"name": name, "status": "failed", "error": repr(exc)}

def main() -> int:
    imports = [
        "torch", "torchvision", "mmcv", "mmdet", "mmtrack", "cv2",
        "numpy", "scipy", "sklearn", "scripts.run_single_video",
        "scripts.aghri_alignment", "mmtrack.models.pose",
        "mmtrack.models.orientation", "mmtrack.models.identifier",
    ]
    results = [check_import(name) for name in imports]
    try:
        import torch
        torch_status = {
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "torch_cuda": str(torch.version.cuda),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        torch_status = {"error": repr(exc)}
    files = {
        "released_reid_checkpoint": str(CHECKPOINT),
        "released_reid_checkpoint_exists": CHECKPOINT.is_file(),
        "pose_module_dir_exists": (ROOT / "mmtrack/models/pose").is_dir(),
        "orientation_module_dir_exists": (ROOT / "mmtrack/models/orientation").is_dir(),
        "detector_config_dir_exists": (ROOT / "configs/rpf/ocl_rpf").is_dir(),
    }
    payload = {"imports": results, "torch": torch_status, "files": files}
    print(json.dumps(payload, indent=2))
    failed = [item for item in results if item["status"] != "ok"]
    return 1 if failed or not CHECKPOINT.is_file() else 0

if __name__ == "__main__":
    raise SystemExit(main())
