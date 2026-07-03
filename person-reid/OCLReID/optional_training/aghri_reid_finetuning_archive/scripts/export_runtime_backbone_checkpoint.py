#!/usr/bin/env python3
"""Export optional AGHRI-trained weights into a runtime-compatible checkpoint.

This script keeps only keys that can be safely loaded into the current
part-OCLReID `PartWeightedClassifier`: backbone keys and any exact-shape
compatible non-identity-head keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch


def state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def force_cpu_part_head():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[attr-defined]
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self  # type: ignore[attr-defined]


def build_part_reid(repo: Path):
    from mmcv import Config
    from mmtrack.models import build_reid

    force_cpu_part_head()
    cfg = Config.fromfile(str(repo / "configs/rpf/ocl_rpf/part_rpf_weighted_yolox_l_r18.py"))
    reid_cfg = cfg.model.reid.copy()
    reid_cfg.pop("init_cfg", None)
    return build_reid(reid_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive/checkpoints/aghri_resnet18_backbone_stage1.pth"))
    parser.add_argument("--mirror-output", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/checkpoints/reid/aghri_resnet18_backbone_stage1.pth"))
    parser.add_argument("--report", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive/reports/RUNTIME_CHECKPOINT_COMPATIBILITY.md"))
    args = parser.parse_args()

    os.chdir(str(args.repo))
    sys.path.insert(0, str(args.repo))
    trained = state_dict(torch.load(str(args.input), map_location="cpu"))
    target_model = build_part_reid(args.repo)
    target_shapes = {k: tuple(v.shape) for k, v in target_model.state_dict().items()}

    exported = {}
    matched = []
    excluded = []
    skipped = []
    mismatched = []
    for key, value in trained.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if not torch.is_tensor(value):
            excluded.append(clean_key)
            continue
        if "classifier" in clean_key:
            excluded.append(clean_key)
            continue
        if clean_key not in target_shapes:
            skipped.append(clean_key)
            continue
        if tuple(value.shape) != target_shapes[clean_key]:
            mismatched.append((clean_key, tuple(value.shape), target_shapes[clean_key]))
            continue
        if clean_key.startswith("backbone.") or clean_key.startswith("head.fcs."):
            exported[clean_key] = value.cpu()
            matched.append(clean_key)
        else:
            excluded.append(clean_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": exported}, str(args.output))
    args.mirror_output.parent.mkdir(parents=True, exist_ok=True)
    if args.mirror_output.resolve() == (args.repo / "checkpoints/reid/resnet18.pth").resolve():
        raise ValueError("Refusing to overwrite the released resnet18.pth checkpoint")
    shutil.copy2(args.output, args.mirror_output)

    exported_hash = sha256_file(args.output)
    released_hash = sha256_file(args.repo / "checkpoints/reid/resnet18.pth")
    mirror_hash = sha256_file(args.mirror_output)
    missing = sorted(set(target_shapes) - set(exported))
    unexpected = sorted(set(exported) - set(target_shapes))
    classifier_keys = sorted(key for key in exported if "classifier" in key)
    optimizer_like_keys = sorted(
        key for key in exported
        if key.startswith("optimizer") or key.startswith("param_groups")
    )
    pass_status = (
        bool(exported)
        and len(unexpected) == 0
        and len(classifier_keys) == 0
        and len(optimizer_like_keys) == 0
        and len(mismatched) == 0
        and exported_hash == mirror_hash
    )
    report = f"""# RUNTIME CHECKPOINT COMPATIBILITY

Selected source checkpoint: `{args.input}`

Exported checkpoint path: `{args.output}`

Runtime copy: `{args.mirror_output}`

Runtime model config: `configs/rpf/ocl_rpf/part_rpf_weighted_yolox_l_r18.py`

## Summary

- Exported key count: {len(exported)}
- Matched key count: {len(matched)}
- Shape-compatible transferred parameters: {sum(v.numel() for v in exported.values())}
- Missing runtime keys after export: {len(missing)}
- Unexpected exported keys: {len(unexpected)}
- Shape mismatches: {len(mismatched)}
- Excluded source keys: {len(excluded)}
- Skipped source keys not present in runtime model: {len(skipped)}
- Temporary classifier keys exported: {len(classifier_keys)}
- Optimizer/scheduler-like keys exported: {len(optimizer_like_keys)}
- Exported SHA256: `{exported_hash}`
- Runtime copy SHA256: `{mirror_hash}`
- Released `resnet18.pth` SHA256 after export: `{released_hash}`

The optional AGHRI export intentionally keeps backbone and compatible shared projection
weights only. The custom part/global identity classifiers remain initialized by
the existing runtime procedure.

## First matched keys

```json
{json.dumps(matched[:40], indent=2)}
```

## Exported keys

```json
{json.dumps(sorted(exported.keys()), indent=2)}
```

## First missing runtime keys

```json
{json.dumps(missing[:40], indent=2)}
```

## Unexpected exported keys

```json
{json.dumps(unexpected, indent=2)}
```

## Excluded keys

```json
{json.dumps(excluded[:80], indent=2)}
```

## Skipped source keys

```json
{json.dumps(skipped[:80], indent=2)}
```

## Shape mismatches

```json
{json.dumps([list(item) for item in mismatched[:40]], indent=2)}
```

## Conclusion

{"RUNTIME CHECKPOINT COMPATIBILITY: PASS" if pass_status else "RUNTIME CHECKPOINT COMPATIBILITY: FAIL"}
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    if not pass_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
