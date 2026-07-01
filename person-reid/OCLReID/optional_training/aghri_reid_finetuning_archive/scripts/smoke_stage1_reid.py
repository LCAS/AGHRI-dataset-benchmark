#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


FREEZE_PREFIXES = (
    "backbone.conv1",
    "backbone.bn1",
    "backbone.layer1",
    "backbone.layer2",
)


def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj if isinstance(obj, dict) else {}


def compatible_load(model, checkpoint: Path):
    source = unwrap_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    target = model.state_dict()
    loadable = {}
    ignored = []
    mismatched = []
    for key, value in source.items():
        if key not in target:
            ignored.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            mismatched.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        loadable[key] = value
    missing, unexpected = model.load_state_dict(loadable, strict=False)
    return loadable, ignored, mismatched, list(missing), list(unexpected)


def freeze_lower_layers(model):
    for name, param in model.named_parameters():
        if name.startswith(FREEZE_PREFIXES):
            param.requires_grad = False
    for name, module in model.named_modules():
        if name.startswith(FREEZE_PREFIXES):
            module.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID"))
    parser.add_argument("--config", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/optional_training/aghri_reid_finetuning_archive/configs/aghri_resnet18_backbone_stage1.py"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/prabuddhi/Desktop/OCLReID/checkpoints/reid/resnet18.pth"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/SMOKE_TEST_REPORT.md"))
    parser.add_argument("--param-report", type=Path, default=Path("/tmp/TRAINABLE_PARAMETER_REPORT.md"))
    args = parser.parse_args()

    os.chdir(str(args.repo))
    sys.path.insert(0, str(args.repo))
    from mmcv import Config
    from mmtrack.models import build_reid

    cfg = Config.fromfile(str(args.config))
    reid_cfg = cfg.model.reid.copy()
    reid_cfg.pop("init_cfg", None)
    model = build_reid(reid_cfg)
    loaded, ignored, mismatched, missing, unexpected = compatible_load(model, args.checkpoint)
    freeze_lower_layers(model)
    smoke_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(smoke_device)

    params = []
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        count = param.numel()
        if param.requires_grad:
            trainable_count += count
        else:
            frozen_count += count
        params.append((name, tuple(param.shape), param.requires_grad, count))

    model.train()
    freeze_lower_layers(model)
    # PK-like toy batch: three identities, four samples each.
    imgs = torch.randn(12, 3, 256, 192, device=smoke_device)
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long, device=smoke_device)
    losses = model.forward_train(imgs, labels)
    total_loss = sum(value for value in losses.values() if torch.is_tensor(value) and value.ndim == 0)
    finite_loss = bool(torch.isfinite(total_loss).item())
    total_loss.backward()
    frozen_grad_violations = [
        name
        for name, param in model.named_parameters()
        if not param.requires_grad and param.grad is not None and torch.any(param.grad != 0)
    ]
    trainable_with_grad = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    ]

    param_lines = [
        "# TRAINABLE PARAMETER REPORT",
        "",
        f"Initial checkpoint: `{args.checkpoint}`",
        "",
        f"- Loaded compatible keys: {len(loaded)}",
        f"- Ignored unexpected keys: {len(ignored)}",
        f"- Shape mismatches: {len(mismatched)}",
        f"- Missing after non-strict load: {len(missing)}",
        f"- Trainable parameters: {trainable_count}",
        f"- Frozen parameters: {frozen_count}",
        "",
        "| parameter | shape | requires_grad | count | initialization_source |",
        "|---|---:|---:|---:|---|",
    ]
    loaded_keys = set(loaded)
    for name, shape, requires_grad, count in params:
        source = "checkpoint" if name in loaded_keys else "random_or_model_init"
        param_lines.append(f"| `{name}` | `{shape}` | {requires_grad} | {count} | {source} |")
    args.param_report.write_text("\n".join(param_lines) + "\n", encoding="utf-8")

    if torch.cuda.is_available():
        status_message = (
            "CUDA is available. Full optional AGHRI training was not launched because "
            "this script is only a model-mechanics smoke test."
        )
    else:
        status_message = (
            "Full optional AGHRI training was not launched because CUDA is unavailable."
        )

    report = f"""# SMOKE TEST REPORT

CUDA available: {torch.cuda.is_available()}
Synthetic smoke device: {smoke_device}

This smoke test used a synthetic PK-like batch to verify model mechanics.
It did not train the optional AGHRI model.

## Results

- Compatible checkpoint keys loaded: {len(loaded)}
- Ignored unexpected keys: {len(ignored)}
- Shape mismatches: {len(mismatched)}
- Finite loss: {finite_loss}
- Total synthetic loss: {float(total_loss.detach()):.6f}
- Trainable tensors with gradients: {len(trainable_with_grad)}
- Frozen gradient violations: {len(frozen_grad_violations)}

## Loss Dictionary

```json
{json.dumps({key: (float(value.detach()) if torch.is_tensor(value) and value.ndim == 0 else str(value)) for key, value in losses.items()}, indent=2)}
```

## Shape Mismatches

```json
{json.dumps([list(item) for item in mismatched[:30]], indent=2)}
```

## Decision

Smoke test {"PASS" if finite_loss and not frozen_grad_violations else "FAIL"}.

{status_message}
"""
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
