# STAGE1 FINAL REPORT

Status date: 2026-06-22

## Status

Stage 1 has completed the identity confirmation, manifest generation, crop
export, training configuration, trainable-parameter audit, CPU model-mechanics
smoke test, and one-epoch real-data GPU smoke test.

The full 30-epoch Stage 1 experiment has **not** been launched yet. The
one-epoch smoke test successfully used:

```text
cuda:0
NVIDIA GeForce RTX 4060 Laptop GPU
```

No runtime checkpoint was exported, and no downstream `part-OCLReID`
validation comparison was run, because this task intentionally stopped after
the one-epoch training smoke test.

## Dataset

Identity mapping source:

```text
aghri_reid_stage1/identity_audit/aghri_identity_map_confirmed.csv
```

Confirmed global identity mapping:

```text
01 -> person_01
02 -> person_02
03 -> person_03
04 -> person_04
05 -> person_05
06 -> person_06
07 -> person_07
08 -> person_08
09 -> person_09
10 -> person_10
```

Stage 1 ReID split:

| ReID split | identities |
|---|---|
| train | `person_07`, `person_08`, `person_10` |
| val | `person_03`, `person_04` |
| reserved_test | `person_01`, `person_02`, `person_05`, `person_06`, `person_09` |

Crop export summary:

| split | crops |
|---|---:|
| train | 1823 |
| val | 305 |

Per identity:

| identity | crops |
|---|---:|
| person_03 | 154 |
| person_04 | 151 |
| person_07 | 477 |
| person_08 | 570 |
| person_10 | 776 |

Leakage checks:

- reserved identities exported: 0
- original final-test scene crops exported: 0
- original `test.txt` scenes excluded before image-size reads/crop export
- reserved identities excluded even when present in original train/val scenes

## Training Configuration

Configuration:

```text
aghri_reid_stage1/configs/aghri_resnet18_backbone_stage1.py
```

Model:

- `BaseReID`
- ResNet18 backbone
- 128-dimensional embedding
- temporary 3-class AGHRI identity classifier
- cross-entropy loss
- triplet loss, margin 0.3

Frozen layers:

```text
backbone.conv1
backbone.bn1
backbone.layer1
backbone.layer2
```

Trainable layers:

```text
backbone.layer3
backbone.layer4
temporary embedding head
temporary AGHRI classifier head
```

Training defaults:

- identity-balanced PK sampling: 3 identities x 4 images
- AdamW
- base LR 1e-4
- head LR multiplier 3.0
- weight decay 1e-4
- max epochs 30
- seed 42

## Smoke Test

Report:

```text
aghri_reid_stage1/reports/SMOKE_TEST_REPORT.md
```

Result: PASS.

Key checks:

- compatible checkpoint keys loaded: 134
- expected shape mismatches: 2 (`head.classifier.weight`, `head.classifier.bias`)
- finite synthetic CE+triplet loss: yes
- backward pass: yes
- frozen gradient violations: 0

## GPU Real-Data Smoke Test

Report:

```text
aghri_reid_stage1/reports/GPU_REAL_DATA_SMOKE_TEST.md
```

Result: PASS.

Key checks:

- CUDA device used: `cuda:0`
- real AGHRI training crops loaded
- PK batches constructed from 3 identities x 4 images
- final logged CE loss: 0.00014
- final logged triplet loss: 0.0
- validation ran on 305 samples
- checkpoint saved: `aghri_reid_stage1/training/work_dirs/gpu_smoke_test/epoch_1.pth`

## Runtime Checkpoint

Not exported because full 30-epoch training did not run.

The export script is available:

```text
aghri_reid_stage1/scripts/export_runtime_backbone_checkpoint.py
```

Expected output after training:

```text
aghri_reid_stage1/checkpoints/aghri_resnet18_backbone_stage1.pth
```

## Downstream Validation Result

Not run. There is no fine-tuned checkpoint yet.

No validation metric claims are made.

## Interpretation

This is now an exploratory identity-disjoint AGHRI backbone fine-tuning setup.
It is ready for Prabuddhi to review the one-epoch smoke result before launching
the full 30-epoch experiment.

Because no full-training checkpoint exists yet:

1. Did AGHRI backbone fine-tuning improve `part-OCLReID`? Not evaluated.
2. Which metrics improved? Not evaluated.
3. Which metrics decreased? Not evaluated.
4. Is improvement consistent across sequences/cameras? Not evaluated.
5. Is there evidence of overfitting? Not evaluated.
6. Should Stage 1 proceed to final test? No, not before training and validation.
7. Should Stage 2 part-aware fine-tuning be attempted? Not yet.

RECOMMENDATION: revise training before final-test evaluation
