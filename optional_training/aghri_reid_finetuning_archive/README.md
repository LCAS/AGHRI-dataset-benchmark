# Optional AGHRI ReID Fine-Tuning Archive

This folder preserves the optional AGHRI ReID fine-tuning experiment. It was not
used as the final reported method. The final main comparison uses the released
checkpoint at `checkpoints/reid/resnet18.pth`.

## Archive Contents

```text
checkpoints/          # optional AGHRI experimental checkpoint
configs/              # MMTrack ReID training config
crops/                # exported train/val ReID crops
identity_audit/       # identity map and contact sheets
manifests/            # train/val crop manifests and ReID txt files
reports/              # training/export/smoke-test reports
scripts/              # manifest, crop, smoke, and checkpoint export scripts
training/             # saved training work directories and checkpoints
downstream_test/      # archived downstream test ablations
```

## Identity Split

```text
train:         person_07, person_08, person_10
validation:    person_03, person_04
reserved test: person_01, person_02, person_05, person_06, person_09
```

Original final-test scenes and reserved-test identities are excluded from crop
export to avoid leakage.

## Step 1: Inspect Identity Audit

Start with:

```text
identity_audit/aghri_identity_map_confirmed.csv
identity_audit/contact_sheets/
```

The identity map records the manually confirmed mapping from local annotation
classes to global person IDs and ReID split membership. Contact sheets provide a
visual check of the identities used for crops.

## Step 2: Build Manifests

Run from the repository root:

```bash
python optional_training/aghri_reid_finetuning_archive/scripts/build_aghri_reid_manifest.py \
  --dataset-root "$AGHRI_DATASET_ROOT" \
  --video-root "$AGHRI_VIDEO_ROOT" \
  --frame-stride 5
```

This creates or updates:

```text
manifests/all_samples_manifest.csv
manifests/train_manifest.csv
manifests/val_manifest.csv
manifests/excluded_samples.csv
```

## Step 3: Export Crops

Run from the repository root:

```bash
python optional_training/aghri_reid_finetuning_archive/scripts/export_aghri_reid_crops.py
```

This writes:

```text
crops/train/
crops/val/
manifests/train_reid.txt
manifests/val_reid.txt
reports/DATASET_EXPORT_REPORT.md
```

The crops are natural-resolution annotation-box crops. Resizing and
normalization are handled by the training pipeline.

## Step 4: Smoke-Test the ReID Model

Run from the repository root:

```bash
python optional_training/aghri_reid_finetuning_archive/scripts/smoke_stage1_reid.py \
  --report optional_training/aghri_reid_finetuning_archive/reports/SMOKE_TEST_REPORT.md \
  --param-report optional_training/aghri_reid_finetuning_archive/reports/TRAINABLE_PARAMETER_REPORT.md
```

This checks that the ReID model can load compatible checkpoint keys, freeze the
intended lower layers, run a synthetic PK-like batch, produce finite losses, and
backpropagate through trainable parameters.

## Step 5: Run a Short Training Experiment

The preserved config is:

```text
configs/aghri_resnet18_backbone_stage1.py
```

Run from the repository root:

```bash
python tools/train.py \
  optional_training/aghri_reid_finetuning_archive/configs/aghri_resnet18_backbone_stage1.py \
  --work-dir optional_training/aghri_reid_finetuning_archive/training/work_dirs/aghri_resnet18_backbone_stage1_5ep \
  --cfg-options runner.max_epochs=5 total_epochs=5
```

Archived training outputs are preserved in:

```text
training/work_dirs/
```

## Step 6: Export a Runtime-Compatible Checkpoint

After training, export only runtime-compatible backbone/shared keys:

```bash
python optional_training/aghri_reid_finetuning_archive/scripts/export_runtime_backbone_checkpoint.py \
  --input optional_training/aghri_reid_finetuning_archive/training/work_dirs/aghri_resnet18_backbone_stage1_5ep/epoch_5.pth \
  --output optional_training/aghri_reid_finetuning_archive/checkpoints/aghri_resnet18_backbone_stage1.pth \
  --mirror-output checkpoints/reid/aghri_resnet18_backbone_stage1.pth \
  --report optional_training/aghri_reid_finetuning_archive/reports/RUNTIME_CHECKPOINT_COMPATIBILITY.md
```

The exporter refuses to overwrite the released `checkpoints/reid/resnet18.pth`.

## Step 7: Run Downstream Ablation

Use the optional checkpoint by setting `REID_CHECKPOINT`:

```bash
REID_CHECKPOINT=/path/to/OCLReID/checkpoints/reid/aghri_resnet18_backbone_stage1.pth \
./methods/normal_part_oclreid/run_aghri_test.sh \
  --results-root optional_training/aghri_reid_finetuning_archive/downstream_test/new_optional_checkpoint_test \
  --save-visualizations \
  --overwrite
```

For a gated optional-checkpoint ablation:

```bash
REID_CHECKPOINT=/path/to/OCLReID/checkpoints/reid/aghri_resnet18_backbone_stage1.pth \
./methods/gated_part_oclreid/run_aghri_test.sh \
  --results-root optional_training/aghri_reid_finetuning_archive/downstream_test/new_optional_checkpoint_gate_test \
  --save-visualizations \
  --overwrite
```

Archived downstream optional-checkpoint results are kept in:

```text
downstream_test/
```
