# GPU REAL-DATA SMOKE TEST

Run date: 2026-06-22

## Device

- Config device: `cuda`
- CUDA device used by training: `cuda:0`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Post-run `nvidia-smi`: 9 MiB / 8188 MiB
- Peak memory reported by training log: 245 MiB

## Command

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n oclreid python tools/train.py \
  aghri_reid_stage1/configs/aghri_resnet18_backbone_stage1.py \
  --work-dir aghri_reid_stage1/training/work_dirs/gpu_smoke_test \
  --gpu-id 0 \
  --cfg-options \
    total_epochs=1 \
    runner.max_epochs=1 \
    checkpoint_config.interval=1
```

The normal config still defaults to 30 epochs. The smoke-test copy saved in the
work directory correctly records:

```text
device: cuda
runner.max_epochs: 1
total_epochs: 1
checkpoint_config.interval: 1
```

## Data

- Train annotation file: `aghri_reid_stage1/manifests/train_reid.txt`
- Validation annotation file: `aghri_reid_stage1/manifests/val_reid.txt`
- Training samples/iterations: 1823
- Validation samples: 305
- PK sampler: 3 identities x 4 images

## Checkpoint Loading

Initial checkpoint:

```text
checkpoints/reid/resnet18.pth
```

Expected classifier mismatch:

```text
head.classifier.weight: [380, 128] -> [3, 128]
head.classifier.bias:   [380]      -> [3]
```

This mismatch is expected and correct. The compatible backbone and embedding
parameters were loaded; the temporary three-class AGHRI classifier was
initialized normally. The released checkpoint was not overwritten.

## Training

The one-epoch real-data GPU smoke test completed.

Final logged training record:

```json
{
  "epoch": 1,
  "iter": 1820,
  "lr": 0.0001,
  "memory": 245,
  "triplet_loss": 0.0,
  "ce_loss": 0.00014,
  "top-1": 100.0,
  "loss": 0.00014
}
```

- CE loss finite: yes
- Triplet loss finite: yes
- Backward pass completed: yes
- Optimizer step completed: yes
- Frozen lower layers: configured through `frozen_stages=2`, `norm_eval=True`,
  and zero LR/decay multipliers for `conv1`, `bn1`, `layer1`, and `layer2`.

## Validation

Validation ran successfully on 305 validation crops.

Final validation metrics:

```json
{
  "mAP": 0.902,
  "R1": 0.984000027179718,
  "R5": 0.9869999885559082,
  "R10": 0.9929999709129333,
  "R20": 0.996999979019165
}
```

## Output

Saved checkpoint:

```text
aghri_reid_stage1/training/work_dirs/gpu_smoke_test/epoch_1.pth
```

Checkpoint exists and contains:

```text
meta
optimizer
state_dict
```

## Warnings

- The local folder is not a Git repository; this warning is harmless for the smoke test.
- OMP/MKL thread warnings were emitted by the training framework; they are normal.
- The 380-identity to 3-identity classifier mismatch is expected for Stage 1.

GPU REAL-DATA SMOKE TEST: PASS
