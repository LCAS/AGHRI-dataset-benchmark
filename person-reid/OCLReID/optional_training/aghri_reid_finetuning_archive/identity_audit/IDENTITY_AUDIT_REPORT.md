# IDENTITY AUDIT REPORT

Audit date: 2026-06-22

## Gate Decision

Identity mapping is **not reliable enough to continue automatically**.

The required confirmed identity file does not exist:

```text
aghri_reid_stage1/identity_audit/aghri_identity_map_confirmed.csv
```

Therefore Stage 1 has stopped after Phase A. No crops were exported, no model
was trained, no runtime checkpoint was created, and no validation/final-test
inference was run.

## Split Files Found

```text
/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists/train.txt
/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists/val.txt
/media/prabuddhi/Backup2/Updated Dataset_PW/split_lists/test.txt
```

Split summary:

| Split | Scenes | Scene/local-class rows |
|---|---:|---:|
| train | 52 | 109 |
| val | 7 | 13 |
| test | 6 | 16 |

Missing split scenes in dataset index: 0

## Findings

1. Globally reliable person identities do **not** already exist in the files
   inspected. Annotation labels are numeric local classes such as `01`, `02`,
   `06`, and `09`.
2. The same numeric local classes occur in multiple scenes and across splits.
   This strongly indicates that `Class=01` in one scene cannot be treated as
   the same real person as `Class=01` in another scene.
3. Some scene names contain candidate human initials/tokens such as `sc`, `mk`,
   `gl`, `oj`, `nj`, `yj`, and `ht`, but the scene filename does not verify the
   mapping from a local annotation class to a real person.
4. Candidate person tokens appear across multiple splits:

```json
{
  "gl": [
    "test",
    "train",
    "val"
  ],
  "mk": [
    "test",
    "train",
    "val"
  ],
  "nj": [
    "test",
    "train",
    "val"
  ],
  "oj": [
    "test",
    "train",
    "val"
  ],
  "sc": [
    "train",
    "val"
  ]
}
```

5. Local annotation classes also appear across multiple splits:

```json
{
  "01": [
    "test",
    "train",
    "val"
  ],
  "02": [
    "test",
    "train",
    "val"
  ],
  "05": [
    "test",
    "train",
    "val"
  ],
  "06": [
    "test",
    "train"
  ],
  "09": [
    "test",
    "train",
    "val"
  ],
  "10": [
    "train",
    "val"
  ]
}
```

## Required Manual Confirmation

Please fill:

```text
aghri_reid_stage1/identity_audit/aghri_identity_map_to_confirm.csv
```

For every train/validation row that should be used:

1. set `approved` to `1`;
2. set `manual_name_or_id` to a stable real-person ID;
3. ensure no final-test person is approved for training;
4. ensure validation identities are either intentionally disjoint from training
   identities or explicitly documented if the experiment is not identity-disjoint.

The helper file:

```text
aghri_reid_stage1/identity_audit/sample_image_paths_for_identity_confirmation.csv
```

contains representative image paths for each scene/local-class row.

## Can Identity-Disjoint Splits Be Established Automatically?

No. Identity-disjoint training, validation, and final-test splits may be
possible, but this cannot be proven from the available annotation labels and
scene filenames alone. Manual confirmation is required before crop export or
training.

## What Was Not Done

- No final-test images were used for training or validation.
- No final-test predictions or metrics were accessed for training.
- No crops were exported.
- No ReID training was run.
- No checkpoint was written or copied into `checkpoints/reid`.
- `run_video.py`, `automate_oclreid.py`, evaluation code, and configs were not
  modified.
