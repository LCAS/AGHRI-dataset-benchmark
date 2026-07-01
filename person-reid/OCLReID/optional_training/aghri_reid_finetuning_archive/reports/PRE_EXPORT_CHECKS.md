# STAGE 1 PRE-EXPORT CHECKS

Frame stride: 5

## Scene/Class Rows By ReID Split

These rows come from `aghri_identity_map_confirmed.csv`.

| ReID split | rows |
|---|---:|
| train | 6 |
| val | 2 |
| reserved_test | 130 |

## Annotated Boxes By Identity

{'person_01': 38308, 'person_02': 48091, 'person_03': 768, 'person_04': 746, 'person_05': 37000, 'person_06': 4865, 'person_07': 2386, 'person_08': 2837, 'person_09': 15143, 'person_10': 3860}

## Valid Boxes Before Temporal Sampling

{'train': 9073, 'val': 1514}

## Crops After Temporal Sampling

{'train': 1823, 'val': 305}

Per identity after temporal sampling:

{'person_03': 154, 'person_04': 151, 'person_07': 477, 'person_08': 570, 'person_10': 776}

Per identity/camera after temporal sampling:

{'person_03:cam_fish_front': 72, 'person_03:cam_fish_right': 51, 'person_03:cam_zed_rgb': 31, 'person_04:cam_fish_front': 39, 'person_04:cam_fish_left': 26, 'person_04:cam_fish_right': 78, 'person_04:cam_zed_rgb': 8, 'person_07:cam_fish_front': 201, 'person_07:cam_fish_left': 202, 'person_07:cam_zed_rgb': 74, 'person_08:cam_fish_front': 255, 'person_08:cam_fish_left': 170, 'person_08:cam_fish_right': 48, 'person_08:cam_zed_rgb': 97, 'person_10:cam_fish_front': 416, 'person_10:cam_fish_left': 12, 'person_10:cam_fish_right': 166, 'person_10:cam_zed_rgb': 182}

## Box Area

- min: 102.0
- median: 2268.0
- max: 131049.5

## Manifest Sources

{'sensor_image_order': 208, 'video_frame_manifest': 52}

Training PK sampling status: PASS.

Validation retrieval image status: PASS.

Identities below 8 sampled crops:

{}

## Leakage Assertions

- No original final-test scene rows in train/val manifests: PASS
- No reserved-test identity in train/val manifests: PASS
- No unapproved identity in train/val manifests: PASS
- No missing image files in train/val manifests: PASS
- No invalid bounding boxes in train/val manifests: PASS
- No final-test scene images accessed for crop export: PASS by construction; original `test.txt` scenes are excluded before image-size reads/crop export.

## Decision

Phase B manifest generation passed leakage assertions. Crop export may proceed
for `train_manifest.csv` and `val_manifest.csv`.
