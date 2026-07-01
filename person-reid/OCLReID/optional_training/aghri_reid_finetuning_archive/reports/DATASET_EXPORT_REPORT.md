# DATASET EXPORT REPORT

## Summary

- Train identities: 3 (person_07, person_08, person_10)
- Validation identities: 2 (person_03, person_04)
- Training crops: 1823
- Validation crops: 305
- Total exported crops: 2128

## Crops Per Identity

{'person_03': 154, 'person_04': 151, 'person_07': 477, 'person_08': 570, 'person_10': 776}

## Crops Per Camera

{'cam_fish_front': 983, 'cam_fish_left': 410, 'cam_fish_right': 343, 'cam_zed_rgb': 392}

## Crops Per Scene

{'footpath1_3walk_mv_11_14_2024_2_label': 305, 'in_straw_2pick_diff_mv_10_10_2024_2_a_label': 477, 'in_straw_3push_diff_mv_11_07_2024_1_label': 342, 'in_vine_1push_3pick_1carry_diff_st_ly_11_06_2024_3_label': 263, 'out_straw_3push_mv_11_07_2024_1_label': 228, 'out_vine_1push_3carry_st_ly_11_06_2024_2_label': 216, 'out_vine_5walk+talk+push_st_ly_11_06_2024_2_label': 297}

## Box/Crop Size

- min area: 144
- median area: 2262
- max area: 119680
- min width: 6
- min height: 16

## Leakage Checks

- Reserved identities exported: 0
- Original final-test scene crops exported: 0
- Leakage rows: 0

## Notes

Crops are annotation-box crops saved at natural cropped resolution. Resizing and
normalization are left to the training pipeline.
