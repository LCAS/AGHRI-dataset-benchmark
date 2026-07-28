# AGHRI DeepFusionMOT/TrackEval Metric Summary

- Batch root: `results/aghri_deepfusionmot_tracking/eight_combo_test_manifest_best_val_policy`
- Dataset root: `/media/prabuddhi/Backup2/Updated Dataset_PW`
- CLEAR/Identity IoU threshold: 0.5
- Metrics are computed with the bundled TrackEval implementation used by DeepFusionMOT.
- Evaluation uses ZED RGB 2D annotation identities and projected tracked 3D boxes.
- Ego motion mode: `aghri_odom`; AGHRI odom compensation was applied on 10048/10048 possible frame transitions.

| ID | Combination | Frames | GT IDs | Trk IDs | HOTA | DetA | AssA | IDSW | MOTP | MOTA | IDF1 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | G YOLO + G SECOND | 1262 | 15 | 47 | 0.1524 | 0.2129 | 0.1107 | 40 | 0.6291 | 0.0065 | 0.1664 | 2007.43 |
| S2 | FT YOLO + G SECOND | 1262 | 15 | 49 | 0.1576 | 0.2244 | 0.1127 | 43 | 0.6278 | -0.0306 | 0.1706 | 2078.88 |
| S3 | G YOLO + FT SECOND | 1262 | 15 | 71 | 0.2317 | 0.3335 | 0.1631 | 53 | 0.6559 | 0.1473 | 0.2581 | 2317.17 |
| S4 | FT YOLO + FT SECOND | 1262 | 15 | 73 | 0.2425 | 0.3511 | 0.1697 | 59 | 0.6536 | 0.1571 | 0.2727 | 2515.66 |
| P1 | G YOLO + G PointPillars | 1262 | 15 | 50 | 0.1098 | 0.1515 | 0.0824 | 37 | 0.6207 | -0.1061 | 0.1271 | 1701.52 |
| P2 | FT YOLO + G PointPillars | 1262 | 15 | 48 | 0.1161 | 0.1533 | 0.0903 | 36 | 0.6195 | -0.1053 | 0.1285 | 1699.82 |
| P3 | G YOLO + FT PointPillars | 1262 | 15 | 40 | 0.3539 | 0.4102 | 0.3089 | 29 | 0.6527 | -0.0020 | 0.3538 | 2795.15 |
| P4 | FT YOLO + FT PointPillars | 1262 | 15 | 40 | 0.3553 | 0.4116 | 0.3102 | 29 | 0.6526 | -0.0053 | 0.3533 | 2800.10 |

## Notes

- These are MOT identity metrics, unlike the previous track-count diagnostics.
- They are not KITTI-camera metrics; they are AGHRI ZED RGB identity metrics using the same TrackEval metric families.
- `Class` values in AGHRI annotation JSON files are treated as ground-truth person IDs.
