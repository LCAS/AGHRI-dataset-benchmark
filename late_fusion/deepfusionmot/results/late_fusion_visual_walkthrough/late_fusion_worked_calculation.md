# Late-Fusion Worked Calculation

This calculation uses one real AGHRI held-out test frame and frozen detector outputs.

Recording: `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label`  
Sample ID: `footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label/1731407186_709929023`

## 1. YOLO 2D Box

`B_C = [277.7835693359375, 106.77847290039062, 346.9267883300781, 252.9573516845703]`

Confidence: `0.916218101978302`.

## 2. PointPillars 3D Box

`B_L = [x, y, z, length, width, height, yaw] = [5.315617561340332, 0.20094463229179382, 0.2806103229522705, 0.4411206543445587, 0.619935154914856, 1.5118756294250488, -0.0062798261642456055]`

Score: `0.7171115279197693`.

Cache metadata box origin: `bottom_center`.  
Geometry origin used in this walkthrough: `center`.

## 3. Box-Origin Convention

For `bottom_center`, `z` is the lower face and the upper face is `z + height`. For `center`, `z` is the geometric centre and the lower/upper faces are `z - height/2` and `z + height/2`.

The projected cuboid was vertically high when the cached PointPillars z value was treated as bottom-centre. Interpreting the same real detector box as centre-origin aligns the projected cuboid with the visible person. This is a box-origin convention correction, not a calibration change.

## 4. Half Dimensions

`[length/2, width/2, height/2] = [0.22056032717227936, 0.309967577457428, 0.7559378147125244]`

## 5. Yaw Rotation Matrix

```text
[[ 0.99998028  0.00627978  0.        ]
 [-0.00627978  0.99998028  0.        ]
 [ 0.          0.          1.        ]]
```

## 6. Corners, Transform and Projection

| corner | local LiDAR box | LiDAR frame | camera frame | pixel u,v | status |
| --- | --- | --- | --- | --- | --- |
| 1 | [-0.220560, -0.309968, -0.755938] | [5.093115, -0.107632, -0.475327] | [-0.112368, 0.770327, 5.168115] | [329.369757, 263.116472] | valid |
| 2 | [0.220560, -0.309968, -0.755938] | [5.534227, -0.110402, -0.475327] | [-0.109598, 0.770327, 5.609227] | [330.421583, 257.519439] | valid |
| 3 | [0.220560, 0.309968, -0.755938] | [5.538120, 0.509521, -0.475327] | [-0.729521, 0.770327, 5.613120] | [277.714195, 257.473958] | valid |
| 4 | [-0.220560, 0.309968, -0.755938] | [5.097008, 0.512291, -0.475327] | [-0.732291, 0.770327, 5.172008] | [272.167834, 263.062899] | valid |
| 5 | [-0.220560, -0.309968, 0.755938] | [5.093115, -0.107632, 1.036548] | [-0.112368, -0.741548, 5.168115] | [329.369757, 123.430525] | valid |
| 6 | [0.220560, -0.309968, 0.755938] | [5.534227, -0.110402, 1.036548] | [-0.109598, -0.741548, 5.609227] | [330.421583, 128.818453] | valid |
| 7 | [0.220560, 0.309968, 0.755938] | [5.538120, 0.509521, 1.036548] | [-0.729521, -0.741548, 5.613120] | [277.714195, 128.862235] | valid |
| 8 | [-0.220560, 0.309968, 0.755938] | [5.097008, 0.512291, 1.036548] | [-0.732291, -0.741548, 5.172008] | [272.167834, 123.482096] | valid |

## 7. Camera-From-LiDAR Matrix

```text
[[ 2.22044605e-16 -1.00000000e+00  0.00000000e+00 -2.20000000e-01]
 [ 2.22044605e-16  0.00000000e+00 -1.00000000e+00  2.95000000e-01]
 [ 1.00000000e+00  2.22044605e-16  2.22044605e-16  7.50000000e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

## 8. Projection Matrix

```text
[[477.29998779   0.         339.74749756   0.        ]
 [  0.         477.49499512 191.94400024   0.        ]
 [  0.           0.           1.           0.        ]]
```

## 9. Projected Envelope and Clipping

Unclipped envelope:

```text
[272.1678344001285, 123.43052469077153, 330.4215834005085, 263.1164718533429]
```

Clipped image envelope:

```text
[272.1678344001285, 123.43052469077153, 330.4215834005085, 263.1164718533429]
```

## 10. Intersection and IoU

Intersection rectangle:

```text
[277.7835693359375, 123.43052469077153, 330.4215834005085, 252.9573516845703]
```

YOLO area: `10107.278228092473` pixels²  
Projected PointPillars area: `8137.23010488877` pixels²  
Intersection area: `6818.034941038834` pixels²  
Union area: `11426.47339194241` pixels²  
Frozen cached association IoU: `0.5966875961787724`  
Note: Cached association IoU already matches the explicit origin-corrected geometry.

```text
IoU = 6818.034941038834 / 11426.47339194241 = 0.5966875961787735
```

Threshold comparison:

```text
0.5966875961787735 >= 0.1 -> MATCHED
```

## 11. Final Output Membership

The YOLO detection is matched. The PointPillars box is matched. The matched 3D output and ALL3D/canonical output contain this PointPillars box. The unmatched camera and unmatched LiDAR outputs do not contain this selected detection.
