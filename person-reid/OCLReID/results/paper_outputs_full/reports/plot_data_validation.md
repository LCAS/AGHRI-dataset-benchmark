# Plot Data Validation

## Source files
- rpf-ReID: `/home/prabuddhi/Desktop/OCLReID/results/test_full/rpf_reid`
- Normal part-OCLReID: `/home/prabuddhi/Desktop/OCLReID/results/test_full/normal_part_oclreid`
- Gated part-OCLReID: `/home/prabuddhi/Desktop/OCLReID/results/test_full/gated_part_oclreid`

## Run counts
- Per-method completed runs: `{'rpf-ReID': 50, 'Normal part-OCLReID': 50, 'Gated part-OCLReID': 50}`
- Common completed runs: `50`
- Excluded completed runs: `{'rpf-ReID': [], 'Normal part-OCLReID': [], 'Gated part-OCLReID': []}`

## Anchor checks
- Gated part-OCLReID Success@0.5 ↑: actual `0.556535830032409`, expected `0.556535830032409`, pass `True`
- Gated part-OCLReID Mean IoU ↑: actual `0.49380522478549277`, expected `0.49380522478549255`, pass `True`
- Gated part-OCLReID Prediction availability ↑: actual `0.6477013563797863`, expected `0.6477013563797863`, pass `True`
- Gated part-OCLReID Absent-target FP rate: actual `0.2761586239847109`, expected `0.2761586239847109`, pass `True`
- Gated part-OCLReID Reacquisition rate: actual `0.42718446601941745`, expected `0.42718446601941745`, pass `True`
- Normal part-OCLReID Success@0.5 ↑: actual `0.48307526107310045`, expected `0.48307526107310045`, pass `True`
- Normal part-OCLReID Mean IoU ↑: actual `0.42822043097417317`, expected `0.4282204309741733`, pass `True`
- Normal part-OCLReID Prediction availability ↑: actual `0.545732805185452`, expected `0.545732805185452`, pass `True`
- Normal part-OCLReID Absent-target FP rate: actual `0.2264691829909221`, expected `0.2264691829909221`, pass `True`

## Figure data
### fig1

- Methods included: `Gated part-OCLReID, Normal part-OCLReID, rpf-ReID`
- Rows: `9`

```text
             method                  metric    value
           rpf-ReID             Success@0.5 0.360461
           rpf-ReID                Mean IoU 0.322440
           rpf-ReID Prediction availability 0.377146
Normal part-OCLReID             Success@0.5 0.483075
Normal part-OCLReID                Mean IoU 0.428220
Normal part-OCLReID Prediction availability 0.545733
 Gated part-OCLReID             Success@0.5 0.556536
 Gated part-OCLReID                Mean IoU 0.493805
 Gated part-OCLReID Prediction availability 0.647701
```

### fig2

- Methods included: `Gated part-OCLReID, Normal part-OCLReID, rpf-ReID`
- Rows: `9`

```text
             method   scenario  Completed runs  GT-visible frames  GT-absent frames  Success@0.5 ↑  Mean IoU ↑  Prediction availability ↑  Absent-target FP rate ↓  Reacquisition rate ↑  Wrong-person rate on visible frames ↓  Correctly localized frames  Prediction-present visible frames  Wrong-person frames  Absent-target FP frames  Reappearance events  Successful reacquisitions
           rpf-ReID   Footpath               2                819                 0       0.012210    0.011482                   0.012210                      NaN                   NaN                               0.000000                          10                                 10                    0                        0                    0                          0
           rpf-ReID Polytunnel               5               1542                 0       0.327497    0.293351                   0.327497                      NaN                   NaN                               0.000000                         505                                505                    0                        0                    0                          0
           rpf-ReID   Vineyard              14               5805              1956       0.393109    0.348949                   0.394143                 0.000000              0.285714                               0.000172                        2282                               2288                    1                        0                   42                         12
Normal part-OCLReID   Footpath               2                819                 0       0.991453    0.832051                   0.993895                      NaN                   NaN                               0.000000                         812                                814                    0                        0                    0                          0
Normal part-OCLReID Polytunnel               5               1542                 0       0.791829    0.707488                   0.791829                      NaN                   NaN                               0.000000                        1221                               1221                    0                        0                    0                          0
Normal part-OCLReID   Vineyard              14               5805              1956       0.410336    0.363484                   0.414815                 0.361963              0.238095                               0.002756                        2382                               2408                   16                      708                   42                         10
 Gated part-OCLReID   Footpath               2                819                 0       0.991453    0.832051                   0.993895                      NaN                   NaN                               0.000000                         812                                814                    0                        0                    0                          0
 Gated part-OCLReID Polytunnel               5               1542                 0       0.837873    0.744461                   0.843709                      NaN                   NaN                               0.005837                        1292                               1301                    9                        0                    0                          0
 Gated part-OCLReID   Vineyard              14               5805              1956       0.515073    0.456222                   0.536434                 0.418200              0.476190                               0.016365                        2990                               3114                   95                      818                   42                         20
```

### fig3

- Methods included: `Gated part-OCLReID, Normal part-OCLReID, rpf-ReID`
- Rows: `12`

```text
             method  camera_label  Completed runs  GT-visible frames  GT-absent frames  Success@0.5 ↑  Mean IoU ↑  Prediction availability ↑  Absent-target FP rate ↓  Reacquisition rate ↑  Wrong-person rate on visible frames ↓  Correctly localized frames  Prediction-present visible frames  Wrong-person frames  Absent-target FP frames  Reappearance events  Successful reacquisitions
           rpf-ReID Front fisheye              15               7996               631       0.429090    0.381045                   0.431841                 0.000000              0.432432                               0.000000                        3431                               3453                    0                        0                   37                         16
           rpf-ReID  Left fisheye              10               2235              2089       0.163311    0.146342                   0.163311                 0.000000              0.100000                               0.000000                         365                                365                    0                        0                   10                          1
           rpf-ReID Right fisheye              11               3350              2808       0.431642    0.386571                   0.507761                 0.005698              0.285714                               0.074328                        1446                               1701                  249                       16                   21                          6
           rpf-ReID ZED front RGB              14               3081               751       0.247971    0.228362                   0.248296                 0.000000              0.285714                               0.000325                         764                                765                    1                        0                   35                         10
Normal part-OCLReID Front fisheye              15               7996               631       0.412456    0.359150                   0.510005                 0.275753              0.189189                               0.093172                        3298                               4078                  745                      174                   37                          7
Normal part-OCLReID  Left fisheye              10               2235              2089       0.655928    0.595563                   0.686353                 0.388224              0.300000                               0.029083                        1466                               1534                   65                      811                   10                          3
Normal part-OCLReID Right fisheye              11               3350              2808       0.599701    0.533192                   0.605373                 0.116097              0.380952                               0.004179                        2009                               2028                   14                      326                   21                          8
Normal part-OCLReID ZED front RGB              14               3081               751       0.414151    0.371947                   0.471600                 0.147803              0.171429                               0.055501                        1276                               1453                  171                      111                   35                          6
 Gated part-OCLReID Front fisheye              15               7996               631       0.478864    0.417218                   0.614182                 0.340729              0.432432                               0.127189                        3829                               4911                 1017                      215                   37                         16
 Gated part-OCLReID  Left fisheye              10               2235              2089       0.767338    0.693873                   0.819239                 0.443274              0.600000                               0.049217                        1715                               1831                  110                      926                   10                          6
 Gated part-OCLReID Right fisheye              11               3350              2808       0.647761    0.577634                   0.666866                 0.149929              0.380952                               0.015821                        2170                               2234                   53                      421                   21                          8
 Gated part-OCLReID ZED front RGB              14               3081               751       0.506005    0.456288                   0.589419                 0.229028              0.400000                               0.081467                        1559                               1816                  251                      172                   35                         14
```

### fig6

- Methods included: `Gated part-OCLReID, Normal part-OCLReID, rpf-ReID`
- Rows: `9`

```text
             method human_count_label  Completed runs  GT-visible frames  GT-absent frames  Success@0.5 ↑  Mean IoU ↑  Prediction availability ↑  Absent-target FP rate ↓  Reacquisition rate ↑  Wrong-person rate on visible frames ↓  Correctly localized frames  Prediction-present visible frames  Wrong-person frames  Absent-target FP frames  Reappearance events  Successful reacquisitions
           rpf-ReID           1 human               6               1633               459       0.018371    0.016882                   0.018371                 0.000000              0.000000                               0.000000                          30                                 30                    0                        0                    3                          0
           rpf-ReID          3 humans              17               6310              3031       0.351506    0.320357                   0.354992                 0.000000              0.354167                               0.000000                        2218                               2240                    0                        0                   48                         17
           rpf-ReID          4 humans              27               8719              2789       0.431013    0.381177                   0.460374                 0.005737              0.307692                               0.028673                        3758                               4014                  250                       16                   52                         16
Normal part-OCLReID           1 human               6               1633               459       0.971219    0.837861                   0.972443                 0.000000              1.000000                               0.000000                        1586                               1588                    0                        0                    3                          3
Normal part-OCLReID          3 humans              17               6310              3031       0.354834    0.323075                   0.472108                 0.138238              0.166667                               0.112203                        2239                               2979                  708                      419                   48                          8
Normal part-OCLReID          4 humans              27               8719              2789       0.484459    0.427592                   0.519096                 0.359627              0.250000                               0.032917                        4224                               4526                  287                     1003                   52                         13
 Gated part-OCLReID           1 human               6               1633               459       0.983466    0.849398                   0.985303                 0.000000              1.000000                               0.000000                        1606                               1609                    0                        0                    3                          3
 Gated part-OCLReID          3 humans              17               6310              3031       0.431537    0.392530                   0.602694                 0.200264              0.354167                               0.163550                        2723                               3803                 1032                      607                   48                         17
 Gated part-OCLReID          4 humans              27               8719              2789       0.567038    0.500499                   0.617043                 0.404087              0.461538                               0.045762                        4944                               5380                  399                     1127                   52                         24
```

### fig7

- Methods included: `Gated part-OCLReID, Normal part-OCLReID, rpf-ReID`
- Rows: `6`

```text
             method robot_motion  Completed runs  GT-visible frames  GT-absent frames  Success@0.5 ↑  Mean IoU ↑  Prediction availability ↑  Absent-target FP rate ↓  Reacquisition rate ↑  Wrong-person rate on visible frames ↓  Correctly localized frames  Prediction-present visible frames  Wrong-person frames  Absent-target FP frames  Reappearance events  Successful reacquisitions
           rpf-ReID   Stationary              46              15848              5820       0.377713    0.337856                   0.395255                 0.002749                  0.33                               0.015775                        5986                               6264                  250                       16                  100                         33
           rpf-ReID       Moving               4                814               459       0.024570    0.022314                   0.024570                 0.000000                  0.00                               0.000000                          20                                 20                    0                        0                    3                          0
Normal part-OCLReID   Stationary              46              15848              5820       0.459048    0.406880                   0.524924                 0.244330                  0.21                               0.062784                        7275                               8319                  995                     1422                  100                         21
Normal part-OCLReID       Moving               4                814               459       0.950860    0.843707                   0.950860                 0.000000                  1.00                               0.000000                         774                                774                    0                        0                    3                          3
 Gated part-OCLReID   Stationary              46              15848              5820       0.535020    0.474644                   0.630805                 0.297938                  0.41                               0.090295                        8479                               9997                 1431                     1734                  100                         41
 Gated part-OCLReID       Moving               4                814               459       0.975430    0.866852                   0.976658                 0.000000                  1.00                               0.000000                         794                                795                    0                        0                    3                          3
```

### fig4

- Methods included: `Gated part-OCLReID, Normal part-OCLReID`
- Rows: `6`

```text
             method                  metric  improvement_over_rpf
Normal part-OCLReID             Success@0.5              0.122614
Normal part-OCLReID                Mean IoU              0.105780
Normal part-OCLReID Prediction availability              0.168587
 Gated part-OCLReID             Success@0.5              0.196075
 Gated part-OCLReID                Mean IoU              0.171365
 Gated part-OCLReID Prediction availability              0.270556
```

## Trade-off table

```text
             Method  Success@IoU 0.5  Mean IoU  Prediction availability  Absent-target false-positive rate  Reacquisition rate  Wrong-person rate on visible frames
           rpf-ReID         0.360461  0.322440                 0.377146                           0.002548            0.320388                             0.015004
Normal part-OCLReID         0.483075  0.428220                 0.545733                           0.226469            0.233010                             0.059717
 Gated part-OCLReID         0.556536  0.493805                 0.647701                           0.276159            0.427184                             0.085884
```

## Checkpoint ablation table

```text
                         Configuration  Success@IoU 0.5  Mean IoU  Prediction availability  Absent-target false-positive rate  Reacquisition rate
          Released checkpoint, no gate         0.483075  0.428220                 0.545733                           0.226469            0.233010
AGHRI experimental checkpoint, no gate         0.483015  0.428818                 0.514524                           0.208313            0.262136
             Released checkpoint, gate         0.556536  0.493805                 0.647701                           0.276159            0.427184
   AGHRI experimental checkpoint, gate         0.550654  0.489106                 0.635518                           0.258958            0.466019
```

## Figure 9 metadata summary
- Scene: `out_vine_4swap+walk_st_ly_11_06_2024_2_label`
- Camera: `cam_fish_front`
- Target: `01`
- Frames: `[516, 517, 519, 521]`
- Panels extracted directly from saved runtime MP4: `True`
- Normal MP4: `/home/prabuddhi/Desktop/OCLReID/results/test_full/normal_part_oclreid/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_fish_front/class_01/inference_visualization.mp4`
- Gated MP4: `/home/prabuddhi/Desktop/OCLReID/results/test_full/gated_part_oclreid/dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label/cam_fish_front/class_01/inference_visualization.mp4`
- Alignment CSV: `/home/prabuddhi/Desktop/OCLReID/results/paper_outputs_full/qualitative/fig9_mp4_alignment_validation.csv`
- Alignment statuses: `['verified', 'verified', 'verified', 'verified']`

Validation result: `PASS`
