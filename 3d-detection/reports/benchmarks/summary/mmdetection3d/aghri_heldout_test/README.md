# AGHRI held-out test results

These four results were computed on the six sequences in the released `test.txt`
list (1,308 LiDAR annotation frames, 3,227 person boxes). No model was trained
or selected on this test export. PointPillars and SECOND were trained for 160
and 80 epochs, respectively, with seed 42 and batch size 4. The saved
checkpoints were selected on the seven-sequence validation split by mean BEV AP
across IoU 0.25, 0.50, and 0.75. The selected epochs were 108 and 36 for
scratch PointPillars and SECOND, and 126 and 2 for their KITTI-pretrained,
AGHRI-fine-tuned counterparts.

| Model | BEV AP@0.25 | 3D AP@0.25 | BEV AP@0.50 | 3D AP@0.50 | Fixed-input latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PointPillars scratch | 0.2813 | 0.2778 | 0.2263 | 0.1581 | 11.274 |
| SECOND scratch | 0.3228 | 0.3189 | 0.2092 | 0.1335 | 15.419 |
| PointPillars KITTI-pretrained, AGHRI-fine-tuned | 0.2935 | 0.2915 | 0.2224 | 0.1525 | 11.790 |
| SECOND KITTI-pretrained, AGHRI-fine-tuned | 0.2981 | 0.2837 | 0.1319 | 0.0874 | 13.824 |

Each CSV/JSON pair preserves the complete metrics, exact checkpoint path,
`eval_split: test`, and `train_time_sec: null`. The corresponding checkpoint
manifest records its SHA256 and six test sequences. The test export manifest
records the split-list SHA256 and frame count. Latency is from 100 repeated
inferences on a fixed test point cloud after 20 warm-ups, using an RTX A6000;
it is not a mean across all test point clouds.

The older `summary_mmdetection3d_aghri_*.csv/json` files in the parent directory
report historical validation evaluations (`eval_split: val`) and are retained
for provenance. They are not the final held-out test results.
