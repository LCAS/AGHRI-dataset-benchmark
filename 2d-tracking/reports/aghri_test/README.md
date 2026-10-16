# AGHRI 2D tracking test reports

This directory contains saved evaluations for the six held-out AGHRI test
sequences. Each sequence has a ZED RGB and a front-fisheye tracking stream.
The detector sources are `zed_yolov8s_pretrained`, `fisheye_yolov8s_pretrained`,
`zed_yolo11s_pretrained`, and `fisheye_yolo11s_pretrained`; `gt` is the
identity-free ground-truth-box input control. Each source/tracker pair covers
six camera-specific sequences for a detector source, or 12 for `gt`.
The manuscript uses the validation-selected YOLOv8s sources and `gt` only.

## Where to find TrackEval metrics

- [`trackeval_detailed/{source}/{tracker}.csv`](trackeval_detailed/) contains one row per camera-specific sequence and a `COMBINED` row. `HOTA___AUC` is the HOTA score; `DetA___AUC` and `AssA___AUC` give detection and association accuracy. The same files provide CLEAR measures (`MOTA`, `MOTP`, `IDSW`, `CLR_Pr`, `CLR_Re`) and Identity measures (`IDF1`, `IDP`, `IDR`). Rate fields in the detailed file use a 0–1 scale; count and sum fields retain their native units.
- [`trackeval_per_sequence.csv`](trackeval_per_sequence.csv) is a compact sequence-level extract. Its `hota_percent`, `mota_percent`, `motp_iou_percent`, and `idf1_percent` columns use a 0–100 scale; `identity_switches` is a count.
- [`trackeval_summary_by_source.csv`](trackeval_summary_by_source.csv) has one row per source/tracker pair. Its `trackeval_hota`, `trackeval_deta`, `trackeval_assa`, `trackeval_mota`, `trackeval_motp`, and `trackeval_idf1` columns come from TrackEval's `COMBINED` result for that source. These rate columns use a 0–100 scale; `trackeval_idsw` is a count. TrackEval's combined HOTA is calculated from the underlying HOTA counts, not by taking an arithmetic mean of sequence HOTA scores.

TrackEval MOTP is the mean IoU of matched boxes, so **higher is better**. The
`motmetrics_*_motp` columns in the source summary are IoU **distances** under
the motmetrics convention, so **lower is better**. Do not compare those MOTP
columns directly without converting their definitions.

## Equal-sequence means and pooled results

[`trackeval_summary_by_source.csv`](trackeval_summary_by_source.csv) also
retains separate `motmetrics_macro_*` and `motmetrics_global_*` columns:

- `motmetrics_macro_*` rate values give each camera-specific sequence equal weight within one source/tracker pair. The motmetrics rate columns use a 0–1 scale. The `motmetrics_macro_num_switches` field is a **sum**, not a mean.
- `motmetrics_global_*` values are recomputed from the combined sequence events for that source/tracker pair. They are pooled/global results, so longer or more densely annotated sequences can have more influence. Global IDF1 is recomputed from identity counts and is not a simple average of the sequence IDF1 values.
- The `trackeval_*` source-summary columns are TrackEval's own `COMBINED` results. They should not be labelled as equal-sequence means. TrackEval and motmetrics columns also use different scales and, for MOTP, different directions.

The [paper-results notebook](../../notebooks/2d_tracking_paper_results.ipynb)
uses the **per-sequence TrackEval rows** in `trackeval_detailed/`. For each
tracker it averages the 12 ZED RGB and front-fisheye camera-specific sequence
rate values equally and sums IDSW. The detector-based versus ground-truth-based
table averages rate values over eight trackers × 12 camera-specific sequences
(96 results) per setting and sums IDSW. These manuscript values are **equal-run
means**, not the pooled or `COMBINED` results in the source summary.

## Tracker-processing throughput

[`tracker_throughput_per_run.csv`](tracker_throughput_per_run.csv) contains each
run's processed frames, tracking time, and frames/s. In
[`tracker_throughput_by_source.csv`](tracker_throughput_by_source.csv),
`macro_fps` is the arithmetic mean of per-run FPS, whereas `pooled_fps` is
total processed frames divided by total tracking time. The manuscript's
per-tracker throughput is the equal mean of the 12 YOLOv8s detector-based
per-run FPS values, six from each camera. Timing covers per-frame tracking
computation, including input adaptation and appearance extraction where used;
it excludes detector inference, frame disk I/O, MOT writing, rendering, and
evaluation.
