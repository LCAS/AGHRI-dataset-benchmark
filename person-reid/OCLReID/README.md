# Person Re-Identification Repository

OCLReID refers to Online Continual Learning based Re-Identification for robot
person following. It selects a target person in a video, follows that person
across frames, and uses ReID features to support identity recovery when the
target becomes difficult to track. This project builds on the original
[MedlarTea/OCLReID](https://github.com/MedlarTea/OCLReID) repository released
for "Person Re-Identification for Robot Person Following with Online Continual
Learning".

This repository contains the OCLReID runtime code, AGHRI video preparation
tools, automated AGHRI evaluation scripts, saved final-test outputs,
and figure generation scripts.

## 1. Repository Layout

Important active folders:

```text
OCLReID/
|-- aghri_video_preparation/       # Build AGHRI MP4s and frame manifests
|-- checkpoints/reid/              # ReID checkpoint files
|-- demo/                          # Demo video and simple demo wrappers
|-- environment/                   # Conda environment and environment checks
|-- methods/
|   |-- rpf_reid/                  # Final-test wrapper for rpf-ReID
|   |-- normal_part_oclreid/       # Final-test wrapper for normal part-OCLReID
|   `-- gated_part_oclreid/        # Final-test wrapper for gated part-OCLReID
|-- notebooks/                     # Jupyter results summary
|-- results/
|   |-- final_comparison/          # Compact summaries and figures
|   |-- paper_outputs_full/        # Full figure/table/report bundle
|   `-- test_full/                 # Full saved final-test predictions and MP4s
`-- scripts/                       # Main runtime, automation, evaluation, figures
```

Set the AGHRI dataset and generated-video locations before running AGHRI
commands:

```bash
export AGHRI_DATASET_ROOT="/path/to/AGHRI_dataset"
export AGHRI_VIDEO_ROOT="/path/to/AGHRI_OCLReID_videos"
```

`AGHRI_DATASET_ROOT` should point to the raw AGHRI dataset. `AGHRI_VIDEO_ROOT`
should point to the folder where the prepared AGHRI MP4 files and frame
manifests are stored or will be generated.

## 2. Environment Setup

The intended environment name is `oclreid`.

```bash
cd /path/to/OCLReID
./environment/create_environment.sh
conda activate oclreid
python environment/verify_environment.py
```

Important: `./environment/create_environment.sh` creates or reuses the
`oclreid` environment and installs the required packages inside it, but it does
not switch your current terminal prompt from `(base)` to `(oclreid)`. Activate
the environment manually with `conda activate oclreid` before running commands
directly.

## 3. Checkpoints

The runtime expects the released ResNet18 ReID checkpoint here:

```text
checkpoints/reid/resnet18.pth
```

This is the default checkpoint used by the demo and AGHRI benchmark wrappers.

## 4. Demo and Single-Video Inference

Run the included demo video:

```bash
cd /path/to/OCLReID
./demo/run_demo.sh --method part-OCLReID
```

The `--method` argument chooses the runtime ReID method. The supported
single-video method names are:

```text
rpf-ReID
part-OCLReID
```

The AGHRI benchmark methods, including the gated part-OCLReID variant, are
described later in Section 6, "The 3 AGHRI Methods".

Live display and saved inference videos are separate outputs. Use `--show-live`
only when you want an OpenCV window during inference:

```bash
./demo/run_demo.sh --method part-OCLReID --show-live
```

For the demo and other single-video runs without `--bbox-file`, the first frame
opens an OpenCV selection window. Draw a bounding box around the target person
and press Enter to initialise tracking.

OpenCV window controls:

- Space: pause or resume display.
- `q` or Esc: close the live window while inference continues headlessly.

By default, the demo wrapper writes predictions and a saved visualization MP4
under a timestamped folder:

```text
results/reproduced_runs/demo/<timestamp>/
```

For clearer file names, use the custom-video wrapper and provide explicit output
paths. This example runs the included demo video but writes named outputs:

```bash
./demo/run_custom_video.sh \
  --video demo/demo_video.mp4 \
  --method part-OCLReID \
  --output results/reproduced_runs/demo_named/part_oclreid_inference.mp4 \
  --output-json results/reproduced_runs/demo_named/part_oclreid_predictions.json
```

The output files are:

- `--output`: annotated inference MP4 with boxes, IDs, and ReID confidence.
- `--output-json`: frame-by-frame prediction JSON.

Optional arguments:

- `--bbox-file /path/to/bbox.json`: provide the initial target box.
- `--start-frame N`: start inference from a later video frame.
- `--show-live`: open the live OpenCV window while still saving the MP4/JSON
  outputs.

## 5. Generate AGHRI Videos and Frame Manifests

The raw AGHRI dataset is image-sequence based. OCLReID inference runs on MP4s, so
the image sequences are converted to videos and paired with frame manifests. The
frame manifests are critical: they map every source image filename to the exact
video frame index used by prediction and evaluation.

This workflow prepares AGHRI videos for the validation and test splits with
camera-specific frame rates: 30 FPS for fisheye cameras and 15 FPS for the ZED
RGB camera.

Generate only manifests when videos already exist:

```bash
./aghri_video_preparation/prepare_aghri_videos.sh \
  --splits test \
  --cameras cam_fish_front cam_fish_left cam_fish_right cam_zed_rgb \
  --manifest-only
```

Generate videos and manifests:

```bash
./aghri_video_preparation/prepare_aghri_videos.sh \
  --splits test \
  --cameras cam_fish_front cam_fish_left cam_fish_right cam_zed_rgb
```

Outputs are written below `AGHRI_VIDEO_ROOT`, for example:

```text
/path/to/AGHRI_OCLReID_videos
```

Each camera stream gets:

```text
<camera>.mp4
<camera>_frame_manifest.json
```

The environment uses FFmpeg 4.2.2. The video generator therefore uses the
FFmpeg 4.2-compatible `-vsync 0` pass-through option, which keeps one output
video frame for each source image.

## 6. The 3 AGHRI Methods

The final comparison has three methods.

| Folder | Method name in code | Explanation |
|---|---|---|
| `methods/rpf_reid/` | `rpf-ReID` | RPF/ReID baseline. |
| `methods/normal_part_oclreid/` | `part-OCLReID` | Original part-aware OCLReID runtime with the released checkpoint. |
| `methods/gated_part_oclreid/` | `part-OCLReID` plus `--association-mode reid_gate` | The same normal part-OCLReID runtime plus a ReID-gated target reassociation fallback. |

The gated method is not a new tracker. It keeps the normal part-OCLReID
prediction, tracker state, online learning, frame mapping, and evaluation logic.
The gate is only a target-reassociation fallback when the normal target state
has no selected target. The frozen gate settings are:

```text
association-reid-threshold: 0.60
association-reid-margin: 0.02
association-min-bbox-score: 0.0
association-min-visible-parts: 1
```

## 7. Run AGHRI Test Experiments

Use the method wrapper scripts for AGHRI test runs. Each wrapper calls the
automation script, runs inference for the selected method, and evaluates each
completed target run automatically.

All method wrappers support:

- `--max-runs N`: run only the first N target runs for a quick check.
- `--target-classes ID ...`: run only selected annotation target IDs, such as
  `--target-classes 01 03`; by default, all eligible annotated IDs are run.
- `--overwrite`: rerun targets that already have completed metrics.
- `--save-visualizations`: save `inference_visualization.mp4` in each target
  result directory.
- `--show-live`: open the live OpenCV inference window while the run continues.
- `--results-root PATH`: choose where new results are written.
- `--dry-run`: print the command that would be executed.

Run one target for each method:

```bash
./methods/rpf_reid/run_aghri_test.sh \
  --max-runs 1 \
  --save-visualizations \
  --overwrite

./methods/normal_part_oclreid/run_aghri_test.sh \
  --max-runs 1 \
  --save-visualizations \
  --overwrite

./methods/gated_part_oclreid/run_aghri_test.sh \
  --max-runs 1 \
  --save-visualizations \
  --overwrite
```

Run the full test set for a method by removing `--max-runs 1`.

By default, reproduced runs are written under:

```text
results/reproduced_runs/
```

To reproduce into a separate directory:

```bash
./methods/gated_part_oclreid/run_aghri_test.sh \
  --results-root results/reproduced_runs/gated_part_oclreid_new \
  --save-visualizations \
  --overwrite
```

To watch a run live while still saving predictions, visualizations, and
evaluation metrics:

```bash
./methods/normal_part_oclreid/run_aghri_test.sh \
  --max-runs 1 \
  --show-live \
  --save-visualizations \
  --overwrite
```

Live display is optional and does not change predictions, tracker state,
learning, metrics, or frame mapping. If no graphical display is available, the
automation warns and continues headlessly.

OpenCV window controls:

- Space: pause or resume display.
- `q` or Esc: close the live window for the remainder of that target run while
  inference continues headlessly.

Automated AGHRI runs do not require manual target-box drawing. The automation
finds the annotated target IDs, runs OCLReID separately for each selected ID,
and initialises each run from the earliest suitable ground-truth target box.
The `--target-classes` option selects annotation IDs from the AGHRI labels, not
runtime tracker IDs assigned after inference starts.
The default initialisation filters are:

```text
width >= 30 px
height >= 70 px
box area >= 0.5% of image area
at least 2 px from every image boundary
```

## 8. Evaluation Outputs

`scripts/run_aghri_experiments.py` evaluates every completed target run after
inference. Each target result directory contains:

```text
predictions.json                  # frame-by-frame target predictions from inference
run_metadata.json                 # method, input video, target, and run configuration
evaluation/summary_metrics.json   # aggregate metrics for the target run
evaluation/per_frame_metrics.csv  # per-frame IoU, target visibility, and prediction status
evaluation/reappearance_events.csv # target disappearance/reappearance and reacquisition events
inference_visualization.mp4       # annotated inference video when --save-visualizations is used
```

Manual evaluation of one existing prediction file is possible with:

```bash
python scripts/evaluate_aghri_results.py \
  --predictions /path/to/predictions.json \
  --annotations /path/to/cam_zed_rgb_ann.json \
  --frame_manifest /path/to/cam_zed_rgb_frame_manifest.json \
  --target_class 01 \
  --initialization_video_frame 0 \
  --fps 10 \
  --output_dir /path/to/evaluation
```

Normal automated runs should not need manual evaluation. The manual command is
mostly useful for debugging a single target.

## 9. Saved Test Dataset Results

The full saved final-test results are active under:

```text
results/test_full/
|-- rpf_reid/
|-- normal_part_oclreid/
`-- gated_part_oclreid/
```

Each method folder contains 50 completed final-test target runs:

```text
rpf_reid:              50 predictions, 50 evaluations, 50 inference MP4s
normal_part_oclreid:   50 predictions, 50 evaluations, 50 inference MP4s
gated_part_oclreid:    50 predictions, 50 evaluations, 50 inference MP4s
```

The runs cover the six final-test bags:

```text
dataset_part1/footpath1_p1_nj+mk+gl_1walk+check_mv_11_12_2024_1_label
dataset_part1/footpath1_p1_oj+mk+gl_1walk+check_st_11_12_2024_1_label
dataset_part3/in_straw_3pick_diff_st_10_24_2024_5_a_label
dataset_part3/out_straw_1push_1walk_1swap_st_11_07_2024_1_b_label
dataset_part4/out_vine_1push_3carry_st_ly_11_06_2024_1_label
dataset_part4/out_vine_4swap+walk_st_ly_11_06_2024_2_label
```

The compact comparison outputs are kept in:

```text
results/final_comparison/
```

This includes the main CSV summaries and selected figures.

The complete figure/table output bundle is kept in:

```text
results/paper_outputs_full/
```

The Jupyter notebook is kept in:

```text
notebooks/aghri_oclreid_results_summary.ipynb
```

It contains the main three-method results table, short explanations of the
three core metrics, the saved final-comparison figures, and notes on which test
bags are used for the full-test and selected-scenario plots.

## 10. Regenerate Figures and Tables

The main figure/table generator is:

```text
scripts/generate_core_metric_paper_outputs_with_gate.py
```

It reads from:

```text
results/test_full/rpf_reid
results/test_full/normal_part_oclreid
results/test_full/gated_part_oclreid
```

and writes to:

```text
results/paper_outputs_full/
```

Regenerate the complete figure/table output bundle:

```bash
python scripts/generate_core_metric_paper_outputs_with_gate.py
```

A smaller convenience script is also kept:

```text
scripts/generate_final_figures.py
```

It is useful for quick checks of the compact `results/final_comparison/`
summaries, but the complete figure/table regeneration should use
`generate_core_metric_paper_outputs_with_gate.py`.

## 11. Validation Commands

Useful syntax checks:

```bash
python -m py_compile scripts/run_single_video.py
python -m py_compile scripts/run_aghri_experiments.py
python -m py_compile scripts/evaluate_aghri_results.py
python -m py_compile scripts/generate_core_metric_paper_outputs_with_gate.py
```

Useful dry runs:

```bash
./demo/run_demo.sh --dry-run
./aghri_video_preparation/prepare_aghri_videos.sh --splits test --manifest-only --dry-run
./methods/rpf_reid/run_aghri_test.sh --max-runs 1 --dry-run
./methods/normal_part_oclreid/run_aghri_test.sh --max-runs 1 --dry-run
./methods/gated_part_oclreid/run_aghri_test.sh --max-runs 1 --dry-run
```

## 12. Notes for Reproduction

- Use `results/test_full/` for the saved complete final-test predictions,
  evaluations, and MP4s.
- Use `results/paper_outputs_full/` for the full generated tables,
  figures, and validation reports.
- Use `scripts/generate_core_metric_paper_outputs_with_gate.py` to regenerate
  the complete figure/table output bundle from the full test outputs.
- Keep the released checkpoint as the default for the three main methods.

## 13. Citation

Use `CITATION.cff` for repository citation metadata and `LICENSE` for licensing
information.
