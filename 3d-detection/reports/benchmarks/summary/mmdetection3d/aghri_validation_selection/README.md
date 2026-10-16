# AGHRI validation checkpoint selection

These are the saved full-training validation summaries for the four AGHRI
LiDAR-only models. The selected checkpoint maximised `aghri/bev_ap_mean`
over the seven released validation sequences. The mean covers BEV IoU
thresholds 0.25, 0.50, and 0.75. The selected epochs are 108 (PointPillars
scratch), 36 (SECOND scratch), 126 (PointPillars KITTI-pretrained and
AGHRI-fine-tuned), and 2 (SECOND KITTI-pretrained and AGHRI-fine-tuned).

These files have `eval_split: val`; their AP values are checkpoint-selection
evidence, not final held-out test scores. Final scores are in the sibling
[`aghri_heldout_test`](../aghri_heldout_test/) directory. The checkpoint paths
in the original cluster summaries are preserved verbatim. The held-out test
checkpoint manifests record SHA256 hashes of the weights actually evaluated.
