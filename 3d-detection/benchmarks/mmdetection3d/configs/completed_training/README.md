# Completed cluster training configuration records

These are copies of the full-run configs used for the four completed AGHRI
3D model trainings. They retain the original cluster paths for provenance.
Scratch runs began from random initialisation; the two fine-tuned runs used
public KITTI-pretrained model weights. All saved summaries used validation
annotations to select the best checkpoint. The held-out test was evaluated
separately without training.

For repository-level execution, use the parent `benchmark_mmdetection3d_aghri_*`
training manifests with `evaluation_split: val`, then the frozen
`benchmark_mmdetection3d_aghri_heldout_test.yaml` test manifest.
