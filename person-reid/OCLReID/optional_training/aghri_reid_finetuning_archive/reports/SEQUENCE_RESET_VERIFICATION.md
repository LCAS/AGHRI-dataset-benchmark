# SEQUENCE RESET VERIFICATION

Not run.

No fine-tuned Stage 1 runtime checkpoint exists yet, because full training was
not launched in the current CPU-only environment. Runtime checkpoint selection
and sequence reset verification should be performed only after:

1. full Stage 1 training completes;
2. `aghri_reid_stage1/checkpoints/aghri_resnet18_backbone_stage1.pth` is exported;
3. compatibility with `part-OCLReID` is verified.

No automated validation or final-test sequence was run for this Stage 1 attempt.
