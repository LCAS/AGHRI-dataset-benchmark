# Figure Captions

## Figure 1
Overall target-tracking performance on the AGHRI test set for rpf-ReID, Normal part-OCLReID, and Gated part-OCLReID. The gated method uses the released ResNet18 checkpoint with the frozen ReID-gated reassociation fallback.

## Figure 2
Core tracking metrics for the selected Footpath, Polytunnel, and Vineyard scenarios. Metrics are recomputed from per-run evaluation files within each scenario rather than copied from whole-test aggregates.

## Figure 3
Camera-wise core metrics over the full AGHRI test set for front fisheye, left fisheye, right fisheye, and ZED RGB cameras.

## Figure 4
Absolute improvement over rpf-ReID for Normal part-OCLReID and Gated part-OCLReID. The gate adds target-reassociation behaviour on top of the released-checkpoint part-OCLReID configuration.

## Figure 5
ZED-front qualitative examples using the original selected frames and saved inference overlays. The part-based method label is shown as Normal part-OCLReID.

## Figure 6
Core metrics grouped by the number of annotated humans in the test scene. Group membership is shared across all three methods.

## Figure 7
Core metrics grouped by robot motion state, using the same stationary/moving scene definitions as the original analysis.

## Figure 8
Front-fisheye qualitative examples using the original selected frames and saved inference overlays. The part-based method label is shown as Normal part-OCLReID.

## Figure 9
Qualitative comparison of Normal part-OCLReID and Gated part-OCLReID following target reappearance. Frames are extracted directly from the saved runtime visualisation videos and aligned using the original frame manifest and per-frame evaluation mapping. Normal part-OCLReID withholds the target output despite the target being visible, whereas Gated part-OCLReID reassociates the correct existing ByteTrack track with the known target identity. The gate is used only when the original state machine returns no target; ByteTrack association itself is unchanged. The frozen gate checks are score >= 0.60, score margin >= 0.02, visible parts >= 1, and minimum bbox score >= 0.0. Confidence values visible in the runtime overlay are target confidence values from the implementation, not claimed here as raw ReID scores.
