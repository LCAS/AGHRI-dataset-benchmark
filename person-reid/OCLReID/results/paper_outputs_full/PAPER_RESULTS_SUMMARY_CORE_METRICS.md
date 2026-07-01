# Paper Results Summary: Core Metrics With Gate

The main quantitative comparison contains exactly three methods: rpf-ReID, Normal part-OCLReID, and Gated part-OCLReID.

- rpf-ReID Success@0.5: 0.360
- Normal part-OCLReID Success@0.5: 0.483
- Gated part-OCLReID Success@0.5: 0.557

The final gated method uses the released ResNet18 checkpoint plus normal part-OCLReID online learning and the frozen ReID-gated target reassociation fallback.
