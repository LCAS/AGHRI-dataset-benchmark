# RUNTIME CHECKPOINT COMPATIBILITY

Selected source checkpoint: `aghri_reid_stage1/training/work_dirs/aghri_resnet18_backbone_stage1_5ep/epoch_1.pth`

Exported checkpoint path: `aghri_reid_stage1/checkpoints/aghri_resnet18_backbone_stage1.pth`

Runtime copy: `checkpoints/reid/aghri_resnet18_backbone_stage1.pth`

Runtime model config: `configs/rpf/ocl_rpf/part_rpf_weighted_yolox_l_r18.py`

## Summary

- Exported key count: 127
- Matched key count: 127
- Shape-compatible transferred parameters: 11318485
- Missing runtime keys after export: 72
- Unexpected exported keys: 0
- Shape mismatches: 0
- Excluded source keys: 2
- Skipped source keys not present in runtime model: 7
- Temporary classifier keys exported: 0
- Optimizer/scheduler-like keys exported: 0
- Exported SHA256: `dfde4150338301af21a0fc4f9346c0f52ef056b8768f627ace598758da644e66`
- Runtime copy SHA256: `dfde4150338301af21a0fc4f9346c0f52ef056b8768f627ace598758da644e66`
- Released `resnet18.pth` SHA256 after export: `907f4055fbaf181901f2f4f2af43fbe86b9d83d1967df67c1e3c37b72ad4ae62`

The Stage 1 export intentionally keeps backbone and compatible shared projection
weights only. The custom part/global identity classifiers remain initialized by
the existing runtime procedure.

## First matched keys

```json
[
  "backbone.conv1.weight",
  "backbone.bn1.weight",
  "backbone.bn1.bias",
  "backbone.bn1.running_mean",
  "backbone.bn1.running_var",
  "backbone.bn1.num_batches_tracked",
  "backbone.layer1.0.conv1.weight",
  "backbone.layer1.0.bn1.weight",
  "backbone.layer1.0.bn1.bias",
  "backbone.layer1.0.bn1.running_mean",
  "backbone.layer1.0.bn1.running_var",
  "backbone.layer1.0.bn1.num_batches_tracked",
  "backbone.layer1.0.conv2.weight",
  "backbone.layer1.0.bn2.weight",
  "backbone.layer1.0.bn2.bias",
  "backbone.layer1.0.bn2.running_mean",
  "backbone.layer1.0.bn2.running_var",
  "backbone.layer1.0.bn2.num_batches_tracked",
  "backbone.layer1.1.conv1.weight",
  "backbone.layer1.1.bn1.weight",
  "backbone.layer1.1.bn1.bias",
  "backbone.layer1.1.bn1.running_mean",
  "backbone.layer1.1.bn1.running_var",
  "backbone.layer1.1.bn1.num_batches_tracked",
  "backbone.layer1.1.conv2.weight",
  "backbone.layer1.1.bn2.weight",
  "backbone.layer1.1.bn2.bias",
  "backbone.layer1.1.bn2.running_mean",
  "backbone.layer1.1.bn2.running_var",
  "backbone.layer1.1.bn2.num_batches_tracked",
  "backbone.layer2.0.conv1.weight",
  "backbone.layer2.0.bn1.weight",
  "backbone.layer2.0.bn1.bias",
  "backbone.layer2.0.bn1.running_mean",
  "backbone.layer2.0.bn1.running_var",
  "backbone.layer2.0.bn1.num_batches_tracked",
  "backbone.layer2.0.conv2.weight",
  "backbone.layer2.0.bn2.weight",
  "backbone.layer2.0.bn2.bias",
  "backbone.layer2.0.bn2.running_mean"
]
```

## Exported keys

```json
[
  "backbone.bn1.bias",
  "backbone.bn1.num_batches_tracked",
  "backbone.bn1.running_mean",
  "backbone.bn1.running_var",
  "backbone.bn1.weight",
  "backbone.conv1.weight",
  "backbone.layer1.0.bn1.bias",
  "backbone.layer1.0.bn1.num_batches_tracked",
  "backbone.layer1.0.bn1.running_mean",
  "backbone.layer1.0.bn1.running_var",
  "backbone.layer1.0.bn1.weight",
  "backbone.layer1.0.bn2.bias",
  "backbone.layer1.0.bn2.num_batches_tracked",
  "backbone.layer1.0.bn2.running_mean",
  "backbone.layer1.0.bn2.running_var",
  "backbone.layer1.0.bn2.weight",
  "backbone.layer1.0.conv1.weight",
  "backbone.layer1.0.conv2.weight",
  "backbone.layer1.1.bn1.bias",
  "backbone.layer1.1.bn1.num_batches_tracked",
  "backbone.layer1.1.bn1.running_mean",
  "backbone.layer1.1.bn1.running_var",
  "backbone.layer1.1.bn1.weight",
  "backbone.layer1.1.bn2.bias",
  "backbone.layer1.1.bn2.num_batches_tracked",
  "backbone.layer1.1.bn2.running_mean",
  "backbone.layer1.1.bn2.running_var",
  "backbone.layer1.1.bn2.weight",
  "backbone.layer1.1.conv1.weight",
  "backbone.layer1.1.conv2.weight",
  "backbone.layer2.0.bn1.bias",
  "backbone.layer2.0.bn1.num_batches_tracked",
  "backbone.layer2.0.bn1.running_mean",
  "backbone.layer2.0.bn1.running_var",
  "backbone.layer2.0.bn1.weight",
  "backbone.layer2.0.bn2.bias",
  "backbone.layer2.0.bn2.num_batches_tracked",
  "backbone.layer2.0.bn2.running_mean",
  "backbone.layer2.0.bn2.running_var",
  "backbone.layer2.0.bn2.weight",
  "backbone.layer2.0.conv1.weight",
  "backbone.layer2.0.conv2.weight",
  "backbone.layer2.0.downsample.0.weight",
  "backbone.layer2.0.downsample.1.bias",
  "backbone.layer2.0.downsample.1.num_batches_tracked",
  "backbone.layer2.0.downsample.1.running_mean",
  "backbone.layer2.0.downsample.1.running_var",
  "backbone.layer2.0.downsample.1.weight",
  "backbone.layer2.1.bn1.bias",
  "backbone.layer2.1.bn1.num_batches_tracked",
  "backbone.layer2.1.bn1.running_mean",
  "backbone.layer2.1.bn1.running_var",
  "backbone.layer2.1.bn1.weight",
  "backbone.layer2.1.bn2.bias",
  "backbone.layer2.1.bn2.num_batches_tracked",
  "backbone.layer2.1.bn2.running_mean",
  "backbone.layer2.1.bn2.running_var",
  "backbone.layer2.1.bn2.weight",
  "backbone.layer2.1.conv1.weight",
  "backbone.layer2.1.conv2.weight",
  "backbone.layer3.0.bn1.bias",
  "backbone.layer3.0.bn1.num_batches_tracked",
  "backbone.layer3.0.bn1.running_mean",
  "backbone.layer3.0.bn1.running_var",
  "backbone.layer3.0.bn1.weight",
  "backbone.layer3.0.bn2.bias",
  "backbone.layer3.0.bn2.num_batches_tracked",
  "backbone.layer3.0.bn2.running_mean",
  "backbone.layer3.0.bn2.running_var",
  "backbone.layer3.0.bn2.weight",
  "backbone.layer3.0.conv1.weight",
  "backbone.layer3.0.conv2.weight",
  "backbone.layer3.0.downsample.0.weight",
  "backbone.layer3.0.downsample.1.bias",
  "backbone.layer3.0.downsample.1.num_batches_tracked",
  "backbone.layer3.0.downsample.1.running_mean",
  "backbone.layer3.0.downsample.1.running_var",
  "backbone.layer3.0.downsample.1.weight",
  "backbone.layer3.1.bn1.bias",
  "backbone.layer3.1.bn1.num_batches_tracked",
  "backbone.layer3.1.bn1.running_mean",
  "backbone.layer3.1.bn1.running_var",
  "backbone.layer3.1.bn1.weight",
  "backbone.layer3.1.bn2.bias",
  "backbone.layer3.1.bn2.num_batches_tracked",
  "backbone.layer3.1.bn2.running_mean",
  "backbone.layer3.1.bn2.running_var",
  "backbone.layer3.1.bn2.weight",
  "backbone.layer3.1.conv1.weight",
  "backbone.layer3.1.conv2.weight",
  "backbone.layer4.0.bn1.bias",
  "backbone.layer4.0.bn1.num_batches_tracked",
  "backbone.layer4.0.bn1.running_mean",
  "backbone.layer4.0.bn1.running_var",
  "backbone.layer4.0.bn1.weight",
  "backbone.layer4.0.bn2.bias",
  "backbone.layer4.0.bn2.num_batches_tracked",
  "backbone.layer4.0.bn2.running_mean",
  "backbone.layer4.0.bn2.running_var",
  "backbone.layer4.0.bn2.weight",
  "backbone.layer4.0.conv1.weight",
  "backbone.layer4.0.conv2.weight",
  "backbone.layer4.0.downsample.0.weight",
  "backbone.layer4.0.downsample.1.bias",
  "backbone.layer4.0.downsample.1.num_batches_tracked",
  "backbone.layer4.0.downsample.1.running_mean",
  "backbone.layer4.0.downsample.1.running_var",
  "backbone.layer4.0.downsample.1.weight",
  "backbone.layer4.1.bn1.bias",
  "backbone.layer4.1.bn1.num_batches_tracked",
  "backbone.layer4.1.bn1.running_mean",
  "backbone.layer4.1.bn1.running_var",
  "backbone.layer4.1.bn1.weight",
  "backbone.layer4.1.bn2.bias",
  "backbone.layer4.1.bn2.num_batches_tracked",
  "backbone.layer4.1.bn2.running_mean",
  "backbone.layer4.1.bn2.running_var",
  "backbone.layer4.1.bn2.weight",
  "backbone.layer4.1.conv1.weight",
  "backbone.layer4.1.conv2.weight",
  "head.fcs.0.bn.bias",
  "head.fcs.0.bn.num_batches_tracked",
  "head.fcs.0.bn.running_mean",
  "head.fcs.0.bn.running_var",
  "head.fcs.0.bn.weight",
  "head.fcs.0.fc.bias",
  "head.fcs.0.fc.weight"
]
```

## First missing runtime keys

```json
[
  "head.concat_parts_identity_classifier.0.bn.bias",
  "head.concat_parts_identity_classifier.0.bn.num_batches_tracked",
  "head.concat_parts_identity_classifier.0.bn.running_mean",
  "head.concat_parts_identity_classifier.0.bn.running_var",
  "head.concat_parts_identity_classifier.0.bn.weight",
  "head.concat_parts_identity_classifier.0.classifier.weight",
  "head.concat_parts_identity_classifier.1.bn.bias",
  "head.concat_parts_identity_classifier.1.bn.num_batches_tracked",
  "head.concat_parts_identity_classifier.1.bn.running_mean",
  "head.concat_parts_identity_classifier.1.bn.running_var",
  "head.concat_parts_identity_classifier.1.bn.weight",
  "head.concat_parts_identity_classifier.1.classifier.weight",
  "head.global_identity_classifier.0.bn.bias",
  "head.global_identity_classifier.0.bn.num_batches_tracked",
  "head.global_identity_classifier.0.bn.running_mean",
  "head.global_identity_classifier.0.bn.running_var",
  "head.global_identity_classifier.0.bn.weight",
  "head.global_identity_classifier.0.classifier.weight",
  "head.global_identity_classifier.1.bn.bias",
  "head.global_identity_classifier.1.bn.num_batches_tracked",
  "head.global_identity_classifier.1.bn.running_mean",
  "head.global_identity_classifier.1.bn.running_var",
  "head.global_identity_classifier.1.bn.weight",
  "head.global_identity_classifier.1.classifier.weight",
  "head.parts_identity_classifier.0.bn.bias",
  "head.parts_identity_classifier.0.bn.num_batches_tracked",
  "head.parts_identity_classifier.0.bn.running_mean",
  "head.parts_identity_classifier.0.bn.running_var",
  "head.parts_identity_classifier.0.bn.weight",
  "head.parts_identity_classifier.0.classifier.weight",
  "head.parts_identity_classifier.1.bn.bias",
  "head.parts_identity_classifier.1.bn.num_batches_tracked",
  "head.parts_identity_classifier.1.bn.running_mean",
  "head.parts_identity_classifier.1.bn.running_var",
  "head.parts_identity_classifier.1.bn.weight",
  "head.parts_identity_classifier.1.classifier.weight",
  "head.parts_identity_classifier.2.bn.bias",
  "head.parts_identity_classifier.2.bn.num_batches_tracked",
  "head.parts_identity_classifier.2.bn.running_mean",
  "head.parts_identity_classifier.2.bn.running_var"
]
```

## Unexpected exported keys

```json
[]
```

## Excluded keys

```json
[
  "head.classifier.weight",
  "head.classifier.bias"
]
```

## Skipped source keys

```json
[
  "head.fc_out.weight",
  "head.fc_out.bias",
  "head.bn.weight",
  "head.bn.bias",
  "head.bn.running_mean",
  "head.bn.running_var",
  "head.bn.num_batches_tracked"
]
```

## Shape mismatches

```json
[]
```

## Conclusion

RUNTIME CHECKPOINT COMPATIBILITY: PASS
