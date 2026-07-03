# TRAINABLE PARAMETER REPORT

Initial checkpoint: `/home/prabuddhi/Desktop/OCLReID/checkpoints/reid/resnet18.pth`

- Loaded compatible keys: 134
- Ignored unexpected keys: 0
- Shape mismatches: 2
- Missing after non-strict load: 2
- Trainable parameters: 10658819
- Frozen parameters: 683072

| parameter | shape | requires_grad | count | initialization_source |
|---|---:|---:|---:|---|
| `backbone.conv1.weight` | `(64, 3, 7, 7)` | False | 9408 | checkpoint |
| `backbone.bn1.weight` | `(64,)` | False | 64 | checkpoint |
| `backbone.bn1.bias` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.0.conv1.weight` | `(64, 64, 3, 3)` | False | 36864 | checkpoint |
| `backbone.layer1.0.bn1.weight` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.0.bn1.bias` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.0.conv2.weight` | `(64, 64, 3, 3)` | False | 36864 | checkpoint |
| `backbone.layer1.0.bn2.weight` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.0.bn2.bias` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.1.conv1.weight` | `(64, 64, 3, 3)` | False | 36864 | checkpoint |
| `backbone.layer1.1.bn1.weight` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.1.bn1.bias` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.1.conv2.weight` | `(64, 64, 3, 3)` | False | 36864 | checkpoint |
| `backbone.layer1.1.bn2.weight` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer1.1.bn2.bias` | `(64,)` | False | 64 | checkpoint |
| `backbone.layer2.0.conv1.weight` | `(128, 64, 3, 3)` | False | 73728 | checkpoint |
| `backbone.layer2.0.bn1.weight` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.0.bn1.bias` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.0.conv2.weight` | `(128, 128, 3, 3)` | False | 147456 | checkpoint |
| `backbone.layer2.0.bn2.weight` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.0.bn2.bias` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.0.downsample.0.weight` | `(128, 64, 1, 1)` | False | 8192 | checkpoint |
| `backbone.layer2.0.downsample.1.weight` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.0.downsample.1.bias` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.1.conv1.weight` | `(128, 128, 3, 3)` | False | 147456 | checkpoint |
| `backbone.layer2.1.bn1.weight` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.1.bn1.bias` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.1.conv2.weight` | `(128, 128, 3, 3)` | False | 147456 | checkpoint |
| `backbone.layer2.1.bn2.weight` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer2.1.bn2.bias` | `(128,)` | False | 128 | checkpoint |
| `backbone.layer3.0.conv1.weight` | `(256, 128, 3, 3)` | True | 294912 | checkpoint |
| `backbone.layer3.0.bn1.weight` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.0.bn1.bias` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.0.conv2.weight` | `(256, 256, 3, 3)` | True | 589824 | checkpoint |
| `backbone.layer3.0.bn2.weight` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.0.bn2.bias` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.0.downsample.0.weight` | `(256, 128, 1, 1)` | True | 32768 | checkpoint |
| `backbone.layer3.0.downsample.1.weight` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.0.downsample.1.bias` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.1.conv1.weight` | `(256, 256, 3, 3)` | True | 589824 | checkpoint |
| `backbone.layer3.1.bn1.weight` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.1.bn1.bias` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.1.conv2.weight` | `(256, 256, 3, 3)` | True | 589824 | checkpoint |
| `backbone.layer3.1.bn2.weight` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer3.1.bn2.bias` | `(256,)` | True | 256 | checkpoint |
| `backbone.layer4.0.conv1.weight` | `(512, 256, 3, 3)` | True | 1179648 | checkpoint |
| `backbone.layer4.0.bn1.weight` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.0.bn1.bias` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.0.conv2.weight` | `(512, 512, 3, 3)` | True | 2359296 | checkpoint |
| `backbone.layer4.0.bn2.weight` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.0.bn2.bias` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.0.downsample.0.weight` | `(512, 256, 1, 1)` | True | 131072 | checkpoint |
| `backbone.layer4.0.downsample.1.weight` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.0.downsample.1.bias` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.1.conv1.weight` | `(512, 512, 3, 3)` | True | 2359296 | checkpoint |
| `backbone.layer4.1.bn1.weight` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.1.bn1.bias` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.1.conv2.weight` | `(512, 512, 3, 3)` | True | 2359296 | checkpoint |
| `backbone.layer4.1.bn2.weight` | `(512,)` | True | 512 | checkpoint |
| `backbone.layer4.1.bn2.bias` | `(512,)` | True | 512 | checkpoint |
| `head.fcs.0.fc.weight` | `(256, 512)` | True | 131072 | checkpoint |
| `head.fcs.0.fc.bias` | `(256,)` | True | 256 | checkpoint |
| `head.fcs.0.bn.weight` | `(256,)` | True | 256 | checkpoint |
| `head.fcs.0.bn.bias` | `(256,)` | True | 256 | checkpoint |
| `head.fc_out.weight` | `(128, 256)` | True | 32768 | checkpoint |
| `head.fc_out.bias` | `(128,)` | True | 128 | checkpoint |
| `head.bn.weight` | `(128,)` | True | 128 | checkpoint |
| `head.bn.bias` | `(128,)` | True | 128 | checkpoint |
| `head.classifier.weight` | `(3, 128)` | True | 384 | random_or_model_init |
| `head.classifier.bias` | `(3,)` | True | 3 | random_or_model_init |
