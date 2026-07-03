# Checkpoint And Model Asset Manifest

| Name | Absolute path | Purpose | Size bytes | SHA-256 | Exists | Required by |
|---|---|---|---:|---|---|---|
| released ResNet18 ReID checkpoint | `/home/prabuddhi/Desktop/OCLReID/checkpoints/reid/resnet18.pth` | initial ReID weights for all three final workflows | 91238949 | `907f4055fbaf181901f2f4f2af43fbe86b9d83d1967df67c1e3c37b72ad4ae62` | True | runtime verification/workflows |
| pose module directory | `/home/prabuddhi/Desktop/OCLReID/mmtrack/models/pose` | pose-estimation code loaded by part-based runtime | 405031348 | `directory` | True | runtime verification/workflows |
| orientation module directory | `/home/prabuddhi/Desktop/OCLReID/mmtrack/models/orientation` | orientation-estimation code loaded by part-based runtime | 9155791 | `directory` | True | runtime verification/workflows |
| YOLOX/person detection configs | `/home/prabuddhi/Desktop/OCLReID/configs/rpf/ocl_rpf` | person detection/tracking configuration files | 38175 | `directory` | True | runtime verification/workflows |
