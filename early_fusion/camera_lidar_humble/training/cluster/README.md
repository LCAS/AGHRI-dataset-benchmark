# AGHRI YOLO11s Cluster Training

This folder contains the cluster handoff for AGHRI ZED RGB YOLO11s fine-tuning. It is intentionally separate from the local training wrapper so the local workflow stays reproducible.

The local YOLO dataset was generated with symlinked images. For cluster use, stage it with dereferenced image files so the job does not depend on `/media/prabuddhi/Backup2` being mounted on the compute node.

```bash
cd /home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble
bash training/cluster/stage_aghri_yolo11s_cluster.sh
```

Expected staged root:

```text
/home/prabuddhi/cluster/aghri_yolo11s_finetune
```

The corresponding in-cluster path used by Slurm is:

```text
/work/users/pwariyapperuma/aghri_yolo11s_finetune
```

Submit from a cluster login node:

```bash
cd /work/users/pwariyapperuma/aghri_yolo11s_finetune
sbatch aghri_yolo11s_train.slurm
```

The Slurm job trains only on the official train split and validates only on the official validation split. It does not touch the official test split.

Completed checkpoint artifacts are expected under:

```text
/work/users/pwariyapperuma/aghri_yolo11s_finetune/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s
```

The final fusion benchmark uses `weights/best.pt` only. Do not substitute `last.pt`.

Local copy command from the mounted cluster tree:

```bash
mkdir -p /home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights
cp -a /home/prabuddhi/cluster/aghri_yolo11s_finetune/checkpoints/aghri_yolo11s_finetune/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt \
  /home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt
sha256sum /home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune/cluster_run/weights/best.pt
```

Expected SHA256:

```text
bbb267ba96e3b94b615ef4e78a0d5319b95c592e932879f2c2bfeedf54fa3110
```
