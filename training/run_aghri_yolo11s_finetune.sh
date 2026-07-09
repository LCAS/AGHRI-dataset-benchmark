#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble"
PYTHON_BIN="/usr/bin/python3"
LOCAL_RUNNER="${REPO_ROOT}/training/run_ultralytics_config.py"
FULL_CONFIG="${REPO_ROOT}/training/configs/aghri_zedrgb_yolo11s_finetune.yaml"
OUTPUT_ROOT="/home/prabuddhi/Desktop/Early_Fus/training_outputs/aghri_yolo11s_finetune"
CHECKPOINT_ROOT="/home/prabuddhi/Desktop/Early_Fus/checkpoints/aghri_yolo11s_finetune"
GENERIC_CKPT="/home/prabuddhi/Desktop/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/yolo11s.pt"
DATASET_YAML="/home/prabuddhi/Desktop/Early_Fus/aghri_zedrgb_yolo_dataset/data.yaml"

MODE="${1:-full}"
mkdir -p "${OUTPUT_ROOT}/summaries" "${OUTPUT_ROOT}/logs" "${CHECKPOINT_ROOT}"

record_environment() {
  {
    echo "python_bin=${PYTHON_BIN}"
    echo "python_path=$(command -v "${PYTHON_BIN}")"
    "${PYTHON_BIN}" --version
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
    df -h /home/prabuddhi/Desktop/Early_Fus /media/prabuddhi/Backup2 || true
    "${PYTHON_BIN}" - <<'PY'
import json
info = {}
try:
    import torch
    info.update({
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    })
except Exception as exc:
    info["torch_error"] = str(exc)
try:
    import ultralytics
    info["ultralytics_version"] = ultralytics.__version__
except Exception as exc:
    info["ultralytics_error"] = str(exc)
print(json.dumps(info, indent=2, sort_keys=True))
PY
  } | tee "${OUTPUT_ROOT}/environment.txt"
}

write_config() {
  local config_path="$1"
  local epochs="$2"
  local project="$3"
  local name="$4"
  "${PYTHON_BIN}" - "$FULL_CONFIG" "$config_path" "$epochs" "$project" "$name" <<'PY'
import sys
from pathlib import Path
import yaml
src, dst, epochs, project, name = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["epochs"] = int(epochs)
cfg["project"] = project
cfg["name"] = name
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
}

case "${MODE}" in
  env)
    record_environment
    ;;
  preflight)
    record_environment
    "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO
model = YOLO("${GENERIC_CKPT}")
results = model.val(data="${DATASET_YAML}", imgsz=640, batch=1, device=0, split="val", workers=0, project="${OUTPUT_ROOT}/preflight", name="generic_yolo11s_val_batch1")
print({"precision": float(results.box.mp), "recall": float(results.box.mr), "map50": float(results.box.map50), "map50_95": float(results.box.map)})
PY
    ;;
  smoke)
    record_environment
    cd "${OUTPUT_ROOT}"
    SMOKE_CONFIG="${OUTPUT_ROOT}/configs/smoke_aghri_zedrgb_yolo11s.yaml"
    write_config "${SMOKE_CONFIG}" 1 "${CHECKPOINT_ROOT}/smoke" "smoke_yolo11s_aghri_zedrgb_seed42"
    "${PYTHON_BIN}" "${LOCAL_RUNNER}" \
      --config "${SMOKE_CONFIG}" \
      --out "${OUTPUT_ROOT}/summaries/smoke_summary.csv" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/smoke_train_eval.log"
    ;;
  full)
    record_environment
    cd "${OUTPUT_ROOT}"
    if [ -e "${CHECKPOINT_ROOT}/yolo11s_aghri_zedrgb_seed42/yolo11s/weights/best.pt" ]; then
      echo "Refusing to overwrite existing completed run. Move it aside or resume intentionally." >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${LOCAL_RUNNER}" \
      --config "${FULL_CONFIG}" \
      --out "${OUTPUT_ROOT}/summaries/full_summary.csv" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/full_train_eval.log"
    ;;
  eval-generic)
    "${PYTHON_BIN}" - <<PY
from ultralytics import YOLO
model = YOLO("${GENERIC_CKPT}")
results = model.val(data="${DATASET_YAML}", imgsz=640, batch=16, device=0, split="val", workers=8, project="${OUTPUT_ROOT}/validation_only", name="generic_yolo11s_val")
print({"precision": float(results.box.mp), "recall": float(results.box.mr), "map50": float(results.box.map50), "map50_95": float(results.box.map), "speed": results.speed})
PY
    ;;
  *)
    echo "Usage: $0 {env|preflight|smoke|full|eval-generic}" >&2
    exit 2
    ;;
esac
