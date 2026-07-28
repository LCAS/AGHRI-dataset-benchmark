#!/usr/bin/env bash
set -euo pipefail

LOCAL_REPO="${LOCAL_REPO:-/home/prabuddhi/Desktop/late_fusion/deepfusionmot}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/home/prabuddhi/Desktop/agri-human-dataset-benchmark}"
MMDET3D_SRC="${MMDET3D_SRC:-${BENCHMARK_ROOT}/3d-detection/benchmarks/mmdetection3d}"
RAW_ROOT="${RAW_ROOT:-/media/prabuddhi/Backup2/Updated Dataset_PW}"
SPLITS_DIR="${SPLITS_DIR:-${RAW_ROOT}/split_lists}"
LOCAL_CLUSTER_ROOT="${LOCAL_CLUSTER_ROOT:-/home/prabuddhi/cluster/aghri_second_finetune}"
WORK_CLUSTER_ROOT="${WORK_CLUSTER_ROOT:-/work/users/pwariyapperuma/aghri_second_finetune}"
GENERIC_CHECKPOINT="${GENERIC_CHECKPOINT:-${LOCAL_REPO}/checkpoints/kitti_pretrained/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth}"
EXPECTED_SHA="${EXPECTED_SHA:-b086d0a369d0b955b3ff45b6849a4370c46dea2b95445671837a989839eb2bbd}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREPARE_DATA=1
DRY_RUN=0
WORKERS="${WORKERS:-8}"

usage() {
  cat <<'EOF'
Usage: scripts/stage_aghri_second_cluster.sh [--dry-run] [--no-prepare-data]

Stages the reproducible AGHRI SECOND fine-tuning workspace. By default it
exports only train/val scenes from the AGHRI scene-level split lists; held-out
test artifacts are removed from the active training data root.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-prepare-data) PREPARE_DATA=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

sha="$(sha256sum "${GENERIC_CHECKPOINT}" | awk '{print $1}')"
if [[ "${sha}" != "${EXPECTED_SHA}" ]]; then
  echo "Generic SECOND checkpoint SHA mismatch: ${sha}" >&2
  exit 3
fi

echo "Staging AGHRI SECOND fine-tune workspace:"
echo "  local cluster root: ${LOCAL_CLUSTER_ROOT}"
echo "  work cluster root:  ${WORK_CLUSTER_ROOT}"
echo "  raw root:           ${RAW_ROOT}"
echo "  dry run:            ${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" "${LOCAL_REPO}/tools/prepare_aghri_second_finetune_dataset.py" \
    --raw-root "${RAW_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --out-root "${LOCAL_CLUSTER_ROOT}/data/aghri_mmdet3d" \
    --workers 1 \
    --dry-run
  exit 0
fi

mkdir -p \
  "${LOCAL_CLUSTER_ROOT}/code" \
  "${LOCAL_CLUSTER_ROOT}/configs" \
  "${LOCAL_CLUSTER_ROOT}/data/aghri_mmdet3d" \
  "${LOCAL_CLUSTER_ROOT}/checkpoints/generic" \
  "${LOCAL_CLUSTER_ROOT}/checkpoints/finetuned" \
  "${LOCAL_CLUSTER_ROOT}/outputs" \
  "${LOCAL_CLUSTER_ROOT}/logs" \
  "${LOCAL_CLUSTER_ROOT}/manifests" \
  "${LOCAL_CLUSTER_ROOT}/environment" \
  "${LOCAL_CLUSTER_ROOT}/scripts" \
  "${LOCAL_CLUSTER_ROOT}/summaries"

RSYNC_OPTS=(-rt --no-owner --no-group --omit-dir-times)
rsync "${RSYNC_OPTS[@]}" --delete "${MMDET3D_SRC}/" "${LOCAL_CLUSTER_ROOT}/code/mmdetection3d/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/tools/prepare_aghri_second_finetune_dataset.py" "${LOCAL_CLUSTER_ROOT}/scripts/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/tools/aghri_second_finetune_common.py" "${LOCAL_CLUSTER_ROOT}/scripts/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/scripts/verify_staged_aghri_second_dataset.py" "${LOCAL_CLUSTER_ROOT}/scripts/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/scripts/bootstrap_aghri_second_environment.sh" "${LOCAL_CLUSTER_ROOT}/scripts/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/config/aghri_second_finetune/second_aghri_finetune_seed42.py" "${LOCAL_CLUSTER_ROOT}/configs/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/config/aghri_second_finetune/benchmark_aghri_second_finetune.yaml" "${LOCAL_CLUSTER_ROOT}/configs/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/config/aghri_second_finetune/benchmark_aghri_second_finetune_smoke.yaml" "${LOCAL_CLUSTER_ROOT}/configs/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/aghri_second_smoke.slurm" "${LOCAL_CLUSTER_ROOT}/"
rsync "${RSYNC_OPTS[@]}" "${LOCAL_REPO}/aghri_second_train.slurm" "${LOCAL_CLUSTER_ROOT}/"
cp -f "${GENERIC_CHECKPOINT}" "${LOCAL_CLUSTER_ROOT}/checkpoints/generic/"

cat > "${LOCAL_CLUSTER_ROOT}/environment/requirements_aghri_second_finetune.txt" <<'EOF'
# Intentionally minimal. The container is expected to provide the CUDA/OpenMMLab stack.
# Add only small missing runtime packages here after a smoke-job failure identifies them.
EOF

cat > "${LOCAL_CLUSTER_ROOT}/manifests/aghri_second_finetune_paths.env" <<EOF
AGHRI_SECOND_CLUSTER_ROOT=${WORK_CLUSTER_ROOT}
AGHRI_SECOND_DATA_ROOT=${WORK_CLUSTER_ROOT}/data/aghri_mmdet3d
AGHRI_SECOND_BASE_CONFIG=${WORK_CLUSTER_ROOT}/code/mmdetection3d/configs/models/second_aghri.py
AGHRI_SECOND_GENERIC_CHECKPOINT=${WORK_CLUSTER_ROOT}/checkpoints/generic/$(basename "${GENERIC_CHECKPOINT}")
AGHRI_SECOND_WORK_DIR=${WORK_CLUSTER_ROOT}/outputs/second_aghri_seed42
EOF

if [[ "${PREPARE_DATA}" == "1" ]]; then
  "${PYTHON_BIN}" "${LOCAL_REPO}/tools/prepare_aghri_second_finetune_dataset.py" \
    --raw-root "${RAW_ROOT}" \
    --splits-dir "${SPLITS_DIR}" \
    --out-root "${LOCAL_CLUSTER_ROOT}/data/aghri_mmdet3d" \
    --workers "${WORKERS}"
fi

"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/verify_staged_aghri_second_dataset.py" \
  --data-root "${LOCAL_CLUSTER_ROOT}/data/aghri_mmdet3d" \
  --json-out "${LOCAL_CLUSTER_ROOT}/manifests/aghri_second_finetune_dataset_verification.json" \
  --md-out "${LOCAL_CLUSTER_ROOT}/summaries/AGHRI_SECOND_FINETUNE_STAGED_DATASET.md"

(
  cd "${LOCAL_CLUSTER_ROOT}"
  find configs scripts checkpoints/generic manifests environment -type f -print0 | sort -z | xargs -0 sha256sum > manifests/SHA256SUMS
)

echo "Staged AGHRI SECOND fine-tune workspace at ${LOCAL_CLUSTER_ROOT}"
