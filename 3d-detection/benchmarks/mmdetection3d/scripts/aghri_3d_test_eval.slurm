#!/bin/bash
#SBATCH --job-name=aghri_3d_test
#SBATCH --output=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_test_stage6/logs/%x_%j.out
#SBATCH --error=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_test_stage6/logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --chdir=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_test_stage6

set -euo pipefail
ROOT=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_test_stage6
SOURCE=/work/users/pwariyapperuma/Dataset_paper/aghri_second_finetune
CODE="${SOURCE}/code/mmdetection3d"
DATA="${ROOT}/data/aghri_mmdet3d"
VENV="${SOURCE}/environment/venv_openmmlab_py38_torch110_cu113"
CONTAINER=/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs
RUN=${AGHRI_3D_TEST_RUN:?Set AGHRI_3D_TEST_RUN}

case "${RUN}" in
  pointpillars_scratch)
    MODEL_CONFIG="${CODE}/configs/models/pointpillars_aghri.py"
    CHECKPOINT_DIR=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_scratch_stage5/outputs/full/pointpillars_scratch
    CHECKPOINT_NAME=best_aghri_bev_ap_mean_epoch_108.pth ;;
  second_scratch)
    MODEL_CONFIG="${CODE}/configs/models/second_aghri.py"
    CHECKPOINT_DIR=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_scratch_stage5/outputs/full/second_scratch
    CHECKPOINT_NAME=best_aghri_bev_ap_mean_epoch_36.pth ;;
  pointpillars_finetuned)
    MODEL_CONFIG="${CODE}/configs/models/pointpillars_aghri.py"
    CHECKPOINT_DIR=/work/users/pwariyapperuma/Dataset_paper/aghri_pointpillars_finetune/outputs/pointpillars_aghri_seed42/full
    CHECKPOINT_NAME=best_aghri_bev_ap_mean_epoch_126.pth ;;
  second_finetuned)
    MODEL_CONFIG="${CODE}/configs/models/second_aghri.py"
    CHECKPOINT_DIR=/work/users/pwariyapperuma/Dataset_paper/aghri_second_finetune/outputs/second_aghri_seed42/full
    CHECKPOINT_NAME=best_aghri_bev_ap_mean_epoch_2.pth ;;
  *) echo "Unknown run: ${RUN}" >&2; exit 2 ;;
esac
CHECKPOINT="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
if [[ ! -s "${CHECKPOINT}" && "${RUN}" == *_finetuned ]]; then
  # The pretrained jobs also existed at the earlier cluster-root location.
  case "${RUN}" in
    pointpillars_finetuned) CHECKPOINT_DIR=/work/users/pwariyapperuma/aghri_pointpillars_finetune/outputs/pointpillars_aghri_seed42/full ;;
    second_finetuned) CHECKPOINT_DIR=/work/users/pwariyapperuma/aghri_second_finetune/outputs/second_aghri_seed42/full ;;
  esac
  CHECKPOINT="${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
fi
SUMMARY="${ROOT}/summaries/${RUN}.csv"
WORK_DIR="${ROOT}/outputs/${RUN}"
CONFIG="${ROOT}/configs/${RUN}.yaml"

for required in "${MODEL_CONFIG}" "${CHECKPOINT}" "${DATA}/infos/agri_person_infos_test.pkl" \
  "${DATA}/test_export_manifest.json" "${ROOT}/split_lists/test.txt" \
  "${CODE}/src/run_benchmark.py" "${CONTAINER}"; do
  test -s "${required}" || { echo "Missing required file: ${required}" >&2; exit 2; }
done
test -L "${VENV}/bin/python" || { echo "Missing environment Python link: ${VENV}" >&2; exit 2; }
test ! -e "${SUMMARY}" && test ! -e "${WORK_DIR}" || {
  echo "Refusing to overwrite existing result for ${RUN}" >&2; exit 2;
}
mkdir -p "${ROOT}/logs" "${ROOT}/configs" "${ROOT}/summaries" "${ROOT}/manifests"

export AGHRI_3D_TEST_RUN="${RUN}" AGHRI_3D_TEST_CHECKPOINT="${CHECKPOINT}"
module load Pyxis Enroot >/dev/null 2>&1 || true
srun --ntasks=1 --gpus=1 --gpu-bind=closest --mpi=none \
  --container-image="${CONTAINER}" \
  --container-mounts=/work/users:/work/users,/dev/shm:/dev/shm \
  --container-workdir="${CODE}" \
  bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail
ROOT=/work/users/pwariyapperuma/Dataset_paper/aghri_3d_test_stage6
SOURCE=/work/users/pwariyapperuma/Dataset_paper/aghri_second_finetune
CODE="${SOURCE}/code/mmdetection3d"
DATA="${ROOT}/data/aghri_mmdet3d"
PYTHON="${SOURCE}/environment/venv_openmmlab_py38_torch110_cu113/bin/python"
RUN="${AGHRI_3D_TEST_RUN}"
CHECKPOINT="${AGHRI_3D_TEST_CHECKPOINT}"
test -x "${PYTHON}" || { echo "Python unavailable inside container: ${PYTHON}" >&2; exit 2; }
export AGHRI_3D_DATA_ROOT="${DATA}" PYTHONNOUSERSITE=1
export PYTHONPATH="${CODE}:${CODE}/aghri3d"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
export TORCH_HOME="${ROOT}/.cache/torch"
export CUMM_CUDA_ARCH_LIST=8.0 TORCH_CUDA_ARCH_LIST=8.0
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}"

"${PYTHON}" - "${ROOT}" "${DATA}" "${CODE}" "${RUN}" "${CHECKPOINT}" <<'PY'
import hashlib, json, pickle, sys
from pathlib import Path
import yaml
root, data, code, run, checkpoint = map(Path, sys.argv[1:])
names = [s.strip() for s in (root/'split_lists/test.txt').read_text().splitlines() if s.strip()]
manifest = json.loads((data/'test_export_manifest.json').read_text())
assert len(names) == 6 and len(set(names)) == 6
assert manifest['sequences'] == names
assert manifest['split_list_sha256'] == hashlib.sha256((root/'split_lists/test.txt').read_bytes()).hexdigest()
with (data/'infos/agri_person_infos_test.pkl').open('rb') as stream:
    payload = pickle.load(stream)
infos = payload['data_list']
assert infos and len(infos) == manifest['frames']
assert {item['scene_name'] for item in infos} == set(names)
for item in infos:
    assert (data/item['lidar_points']['lidar_path']).is_file(), item['sample_idx']
assert Path(checkpoint).is_file()
model_config = code/'configs/models'/('pointpillars_aghri.py' if 'pointpillars' in str(run) else 'second_aghri.py')
config = {
    'device': 0,
    'seed': 42,
    'benchmark_fixed_input': True,
    'benchmark_warmup': 20,
    'benchmark_iters': 100,
    'summary_primary_mode': 'bev',
    'summary_primary_iou': 0.50,
    'prefer_best_checkpoint': False,
    'project': str(root/'outputs'),
    'name': 'held_out_test',
    'cfg_options': {
        'test_dataloader.dataset.ann_file': 'infos/agri_person_infos_test.pkl',
        'test_evaluator.ann_file': str(data/'infos/agri_person_infos_test.pkl'),
    },
    'models': [{
        'name': str(run), 'config': str(model_config), 'train': False,
        'checkpoint': str(checkpoint), 'work_dir': str(root/'outputs'/run),
    }],
}
with (root/'configs'/f'{run}.yaml').open('w') as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
sha = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b''):
        sha.update(chunk)
(root/'manifests'/f'{run}_checkpoint.json').write_text(json.dumps({
    'run': str(run), 'checkpoint': str(checkpoint), 'sha256': sha.hexdigest(),
    'test_sequences': names, 'test_frames': len(infos),
}, indent=2)+'\n')
print('Test preflight OK:', run, 'frames:', len(infos), 'checkpoint SHA256:', sha.hexdigest())
PY

"${PYTHON}" "${CODE}/src/run_benchmark.py" \
  --config "${ROOT}/configs/${RUN}.yaml" \
  --out "${ROOT}/summaries/${RUN}.csv"

"${PYTHON}" - "${ROOT}/summaries/${RUN}.json" "${CHECKPOINT}" <<'PY'
import json, sys
from pathlib import Path
rows=json.loads(Path(sys.argv[1]).read_text())
assert len(rows)==1 and rows[0]['eval_split']=='test', rows
assert rows[0]['checkpoint_used']==sys.argv[2], rows[0]['checkpoint_used']
assert rows[0]['train_time_sec'] is None
print('Held-out test evaluation complete:', rows[0]['model'])
PY
CONTAINER_SCRIPT
