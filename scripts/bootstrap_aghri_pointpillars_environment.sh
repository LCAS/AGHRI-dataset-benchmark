#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGHRI_POINTPILLARS_CLUSTER_ROOT:-/work/users/pwariyapperuma/aghri_pointpillars_finetune}"
VENV="${AGHRI_POINTPILLARS_VENV:-${ROOT}/environment/venv}"
MMDET3D_ROOT="${ROOT}/code/mmdetection3d"

python3 -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ -f "${ROOT}/environment/requirements_aghri_pointpillars_finetune.txt" ]]; then
  python -m pip install --no-deps -r "${ROOT}/environment/requirements_aghri_pointpillars_finetune.txt"
fi

python -m pip install --no-deps -e "${MMDET3D_ROOT}"
python - <<'PY'
import importlib
for name in ("torch", "mmcv", "mmengine", "mmdet", "mmdet3d", "spconv", "numba"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
PY
