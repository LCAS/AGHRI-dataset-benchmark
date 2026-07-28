#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGHRI_SECOND_CLUSTER_ROOT:-/work/users/pwariyapperuma/aghri_second_finetune}"
VENV="${AGHRI_SECOND_VENV:-${ROOT}/environment/venv}"
MMDET3D_ROOT="${ROOT}/code/mmdetection3d"

python3 -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ -f "${ROOT}/environment/requirements_aghri_second_finetune.txt" ]]; then
  python -m pip install --no-deps -r "${ROOT}/environment/requirements_aghri_second_finetune.txt"
fi

python -m pip install --no-deps -e "${MMDET3D_ROOT}"
python - <<'PY'
import importlib
for name in ("torch", "mmengine", "mmdet3d"):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
PY
