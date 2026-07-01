#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
ENV_FILE="${ROOT}/environment/environment.yml"

echo "Repository: ${ROOT}"
echo "Conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null; then
  echo "Environment already exists: ${ENV_NAME}"
else
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi
conda run -n "${ENV_NAME}" pip install -r "${ROOT}/requirements.txt"
conda run -n "${ENV_NAME}" pip install -r "${ROOT}/requirements/build.txt"
conda run -n "${ENV_NAME}" pip install -e "${ROOT}"
if [[ -f "${ROOT}/mmtrack/models/orientation/setup.py" ]]; then
  conda run -n "${ENV_NAME}" pip install -e "${ROOT}/mmtrack/models/orientation"
fi
echo "Run: conda run -n ${ENV_NAME} python environment/verify_environment.py"
