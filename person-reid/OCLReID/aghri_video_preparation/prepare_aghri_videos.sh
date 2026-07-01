#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
AGHRI_DATASET_ROOT="${AGHRI_DATASET_ROOT:-/media/prabuddhi/Backup2/Updated Dataset_PW}"
AGHRI_VIDEO_ROOT="${AGHRI_VIDEO_ROOT:-/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos}"
DRY_RUN=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/aghri_video_preparation/prepare_aghri_videos.py" --dataset-root "${AGHRI_DATASET_ROOT}" --output-root "${AGHRI_VIDEO_ROOT}" "${ARGS[@]}")
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi
"${CMD[@]}"
