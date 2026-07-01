#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
REID_CHECKPOINT="${REID_CHECKPOINT:-${ROOT}/checkpoints/reid/resnet18.pth}"
AGHRI_DATASET_ROOT="${AGHRI_DATASET_ROOT:-/media/prabuddhi/Backup2/Updated Dataset_PW}"
AGHRI_VIDEO_ROOT="${AGHRI_VIDEO_ROOT:-/media/prabuddhi/Backup2/AGHRI_OCLReID_Videos}"
RESULTS_ROOT="${OCLREID_RESULTS_ROOT:-${ROOT}/results/reproduced_runs/gated_part_oclreid}"
DRY_RUN=0
MAX_RUNS=()
OVERWRITE=()
SAVE_VIS=()
SHOW_LIVE=()
TARGET_CLASSES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --max-runs) MAX_RUNS=(--max-runs "$2"); shift 2 ;;
    --target-classes)
      shift
      TARGET_CLASSES=(--target-classes)
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TARGET_CLASSES+=("$1")
        shift
      done
      ;;
    --overwrite) OVERWRITE=(--overwrite); shift ;;
    --save-visualizations) SAVE_VIS=(--save-visualizations); shift ;;
    --show-live) SHOW_LIVE=(--show-live); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_aghri_experiments.py" --splits test --methods part-OCLReID --dataset-root "${AGHRI_DATASET_ROOT}" --video-root "${AGHRI_VIDEO_ROOT}" --results-root "${RESULTS_ROOT}" --reid-checkpoint "${REID_CHECKPOINT}" "${MAX_RUNS[@]}" "${TARGET_CLASSES[@]}" "${OVERWRITE[@]}" "${SAVE_VIS[@]}" "${SHOW_LIVE[@]}")
if [[ "True" == "True" ]]; then CMD+=(--association-mode reid_gate --association-reid-threshold 0.60 --association-reid-margin 0.02 --association-min-bbox-score 0.0 --association-min-visible-parts 1); fi
if [[ "${DRY_RUN}" -eq 1 ]]; then CMD+=(--dry-run); fi
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi
"${CMD[@]}"
