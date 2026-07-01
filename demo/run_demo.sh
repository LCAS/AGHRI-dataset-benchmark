#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
REID_CHECKPOINT="${REID_CHECKPOINT:-${ROOT}/checkpoints/reid/resnet18.pth}"
RESULTS_ROOT="${OCLREID_RESULTS_ROOT:-${ROOT}/results/reproduced_runs}"

DRY_RUN=0
METHOD="part-OCLReID"
SHOW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --show-live) SHOW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
VIDEO="${ROOT}/demo/demo_video.mp4"
[[ -f "${VIDEO}" ]] || { echo "Missing demo video: ${VIDEO}" >&2; exit 1; }
[[ -f "${REID_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${REID_CHECKPOINT}" >&2; exit 1; }
OUT_DIR="${RESULTS_ROOT}/demo/$(date +%Y%m%d_%H%M%S)"
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_single_video.py" --input "${VIDEO}" --method "${METHOD}" --reid-checkpoint "${REID_CHECKPOINT}" --output_json "${OUT_DIR}/predictions.json" --visualization-video "${OUT_DIR}/inference_visualization.mp4")
if [[ "${SHOW}" -eq 1 ]]; then CMD+=(--show_result); fi
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi
mkdir -p "${OUT_DIR}"
"${CMD[@]}"
