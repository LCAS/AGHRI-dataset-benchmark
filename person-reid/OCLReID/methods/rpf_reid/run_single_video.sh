#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
REID_CHECKPOINT="${REID_CHECKPOINT:-${ROOT}/checkpoints/reid/resnet18.pth}"
VIDEO=""
OUT_DIR="${OCLREID_RESULTS_ROOT:-${ROOT}/results/reproduced_runs/rpf_reid/single_video}/$(date +%Y%m%d_%H%M%S)"
BBOX_FILE=""
START_FRAME=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --bbox-file) BBOX_FILE="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${VIDEO}" && -f "${VIDEO}" ]] || { echo "--video is required" >&2; exit 2; }
mkdir -p "${OUT_DIR}"
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_single_video.py" --input "${VIDEO}" --method rpf-ReID --reid-checkpoint "${REID_CHECKPOINT}" --start_frame "${START_FRAME}" --output_json "${OUT_DIR}/predictions.json" --visualization-video "${OUT_DIR}/inference_visualization.mp4")
if [[ -n "${BBOX_FILE}" ]]; then CMD+=(--gt_bbox_file "${BBOX_FILE}"); fi
if [[ "False" == "True" ]]; then CMD+=(); fi
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi
"${CMD[@]}"
