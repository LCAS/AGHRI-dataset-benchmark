#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
REID_CHECKPOINT="${REID_CHECKPOINT:-${ROOT}/checkpoints/reid/resnet18.pth}"
RESULTS_ROOT="${OCLREID_RESULTS_ROOT:-${ROOT}/results/reproduced_runs}"

VIDEO=""; METHOD="part-OCLReID"; OUTPUT_MP4=""; OUTPUT_JSON=""; BBOX_FILE=""; START_FRAME=0; SHOW=0; DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --output) OUTPUT_MP4="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --bbox-file) BBOX_FILE="$2"; shift 2 ;;
    --start-frame) START_FRAME="$2"; shift 2 ;;
    --show-live) SHOW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${VIDEO}" && -f "${VIDEO}" ]] || { echo "--video must point to an MP4" >&2; exit 2; }
[[ -f "${REID_CHECKPOINT}" ]] || { echo "Missing checkpoint: ${REID_CHECKPOINT}" >&2; exit 1; }
OUT_DIR="${RESULTS_ROOT}/custom_video/$(date +%Y%m%d_%H%M%S)"
OUTPUT_MP4="${OUTPUT_MP4:-${OUT_DIR}/inference_visualization.mp4}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUT_DIR}/predictions.json}"
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/scripts/run_single_video.py" --input "${VIDEO}" --method "${METHOD}" --reid-checkpoint "${REID_CHECKPOINT}" --start_frame "${START_FRAME}" --output_json "${OUTPUT_JSON}" --visualization-video "${OUTPUT_MP4}")
if [[ -n "${BBOX_FILE}" ]]; then [[ -f "${BBOX_FILE}" ]] || { echo "Missing bbox file: ${BBOX_FILE}" >&2; exit 1; }; CMD+=(--gt_bbox_file "${BBOX_FILE}"); fi
if [[ "${SHOW}" -eq 1 ]]; then CMD+=(--show_result); fi
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi
mkdir -p "${OUT_DIR}"
"${CMD[@]}"
