#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-oclreid}"
PREDICTIONS=""; ANNOTATIONS=""; MANIFEST=""; TARGET_CLASS=""; INIT_FRAME=""; FPS=""; OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions) PREDICTIONS="$2"; shift 2 ;;
    --annotations) ANNOTATIONS="$2"; shift 2 ;;
    --frame-manifest) MANIFEST="$2"; shift 2 ;;
    --target-class) TARGET_CLASS="$2"; shift 2 ;;
    --initialization-video-frame) INIT_FRAME="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
CMD=(conda run -n "${ENV_NAME}" python "${ROOT}/scripts/evaluate_aghri_results.py" --predictions "${PREDICTIONS}" --annotations "${ANNOTATIONS}" --frame_manifest "${MANIFEST}" --target_class "${TARGET_CLASS}" --initialization_video_frame "${INIT_FRAME}" --fps "${FPS}" --output_dir "${OUTPUT_DIR}")
printf 'Command:'; printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"
