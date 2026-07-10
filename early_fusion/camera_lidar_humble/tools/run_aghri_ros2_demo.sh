#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/prabuddhi/Desktop/Early_Fus/camera_lidar_humble"
GENERIC_CONFIG="${REPO_ROOT}/config/aghri_zed_livox.yaml"
FINETUNED_CONFIG="${REPO_ROOT}/config/aghri_zed_livox_yolo11s_finetuned.yaml"

checkpoint="finetuned"
bag=""
rate="1.0"
loop="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      bag="$2"
      shift 2
      ;;
    --checkpoint)
      checkpoint="$2"
      shift 2
      ;;
    --rate)
      rate="$2"
      shift 2
      ;;
    --loop)
      loop="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${bag}" ]]; then
  echo "Usage: $0 --bag <BAG_PATH> [--checkpoint generic|finetuned] [--rate 1.0] [--loop]" >&2
  exit 2
fi

case "${checkpoint}" in
  generic)
    params_file="${GENERIC_CONFIG}"
    ;;
  finetuned)
    params_file="${FINETUNED_CONFIG}"
    ;;
  *)
    echo "--checkpoint must be 'generic' or 'finetuned'" >&2
    exit 2
    ;;
esac

ros2 launch camera_lidar_fusion aghri_test_bag_fusion_rviz.launch.py \
  bag_path:="${bag}" \
  params_file:="${params_file}" \
  playback_rate:="${rate}" \
  loop:="${loop}"
