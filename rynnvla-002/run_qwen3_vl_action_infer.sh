#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_qwen3_vl_action_infer.sh \
#     /path/to/model \
#     /path/to/front.png \
#     /path/to/wrist.png \
#     "pick up the red block and place it in the tray" \
#     /path/to/output.json
#
# Positional args:
#   1: model_path
#   2: front_image
#   3: wrist_image
#   4: prompt
#   5: save_json (optional)

MODEL_PATH="${1:-/home/yunteng.li/RynnBrain/RynnBrain/RynnBrain-Plan-8B}"
FRONT_IMAGE="${2:-/home/yunteng.li/RynnBrain/RynnBrain/planning/clean.jpg}"
WRIST_IMAGE="${3:-/home/yunteng.li/RynnBrain/RynnBrain/planning/clean.jpg}"
PROMPT="${4:-pick up the red block and place it in the tray}"
SAVE_JSON="${5:-/home/yunteng.li/rynnVLA-qwen3vl-8B/rynnvla-002/output/output.json}"

ACTION_DIM="${ACTION_DIM:-6}"
TIME_HORIZON="${TIME_HORIZON:-20}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

# Optional: set visible GPU ids, e.g. "0" or "0,1". Empty means do not override.
GPU_IDS="${GPU_IDS:-0, 1, 2, 3}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  echo "Using GPUs (CUDA_VISIBLE_DEVICES): ${CUDA_VISIBLE_DEVICES}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "${SCRIPT_DIR}/run_qwen3_vl_action_infer.py" \
  --model_path "${MODEL_PATH}" \
  --front_image "${FRONT_IMAGE}" \
  --wrist_image "${WRIST_IMAGE}" \
  --prompt "${PROMPT}" \
  --action_dim "${ACTION_DIM}" \
  --time_horizon "${TIME_HORIZON}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --save_json "${SAVE_JSON}"
