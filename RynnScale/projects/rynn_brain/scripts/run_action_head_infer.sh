#!/bin/bash
set -euo pipefail

REPO_ROOT=/home/yunteng.li/rynnVLA-qwen3vl-8B/RynnScale
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:$PYTHONPATH"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_ROOT}/work_dirs/vla_action_head_8b/checkpoint-2}
FRONT_IMAGE=${FRONT_IMAGE:-/path/to/front.jpg}
WRIST_IMAGE=${WRIST_IMAGE:-/path/to/wrist.jpg}
PROMPT=${PROMPT:-"What action should the robot take to grab the block?"}
STATE=${STATE:-"[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]"}
DEVICE=${DEVICE:-cuda:0}
DEVICE_MAP=${DEVICE_MAP:-}
OUTPUT_JSON=${OUTPUT_JSON:-/tmp/action_pred.json}
DTYPE=${DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
MM_MAX_LENGTH=${MM_MAX_LENGTH:-1024}
FPS=${FPS:-2}
MAX_FRAMES=${MAX_FRAMES:-1}

PY_ARGS=(
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --front_image "${FRONT_IMAGE}"
  --wrist_image "${WRIST_IMAGE}"
  --prompt "${PROMPT}"
  --state "${STATE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --mm_max_length "${MM_MAX_LENGTH}"
  --fps "${FPS}"
  --max_frames "${MAX_FRAMES}"
  --output_json "${OUTPUT_JSON}"
)

if [[ -n "${DEVICE_MAP}" ]]; then
  PY_ARGS+=(--device_map "${DEVICE_MAP}")
fi

python projects/rynn_brain/scripts/run_action_head_infer.py \
  "${PY_ARGS[@]}"
