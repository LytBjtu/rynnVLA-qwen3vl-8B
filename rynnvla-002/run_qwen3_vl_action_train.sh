#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL + ActionHead joint training launcher.
#
# Usage:
#   bash run_qwen3_vl_action_train.sh \
#     /path/to/qwen3vl_model \
#     /path/to/train.yaml \
#     /path/to/val_ind.yaml \
#     /path/to/val_ood.yaml \
#     /path/to/output_dir
#
# Positional args:
#   1: qwen_model_path
#   2: data_config_train
#   3: data_config_val_ind
#   4: data_config_val_ood
#   5: output_dir

QWEN_MODEL_PATH="${1:?Please provide qwen model path}"
DATA_CONFIG_TRAIN="${2:?Please provide train yaml}"
DATA_CONFIG_VAL_IND="${3:?Please provide val_ind yaml}"
DATA_CONFIG_VAL_OOD="${4:?Please provide val_ood yaml}"
OUTPUT_DIR="${5:?Please provide output dir}"

# Distributed settings
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-16666}"
NODE_RANK="${NODE_RANK:-0}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.03}"
LR="${LR:-8e-6}"
MIN_LR="${MIN_LR:-8e-6}"
WD="${WD:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Model/sequence settings
ACTION_DIM="${ACTION_DIM:-7}"
TIME_HORIZON="${TIME_HORIZON:-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
RESOLUTION="${RESOLUTION:-256}"
PRECISION="${PRECISION:-bf16}"

# Data pipeline toggles
PREPROCESS="${PREPROCESS:-false}"          # true: pretokenized data; false: on-the-fly build
WITH_STATE="${WITH_STATE:-1}"              # 1/0
WITH_WRIST="${WITH_WRIST:-1}"              # 1/0
WITH_WORLD_MODEL="${WITH_WORLD_MODEL:-0}"  # 1/0

# Optional
TOKENIZER_PATH="${TOKENIZER_PATH:-../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af}"
RESUME_PATH="${RESUME_PATH:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  torchrun
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  --node_rank="${NODE_RANK}"
  pretrain_solver_awm_w_ck_action_head.py

  --disable_length_clustering
  --train_only True

  --vlm_arch qwen3_vl
  --qwen_model_path "${QWEN_MODEL_PATH}"
  --init_from "${QWEN_MODEL_PATH}"

  --with_action
  --resolution "${RESOLUTION}"
  --preprocess "${PREPROCESS}"

  --tokenizer_path "${TOKENIZER_PATH}"
  --batch_size "${BATCH_SIZE}"
  --accum_iter "${ACCUM_ITER}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --lr "${LR}"
  --min_lr "${MIN_LR}"
  --wd "${WD}"
  --clip_grad "${CLIP_GRAD}"
  --action_dim "${ACTION_DIM}"
  --time_horizon "${TIME_HORIZON}"
  --data_config_train "${DATA_CONFIG_TRAIN}"
  --data_config_val_ind "${DATA_CONFIG_VAL_IND}"
  --data_config_val_ood "${DATA_CONFIG_VAL_OOD}"
  --num_workers "${NUM_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  --checkpointing
  --max_seq_len "${MAX_SEQ_LEN}"
  --precision "${PRECISION}"
)

if [[ "${WITH_STATE}" == "1" ]]; then
  CMD+=(--with_state)
fi

if [[ "${WITH_WRIST}" == "1" ]]; then
  CMD+=(--with_wrist)
fi

if [[ "${WITH_WORLD_MODEL}" == "1" ]]; then
  CMD+=(--with_world_model)
fi

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume_path "${RESUME_PATH}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARRAY[@]}")
fi

printf 'Launching command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
