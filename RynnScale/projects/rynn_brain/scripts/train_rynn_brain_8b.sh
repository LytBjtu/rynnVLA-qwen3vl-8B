#!/bin/bash
set -euo pipefail

export DEBUG=${DEBUG:-1}
if [[ -n "${DEBUG}" && "${DEBUG}" -eq 1 ]]; then
    export WORLD_SIZE=1
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT=${MASTER_PORT:-16666}
    export RANK=0
fi

: "${WORLD_SIZE:=1}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=16666}"
: "${RANK:=0}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

IFS=',' read -ra _CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE=${#_CUDA_DEVICES[@]}

REPO_ROOT=/home/yunteng.li/rynnVLA-qwen3vl-8B/RynnScale
export PYTHONPATH="${REPO_ROOT}:$PYTHONPATH"

# Avoid CUDA memory fragmentation OOM when DeepSpeed needs a large contiguous buffer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "RANK: ${RANK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/RynnBrain-8B}
DATA_PATH=${DATA_PATH:-${REPO_ROOT}/work_dirs/llava_sample/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/work_dirs/vla_action_head_8b}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/zero3.json}

MODEL_ARGS=(
    --model_path "${MODEL_PATH}"
    --model_type qwen3_vl
    --use_action_head True
    --action_dim 7
    --time_horizon 5
    --action_loss_weight 1.0
    --lm_loss_weight 0.0
)

FROZEN_ARGS=(
    --frozen_parameters
    '^model\.visual\..*'
    '^model\.language_model\.embed_tokens\..*'
    '^model\.language_model\.layers\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27)\..*'
)

DATA_ARGS=(
    --data_type VLADataset
    --data_path "${DATA_PATH}"
    --model_max_length 2048
    --mm_max_length 1024
    --fps 2
    --max_frames 1
    --micro_batch_size 1
    --gradient_accumulation_steps 16
    --num_train_epochs 1
)

OPTIMIZER_ARGS=(
    --learning_rate 2e-6
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
)

TRAINING_ARGS=(
    --deepspeed "${DEEPSPEED_CONFIG}"
    --gradient_checkpointing True
    --bf16 True
    --fp16 False
    --dataloader_num_workers 8
    --decoder_load_balancing False
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
    --sequence_packing False
)

LOG_ARGS=(
    --output_dir "${OUTPUT_DIR}"
    --logging_steps 1
    --report_to tensorboard
    --save_strategy steps
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes "${WORLD_SIZE}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --node_rank "${RANK}" \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m rynn_scale.api.train \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${FROZEN_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${LOG_ARGS[@]}"
