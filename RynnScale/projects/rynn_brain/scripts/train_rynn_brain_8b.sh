#!/bin/bash
export DEBUG=${DEBUG:-1}
if [[ -n $DEBUG && $DEBUG -eq 1 ]]; then
    WORLD_SIZE=1
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=${MASTER_PORT:-16666}
    RANK=0
fi

: "${CUDA_VISIBLE_DEVICES:=3,5,6,7}"
IFS=',' read -ra _CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE=${#_CUDA_DEVICES[@]}

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
export PYTHONPATH="/home/yunteng.li/RynnScale:$PYTHONPATH"

# Avoid CUDA memory fragmentation OOM when DeepSpeed needs a large contiguous buffer
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE

MODEL_PATH=/home/yunteng.li/RynnScale/RynnBrain-8B
OUTPUT_DIR=/home/yunteng.li/RynnScale/output_dir
DATA_PATH=/tmp/yunteng.li/llava_sample/train.jsonl

FROZEN_ARGS=(
    --frozen_parameters
    '^model\.visual\..*'
    '^model\.language_model\.embed_tokens\..*'
    '^model\.language_model\.layers\.(0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27)\..*'
)

DATA_ARGS=(
    --data_type VLMDataset
    --data_path $DATA_PATH
    --model_max_length 2048
    --mm_max_length 1024
    --fps 2
    --max_frames 1
    --micro_batch_size 1
    --gradient_accumulation_steps 16
)

OPTIMIZER_ARGS=(
    --learning_rate 2e-6
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed /home/yunteng.li/RynnScale/configs/zero2.json
    --gradient_checkpointing True
    --bf16 True
    --fp16 False
    --dataloader_num_workers 8
    --decoder_load_balancing False
    --loss_reduction_scope sequence
    --average_tokens_across_devices True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --logging_steps 1
    --report_to tensorboard
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    --rdzv_conf="timeout=7200,join_timeout=7200" \
    -m rynn_scale.api.train \
    --model_path $MODEL_PATH \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${FROZEN_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOG_ARGS[@]}