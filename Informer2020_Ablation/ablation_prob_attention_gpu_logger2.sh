#!/bin/bash

LOG_ID=$1
if [ -z "$LOG_ID" ]; then
  echo "Usage: $0 <log_id>"
  exit 1
fi

LABEL_LEN=336
D_LAYERS=1
E_LAYERS=2
DES=AbProb
ITR=5
FEATURES=S
DATA=ETTh1
MODEL=informer

mkdir -p logs/gpu_logs

function start_gpu_logger() {
  local out_file=$1
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$out_file" &
  echo $!
}

function stop_gpu_logger() {
  local pid=$1
  kill "$pid"
}

# pred_len=336
for S in 336 720 1440; do
  for A in prob full; do
    echo "Running seq_len=$S, pred_len=336, attn=$A ..."

    GPU_LOG="logs/gpu_logs/${LOG_ID}_seq${S}_pred336_attn${A}_gpu.csv"
    LOG_FILE="logs/informer_${LOG_ID}_seq${S}_pred336_attn${A}.log"

    GPU_LOG_PID=$(start_gpu_logger "$GPU_LOG")
    sleep 2

    python -u main_informer.py \
      --model $MODEL \
      --data $DATA \
      --features $FEATURES \
      --seq_len $S \
      --label_len $LABEL_LEN \
      --pred_len 336 \
      --e_layers $E_LAYERS \
      --d_layers $D_LAYERS \
      --attn $A \
      --des $DES \
      --itr $ITR \
      > "$LOG_FILE"

    stop_gpu_logger "$GPU_LOG_PID"
    sleep 2
  done
done

# pred_len=720
for S in 720 1440 2880; do
  for A in prob full; do
    echo "Running seq_len=$S, pred_len=720, attn=$A ..."

    GPU_LOG="logs/gpu_logs/${LOG_ID}_seq${S}_pred720_attn${A}_gpu.csv"
    LOG_FILE="logs/informer_${LOG_ID}_seq${S}_pred720_attn${A}.log"

    GPU_LOG_PID=$(start_gpu_logger "$GPU_LOG")
    sleep 2

    python -u main_informer.py \
      --model $MODEL \
      --data $DATA \
      --features $FEATURES \
      --seq_len $S \
      --label_len $LABEL_LEN \
      --pred_len 720 \
      --e_layers $E_LAYERS \
      --d_layers $D_LAYERS \
      --attn $A \
      --des $DES \
      --itr $ITR \
      > "$LOG_FILE"

    stop_gpu_logger "$GPU_LOG_PID"
    sleep 2
  done
done

echo "All ablation experiments finished."
