#!/bin/bash

# 第1引数にログ名のベース（例: ablation_prob_attention_20250519_150000）
LOG_ID=$1
if [ -z "$LOG_ID" ]; then
  echo "Usage: $0 <log_id>"
  exit 1
fi

# ===== 共通設定 =====
LABEL_LEN=336
D_LAYERS=1
E_LAYERS=2
DES=AbProb
ITR=5
FEATURES=S
DATA=ETTh1
MODEL=informer

# ===== pred_len=336 =====
for S in 336 720 1440; do
  for A in prob full; do
    echo "Running seq_len=$S, pred_len=336, attn=$A ..."
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
      > logs/informer_${LOG_ID}_seq${S}_pred336_attn${A}.log
  done
done

# ===== pred_len=720 =====
for S in 720 1440 2880; do
  for A in prob full; do
    echo "Running seq_len=$S, pred_len=720, attn=$A ..."
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
      > logs/informer_${LOG_ID}_seq${S}_pred720_attn${A}.log
  done
done

echo "All ablation experiments finished."

