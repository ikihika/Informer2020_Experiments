#!/bin/bash

# ===== 共通設定 =====
LABEL_LEN=336
D_LAYERS=1
E_LAYERS=2
DES=AbProb
ITR=5
FEATURES=S
DATA=ETTh1
MODEL=informer

# ===== seq_len=2880 で pred_len=720 =====
for A in prob full; do
  echo "Running seq_len=2880, pred_len=720, attn=$A ..."
  python -u main_informer.py \
    --model $MODEL \
    --data $DATA \
    --features $FEATURES \
    --seq_len 2880 \
    --label_len $LABEL_LEN \
    --pred_len 720 \
    --e_layers $E_LAYERS \
    --d_layers $D_LAYERS \
    --attn $A \
    --des $DES \
    --itr $ITR \
    > logs/${MODEL}_seq2880_pred720_attn${A}.log
done

echo "Experiment with seq_len=2880 finished."

