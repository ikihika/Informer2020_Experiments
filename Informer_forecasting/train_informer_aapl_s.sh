#!/bin/bash

# 自動でログIDを生成（例: aapl_20250530_145600）
LOG_ID="aapl_$(date +%Y%m%d_%H%M%S)"

# ===== 共通設定 =====
MODEL=informer
DATA=custom
FEATURES=S
TARGET=Close            # AAPL_S.csvのカラム名（例: Close）
ROOT_PATH=./data/
DATA_PATH=AAPL_S.csv
FREQ=d
ENC_IN=1
DEC_IN=1
C_OUT=1
DES=aapl_test
ITR=1
TRAIN_EPOCHS=10
BATCH_SIZE=16
PATIENCE=3
LR=0.001
LOSS=mse
LRADJ=type1
USE_GPU=True
GPU=0
NUM_WORKERS=4
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=2048
FACTOR=5
PADDING=0
ATTN=prob
EMBED=timeF
ACTIVATION=gelu

# シーケンス設定
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=24

# ログ保存用フォルダ作成
mkdir -p logs

echo "Running training on AAPL.csv with log ID: $LOG_ID"

python -u main_informer.py \
  --model $MODEL \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --target $TARGET \
  --freq $FREQ \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --enc_in $ENC_IN \
  --dec_in $DEC_IN \
  --c_out $C_OUT \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_layers $D_LAYERS \
  --d_ff $D_FF \
  --factor $FACTOR \
  --padding $PADDING \
  --dropout 0.05 \
  --attn $ATTN \
  --embed $EMBED \
  --activation $ACTIVATION \
  --itr $ITR \
  --des ${DES}_${LOG_ID} \
  --train_epochs $TRAIN_EPOCHS \
  --batch_size $BATCH_SIZE \
  --patience $PATIENCE \
  --learning_rate $LR \
  --loss $LOSS \
  --lradj $LRADJ \
  --use_gpu $USE_GPU \
  --gpu $GPU \
  --num_workers $NUM_WORKERS \
  > logs/train_log_${LOG_ID}.txt

echo "Training complete. Log saved to logs/train_log_${LOG_ID}.txt"

