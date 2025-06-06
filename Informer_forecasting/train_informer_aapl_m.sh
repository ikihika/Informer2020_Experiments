#!/bin/bash

# 自動でログIDを生成（例: aapl_M_20250530_145600）
LOG_ID="aapl_M_$(date +%Y%m%d_%H%M%S)"

# ===== 共通設定 =====
MODEL=informer
DATA=custom
FEATURES=M             # ★ここを 'M' に変更 (Multi-variant input)
TARGET=Close           # 予測対象とするメインのカラム名（例: Close）
ROOT_PATH=./data/
DATA_PATH=AAPL_M_2022-06-04_2025-06-04.csv     
FREQ=d

# ★ここが FEATURES=M の最も重要な変更点です
# 入力特徴量の数 (Close, High, Low, Open, Volume の5つ)
ENC_IN=5
DEC_IN=5
# 出力特徴量の数 (TARGET=Close なので、終値1つを予測するなら 1)
# もし全ての入力特徴量（5つ）を同時に予測したい場合は C_OUT=5 に変更してください。
C_OUT=1

DES=aapl_M_test        # ログIDと区別するためDescriptive nameを変更
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

echo "Running training on AAPL.csv with LOG ID: $LOG_ID (FEATURES=M)"

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
