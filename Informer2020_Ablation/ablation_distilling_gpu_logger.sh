#!/bin/bash

# 第1引数にログ名のベース（例: ablation_prob_attention_20250519_150000）
LOG_ID=$1
if [ -z "$LOG_ID" ]; then
  echo "Usage: $0 <log_id>"
  exit 1
fi

# ===== 共通設定 =====
MODEL=informer
DATA=ETTh1
FEATURES=M
TARGET=OT
ROOT_PATH=./data/ETT
DATA_PATH=ETTh1.csv
FREQ=h
ENC_IN=7
DEC_IN=7
C_OUT=7
DES=exp_distil_ablation
ITR=1
TRAIN_EPOCHS=10
BATCH_SIZE=32
PATIENCE=3
LR=0.0001
LOSS=mse
LRADJ=type1
USE_GPU=True
GPU=0
USE_MULTI_GPU=False
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

# ===== ログ保存用フォルダ作成 =====
mkdir -p logs

# ===== 各予測長に対して、シーケンス長と distil モードをループ =====
for P in 336 480; do
  for S in 336 480 720 960 1200; do
    LABEL_LEN=$((S / 2))

    for D in with_distil no_distil; do
      if [ "$D" == "with_distil" ]; then
        DISTIL_FLAG=""
        DES_TAG="with_distil"
      else
        DISTIL_FLAG="--distil"
        DES_TAG="no_distil"
      fi

      echo "Running: pred_len=$P, seq_len=$S, distil=$D"
      LOG_NAME=logs/informer_${LOG_ID}_pl${P}_sl${S}_${D}.txt

      python -u main_informer.py \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --features $FEATURES \
        --target $TARGET \
        --freq $FREQ \
        --seq_len $S \
        --label_len $LABEL_LEN \
        --pred_len $P \
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
        $DISTIL_FLAG \
        --attn $ATTN \
        --embed $EMBED \
        --activation $ACTIVATION \
        --do_predict \
        --itr $ITR \
        --des ${DES}_${DES_TAG}_pl${P}_sl${S} \
        --train_epochs $TRAIN_EPOCHS \
        --batch_size $BATCH_SIZE \
        --patience $PATIENCE \
        --learning_rate $LR \
        --loss $LOSS \
        --lradj $LRADJ \
        --use_gpu $USE_GPU \
        --gpu $GPU \
        --num_workers $NUM_WORKERS > $LOG_NAME

    done
  done
done

echo "All distilling ablation experiments completed."
