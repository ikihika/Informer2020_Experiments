#!/bin/bash

# 自動でログIDを生成（例: weather_M_20250605_145600）
LOG_ID="weather_M_$(date +%Y%m%d_%H%M%S)"

# ===== 共通設定 =====
MODEL=informer
DATA=custom
FEATURES=M             # Multi-variant input
TARGET=temperature_2m_max # 予測対象とするメインのカラム名 (例: 最高気温)
                        # もし全ての天候特徴量を予測したい場合は、inference_weather.pyでC_OUTを変更し、
                        # ここは予測対象の一つを指定し続けるか、空にするか、exp_main.pyの仕様に従う
ROOT_PATH=./data/
DATA_PATH=weather_M_2025_06_06.csv # 整形済みの天気データファイル名
FREQ=d                 # 日次データ

# 入力特徴量の数 (temperature_2m_max, temperature_2m_min, precipitation_sum の3つ)
ENC_IN=3
DEC_IN=3
# 出力特徴量の数 (TARGET=temperature_2m_max なので、1つを予測するなら 1)
# もしENC_INと同じ3つの特徴量を全て同時に予測したい場合は C_OUT=3 に変更してください。
C_OUT=3

DES=weather_M_test     # ログIDと区別するためDescriptive nameを変更
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
E_LAYERS=2             # 訓練ログからELAYERS=2, DLAYERS=1が確認されているため、これを維持
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
PRED_LEN=24            # 予測する未来の長さ（例: 24日分）

# ログ保存用フォルダ作成
mkdir -p logs

echo "Running training on weather_M.csv with LOG ID: $LOG_ID (FEATURES=M)"

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
