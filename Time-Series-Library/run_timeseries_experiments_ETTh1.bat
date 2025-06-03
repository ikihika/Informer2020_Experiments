@echo off
setlocal enabledelayedexpansion

REM モデルと予測長の設定
set MODELS=Informer Reformer Transformer
set PREDS=96 192 336

REM ログ保存ディレクトリ
if not exist logs (
    mkdir logs
)

REM 各実験の実行
for %%M in (%MODELS%) do (
    for %%P in (%PREDS%) do (
        echo Running model=%%M with pred_len=%%P ...
        python -u run.py ^
            --task_name long_term_forecast ^
            --is_training 1 ^
            --root_path ./dataset/ETT-small/ ^
            --data_path ETTh1.csv ^
            --model_id ETTh1_%%P ^
            --model %%M ^
            --data ETTh1 ^
            --features M ^
            --seq_len 96 ^
            --label_len 48 ^
            --pred_len %%P ^
            --e_layers 2 ^
            --d_layers 1 ^
            --factor 3 ^
            --enc_in 7 ^
            --dec_in 7 ^
            --c_out 7 ^
            --des Exp ^
            --itr 1 ^
            > logs/%%M_ETTh1_%%P.txt
    )
)

echo All experiments completed.
pause