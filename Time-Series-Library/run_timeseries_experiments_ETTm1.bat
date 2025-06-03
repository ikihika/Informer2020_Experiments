@echo off
setlocal enabledelayedexpansion

set MODELS=Informer Reformer Transformer
set PREDS=96 192 336

if not exist logs (
    mkdir logs
)

for %%M in (%MODELS%) do (
    for %%P in (%PREDS%) do (
        echo Running model=%%M with pred_len=%%P on ETTm1...
        python -u run.py ^
            --task_name long_term_forecast ^
            --is_training 1 ^
            --root_path ./dataset/ETT-small/ ^
            --data_path ETTm1.csv ^
            --model_id ETTm1_%%P ^
            --model %%M ^
            --data ETTm1 ^
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
            > logs/%%M_ETTm1_%%P.txt
    )
)

echo All ETTm1 experiments completed.
pause