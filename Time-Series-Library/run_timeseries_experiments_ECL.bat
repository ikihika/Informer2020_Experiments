@echo off
setlocal enabledelayedexpansion

set MODELS=Informer Reformer Transformer
set PREDS=96 192 336

if not exist logs (
    mkdir logs
)

for %%M in (%MODELS%) do (
    for %%P in (%PREDS%) do (
        echo Running model=%%M with pred_len=%%P on electricity...
        python -u run.py ^
            --task_name long_term_forecast ^
            --is_training 1 ^
            --root_path ./dataset/electricity/ ^
            --data_path electricity.csv ^
            --model_id ECL_96_%%P ^
            --model %%M ^
            --data custom ^
            --features M ^
            --seq_len 96 ^
            --label_len 48 ^
            --pred_len %%P ^
            --e_layers 2 ^
            --d_layers 1 ^
            --factor 3 ^
            --enc_in 321 ^
            --dec_in 321 ^
            --c_out 321 ^
            --des Exp ^
            --itr 1 ^
            > logs/%%M_ECL_96_%%P.txt
    )
)

echo All electricity experiments completed.
pause