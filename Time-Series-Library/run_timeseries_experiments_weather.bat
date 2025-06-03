@echo off
setlocal enabledelayedexpansion

set MODEL=Transformer
set PREDS=96

if not exist logs (
    mkdir logs
)

for %%P in (%PREDS%) do (
    echo Running model=%MODEL% with pred_len=%%P on weather...
    python -u run.py ^
        --task_name long_term_forecast ^
        --is_training 1 ^
        --root_path ./dataset/weather/ ^
        --data_path weather.csv ^
        --model_id weather_96_%%P ^
        --model %MODEL% ^
        --data custom ^
        --features M ^
        --seq_len 96 ^
        --label_len 48 ^
        --pred_len %%P ^
        --e_layers 2 ^
        --d_layers 1 ^
        --factor 3 ^
        --enc_in 21 ^
        --dec_in 21 ^
        --c_out 21 ^
        --des Exp ^
        --itr 1 ^
        > logs/%MODEL%_weather_%%P.txt
)

echo All weather experiments completed.
pause