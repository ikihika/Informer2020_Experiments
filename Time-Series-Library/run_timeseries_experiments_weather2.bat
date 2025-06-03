@echo off
setlocal enabledelayedexpansion

:: モデルと予測長の設定
set MODELS=Informer Reformer
set PREDS=96 192 336

:: logs フォルダがなければ作成
if not exist logs (
    mkdir logs
)

:: モデルごと、予測長ごとにループ
for %%M in (%MODELS%) do (
    for %%P in (%PREDS%) do (
        echo Running model=%%M with pred_len=%%P on weather...
        python -u run.py ^
            --task_name long_term_forecast ^
            --is_training 1 ^
            --root_path ./dataset/weather/ ^
            --data_path weather.csv ^
            --model_id weather_96_%%P ^
            --model %%M ^
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
            > logs/%%M_weather_%%P.txt
    )
)

echo All weather experiments completed.
pause
