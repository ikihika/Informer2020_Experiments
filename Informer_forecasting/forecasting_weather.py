import torch
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta # timedeltaを追加

# InformerのプロジェクトルートをPythonパスに追加
import sys
sys.path.append('/home/limu-pytorch/Documents/Informer_forecasting/')

from models.model import Informer
from data.data_loader import Dataset_Custom, time_features
from sklearn.preprocessing import StandardScaler

import models.model # Informerがロードされたパス確認用
print(f"Loaded Informer from: {models.model.__file__}")

# ===== 推論のための設定（train_informer_weather_M.shと一致させる） =====
class Args:
    model = 'informer'
    data = 'custom'
    root_path = './data/'
    data_path = 'weather_M_2025_06_06.csv' # 整形済みの天気データファイル名
    features = 'M'
    target = 'temperature_2m_max' # 予測対象
    freq = 'd'
    seq_len = 96
    label_len = 48
    pred_len = 24
    enc_in = 3 # 入力特徴量の数 (temperature_2m_max, temperature_2m_min, precipitation_sum)
    dec_in = 3
    c_out = 3 # 出力特徴量の数 (最高気温のみ予測なら1)
              # もし全特徴量（3つ）を予測したい場合は C_OUT=3 に変更
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    factor = 5
    padding = 0
    dropout = float(0.05)
    attn = 'prob'
    embed = 'timeF'
    activation = 'gelu'
    mix = True
    output_attention = False
    moving_avg = 25
    use_gpu = True
    gpu = 0
    devices = [0]
    itr = 1
    train_epochs = 10
    batch_size = 1
    patience = 3
    learning_rate = 0.001
    loss = 'mse'
    lradj = 'type1'
    use_wandb = False
    distil = True

args = Args()

# GPU設定
if args.use_gpu:
    args.gpu = 0
    torch.cuda.set_device(args.gpu)

# モデルのロード
# train_informer_weather_M.shを実行して生成されたチェックポイントのパスに修正してください。
# 例: checkpoints/informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_weather_M_test_weather_M_20250605_xxxxxx_0/checkpoint.pth
model_path = 'checkpoints/informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_weather_M_test_weather_M_20250606_111803_0/checkpoint.pth' # <<< ここを修正

print(f"Loading model from: {model_path}")
model = Informer(
    args.enc_in,
    args.dec_in,
    args.c_out,
    args.seq_len,
    args.label_len,
    args.pred_len,
    factor=args.factor,
    d_model=args.d_model,
    n_heads=args.n_heads,
    e_layers=args.e_layers,
    d_layers=args.d_layers,
    d_ff=args.d_ff,
    dropout=args.dropout,
    attn=args.attn,
    embed=args.embed,
    freq=args.freq,
    activation=args.activation,
    output_attention=args.output_attention,
    distil=args.distil,
    mix=args.mix,
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:' + str(args.gpu)) if args.use_gpu else torch.device('cpu')))
model.eval()
if args.use_gpu:
    model.to(f'cuda:{args.gpu}')
print("Model loaded successfully.")

# --- データロードと前処理 ---
print(f"Loading historical data from: {os.path.join(args.root_path, args.data_path)}")
df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
df_raw['date'] = pd.to_datetime(df_raw['date'])

cols_data = [col for col in df_raw.columns if col not in ['date']]
data = df_raw[cols_data].values

scaler = StandardScaler()
scaler.fit(data) # 全データでスケーラーをフィット
data_scaled = scaler.transform(data)

print(f"Loaded {len(df_raw)} data points. Preparing for prediction.")

# --- 推論用入力データの準備 ---
# 訓練データの最終点を「現在」として、そこから pred_len 分の未来を予測します。

# エンコーダへの入力 (最後の seq_len データポイント)
enc_in_data = data_scaled[-args.seq_len:]
enc_in_data_tensor = torch.tensor(enc_in_data).float().unsqueeze(0)

# デコーダへの入力 (label_len + pred_len)
dec_in_data = data_scaled[-args.label_len:]
dec_in_data_zero_padded = np.concatenate([dec_in_data, np.zeros((args.pred_len, args.enc_in))], axis=0)
dec_in_data_tensor = torch.tensor(dec_in_data_zero_padded).float().unsqueeze(0)

# 時刻特徴量の生成
enc_in_dates = df_raw['date'].iloc[-args.seq_len:].values
enc_in_time_feat = time_features(pd.to_datetime(enc_in_dates), freq=args.freq)
enc_in_time_feat_tensor = torch.tensor(enc_in_time_feat).float().unsqueeze(0)

last_known_date = df_raw['date'].iloc[-1]
# 推論の開始日付を、最終学習日付の翌日とする
future_dates_start = last_known_date + pd.Timedelta(days=1)

# ここを特定の未来日付から予測するように変更する場合（例: 2025年6月6日から）
# future_dates_start = pd.Timestamp('2025-06-06') 
# 注意: この場合、enc_in_dataとdec_in_dataは
# '2025-06-05'までの実測データで準備する必要があります。
# 現在のデータ収集は2024年7月20日までのため、
# 2025年の予測は、データにない期間の予測となります。
# 発表ではこの点を明確に伝えるか、2025年の最新データを別途入手してください。

future_dates = pd.date_range(start=future_dates_start, periods=args.pred_len, freq=args.freq)
dec_in_dates = np.concatenate([enc_in_dates[-args.label_len:], future_dates.values], axis=0)
dec_in_time_feat = time_features(pd.to_datetime(dec_in_dates), freq=args.freq)
dec_in_time_feat_tensor = torch.tensor(dec_in_time_feat).float().unsqueeze(0)

if args.use_gpu:
    enc_in_data_tensor = enc_in_data_tensor.cuda()
    dec_in_data_tensor = dec_in_data_tensor.cuda()
    enc_in_time_feat_tensor = enc_in_time_feat_tensor.cuda()
    dec_in_time_feat_tensor = dec_in_time_feat_tensor.cuda()

with torch.no_grad():
    outputs = model(enc_in_data_tensor, enc_in_time_feat_tensor,
                    dec_in_data_tensor, dec_in_time_feat_tensor)

# --- 予測結果の逆正規化 ---
if args.c_out == 1:
    dummy_output_data = np.zeros((args.pred_len, args.enc_in))
    target_idx = cols_data.index(args.target)
    dummy_output_data[:, target_idx] = outputs.cpu().numpy().squeeze(-1)
    
    predicted_data_original_scale = scaler.inverse_transform(dummy_output_data)
    predicted_target_values = predicted_data_original_scale[:, target_idx]
else: # args.c_out == args.enc_in の場合
    predicted_data_original_scale = scaler.inverse_transform(outputs.cpu().numpy().squeeze(0))
    # 複数特徴量を予測している場合、ここではターゲットとして指定されたもののみ取り出す
    predicted_target_values = predicted_data_original_scale[:, cols_data.index(args.target)]


# --- 予測結果の表示とファイル保存 ---
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"weather_prediction_results_{current_time_str}.txt"
output_filepath = os.path.join("./Prediction_results", output_filename)

os.makedirs("./Prediction_results", exist_ok=True)

print(f"\n--- Prediction Results for the Next {args.pred_len} Days (from {future_dates_start.strftime('%Y-%m-%d')}) ---")

output_lines = []
output_lines.append(f"--- Prediction Results for the Next {args.pred_len} Days (from {future_dates_start.strftime('%Y-%m-%d')}) ---")

for i, pred_date in enumerate(future_dates):
    line = ""
    if args.c_out == 1:
        line = f"Date: {pred_date.strftime('%Y-%m-%d')}, Predicted {args.target}: {predicted_target_values[i]:.2f}"
    else:
        # ここで全ての予測特徴量を表示するように変更
        predicted_values_for_date = predicted_data_original_scale[i]
        line = f"Date: {pred_date.strftime('%Y-%m-%d')}"
        for j, col_name in enumerate(cols_data):
            # args.target が 'temperature_2m_max' なので、それ以外の列も表示するとユーザーは混乱するかも。
            # ここではシンプルにターゲットのみ表示を継続するか、明示的に全列表示するか選択
            line += f", Predicted {col_name}: {predicted_values_for_date[j]:.2f}"
        # line = line.strip() # 最後の改行を削除
    print(line)
    output_lines.append(line)

final_message_1 = "\nInference complete. Please note that these are model predictions for the future, and actual values are unknown."
final_message_2 = "The accuracy of these true future predictions can only be verified when the actual data becomes available."

print(final_message_1)
print(final_message_2)

output_lines.append(final_message_1)
output_lines.append(final_message_2)

with open(output_filepath, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"\nPrediction results saved to: {output_filepath}")