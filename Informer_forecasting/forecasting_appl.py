import torch
import pandas as pd
import numpy as np
import os
import joblib # スケーラーの保存・ロード用
from datetime import datetime # ★追加S

# InformerのプロジェクトルートをPythonパスに追加（必要に応じて）
# これは、main_informer.pyなどが実行される場所によって変わります
import sys
# もしdata/data_loader.pyなどがimportできない場合、以下の行のコメントを解除し、パスを調整してください
sys.path.append('/home/limu-pytorch/Documents/Informer_forecasting/')
print(f"Current sys.path: {sys.path}") # 追加されたパスを確認

from models.model import Informer # Informerモデルの定義
from data.data_loader import Dataset_Custom, time_features # データセットと時刻特徴量生成
from sklearn.preprocessing import StandardScaler, MinMaxScaler # スケーラー

# どのInformerクラスがロードされたか確認
# Informerクラスが定義されているモジュール（models.model）のファイルパスを取得します
import models.model # models.model モジュールをインポート
print(f"Loaded Informer from: {models.model.__file__}")

# models/embed.py の PositionalEmbedding も確認
# from models.embed import PositionalEmbedding, DataEmbedding # 明示的にインポートしてみる
# print(f"Loaded PositionalEmbedding from: {PositionalEmbedding.__module__} at {PositionalEmbedding.__file__}")

# ===== 推論のための設定（train_informer_aapl_M.shと一致させる） =====
# これらの設定は、訓練スクリプト (train_informer_aapl_M.sh) の設定と完全に一致させる必要があります。
# 訓練時と同じモデルの構造、入力/出力次元、シーケンス長、ハイパーパラメータなどです。
class Args:
    model = 'informer'
    data = 'custom'
    root_path = './data/'
    data_path = 'AAPL_M_2022-06-04_2025-06-04.csv' # FEATURES=M 用に整形されたCSVファイル名
    features = 'M' # Multi-variant input
    target = 'Close' # 予測対象
    freq = 'd' # 日次データ
    seq_len = 96 # エンコーダ入力シーケンス長
    label_len = 48 # デコーダ入力シーケンス長（真値部分）
    pred_len = 24 # 予測する未来の長さ（今回は24日分）
    enc_in = 5 # 入力特徴量の数 (Close, High, Low, Open, Volume)
    dec_in = 5 # デコーダ入力特徴量の数
    c_out = 1 # 出力特徴量の数 (Closeのみ予測なら1)
    d_model = 512 # ★★★この値が正しく渡されているか確認が必要です★★★
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
    mix = True # Informerのattn mix
    output_attention = False # Attentionマップの出力
    moving_avg = 25 # Informerの移動平均レイヤー用
    use_gpu = True
    gpu = 0
    devices = [0]
    itr = 1 # ダミー値、推論では関係ない
    train_epochs = 10 # ダミー値
    batch_size = 1 # 推論時は1でOK
    patience = 3 # ダミー値
    learning_rate = 0.001 # ダミー値
    loss = 'mse' # ダミー値
    lradj = 'type1' # ダミー値
    use_wandb = False # 必要に応じて変更
    distil = True

args = Args()

# GPU設定
if args.use_gpu:
    args.gpu = 0
    torch.cuda.set_device(args.gpu)

# モデルのロード
# 訓練時に保存されたチェックポイントのパスを指定してください。
# 例: checkpoints/informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc5_ebtimeF_dtTrue_mxTrue_aapl_M_test_aapl_M_20250530_xxxxxx_0/checkpoint.pth
# train_informer_aapl_M.sh の実行ログから正確なパスを見つけることができます。
# 通常、desパラメータとLOG_IDがディレクトリ名に含まれます。
model_path = 'checkpoints/informer_custom_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_aapl_M_test_aapl_M_20250605_154155_0/checkpoint.pth' # <<< ここをあなたのパスに修正

print(f"Loading model from: {model_path}")
model = Informer(
    args.enc_in,
    args.dec_in,
    args.c_out,
    args.seq_len,
    args.label_len,
    args.pred_len,
    # ここから、Informerの__init__の引数名と一致するように明示的にキーワード引数で渡します。
    # こうすることで、順番の間違いによる問題を防ぎます。
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
    # Informerの__init__に 'mode' 引数が追加されているようですが、Argsにはありません。
    # train_informer_aapl_M.sh で訓練した際の 'mode' のデフォルト値を確認し、
    # 必要であれば Args に追加するか、Informer のデフォルト値に任せてください。
    # もし訓練時に 'mode' が指定されていなければ、この行は不要です。
    # mode='NAR', # もし必要ならこの行を追加
    # deviceはInformerの__init__でデフォルト値がcuda:0になっているので、通常は指定不要です。
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:' + str(args.gpu)) if args.use_gpu else torch.device('cpu')))
model.eval() # 推論モードに設定
if args.use_gpu:
    model.to(f'cuda:{args.gpu}')

print("Model loaded successfully.")

# --- データロードと前処理 ---
# 訓練に使用した全歴史データをロードし、スケーラーをfitします。
# 理想的には、訓練時に使われたスケーラーを保存・ロードすべきです。
print(f"Loading historical data from: {os.path.join(args.root_path, args.data_path)}")
df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
df_raw['date'] = pd.to_datetime(df_raw['date']) # 日付列をdatetime型に変換

# 予測に使う特徴量を選択
# 'date'列は時刻特徴量生成に使うので除外します。
cols_data = [col for col in df_raw.columns if col not in ['date']]
data = df_raw[cols_data].values

# スケーラーの初期化とfit
# 訓練時と同じ種類のスケーラーを使用すること（例: StandardScaler or MinMaxScaler）
# Informerの公式実装ではStandardScalerがよく使われます
scaler = StandardScaler()
# 全データでfit
scaler.fit(data)
# データを正規化
data_scaled = scaler.transform(data)

print(f"Loaded {len(df_raw)} data points. Preparing for prediction.")

# --- 推論用入力データの準備 ---
# 訓練データの最終点を「現在」として、そこから pred_len 分の未来を予測します。
# data_scaledの最後のseq_len_window分をエンコーダへの入力とします。

# エンコーダへの入力 (最後の seq_len データポイント)
enc_in_data = data_scaled[-args.seq_len:]
enc_in_data_tensor = torch.tensor(enc_in_data).float().unsqueeze(0) # unsqueeze(0)でバッチ次元を追加

# デコーダへの入力 (label_len + pred_len)
# デコーダの真値部分は、エンコーダ入力の最後のlabel_len部分
# 予測部分はゼロ埋め
dec_in_data = data_scaled[-args.label_len:]
# 予測期間分のゼロ埋めを追加
dec_in_data_zero_padded = np.concatenate([dec_in_data, np.zeros((args.pred_len, args.enc_in))], axis=0)
dec_in_data_tensor = torch.tensor(dec_in_data_zero_padded).float().unsqueeze(0)

# 時刻特徴量の生成
# エンコーダの時刻特徴量 (最後のseq_lenの日付)
enc_in_dates = df_raw['date'].iloc[-args.seq_len:].values
enc_in_time_feat = time_features(pd.to_datetime(enc_in_dates), freq=args.freq)
enc_in_time_feat_tensor = torch.tensor(enc_in_time_feat).float().unsqueeze(0)

# デコーダの時刻特徴量 (最後のlabel_lenの日付 + 未来のpred_lenの日付)
# 未来の日付を生成
last_known_date = df_raw['date'].iloc[-1]
future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=args.pred_len, freq=args.freq)
dec_in_dates = np.concatenate([enc_in_dates[-args.label_len:], future_dates.values], axis=0) # label_len + pred_len の日付
dec_in_time_feat = time_features(pd.to_datetime(dec_in_dates), freq=args.freq)
dec_in_time_feat_tensor = torch.tensor(dec_in_time_feat).float().unsqueeze(0)

# GPUへ転送
if args.use_gpu:
    enc_in_data_tensor = enc_in_data_tensor.cuda()
    dec_in_data_tensor = dec_in_data_tensor.cuda()
    enc_in_time_feat_tensor = enc_in_time_feat_tensor.cuda()
    dec_in_time_feat_tensor = dec_in_time_feat_tensor.cuda()

# --- 予測実行 ---
with torch.no_grad():
    outputs = model(enc_in_data_tensor, enc_in_time_feat_tensor,
                    dec_in_data_tensor, dec_in_time_feat_tensor)

# --- 予測結果の逆正規化 ---
# モデルの出力は正規化された値なので、元のスケールに戻します。
# モデルの出力は (batch_size, pred_len, c_out) の形をしています。
# scaler.inverse_transform は元のデータ形式 (featuresの数) を期待するので、
# 出力だけを逆変換できるように調整が必要です。

# c_out が 1 の場合 (TARGET=Closeのみ予測)
if args.c_out == 1:
    # outputs: (1, pred_len, 1)
    # scaler.inverse_transformは入力と同じ次元の出力を持つため、元の特徴量数に合わせる
    # 予測されたターゲット列以外の部分には、最後の既知の値をコピーして埋めます
    dummy_output_data = np.zeros((args.pred_len, args.enc_in))
    # ターゲット列（例: Close）のインデックスを特定
    target_idx = cols_data.index(args.target)
    # 修正前: dummy_output_data[:, target_idx] = outputs.cpu().numpy().squeeze(0) # outputsは(pred_len, 1)
    dummy_output_data[:, target_idx] = outputs.cpu().numpy().squeeze(-1) # outputsは(pred_len, 1)から(pred_len,)へ
    
    # 逆正規化
    predicted_data_original_scale = scaler.inverse_transform(dummy_output_data)
    
    # 予測されたターゲット列の値だけを取り出す
    predicted_target_values = predicted_data_original_scale[:, target_idx]

# c_out が enc_in と同じ場合 (全ての入力特徴量を予測)
else: # args.c_out == args.enc_in
    # outputs: (1, pred_len, enc_in)
    predicted_data_original_scale = scaler.inverse_transform(outputs.cpu().numpy().squeeze(0))
    predicted_target_values = predicted_data_original_scale[:, cols_data.index(args.target)]


# --- 予測結果の表示とファイル保存 ---
# 予測結果を保存するファイル名を生成
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"prediction_results_{current_time_str}.txt"
output_filepath = os.path.join("./Prediction_results", output_filename) # resultsフォルダに保存する例

# resultsフォルダが存在しない場合は作成
os.makedirs("./Prediction_results", exist_ok=True)

print(f"\n--- Prediction Results for the Next {args.pred_len} Days ---")

# ファイルに書き込むための文字列を構築
output_lines = []
output_lines.append(f"--- Prediction Results for the Next {args.pred_len} Days ---")

for i, pred_date in enumerate(future_dates):
    line = ""
    if args.c_out == 1:
        line = f"Date: {pred_date.strftime('%Y-%m-%d')}, Predicted {args.target}: {predicted_target_values[i]:.2f}"
    else: # 複数の特徴量を予測している場合
        predicted_values_for_date = predicted_data_original_scale[i]
        line = f"Date: {pred_date.strftime('%Y-%m-%d')}\n"
        for j, col_name in enumerate(cols_data):
            line += f"  Predicted {col_name}: {predicted_values_for_date[j]:.2f}\n"
        line = line.strip() # 最後の改行を削除
    print(line) # 画面にも表示
    output_lines.append(line) # ファイル保存用リストに追加

final_message_1 = "\nInference complete. Please note that these are model predictions for the future, and actual values are unknown."
final_message_2 = "The accuracy of these true future predictions can only be verified when the actual data becomes available."

print(final_message_1)
print(final_message_2)

output_lines.append(final_message_1)
output_lines.append(final_message_2)

# ファイルに書き込み
with open(output_filepath, 'w') as f:
    for line in output_lines:
        f.write(line + '\n')

print(f"\nPrediction results saved to: {output_filepath}")