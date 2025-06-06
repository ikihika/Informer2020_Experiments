import pandas as pd
import os
import glob # 複数のCSVファイルを扱うために追加

# 元のCSVファイルのパス（ワイルドカードを使って最新のものを取得）
# weather_data/weather_fukuoka_june20±30days_YYYY-MM-DD_HH-MM-SS.csv
input_csv_pattern = "weather_data/weather_fukuoka_june20±30days_*.csv"
# 出力CSVファイルのパス
output_csv_path = "data/weather_M_2025_06_06.csv"

print(f"Searching for raw weather data in: {input_csv_pattern}")

# 最新のファイルを見つける
list_of_files = glob.glob(input_csv_pattern)
if not list_of_files:
    print(f"Error: No weather CSV files found matching pattern '{input_csv_pattern}'.")
    print("Please ensure your weather data collection script has been run and generated files in 'weather_data/' directory.")
    exit()

# 更新日時でソートし、最新のファイルを選択
latest_file = max(list_of_files, key=os.path.getctime)
input_csv_path = latest_file
print(f"Loading latest weather data from: {input_csv_path}")

try:
    df_raw = pd.read_csv(input_csv_path)

    # 'time' 列を 'date' にリネーム
    if 'time' in df_raw.columns:
        df_raw.rename(columns={"time": "date"}, inplace=True)
    else:
        print("Warning: 'time' column not found. Assuming 'date' column is already correct.")

    # 日付列をdatetime型に変換
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    # 必要な列を FEATURES=M に合わせて選択
    # date列は必ず含める
    # 今回の天気データは 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'
    # これらをすべて含める
    features_to_include = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
    required_cols = ["date"] + features_to_include
    
    # 実際にDataFrameにある列だけを選択するようにします
    cols_to_select = [col for col in required_cols if col in df_raw.columns]

    if not all(col in df_raw.columns for col in features_to_include):
        print(f"Warning: Not all required weather feature columns ({features_to_include}) found in input CSV.")
        print(f"Available columns: {df_raw.columns.tolist()}")
        print("Proceeding with available weather columns. Please ensure your script's ENC_IN/DEC_IN/C_OUT match the number of selected features.")
        # 見つかった特徴量のみに features_to_include を更新
        features_to_include = [col for col in features_to_include if col in df_raw.columns]
        cols_to_select = ["date"] + features_to_include

    df_processed = df_raw[cols_to_select].copy()

    # 日付でソート (データ収集スクリプトでソート済のはずですが念のため)
    df_processed.sort_values(by='date', inplace=True)

    # 欠損値があれば補完（ここでは簡単な前方補完、必要に応じて変更）
    df_processed.fillna(method='ffill', inplace=True)
    df_processed.fillna(method='bfill', inplace=True) # 前方で埋まらない場合は後方で

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 処理済みデータを新しいCSVファイルとして保存
    df_processed.to_csv(output_csv_path, index=False)

    print(f"Weather data successfully processed and saved to: {output_csv_path}")
    print("Processed weather data head:")
    print(df_processed.head())
    print("\nProcessed weather data tail:")
    print(df_processed.tail())
    print(f"Number of features in processed data (excluding date): {len(features_to_include)}")

except FileNotFoundError:
    print(f"Error: Input CSV file not found at '{input_csv_path}'")
except Exception as e:
    print(f"An error occurred: {e}")