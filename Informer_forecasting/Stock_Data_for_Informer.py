import pandas as pd
import os

# 元のCSVファイルのパス
input_csv_path = "stock_data/AAPL_1d_2022-06-04_2025-06-04.csv"
# 出力CSVファイルのパス
output_csv_path = "data/AAPL_M_2022-06-04_2025-06-04.csv"

print(f"Loading data from: {input_csv_path}")

try:
    # CSVファイルを読み込む際に、ヘッダーを2行目（インデックス1）として指定し、
    # 最初の行（インデックス0）をスキップします。
    # header=[0, 1] とすることで、複数行のヘッダーを読み込むことができますが、
    # それを後で結合して扱いやすい列名にする必要があります。
    # 最もシンプルなのは、`names`引数で列名を直接指定し、`skiprows`で不要な行をスキップする方法です。
    # または、`header=0`として最初の行をヘッダーとして読み込み、2行目以降の不要なデータ行を削除します。
    # 今回のCSVは1行目が実質的なヘッダーなので、header=0で読み込み、2行目をデータとして捨てるのが良いでしょう。
    
    # 1行目をヘッダーとして読み込み、2行目（'AAPL'の行）をスキップする
    df_raw = pd.read_csv(input_csv_path, header=0, skiprows=[1])

    # 列名を確認し、'time'を'date'にリネーム
    if 'time' in df_raw.columns:
        df_raw.rename(columns={"time": "date"}, inplace=True)
    else:
        print("Warning: 'time' column not found. Please check your CSV header.")

    # 必要な列を FEATURES=M に合わせて選択
    # date列は必ず含める
    # その他の列は、Informerが予測に使う特徴量として含める
    # 今回は Close, High, Low, Open, Volume 全てを含めます
    required_cols = ["date", "Close", "High", "Low", "Open", "Volume"]
    
    # 実際にDataFrameにある列だけを選択するようにします
    cols_to_select = [col for col in required_cols if col in df_raw.columns]

    if not all(col in df_raw.columns for col in required_cols[1:]):
        print(f"Warning: Not all required columns for FEATURES=M ({required_cols[1:]}) found in input CSV. Available columns: {df_raw.columns.tolist()}")
        print("Proceeding with available columns. Please ensure your script's ENC_IN/DEC_IN/C_OUT match the number of selected features.")

    df_processed = df_raw[cols_to_select].copy()

    # 日付列をdatetime型に変換し、欠損値がある場合はNaNを削除（または補間）
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed.dropna(inplace=True) # 簡単な欠損値処理
    
    # 日付でソート
    df_processed.sort_values(by='date', inplace=True)

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 処理済みデータを新しいCSVファイルとして保存
    df_processed.to_csv(output_csv_path, index=False)

    print(f"Data successfully processed and saved to: {output_csv_path}")
    print("Processed data head:")
    print(df_processed.head())
    print("\nProcessed data tail:")
    print(df_processed.tail())

except FileNotFoundError:
    print(f"Error: Input CSV file not found at '{input_csv_path}'")
except Exception as e:
    print(f"An error occurred: {e}")