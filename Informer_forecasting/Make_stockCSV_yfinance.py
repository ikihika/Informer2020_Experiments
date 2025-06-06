import yfinance as yf
import pandas as pd
import os

def get_stock_data_yf(symbol, start_date, end_date, interval='1d'):
    """
    Yahoo Financeから株価データを取得し、DataFrameで返す
    :param symbol: ティッカーシンボル（例：'AAPL')
    :param start_date: 開始日（例：'2023-01-01')
    :param end_date: 終了日（例：'2024-01-01')
    :param interval: 取得間隔（'1d', '1h', '1m' など）
    :return: DataFrame（連続日付・補完済み）
    """
    print(f"Downloading {symbol} from {start_date} to {end_date} with interval {interval} ...")
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    df.reset_index(inplace=True)

    # カラム名を統一
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'time'}, inplace=True)
    elif 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'time'}, inplace=True)

    # 日付をインデックスに設定して全日補完
    df.set_index('time', inplace=True)
    full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(full_dates)
    df.index.name = 'time'

    # 欠損値を前方補完
    df.ffill(inplace=True)

    # 欠損が残っていれば（先頭NaNなど）さらに後方補完
    df.bfill(inplace=True)

    df.reset_index(inplace=True)
    return df


def save_to_csv(df, symbol, interval, start_date, end_date):
    """
    指定されたフォルダにCSVファイルとして保存する
    """
    output_dir = "stock_data"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{symbol}_{interval}_{start_date}_{end_date}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")


if __name__ == "__main__":
    # ユーザー定義（必要に応じて編集）
    symbol = "AAPL"
    start_date = "2022-06-04"
    end_date = "2025-06-04"
    interval = "1d"  # 例: '1d', '1h', '1wk'

    df = get_stock_data_yf(symbol, start_date, end_date, interval)
    save_to_csv(df, symbol, interval, start_date, end_date)
