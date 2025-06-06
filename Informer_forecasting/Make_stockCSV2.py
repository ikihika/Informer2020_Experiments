import requests
import pandas as pd
import time
import os
import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='stock_data_fetch3.log',
    filemode='a'
)

def get_stock_data(symbol, interval, api_key, month, max_retries=3):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}"
        f"&interval={interval}"
        f"&adjusted=false"
        f"&extended_hours=false"
        f"&outputsize=full"
        f"&datatype=csv"
        f"&apikey={api_key}"
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}: Fetching data for {symbol} - {month}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.text.splitlines()

                if not data or not data[0].startswith("time"):
                    logging.warning(f"Unexpected response format:\n{response.text[:300]}")
                    return None

                logging.info(f"Success: Data retrieved for {symbol} - {month}")
                return data
            else:
                logging.warning(f"Status {response.status_code} on attempt {attempt} for {symbol} - {month}")
        
        except requests.RequestException as e:
            logging.error(f"Exception on attempt {attempt} for {symbol} - {month} - {str(e)}")

        time.sleep(10)

    logging.error(f"Failed to fetch data for {symbol} - {month} after {max_retries} attempts.")
    return None

def convert_to_dataframe(raw_data):
    header = raw_data[0].split(',')
    expected_cols = len(header)

    data_rows = []
    for row in raw_data[1:]:
        cols = row.split(',')
        if len(cols) == expected_cols:
            data_rows.append(cols)
        else:
            logging.warning(f"Skipping malformed row (expected {expected_cols} columns): {row}")

    if not data_rows:
        raise ValueError("No valid data rows found.")

    df = pd.DataFrame(data_rows, columns=header)
    df['time'] = pd.to_datetime(df['time'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

    df.sort_values(by='time', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def get_full_data(symbol, interval, api_key, start_year=2023, end_year=2024):
    all_data = []
    header = None

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_str = f"{year}-{month:02d}"
            raw_stock_data = get_stock_data(symbol, interval, api_key, month_str)
            if raw_stock_data:
                if header is None:
                    header = raw_stock_data[0]
                all_data.extend(raw_stock_data[1:])
            time.sleep(15)

    if not header:
        raise ValueError("No header found. All API calls failed?")

    all_data.insert(0, header)
    stock_dataframe = convert_to_dataframe(all_data)
    
    return stock_dataframe

# パラメータ指定
symbol = "AAPL"
interval = "15min"
api_key = "5S5MM38JANJN6T0U"

# データ取得
stock_dataframe = get_full_data(symbol, interval, api_key, start_year=2023, end_year=2024)
stock_dataframe.rename(columns={'time': 'date'}, inplace=True)
df = stock_dataframe

# 保存
output_directory = './data/'  # 絶対パスに変更可能
os.makedirs(output_directory, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"{symbol}_{interval}_{timestamp}.csv"
output_path = os.path.join(output_directory, csv_filename)

df.to_csv(output_path, index=False)
logging.info(f"CSV saved to {output_path}")

# 確認用出力
print(df.head())
