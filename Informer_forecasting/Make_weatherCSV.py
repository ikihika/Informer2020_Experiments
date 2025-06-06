import requests
import pandas as pd
import time
import os
import datetime
import logging
from datetime import timedelta

# ログ設定（INFOレベルで表示）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='weather_data_fetch.log',
    filemode='a'
)

def get_weather_data(latitude, longitude, start_date, end_date, max_retries=3):
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Asia/Tokyo"
    }

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}: Fetching data for {start_date} to {end_date}")
            response = requests.get(base_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # データが空の場合のチェックを追加
                if not data.get("daily") or not data["daily"].get("time"):
                    logging.warning(f"No daily data returned for {start_date} to {end_date}. Status: {response.status_code}")
                    return None # データがない場合はNoneを返す
                
                df = pd.DataFrame(data["daily"])
                df["time"] = pd.to_datetime(df["time"])
                logging.info(f"Success: Data retrieved for {start_date} to {end_date}")
                return df
            else:
                logging.warning(f"Warning: Status {response.status_code} on attempt {attempt}")
        except requests.RequestException as e:
            logging.error(f"Error: Exception occurred on attempt {attempt} - {str(e)}")

        time.sleep(5) # 失敗時にはリトライ前に待機

    logging.error(f"Failed to fetch data for {start_date} to {end_date} after {max_retries} attempts.")
    return None

def collect_fukuoka_weather_data(target_month=6, target_day=20, days_range=30, years_back=10):
    all_dfs = []
    now = datetime.datetime.now()
    # target_date = datetime.datetime(now.year, target_month, target_day)
    today_date_str = now.strftime("%Y-%m-%d") # 今日の日付文字列
    yesterday_date_str = (now - timedelta(days=1)).strftime("%Y-%m-%d") # 昨日の日付文字列

    # 福岡市の緯度・経度
    latitude = 33.5902
    longitude = 130.4017

    """for year_offset in range(1, years_back + 1):
        year = now.year - year_offset
        start = (target_date - timedelta(days=days_range)).replace(year=year).strftime("%Y-%m-%d")
        end = (target_date + timedelta(days=days_range)).replace(year=year).strftime("%Y-%m-%d")

        df = get_weather_data(latitude, longitude, start, end)
        if df is not None:
            all_dfs.append(df)"""
    
    # 過去のデータ（現在年-1 から years_back 年前まで）
    # ループ範囲を now.year - 1 から now.year - years_back までに変更
    for year in range(now.year - 1, now.year - years_back - 1, -1):
        # ターゲット日付の年を現在のループ年に設定
        current_year_target_date = datetime.datetime(year, target_month, target_day)
        
        start = (current_year_target_date - timedelta(days=days_range)).strftime("%Y-%m-%d")
        end = (current_year_target_date + timedelta(days=days_range)).strftime("%Y-%m-%d")

        # APIの最大期間チェック（Open-Meteoは過去のデータのみを提供）
        # startがendよりも新しい日付にならないようにする (特に月末付近でdays_rangeが大きい場合)
        if start > end:
            start, end = end, start # 日付が逆転しないように入れ替え
        
        df = get_weather_data(latitude, longitude, start, end)
        if df is not None:
            all_dfs.append(df)
            logging.info(f"Successfully collected data for year {year}")
        else:
            logging.warning(f"No data collected for year {year}")

    # 今年のデータ（取得可能な範囲まで）
    logging.info(f"Attempting to collect data for current year {now.year}.")
    current_year_target_date = datetime.datetime(now.year, target_month, target_day)
    
    # 今年の開始日
    start_this_year = (current_year_target_date - timedelta(days=days_range)).strftime("%Y-%m-%d")
    
    # 今年の終了日 (APIで取得可能なのは昨日の日付まで)
    # 6月20日+30日後の日付が、今日の前日よりも未来になる場合、今日の前日までとする
    end_this_year_theoretical = (current_year_target_date + timedelta(days=days_range)).strftime("%Y-%m-%d")
    end_this_year = min(end_this_year_theoretical, yesterday_date_str) # 昨日の日付を上限とする

    # 今年のデータ取得
    if start_this_year <= end_this_year: # 開始日 <= 終了日 の場合のみリクエスト
        logging.info(f"Fetching current year data from {start_this_year} to {end_this_year}")
        df_this_year = get_weather_data(latitude, longitude, start_this_year, end_this_year)
        if df_this_year is not None:
            all_dfs.append(df_this_year)
            logging.info(f"Successfully collected data for current year {now.year}")
        else:
            logging.warning(f"No data collected for current year {now.year} or period invalid.")
    else:
        logging.info(f"Current year data range ({start_this_year} to {end_this_year}) is invalid or in the future. Skipping.")

    if not all_dfs:
        logging.error("No data collected.")
        return None

    # result_df = pd.concat(all_dfs).sort_values(by="time").reset_index(drop=True)
    result_df = pd.concat(all_dfs).sort_values(by="time").drop_duplicates(subset=['time']).reset_index(drop=True)
    return result_df

# データ収集
# df = collect_fukuoka_weather_data()
df = collect_fukuoka_weather_data(days_range=30, years_back=10)

if df is not None:
    df.rename(columns={"time": "date"}, inplace=True)

    # 出力ディレクトリ
    output_directory = "weather_data"
    os.makedirs(output_directory, exist_ok=True)

    # 現在の日時をファイル名に使用
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ファイル名を地域名・期間・タイムスタンプから生成
    csv_filename = f"weather_fukuoka_june20±30days_{timestamp}.csv"
    output_path = os.path.join(output_directory, csv_filename)

    # CSV出力
    df.to_csv(output_path, index=False)

    print(f"Weather data successfully collected and saved to: {output_path}")
    print("\nCollected weather data head:")
    print(df.head())
    print("\nCollected weather data tail:")
    print(df.tail())
    print(f"Total rows collected: {len(df)}")
else:
    print("天気データの取得に失敗しました。詳細は weather_data_fetch.log を確認してください。")

