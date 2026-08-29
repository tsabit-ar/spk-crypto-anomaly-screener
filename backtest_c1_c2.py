import requests
import pandas as pd
import time

PROXY_BASE = "https://binance-fapi-proxy.vercel.app"

def get_historical_data(symbol, days=29):
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    print(f"--> Mengambil data historis untuk {symbol}...")

    # 1. Ekstrak C1: Funding Rate
    url_funding = f"{PROXY_BASE}/fapi/v1/fundingRate"
    params_funding = {"symbol": symbol, "startTime": start_time, "endTime": end_time, "limit": 500}
    try:
        res_funding = requests.get(url_funding, params=params_funding).json()
    except Exception as e:
        print(f"[{symbol}] Gagal fetch C1 (Request error): {e}")
        return None

    if isinstance(res_funding, dict) and "code" in res_funding:
        print(f"[{symbol}] Gagal C1 API: {res_funding.get('msg', res_funding)}")
        return None

    df_funding = pd.DataFrame(res_funding)
    if df_funding.empty or 'fundingTime' not in df_funding.columns:
        print(f"[{symbol}] Data C1 kosong.")
        return None

    df_funding['timestamp'] = pd.to_datetime(df_funding['fundingTime'], unit='ms')
    df_funding['fundingRate'] = df_funding['fundingRate'].astype(float)
    df_funding = df_funding[['timestamp', 'fundingRate']].sort_values('timestamp')

    # 2. Ekstrak C2: Open Interest History
    url_oi = f"{PROXY_BASE}/futures/data/openInterestHist"
    params_oi = {"symbol": symbol, "period": "4h", "limit": 500, "startTime": start_time, "endTime": end_time}
    try:
        res_oi = requests.get(url_oi, params=params_oi).json()
    except Exception as e:
        print(f"[{symbol}] Gagal fetch C2 (Request error): {e}")
        return None

    if isinstance(res_oi, dict) and "code" in res_oi:
        print(f"[{symbol}] Gagal C2 API: {res_oi.get('msg', res_oi)}")
        return None

    df_oi = pd.DataFrame(res_oi)
    if df_oi.empty or 'sumOpenInterest' not in df_oi.columns:
        print(f"[{symbol}] Data C2 kosong.")
        return None

    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms')
    df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
    df_oi['oi_shift'] = df_oi['sumOpenInterest'].shift(1)
    df_oi['delta_oi_4h_pct'] = ((df_oi['sumOpenInterest'] - df_oi['oi_shift']) / df_oi['oi_shift']) * 100
    df_oi = df_oi[['timestamp', 'delta_oi_4h_pct']].dropna().sort_values('timestamp')

    # 3. Sinkronisasi Waktu
    df_merged = pd.merge_asof(df_oi, df_funding, on='timestamp', direction='backward')
    df_merged['symbol'] = symbol
    time.sleep(1)
    return df_merged

import requests

def get_all_active_tickers():
    print("--> Mengunduh daftar seluruh koin USDT-M aktif...")
    url = "https://binance-fapi-proxy.vercel.app/fapi/v1/exchangeInfo"
    
    try:
        res = requests.get(url).json()
        if "symbols" not in res:
            return []
            
        # Filter hanya koin USDT (USDT-M) yang statusnya sedang TRADING
        active_symbols = [
            s['symbol'] for s in res['symbols'] 
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL'
        ]
        print(f"Berhasil menemukan {len(active_symbols)} koin aktif.\n")
        return active_symbols
    except Exception as e:
        print(f"Gagal mengambil daftar koin: {e}")
        return []


def calculate_forward_performance(symbol, signal_timestamp_ms):
    url_klines = f"{PROXY_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1h", "startTime": signal_timestamp_ms, "limit": 168}
    
    try:
        res = requests.get(url_klines, params=params).json()
    except Exception as e:
        return {"Status": f"ERROR: {e}"}

    if isinstance(res, dict) and 'code' in res:
        return {"Status": f"API ERROR: {res.get('msg')}"}

    if not isinstance(res, list) or len(res) == 0:
        return {"Status": "NO_KLINES_DATA"}

    df = pd.DataFrame(res, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbb', 'tbq', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['high'] = df['high'].astype(float)
    
    entry_price = float(df.iloc[0]['open'])
    entry_time = df.iloc[0]['timestamp']
    
    peak_idx = df['high'].idxmax()
    peak_price = df.loc[peak_idx, 'high']
    peak_time = df.loc[peak_idx, 'timestamp']
    max_return_pct = round(((peak_price - entry_price) / entry_price) * 100, 2)
    
    # Target pump minimal 10%
    pump_start_df = df[df['high'] > (entry_price * 1.10)]
    
    if pump_start_df.empty:
        return {
            "Status": "GAGAL (<10%)",
            "Entry": entry_price,
            "Max Peak": peak_price,
            "Max Return": f"{max_return_pct}%"
        }
        
    start_pump_time = pump_start_df.iloc[0]['timestamp']
    rally_duration = peak_time - start_pump_time
    rally_duration = rally_duration if rally_duration >= pd.Timedelta(0) else pd.Timedelta(0)
    
    return {
        "Status": "BERHASIL PUMP \u2705",
        "Entry": entry_price,
        "Max Peak": peak_price,
        "Max Return": f"{max_return_pct}%",
        "Time to Pump": str(start_pump_time - entry_time),
        "Rally Duration": str(rally_duration)
    }

print("=== MEMULAI BACKTEST HISTORIS C1 & C2 ===\n")
# Ubah baris ini
symbols_to_test = get_all_active_tickers()
all_data = []

for sym in symbols_to_test:
    data = get_historical_data(sym)
    if data is not None and not data.empty:
        all_data.append(data)

if not all_data:
    print("\n[!] Gagal: Tidak ada dataset yang berhasil ditarik dari API/Proxy.")
    exit()

final_df = pd.concat(all_data, ignore_index=True)

# Filter Anomali: Delta OI 4H > 5% dan Funding Rate < -0.05% (-0.0005)
anomaly_signals = final_df[(final_df['delta_oi_4h_pct'] > 5.0) & (final_df['fundingRate'] < -0.0005)]

print(f"\nTotal titik sinyal anomali terdeteksi: {len(anomaly_signals)}")

if anomaly_signals.empty:
    print("Tidak ditemukan titik anomali dengan ambang batas tersebut pada rentang data ini.")
else:
    for index, row in anomaly_signals.iterrows():
        signal_time_ms = int(row['timestamp'].timestamp() * 1000)
        print(f"\n--- Sinyal {row['symbol']} pada {row['timestamp']} ---")
        print(f"C1 (Funding Rate): {row['fundingRate']} | C2 (Delta OI 4H): {round(row['delta_oi_4h_pct'], 2)}%")
        
        result = calculate_forward_performance(row['symbol'], signal_time_ms)
        for key, val in result.items():
            print(f"  {key}: {val}")
        time.sleep(1)