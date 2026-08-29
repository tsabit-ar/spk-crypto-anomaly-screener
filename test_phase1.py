"""Test script for Phase 1: Binance Futures API Ingestion Test.

Validates connectivity, response structures, and data type parsing
for all 5 public endpoints using 'SOLUSDT' sample.
"""

import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.ingestion import (
    fetch_bulk_24h_tickers,
    fetch_funding_rate,
    fetch_delta_oi_4h,
    fetch_klines_1h,
    fetch_depth_2pct,
)

SAMPLE_SYMBOL = "SOLUSDT"
SEPARATOR = "=" * 70


def test_bulk_24h_tickers() -> bool:
    """Test GET /fapi/v1/ticker/24hr endpoint."""
    print(f"\n[1/5] Testing fetch_bulk_24h_tickers() ...")
    try:
        data = fetch_bulk_24h_tickers()
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(f"Expected non-empty list, received {type(data)} with len={len(data) if isinstance(data, list) else 0}")

        # Find sample symbol in tickers
        sample_item = next((item for item in data if item.get("symbol") == SAMPLE_SYMBOL), None)
        if not sample_item:
            raise ValueError(f"Sample symbol '{SAMPLE_SYMBOL}' not found in 24h tickers list.")

        # Data type parsing
        last_price = float(sample_item["lastPrice"])
        price_change_pct = float(sample_item["priceChangePercent"])
        quote_volume = float(sample_item["quoteVolume"])
        count = int(sample_item["count"])

        print(f"      Status: [PASS]")
        print(f"      Total Tickers Fetched: {len(data)}")
        print(f"      Sample ({SAMPLE_SYMBOL}):")
        print(f"        - Last Price: ${last_price:,.4f}")
        print(f"        - 24h Change: {price_change_pct:+.2f}%")
        print(f"        - 24h Quote Volume: ${quote_volume:,.2f} USDT")
        print(f"        - 24h Total Trades: {count:,}")
        return True
    except Exception as e:
        print(f"      Status: [FAIL] - Error: {e}")
        return False


def test_funding_rate() -> bool:
    """Test GET /fapi/v1/premiumIndex endpoint."""
    print(f"\n[2/5] Testing fetch_funding_rate('{SAMPLE_SYMBOL}') ...")
    try:
        data = fetch_funding_rate(SAMPLE_SYMBOL)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, received {type(data)}")

        # Data type parsing
        mark_price = float(data["markPrice"])
        index_price = float(data["indexPrice"])
        funding_rate = float(data["lastFundingRate"])
        next_funding_time_ms = int(data["nextFundingTime"])
        next_funding_dt = datetime.fromtimestamp(next_funding_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"      Status: [PASS]")
        print(f"      Sample ({SAMPLE_SYMBOL}):")
        print(f"        - Mark Price: ${mark_price:,.4f}")
        print(f"        - Index Price: ${index_price:,.4f}")
        print(f"        - Current Funding Rate: {funding_rate * 100:+.4f}%")
        print(f"        - Next Funding Time: {next_funding_dt}")
        return True
    except Exception as e:
        print(f"      Status: [FAIL] - Error: {e}")
        return False


def test_delta_oi_4h() -> bool:
    """Test GET /futures/data/openInterestHist endpoint."""
    print(f"\n[3/5] Testing fetch_delta_oi_4h('{SAMPLE_SYMBOL}') ...")
    try:
        data = fetch_delta_oi_4h(SAMPLE_SYMBOL)
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError(f"Expected list with at least 2 records, received len={len(data) if isinstance(data, list) else 0}")

        # Parsing records into pandas DataFrame for clean metric calculation
        records = []
        for record in data:
            records.append({
                "timestamp": int(record["timestamp"]),
                "sumOpenInterest": float(record["sumOpenInterest"]),
                "sumOpenInterestValue": float(record["sumOpenInterestValue"]),
            })
        df_oi = pd.DataFrame(records)

        oi_initial = df_oi.iloc[0]["sumOpenInterest"]
        oi_latest = df_oi.iloc[-1]["sumOpenInterest"]
        oi_value_latest = df_oi.iloc[-1]["sumOpenInterestValue"]
        delta_oi = oi_latest - oi_initial
        delta_oi_pct = (delta_oi / oi_initial) * 100 if oi_initial > 0 else 0.0

        print(f"      Status: [PASS]")
        print(f"      Records Fetched: {len(data)} (Interval: 1h, Window: ~4h)")
        print(f"      Sample ({SAMPLE_SYMBOL}):")
        print(f"        - Latest OI: {oi_latest:,.2f} SOL (${oi_value_latest:,.2f} USDT)")
        print(f"        - 4H Delta OI: {delta_oi:+,.2f} SOL ({delta_oi_pct:+.2f}%)")
        return True
    except Exception as e:
        print(f"      Status: [FAIL] - Error: {e}")
        return False


def test_klines_1h() -> bool:
    """Test GET /fapi/v1/klines endpoint."""
    print(f"\n[4/5] Testing fetch_klines_1h('{SAMPLE_SYMBOL}', limit=20) ...")
    try:
        data = fetch_klines_1h(SAMPLE_SYMBOL, limit=20)
        if not isinstance(data, list) or len(data) != 20:
            raise ValueError(f"Expected list of 20 candles, received {len(data) if isinstance(data, list) else type(data)}")

        # Convert to DataFrame
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df_klines = pd.DataFrame(data, columns=columns)
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        df_klines[numeric_cols] = df_klines[numeric_cols].astype(float)
        df_klines["trades"] = df_klines["trades"].astype(int)

        latest_candle = df_klines.iloc[-1]
        candle_time = datetime.fromtimestamp(int(latest_candle["open_time"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"      Status: [PASS]")
        print(f"      Candles Fetched: {len(df_klines)}")
        print(f"      Latest 1H Candle ({candle_time}):")
        print(f"        - Open:  ${latest_candle['open']:,.4f}")
        print(f"        - High:  ${latest_candle['high']:,.4f}")
        print(f"        - Low:   ${latest_candle['low']:,.4f}")
        print(f"        - Close: ${latest_candle['close']:,.4f}")
        print(f"        - Volume: {latest_candle['volume']:,.2f} SOL (${latest_candle['quote_volume']:,.2f} USDT)")
        return True
    except Exception as e:
        print(f"      Status: [FAIL] - Error: {e}")
        return False


def test_depth_2pct() -> bool:
    """Test GET /fapi/v1/depth endpoint."""
    print(f"\n[5/5] Testing fetch_depth_2pct('{SAMPLE_SYMBOL}') ...")
    try:
        data = fetch_depth_2pct(SAMPLE_SYMBOL)
        if not isinstance(data, dict) or "bids" not in data or "asks" not in data:
            raise ValueError(f"Expected dict with 'bids' and 'asks', received keys={list(data.keys()) if isinstance(data, dict) else type(data)}")

        bids = np.array([[float(p), float(q)] for p, q in data["bids"]])
        asks = np.array([[float(p), float(q)] for p, q in data["asks"]])

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        # Filter depth within +/- 2% of mid price
        bid_mask = bids[:, 0] >= (mid_price * 0.98)
        ask_mask = asks[:, 0] <= (mid_price * 1.02)

        bid_liquidity_2pct_usdt = np.sum(bids[bid_mask, 0] * bids[bid_mask, 1])
        ask_liquidity_2pct_usdt = np.sum(asks[ask_mask, 0] * asks[ask_mask, 1])

        print(f"      Status: [PASS]")
        print(f"      Orderbook Levels Fetched: {len(bids)} Bids / {len(asks)} Asks")
        print(f"      Sample ({SAMPLE_SYMBOL}):")
        print(f"        - Best Bid: ${best_bid:,.4f} | Best Ask: ${best_ask:,.4f}")
        print(f"        - Spread: ${spread:,.4f} ({spread_bps:.2f} bps)")
        print(f"        - +/- 2% Bid Liquidity: ${bid_liquidity_2pct_usdt:,.2f} USDT")
        print(f"        - +/- 2% Ask Liquidity: ${ask_liquidity_2pct_usdt:,.2f} USDT")
        return True
    except Exception as e:
        print(f"      Status: [FAIL] - Error: {e}")
        return False


def main():
    """Main test runner."""
    print(SEPARATOR)
    print(" SPK Crypto Anomaly Screener - Phase 1 Ingestion Test")
    print(" Target: Binance Futures USDT-M (Public Endpoints - $0 Cost)")
    print(f" Execution Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(SEPARATOR)

    results = {
        "Bulk 24h Tickers (/fapi/v1/ticker/24hr)": test_bulk_24h_tickers(),
        "Funding Rate & Mark Price (/fapi/v1/premiumIndex)": test_funding_rate(),
        "4H Delta Open Interest (/futures/data/openInterestHist)": test_delta_oi_4h(),
        "1H Klines (/fapi/v1/klines)": test_klines_1h(),
        "Orderbook Depth (/fapi/v1/depth)": test_depth_2pct(),
    }

    print("\n" + SEPARATOR)
    print(" SUMMARY OF PHASE 1 INGESTION TEST")
    print(SEPARATOR)

    all_passed = True
    for test_name, status in results.items():
        status_str = "[PASS] SUCCESS" if status else "[FAIL] FAILED"
        print(f"  {status_str:<15} : {test_name}")
        if not status:
            all_passed = False

    print(SEPARATOR)
    if all_passed:
        print(" RESULT: ALL 5 ENDPOINTS PASSED SUCCESSFULLY! Ready for Phase 2.")
        sys.exit(0)
    else:
        print(" RESULT: SOME ENDPOINTS FAILED. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
