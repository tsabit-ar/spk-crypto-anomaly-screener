"""Feature engineering module for SPK Crypto Anomaly Screener.

Extracts the 5 multi-criteria decision metrics (C1 - C5) from Binance Futures:
- C1: Funding Rate (%) [Cost]
- C2: Delta OI 4H (%) [Benefit]
- C3: Bollinger Band Width 1H (%) [Cost]
- C4: Depth Imbalance Ratio (Bids / Asks +/-2%) [Benefit]
- C5: Volume / OI Velocity Ratio [Benefit]
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.ingestion import (
    fetch_funding_rate,
    fetch_delta_oi_4h,
    fetch_klines_1h,
    fetch_depth_2pct,
)

EPSILON = 1e-9


def extract_coin_metrics(
    symbol: str,
    base_ticker_data: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract and compute all 5 criteria metrics for a single coin symbol.

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').
        base_ticker_data: Optional base 24h ticker metadata dictionary.

    Returns:
        Dictionary containing symbol, C1..C5 metrics, and raw metadata,
        or None if an error occurs during extraction.
    """
    symbol = symbol.upper()

    try:
        # 1. C1: Funding Rate (%)
        fr_data = fetch_funding_rate(symbol)
        funding_rate = float(fr_data.get("lastFundingRate", 0.0))
        c1_funding_rate_pct = funding_rate * 100.0  # In percent

        # 2. C2: Delta Open Interest 4H (%)
        oi_data = fetch_delta_oi_4h(symbol)
        if not isinstance(oi_data, list) or len(oi_data) < 2:
            return None

        oi_initial = float(oi_data[0].get("sumOpenInterest", 0.0))
        oi_latest = float(oi_data[-1].get("sumOpenInterest", 0.0))
        oi_value_latest = float(oi_data[-1].get("sumOpenInterestValue", 0.0))

        if oi_initial <= 0:
            return None

        c2_delta_oi_4h_pct = ((oi_latest - oi_initial) / (oi_initial + EPSILON)) * 100.0

        # 3. C3: Bollinger Band Width 1H (%) & 1H Kline data
        klines_data = fetch_klines_1h(symbol, limit=20)
        if not isinstance(klines_data, list) or len(klines_data) < 20:
            return None

        # Kline close prices (index 4)
        closes = np.array([float(k[4]) for k in klines_data], dtype=np.float64)
        latest_1h_quote_vol = float(klines_data[-1][7])  # Index 7: Quote asset volume
        latest_close = float(closes[-1])

        sma_20 = np.mean(closes)
        std_20 = np.std(closes)
        # BBW = (Upper Band - Lower Band) / SMA * 100 = (4 * std) / SMA * 100
        c3_bbw_1h_pct = (4.0 * std_20 / (sma_20 + EPSILON)) * 100.0

        # 4. C4: Orderbook Depth Imbalance Ratio (+/- 2%)
        depth_data = fetch_depth_2pct(symbol)
        if not isinstance(depth_data, dict) or "bids" not in depth_data or "asks" not in depth_data:
            return None

        bids = np.array([[float(p), float(q)] for p, q in depth_data["bids"]], dtype=np.float64)
        asks = np.array([[float(p), float(q)] for p, q in depth_data["asks"]], dtype=np.float64)

        if len(bids) == 0 or len(asks) == 0:
            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2.0

        bid_mask = bids[:, 0] >= (mid_price * 0.98)
        ask_mask = asks[:, 0] <= (mid_price * 1.02)

        bid_liq_2pct = np.sum(bids[bid_mask, 0] * bids[bid_mask, 1]) if np.any(bid_mask) else 0.0
        ask_liq_2pct = np.sum(asks[ask_mask, 0] * asks[ask_mask, 1]) if np.any(ask_mask) else 0.0

        c4_depth_imbalance_ratio = (bid_liq_2pct + EPSILON) / (ask_liq_2pct + EPSILON)

        # 5. C5: Volume / OI Velocity Ratio (1H Volume / Latest OI Value)
        c5_volume_oi_velocity = latest_1h_quote_vol / (oi_value_latest + EPSILON)

        # Base ticker metadata if available
        price_change_24h = float(base_ticker_data.get("price_change_pct", 0.0)) if base_ticker_data else 0.0
        quote_volume_24h = float(base_ticker_data.get("quote_volume", 0.0)) if base_ticker_data else 0.0

        return {
            "symbol": symbol,
            "C1": c1_funding_rate_pct,
            "C2": c2_delta_oi_4h_pct,
            "C3": c3_bbw_1h_pct,
            "C4": c4_depth_imbalance_ratio,
            "C5": c5_volume_oi_velocity,
            # Metadata for reporting / presentation
            "last_price": latest_close,
            "price_change_24h": price_change_24h,
            "quote_volume_24h": quote_volume_24h,
            "oi_value_usdt": oi_value_latest,
            "bid_liq_2pct": bid_liq_2pct,
            "ask_liq_2pct": ask_liq_2pct,
        }

    except Exception:
        return None


def extract_batch_metrics(
    candidate_symbols_or_df: Any,
    max_workers: int = 5,
) -> pd.DataFrame:
    """Extract metrics for a batch of symbols concurrently.

    Args:
        candidate_symbols_or_df: Either a list of symbol strings or a DataFrame
                                from prefilter module.
        max_workers: Max concurrent threads for API requests (default: 5).

    Returns:
        pd.DataFrame containing computed C1..C5 metrics for valid symbols.
    """
    tasks = []

    if isinstance(candidate_symbols_or_df, pd.DataFrame):
        for _, row in candidate_symbols_or_df.iterrows():
            tasks.append((row["symbol"], row.to_dict()))
    else:
        for sym in candidate_symbols_or_df:
            tasks.append((str(sym), None))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sym = {
            executor.submit(extract_coin_metrics, sym, meta): sym
            for sym, meta in tasks
        }
        for future in as_completed(future_to_sym):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
