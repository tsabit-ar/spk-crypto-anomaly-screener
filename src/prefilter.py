"""Pre-filtering module to filter and rank the top universe candidates from Binance Futures.

Applies liquidity, volume, and volatility criteria to eliminate noise,
stablecoins, and illiquid tokens, returning the top dynamic candidates.
"""

from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd

# Stablecoins, pegged tokens, fiat pairs, and mega-cap majors to exclude
EXCLUDED_SYMBOLS: Set[str] = {
    "BTCUSDT",
    "ETHUSDT",
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "USDPUSDT",
    "EURUSDT",
    "DAIUSDT",
    "BUSDUSDT",
    "AEURUSDT",
    "USTCUSDT",
    "USD1USDT",
}

# Threshold constants for pre-filtering
MIN_QUOTE_VOLUME_USDT = 10_000_000.0  # Min $10M 24h quote volume
MIN_TRADE_COUNT = 15_000               # Min 15k trades in 24h


def filter_top_universe(
    bulk_tickers_data: List[Dict[str, Any]],
    limit: int = 25,
    min_volume: float = MIN_QUOTE_VOLUME_USDT,
    min_trades: int = MIN_TRADE_COUNT,
    excluded_symbols: Set[str] = EXCLUDED_SYMBOLS,
) -> pd.DataFrame:
    """Filter and rank bulk 24h ticker data to select top dynamic candidates.

    Args:
        bulk_tickers_data: Raw list of 24h ticker dictionaries from Binance Futures API.
        limit: Number of top candidate symbols to return (default: 25).
        min_volume: Minimum 24h quote volume in USDT (default: 10,000,000).
        min_trades: Minimum 24h trade count (default: 15,000).
        excluded_symbols: Set of symbols to exclude (stablecoins, majors).

    Returns:
        pd.DataFrame: Top candidate coins sorted by composite dynamic score.
    """
    candidates = []

    for item in bulk_tickers_data:
        symbol = str(item.get("symbol", "")).upper()

        # Hard filters: USDT pair, ASCII alphanumeric only, not in exclusion list
        if (
            not symbol.endswith("USDT")
            or not symbol.isascii()
            or not symbol.isalnum()
            or symbol in excluded_symbols
        ):
            continue

        try:
            last_price = float(item.get("lastPrice", 0))
            high_price = float(item.get("highPrice", 0))
            low_price = float(item.get("lowPrice", 0))
            price_change_pct = float(item.get("priceChangePercent", 0))
            quote_volume = float(item.get("quoteVolume", 0))
            count = int(item.get("count", 0))

            # Liquidity thresholds
            if last_price <= 0 or quote_volume < min_volume or count < min_trades:
                continue

            # Daily Volatility: (High - Low) / LastPrice
            volatility = (high_price - low_price) / last_price if last_price > 0 else 0.0

            # Log-Volume: log10(quoteVolume)
            log_vol = float(np.log10(quote_volume)) if quote_volume > 0 else 0.0

            candidates.append({
                "symbol": symbol,
                "last_price": last_price,
                "high_price": high_price,
                "low_price": low_price,
                "price_change_pct": price_change_pct,
                "quote_volume": quote_volume,
                "count": count,
                "log_volume": log_vol,
                "volatility": volatility,
            })
        except (ValueError, TypeError):
            continue

    if not candidates:
        return pd.DataFrame()

    df = pd.DataFrame(candidates)

    # Min-Max Normalization
    log_vol_min, log_vol_max = df["log_volume"].min(), df["log_volume"].max()
    vol_min, vol_max = df["volatility"].min(), df["volatility"].max()

    eps = 1e-9
    df["norm_log_volume"] = (df["log_volume"] - log_vol_min) / (log_vol_max - log_vol_min + eps)
    df["norm_volatility"] = (df["volatility"] - vol_min) / (vol_max - vol_min + eps)

    # Composite Score (50% log-volume + 50% volatility)
    df["composite_score"] = 0.5 * df["norm_log_volume"] + 0.5 * df["norm_volatility"]

    # Sort descending and select top candidates
    df_sorted = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    return df_sorted.head(limit)
