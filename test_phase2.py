"""Test script for Phase 2: Feature Engineering, Pre-Filter, and TOPSIS Engine.

Part 1: Mathematical Unit Test (Synthetic / Mock Dataset)
Part 2: Live Pipeline Integration Test (Binance Futures Live Data)
"""

import sys
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Ensure Windows stdout handles UTF-8 gracefully
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ingestion import fetch_bulk_24h_tickers
from src.prefilter import filter_top_universe
from src.features import extract_batch_metrics
from src.topsis import TopsisEngine

SEPARATOR = "=" * 78


def test_topsis_mathematical_mock() -> bool:
    """Part 1: Verify TOPSIS mathematical engine with synthetic mock data."""
    print(f"\n[PART 1] TOPSIS Engine Mathematical Unit Test (Synthetic Data)")
    print("-" * 78)

    # Synthetic candidate scenarios:
    # - COIN_IDEAL: Low Funding (Cost), High Delta OI (Benefit), Squeeze BBW (Cost), High Depth Imbalance (Benefit), High Velocity (Benefit)
    # - COIN_MODERATE: Average on all criteria
    # - COIN_WORST: High Funding (Cost), Negative Delta OI (Benefit), Wide BBW (Cost), Heavy Ask Wall (Benefit), Low Velocity (Benefit)
    mock_data = pd.DataFrame([
        {
            "symbol": "COIN_IDEAL",
            "C1": -0.05,   # Low funding rate -> Best for Cost
            "C2": 45.0,    # +45% 4h OI growth -> Best for Benefit
            "C3": 1.10,    # 1.1% tight BBW compression -> Best for Cost
            "C4": 3.80,    # 3.8x Bids over Asks -> Best for Benefit
            "C5": 2.50,    # 2.5x Velocity -> Best for Benefit
        },
        {
            "symbol": "COIN_MODERATE",
            "C1": 0.01,
            "C2": 10.0,
            "C3": 3.20,
            "C4": 1.20,
            "C5": 0.70,
        },
        {
            "symbol": "COIN_WORST",
            "C1": 0.08,    # High funding rate -> Worst for Cost
            "C2": -20.0,   # -20% OI decline -> Worst for Benefit
            "C3": 7.50,    # 7.5% expanded BBW -> Worst for Cost
            "C4": 0.35,    # 0.35x Bids (Ask heavy) -> Worst for Benefit
            "C5": 0.05,    # 0.05x Velocity -> Worst for Benefit
        },
    ])

    engine = TopsisEngine()
    ranked_df = engine.rank_candidates(mock_data)

    print("Synthetic Input & Output Scores:")
    display_cols = ["rank", "symbol", "C1", "C2", "C3", "C4", "C5", "D_pos", "D_neg", "topsis_score"]
    print(ranked_df[display_cols].to_string(index=False))

    # Assertions
    top_coin = ranked_df.iloc[0]["symbol"]
    worst_coin = ranked_df.iloc[-1]["symbol"]
    score_ideal = ranked_df.iloc[0]["topsis_score"]
    score_worst = ranked_df.iloc[-1]["topsis_score"]

    if top_coin != "COIN_IDEAL" or worst_coin != "COIN_WORST":
        print(f"\n[FAIL] Incorrect ranking order: Top={top_coin}, Worst={worst_coin}")
        return False

    if score_ideal <= score_worst:
        print(f"\n[FAIL] Score ordering error: Ideal score ({score_ideal:.4f}) <= Worst score ({score_worst:.4f})")
        return False

    print(f"\n[PASS] Mathematical unit test validated successfully! Top coin: {top_coin} (Ci={score_ideal:.4f}), Worst: {worst_coin} (Ci={score_worst:.4f})")
    return True


def test_live_pipeline_integration() -> bool:
    """Part 2: Run end-to-end live pipeline from Binance Futures API."""
    print(f"\n[PART 2] Live Pipeline Integration Test (Binance Futures Live Ingestion)")
    print("-" * 78)

    t0 = time.time()

    # Step 1: Bulk Ingestion
    print("1. Fetching 24h bulk ticker data from Binance Futures ...")
    bulk_data = fetch_bulk_24h_tickers()
    print(f"   -> Successfully ingested {len(bulk_data)} tickers.")

    # Step 2: Pre-filter Universe
    print("2. Pre-filtering Top 25 dynamic & liquid universe candidates ...")
    top_candidates_df = filter_top_universe(bulk_data, limit=25)
    if top_candidates_df.empty:
        print("   -> [FAIL] Pre-filter returned 0 candidates.")
        return False
    print(f"   -> Selected {len(top_candidates_df)} candidate symbols.")
    print(f"   -> Universe symbols: {', '.join(top_candidates_df['symbol'].tolist()[:10])} ...")

    # Step 3: Feature Extraction (Concurrent)
    print("3. Extracting 5 multi-criteria features (C1 - C5) for all candidates ...")
    features_df = extract_batch_metrics(top_candidates_df, max_workers=6)
    if features_df.empty or len(features_df) < 5:
        print(f"   -> [FAIL] Feature extraction returned insufficient data (len={len(features_df)}).")
        return False
    print(f"   -> Successfully extracted complete metrics for {len(features_df)} coins.")

    # Step 4: TOPSIS Ranking Engine
    print("4. Executing TOPSIS Multi-Criteria Decision Engine ...")
    engine = TopsisEngine()
    ranked_df = engine.rank_candidates(features_df)

    elapsed = time.time() - t0
    print(f"   -> Pipeline execution completed in {elapsed:.2f} seconds.")

    # Step 5: Format and Print Top 5 Anomaly Screener Results
    print("\n" + SEPARATOR)
    print(" TOP 5 CRYPTO ANOMALY CANDIDATES (BINANCE FUTURES USDT-M)")
    print(SEPARATOR)

    top5 = ranked_df.head(5)
    for idx, row in top5.iterrows():
        print(f" Rank #{int(row['rank'])} | {row['symbol']:<10} | TOPSIS Score (Ci): {row['topsis_score']:.4f}")
        print(f"   * Price: ${row['last_price']:<10,.4f} | 24h Change: {row['price_change_24h']:+.2f}% | 24h Vol: ${row['quote_volume_24h']:,.0f}")
        print(f"   * C1 [Cost]    Funding Rate:        {row['C1']:+.4f}%")
        print(f"   * C2 [Benefit] 4H Delta OI:         {row['C2']:+.2f}% (OI Value: ${row['oi_value_usdt']:,.0f})")
        print(f"   * C3 [Cost]    1H BBW Compression:  {row['C3']:.2f}%")
        print(f"   * C4 [Benefit] 2% Depth Imbalance:  {row['C4']:.2f}x (Bid ${row['bid_liq_2pct']:,.0f} / Ask ${row['ask_liq_2pct']:,.0f})")
        print(f"   * C5 [Benefit] Volume/OI Velocity:  {row['C5']:.4f}")
        print("-" * 78)

    # Print summary table
    summary_cols = ["rank", "symbol", "topsis_score", "C1", "C2", "C3", "C4", "C5"]
    print("\nSummary Metrics Table (Top 5):")
    print(top5[summary_cols].to_string(index=False))

    print("\n[PASS] Live pipeline integration test completed successfully!")
    return True


def main():
    """Main runner for Phase 2 tests."""
    print(SEPARATOR)
    print(" SPK Crypto Anomaly Screener - Phase 2 Verification Test")
    print(" Modules: prefilter.py | features.py | topsis.py")
    print(f" Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(SEPARATOR)

    mock_ok = test_topsis_mathematical_mock()
    pipeline_ok = test_live_pipeline_integration()

    print("\n" + SEPARATOR)
    print(" FINAL RESULT FOR PHASE 2")
    print(SEPARATOR)
    print(f" Part 1: TOPSIS Unit Test    : {'[PASS] SUCCESS' if mock_ok else '[FAIL] FAILED'}")
    print(f" Part 2: Live Pipeline Test  : {'[PASS] SUCCESS' if pipeline_ok else '[FAIL] FAILED'}")
    print(SEPARATOR)

    if mock_ok and pipeline_ok:
        print(" ALL PHASE 2 TESTS PASSED! Ready for Phase 3.")
        sys.exit(0)
    else:
        print(" SOME TESTS FAILED. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
