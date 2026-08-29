"""Main Pipeline Orchestrator for SPK Crypto Anomaly Screener.

Executes the full end-to-end screening workflow:
1. Ingests 24h bulk tickers from Binance Futures (Public API, $0 cost).
2. Pre-filters Top 25 dynamic & liquid universe candidates.
3. Concurrently extracts 5 multi-criteria market features (C1 - C5).
4. Evaluates and ranks candidates using TOPSIS Multi-Criteria Decision Engine.
5. Filters candidates meeting the anomaly threshold (Ci >= 0.65).
6. Applies SQLite state cooldown logic and dispatches Telegram alerts.
"""

import argparse
from datetime import datetime, timezone
import os
import sys
import time
import pandas as pd

# UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Optional dotenv loader if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.database import init_db, should_send_alert, DEFAULT_DB_PATH
from src.features import extract_batch_metrics
from src.ingestion import fetch_bulk_24h_tickers
from src.prefilter import filter_top_universe
from src.telegram_bot import TelegramDispatcher
from src.topsis import TopsisEngine

SEPARATOR = "=" * 80


def run_pipeline(
    scan_limit: int = 25,
    min_ci_threshold: float = 0.65,
    cooldown_hours: float = 4.0,
    delta_bypass: float = 0.15,
    db_path: str = DEFAULT_DB_PATH,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Execute complete screener pipeline run.

    Args:
        scan_limit: Number of universe candidates to pre-filter (default: 25).
        min_ci_threshold: Minimum TOPSIS Ci score to trigger anomaly alert (default: 0.65).
        cooldown_hours: Minimum hours between repeat alerts for same coin (default: 4.0).
        delta_bypass: Score jump required to bypass cooldown (default: 0.15).
        db_path: SQLite database file path.
        dry_run: If True, do not send live Telegram alerts.

    Returns:
        pd.DataFrame: Ranked candidates DataFrame.
    """
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(SEPARATOR)
    print(" 🚀 SPK CRYPTO ANOMALY SCREENER (BINANCE FUTURES USDT-M)")
    print(f" ⏱️ Execution Time : {now_utc}")
    print(f" ⚙️ Config          : Limit={scan_limit} | Min Ci={min_ci_threshold:.2f} | Cooldown={cooldown_hours}h | Bypass=+{delta_bypass:.2f}")
    print(SEPARATOR)

    # 1. Initialize SQLite State Storage
    init_db(db_path)
    dispatcher = TelegramDispatcher()

    # 2. Ingest 24h Bulk Tickers
    print("\n[Step 1/5] Ingesting Binance Futures 24h bulk ticker data...")
    bulk_data = fetch_bulk_24h_tickers()
    print(f"           -> Fetched {len(bulk_data)} futures contracts.")

    # 3. Pre-filter Universe
    print(f"\n[Step 2/5] Pre-filtering top {scan_limit} dynamic universe candidates...")
    candidates_df = filter_top_universe(bulk_data, limit=scan_limit)
    if candidates_df.empty:
        print("           [!] No candidates passed the pre-filter criteria.")
        return pd.DataFrame()
    print(f"           -> Selected {len(candidates_df)} liquid/active candidates.")

    # 4. Extract Multi-Criteria Features
    print("\n[Step 3/5] Concurrently extracting 5 anomaly criteria (C1 - C5)...")
    features_df = extract_batch_metrics(candidates_df, max_workers=6)
    if features_df.empty:
        print("           [!] Failed to extract features for candidates.")
        return pd.DataFrame()
    print(f"           -> Completed metric extraction for {len(features_df)} pairs.")

    # 5. TOPSIS Decision Engine Ranking
    print("\n[Step 4/5] Computing TOPSIS multi-criteria preference scores (Ci)...")
    engine = TopsisEngine()
    ranked_df = engine.rank_candidates(features_df)

    # 6. Anomaly Filtering & State Management
    print(f"\n[Step 5/5] Evaluating anomaly candidates (Threshold: Ci >= {min_ci_threshold:.2f})...")
    anomalies = ranked_df[ranked_df["topsis_score"] >= min_ci_threshold].copy()

    dispatched_count = 0
    suppressed_count = 0

    if anomalies.empty:
        print("           -> No coins exceeded the anomaly threshold in this scan cycle.")
    else:
        print(f"           -> Found {len(anomalies)} candidate(s) meeting anomaly threshold:")
        for _, row in anomalies.iterrows():
            sym = row["symbol"]
            ci = float(row["topsis_score"])
            price = float(row["last_price"])

            should_alert = should_send_alert(
                symbol=sym,
                current_ci=ci,
                current_price=price,
                cooldown_hours=cooldown_hours,
                delta_bypass=delta_bypass,
                db_path=db_path,
            )

            if should_alert:
                if dispatcher.is_configured and not dry_run:
                    sent = dispatcher.send_alert(row.to_dict())
                    status_text = "DISPATCHED (Telegram Sent)" if sent else "FAILED (Telegram Error)"
                else:
                    status_text = "DISPATCHED (Dry Run / No Token)"
                dispatched_count += 1
            else:
                status_text = "SUPPRESSED (Active Cooldown)"
                suppressed_count += 1

            print(f"              • {sym:<10} | Ci: {ci:.4f} | Rank: #{int(row['rank'])} -> {status_text}")

    # Summary Table Display
    print("\n" + SEPARATOR)
    print(" 📊 TOP 5 ANOMALY SCREENER LEADERBOARD")
    print(SEPARATOR)
    top5 = ranked_df.head(5)
    summary_cols = ["rank", "symbol", "topsis_score", "last_price", "C1", "C2", "C3", "C4", "C5"]
    print(top5[summary_cols].to_string(index=False))

    elapsed = time.time() - t_start
    print(SEPARATOR)
    print(f" ✅ Pipeline Completed in {elapsed:.2f}s | Anomalies Detected: {len(anomalies)} | Dispatched: {dispatched_count} | Suppressed: {suppressed_count}")
    print(SEPARATOR + "\n")

    return ranked_df


def main():
    """CLI Entrypoint for orchestrator."""
    parser = argparse.ArgumentParser(description="SPK Crypto Anomaly Screener Orchestrator")
    parser.add_argument("--limit", type=int, default=int(os.getenv("SCAN_LIMIT", "25")), help="Universe scan limit (default: 25)")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MIN_CI_ALERT_THRESHOLD", os.getenv("MIN_CI_THRESHOLD", "0.65"))), help="Minimum Ci anomaly threshold (default: 0.65)")
    parser.add_argument("--cooldown", type=float, default=float(os.getenv("COOLDOWN_HOURS", "4.0")), help="Alert cooldown in hours (default: 4.0)")
    parser.add_argument("--bypass", type=float, default=float(os.getenv("DELTA_BYPASS", "0.15")), help="Ci jump to bypass cooldown (default: 0.15)")
    parser.add_argument("--db", type=str, default=os.getenv("DB_PATH", DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending real Telegram alerts")

    args = parser.parse_args()

    run_pipeline(
        scan_limit=args.limit,
        min_ci_threshold=args.threshold,
        cooldown_hours=args.cooldown,
        delta_bypass=args.bypass,
        db_path=args.db,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
