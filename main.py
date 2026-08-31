"""Main Pipeline Orchestrator for SPK Crypto Anomaly Screener.

Executes the full end-to-end dual screening workflow (Long & Short):
1. Ingests 24h bulk tickers from Binance Futures (via Vercel proxy / public API, $0 cost).
2. Pre-filters Top 25 dynamic & liquid universe candidates.
3. Concurrently extracts multi-criteria market features (C1 - C5) into a unified snapshot.
4. Concurrently computes Dual TOPSIS Rankings:
   - Long Strategy (topsis_rank): Long Squeeze anomalies.
   - Short Strategy (topsis_short_rank): Overburdened / Overheated anomalies.
5. Evaluates and filters candidates meeting the anomaly threshold (Ci >= 0.65).
6. Applies SQLite state cooldown logic per (symbol, signal_type) and dispatches Telegram alerts.
7. Supports single-run execution and continuous daemon/scheduler loop mode.
"""

import argparse
from datetime import datetime, timedelta, timezone
import os
import sys
import time
from typing import Dict
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
from src.topsis import topsis_rank, topsis_short_rank

SEPARATOR = "=" * 80


def run_pipeline(
    scan_limit: int = 25,
    min_ci_threshold: float = 0.65,
    cooldown_hours: float = 4.0,
    delta_bypass: float = 0.15,
    db_path: str = DEFAULT_DB_PATH,
    dry_run: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Execute complete dual-direction screener pipeline run (Long & Short).

    Args:
        scan_limit: Number of universe candidates to pre-filter (default: 25).
        min_ci_threshold: Minimum TOPSIS Ci score to trigger anomaly alert (default: 0.65).
        cooldown_hours: Minimum hours between repeat alerts for same coin and direction (default: 4.0).
        delta_bypass: Score jump required to bypass cooldown (default: 0.15).
        db_path: SQLite database file path.
        dry_run: If True, do not send live Telegram alerts.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing 'long' and 'short' ranked DataFrames.
    """
    t_start = time.time()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print(SEPARATOR)
    print(" 🚀 SPK CRYPTO ANOMALY SCREENER (BINANCE FUTURES USDT-M) [DUAL LONG/SHORT]")
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
        return {"long": pd.DataFrame(), "short": pd.DataFrame()}
    print(f"           -> Selected {len(candidates_df)} liquid/active candidates.")

    # 4. Extract Multi-Criteria Features (Snapshot)
    print("\n[Step 3/5] Concurrently extracting 5 anomaly criteria (C1 - C5) snapshot...")
    features_df = extract_batch_metrics(candidates_df, max_workers=6)
    if features_df.empty:
        print("           [!] Failed to extract features for candidates.")
        return {"long": pd.DataFrame(), "short": pd.DataFrame()}
    print(f"           -> Completed metric snapshot extraction for {len(features_df)} pairs.")

    # 5. Dual TOPSIS Ranking (Long & Short)
    print("\n[Step 4/5] Computing Dual TOPSIS Rankings (topsis_rank & topsis_short_rank)...")
    long_ranked_df = topsis_rank(features_df)
    short_ranked_df = topsis_short_rank(features_df)
    print(f"           -> Long Ranked: {len(long_ranked_df)} pairs | Short Ranked: {len(short_ranked_df)} pairs.")

    # 6. Dual Anomaly Filtering & State Management
    print(f"\n[Step 5/5] Evaluating Long & Short anomalies (Threshold: Ci >= {min_ci_threshold:.2f})...")

    total_dispatched = 0
    total_suppressed = 0

    # --- Process LONG Anomalies ---
    long_anomalies = long_ranked_df[long_ranked_df["topsis_score"] >= min_ci_threshold].copy() if not long_ranked_df.empty else pd.DataFrame()
    print(f"\n  [🟢 LONG SQUEEZE SIGNALS] - Found {len(long_anomalies)} candidate(s):")
    if long_anomalies.empty:
        print("    -> No coins met the Long anomaly threshold in this scan cycle.")
    else:
        for _, row in long_anomalies.iterrows():
            sym = row["symbol"]
            ci = float(row["topsis_score"])
            price = float(row["last_price"])

            should_alert = should_send_alert(
                symbol=sym,
                current_ci=ci,
                current_price=price,
                cooldown_hours=cooldown_hours,
                delta_bypass=delta_bypass,
                signal_type="LONG",
                db_path=db_path,
            )

            if should_alert:
                if dispatcher.is_configured and not dry_run:
                    sent = dispatcher.send_alert(row.to_dict(), signal_type="LONG")
                    status_text = "DISPATCHED (Telegram Sent)" if sent else "FAILED (Telegram Error)"
                else:
                    status_text = "DISPATCHED (Dry Run / No Token)"
                total_dispatched += 1
            else:
                status_text = "SUPPRESSED (Active Cooldown)"
                total_suppressed += 1

            print(f"    • {sym:<10} | Ci: {ci:.4f} | Rank: #{int(row['rank'])} -> {status_text}")

    # --- Process SHORT Anomalies ---
    short_anomalies = short_ranked_df[short_ranked_df["topsis_score"] >= min_ci_threshold].copy() if not short_ranked_df.empty else pd.DataFrame()
    print(f"\n  [🔴 SHORT WARNING SIGNALS] - Found {len(short_anomalies)} candidate(s):")
    if short_anomalies.empty:
        print("    -> No coins met the Short anomaly threshold in this scan cycle.")
    else:
        for _, row in short_anomalies.iterrows():
            sym = row["symbol"]
            ci = float(row["topsis_score"])
            price = float(row["last_price"])

            should_alert = should_send_alert(
                symbol=sym,
                current_ci=ci,
                current_price=price,
                cooldown_hours=cooldown_hours,
                delta_bypass=delta_bypass,
                signal_type="SHORT",
                db_path=db_path,
            )

            if should_alert:
                if dispatcher.is_configured and not dry_run:
                    sent = dispatcher.send_alert(row.to_dict(), signal_type="SHORT")
                    status_text = "DISPATCHED (Telegram Sent)" if sent else "FAILED (Telegram Error)"
                else:
                    status_text = "DISPATCHED (Dry Run / No Token)"
                total_dispatched += 1
            else:
                status_text = "SUPPRESSED (Active Cooldown)"
                total_suppressed += 1

            print(f"    • {sym:<10} | Ci: {ci:.4f} | Rank: #{int(row['rank'])} -> {status_text}")

    # Summary Tables Display
    summary_cols = ["rank", "symbol", "topsis_score", "last_price", "C1", "C2", "C3", "C4", "C5"]

    print("\n" + SEPARATOR)
    print(" 📊 TOP 5 LONG ANOMALIES [🟢 LONG SQUEEZE]")
    print(SEPARATOR)
    if not long_ranked_df.empty:
        print(long_ranked_df.head(5)[summary_cols].to_string(index=False))
    else:
        print("No candidates available.")

    print("\n" + SEPARATOR)
    print(" 📊 TOP 5 SHORT ANOMALIES [🔴 SHORT WARNING]")
    print(SEPARATOR)
    if not short_ranked_df.empty:
        print(short_ranked_df.head(5)[summary_cols].to_string(index=False))
    else:
        print("No candidates available (C1 >= 0 required).")

    elapsed = time.time() - t_start
    total_anomalies = len(long_anomalies) + len(short_anomalies)
    print("\n" + SEPARATOR)
    print(f" ✅ Pipeline Completed in {elapsed:.2f}s | Anomalies: {total_anomalies} (Long: {len(long_anomalies)}, Short: {len(short_anomalies)}) | Dispatched: {total_dispatched} | Suppressed: {total_suppressed}")
    print(SEPARATOR + "\n")

    return {"long": long_ranked_df, "short": short_ranked_df}


def start_scheduler(
    interval_seconds: int = 900,
    scan_limit: int = 25,
    min_ci_threshold: float = 0.65,
    cooldown_hours: float = 4.0,
    delta_bypass: float = 0.15,
    db_path: str = DEFAULT_DB_PATH,
    dry_run: bool = False,
) -> None:
    """Run continuous dual-direction anomaly screener loop at fixed intervals.

    Args:
        interval_seconds: Delay between scans in seconds (default: 900 / 15 mins).
        scan_limit: Number of universe candidates to pre-filter (default: 25).
        min_ci_threshold: Minimum TOPSIS Ci score to trigger anomaly alert (default: 0.65).
        cooldown_hours: Minimum hours between repeat alerts for same coin/direction (default: 4.0).
        delta_bypass: Score jump required to bypass cooldown (default: 0.15).
        db_path: SQLite database file path.
        dry_run: If True, do not send live Telegram alerts.
    """
    interval_mins = interval_seconds / 60.0
    print("\n" + SEPARATOR)
    print(f" 🔄 STARTING DUAL DAEMON SCREENER SCHEDULER (Interval: {interval_seconds}s / {interval_mins:.1f} mins)")
    print(" Press Ctrl+C to stop the scheduler.")
    print(SEPARATOR + "\n")

    iteration = 1
    try:
        while True:
            cycle_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n>>> [Scan Cycle #{iteration}] Started at {cycle_start}")
            try:
                run_pipeline(
                    scan_limit=scan_limit,
                    min_ci_threshold=min_ci_threshold,
                    cooldown_hours=cooldown_hours,
                    delta_bypass=delta_bypass,
                    db_path=db_path,
                    dry_run=dry_run,
                )
            except Exception as e:
                print(f"\n[!] Error during pipeline cycle #{iteration}: {e}")
                print("    Recovering automatically for next scheduled cycle...")

            next_run_dt = datetime.now(timezone.utc) + timedelta(seconds=interval_seconds)
            next_run_str = next_run_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"⏳ Next scan in {interval_mins:.1f} minutes ({next_run_str}). Sleeping...\n")

            time.sleep(interval_seconds)
            iteration += 1

    except KeyboardInterrupt:
        print("\n" + SEPARATOR)
        print(" 🛑 Scheduler stopped by user (Ctrl+C). Exiting gracefully.")
        print(SEPARATOR + "\n")
        sys.exit(0)


def main():
    """CLI Entrypoint for orchestrator."""
    parser = argparse.ArgumentParser(description="SPK Crypto Anomaly Screener Orchestrator (Dual Long/Short)")
    parser.add_argument("--limit", type=int, default=int(os.getenv("SCAN_LIMIT", "25")), help="Universe scan limit (default: 25)")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("MIN_CI_ALERT_THRESHOLD", os.getenv("MIN_CI_THRESHOLD", "0.65"))), help="Minimum Ci anomaly threshold (default: 0.65)")
    parser.add_argument("--cooldown", type=float, default=float(os.getenv("COOLDOWN_HOURS", "4.0")), help="Alert cooldown in hours (default: 4.0)")
    parser.add_argument("--bypass", type=float, default=float(os.getenv("DELTA_BYPASS", "0.15")), help="Ci jump to bypass cooldown (default: 0.15)")
    parser.add_argument("--db", type=str, default=os.getenv("DB_PATH", DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending real Telegram alerts")
    parser.add_argument("--loop", action="store_true", help="Run in continuous daemon loop mode")
    parser.add_argument("--interval", type=int, default=int(os.getenv("SCAN_INTERVAL_SECONDS", "900")), help="Loop interval in seconds (default: 900 / 15 mins)")

    args = parser.parse_args()

    if args.loop:
        start_scheduler(
            interval_seconds=args.interval,
            scan_limit=args.limit,
            min_ci_threshold=args.threshold,
            cooldown_hours=args.cooldown,
            delta_bypass=args.bypass,
            db_path=args.db,
            dry_run=args.dry_run,
        )
    else:
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
