"""Comprehensive verification test for Short Signal Integration (Phase 3+).

Validates:
1. TOPSIS Short ranking logic & C1 < 0 filtering.
2. SQLite schema migration and (symbol, signal_type) independent cooldowns.
3. Telegram message formatting for Long [🟢 LONG SQUEEZE] vs Short [🔴 SHORT WARNING].
4. Live dual pipeline execution.
"""

from datetime import datetime, timedelta, timezone
import os
import sys
import pandas as pd
import numpy as np

# UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.database import (
    init_db,
    should_send_alert,
    get_alert_record,
    clear_alerts,
)
from src.topsis import topsis_rank, topsis_short_rank
from src.telegram_bot import TelegramDispatcher
from main import run_pipeline

TEST_DB_PATH = "test_short_screener.db"
SEPARATOR = "=" * 80


def test_topsis_short_logic() -> bool:
    """Test TOPSIS Short ranking algorithm and C1 < 0 filtering."""
    print(f"\n[TEST 1] TOPSIS Short Engine Logic & Filtering")
    print("-" * 80)

    # Synthetic candidate scenarios:
    # - COIN_OVERHEATED: High positive funding (C1=0.08%), High OI (C2=30%), High BBW expansion (C3=8.5%), Heavy Ask Wall (ask > bid -> ask/bid = 3.0), High Velocity (C5=1.5) -> Should be Rank #1 Short
    # - COIN_MODERATE: Positive funding (C1=0.01%), moderate OI (C2=5%), moderate BBW (C3=2.5%), balanced depth (1.0), moderate velocity (0.5)
    # - COIN_NEGATIVE_FUNDING: Negative funding (C1=-0.05%) -> MUST be discarded by Short filter!
    mock_df = pd.DataFrame([
        {
            "symbol": "COIN_OVERHEATED",
            "C1": 0.08,
            "C2": 30.0,
            "C3": 8.5,
            "C4": 0.33,  # Raw C4 (bid/ask). In short, ask/bid will be ~3.0
            "C5": 1.5,
            "bid_liq_2pct": 100000.0,
            "ask_liq_2pct": 300000.0,
            "last_price": 50.0,
            "price_change_24h": 15.0,
            "quote_volume_24h": 10000000.0,
            "oi_value_usdt": 5000000.0,
        },
        {
            "symbol": "COIN_MODERATE",
            "C1": 0.01,
            "C2": 5.0,
            "C3": 2.5,
            "C4": 1.0,
            "C5": 0.5,
            "bid_liq_2pct": 200000.0,
            "ask_liq_2pct": 200000.0,
            "last_price": 10.0,
            "price_change_24h": 2.0,
            "quote_volume_24h": 5000000.0,
            "oi_value_usdt": 2000000.0,
        },
        {
            "symbol": "COIN_NEGATIVE_FUNDING",
            "C1": -0.05,  # Negative funding -> MUST be filtered out of Short
            "C2": 40.0,
            "C3": 9.0,
            "C4": 0.2,
            "C5": 2.0,
            "bid_liq_2pct": 50000.0,
            "ask_liq_2pct": 250000.0,
            "last_price": 5.0,
            "price_change_24h": -8.0,
            "quote_volume_24h": 8000000.0,
            "oi_value_usdt": 3000000.0,
        },
    ])

    short_ranked = topsis_short_rank(mock_df)

    # 1. Check C1 < 0 filtering
    symbols = short_ranked["symbol"].tolist()
    if "COIN_NEGATIVE_FUNDING" in symbols:
        print("  -> [FAIL] COIN_NEGATIVE_FUNDING (C1 < 0) was NOT filtered out!")
        return False
    print("  -> [PASS] Negative funding coin correctly discarded from Short screening.")

    # 2. Check Rank #1
    top_short = short_ranked.iloc[0]["symbol"]
    if top_short != "COIN_OVERHEATED":
        print(f"  -> [FAIL] Expected COIN_OVERHEATED to be Rank #1 Short, got: {top_short}")
        return False
    print(f"  -> [PASS] COIN_OVERHEATED correctly ranked #1 Short with Ci={short_ranked.iloc[0]['topsis_score']:.4f}")

    # 3. Check C4 inversion
    c4_val = short_ranked.loc[short_ranked["symbol"] == "COIN_OVERHEATED", "C4"].values[0]
    expected_c4 = 300000.0 / 100000.0  # 3.0
    if abs(c4_val - expected_c4) > 0.01:
        print(f"  -> [FAIL] C4 not inverted to Ask/Bid properly! Expected ~3.0, got: {c4_val}")
        return False
    print(f"  -> [PASS] C4 correctly inverted to Ask/Bid ratio ({c4_val:.2f}x).")

    return True


def test_sqlite_dual_state() -> bool:
    """Test SQLite composite key (symbol, signal_type) and independent cooldowns."""
    print(f"\n[TEST 2] SQLite Composite (symbol, signal_type) State & Cooldowns")
    print("-" * 80)

    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    init_db(TEST_DB_PATH)
    t0 = datetime(2026, 8, 31, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Alert LONG for 'SOLUSDT'
    print(" 2.1 Inserting LONG alert for 'SOLUSDT' (Ci=0.75)...")
    res_long = should_send_alert("SOLUSDT", current_ci=0.75, current_price=105.0, signal_type="LONG", db_path=TEST_DB_PATH, current_time=t0)
    if not res_long:
        print("     -> [FAIL] Initial LONG alert failed.")
        return False

    # 2. Alert SHORT for 'SOLUSDT' at same timestamp -> MUST return True because signal_type is different!
    print(" 2.2 Inserting SHORT alert for 'SOLUSDT' at same time (Ci=0.80)...")
    res_short = should_send_alert("SOLUSDT", current_ci=0.80, current_price=105.0, signal_type="SHORT", db_path=TEST_DB_PATH, current_time=t0)
    if not res_short:
        print("     -> [FAIL] SHORT alert was blocked by LONG alert state! (Should be independent).")
        return False
    print("     -> [PASS] LONG and SHORT maintain independent states for the same symbol.")

    # 3. Immediate repeat LONG -> Must be SUPPRESSED
    t_repeat = t0 + timedelta(minutes=20)
    print(" 2.3 Testing repeat LONG alert after 20 mins (Cooldown 4h)...")
    res_repeat_long = should_send_alert("SOLUSDT", current_ci=0.75, current_price=106.0, signal_type="LONG", db_path=TEST_DB_PATH, current_time=t_repeat)
    if res_repeat_long:
        print("     -> [FAIL] Repeat LONG alert was not suppressed.")
        return False
    print("     -> [PASS] Repeat LONG alert properly suppressed.")

    # 4. Immediate repeat SHORT with delta bypass (+0.18 jump) -> Must be DISPATCHED
    print(" 2.4 Testing SHORT alert with Ci surge (0.80 -> 0.98, delta +0.18)...")
    res_bypass_short = should_send_alert("SOLUSDT", current_ci=0.98, current_price=108.0, signal_type="SHORT", delta_bypass=0.15, db_path=TEST_DB_PATH, current_time=t_repeat)
    if not res_bypass_short:
        print("     -> [FAIL] Delta bypass for SHORT failed.")
        return False
    print("     -> [PASS] SHORT alert correctly triggered on delta bypass.")

    # Cleanup test db
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    return True


def test_telegram_formatting() -> bool:
    """Test Telegram message formatting for both Long and Short."""
    print(f"\n[TEST 3] Telegram Message Formatter (Long & Short Matrix)")
    print("-" * 80)

    dispatcher = TelegramDispatcher(bot_token="", chat_id="")

    sample_long = {
        "symbol": "BTCUSDT",
        "rank": 1,
        "topsis_score": 0.82,
        "last_price": 60000.0,
        "price_change_24h": 3.5,
        "quote_volume_24h": 2000000000.0,
        "oi_value_usdt": 500000000.0,
        "C1": -0.025,
        "C2": 15.0,
        "C3": 1.8,
        "C4": 2.1,
        "C5": 0.05,
    }

    sample_short = {
        "symbol": "ETHUSDT",
        "rank": 1,
        "topsis_score": 0.78,
        "last_price": 3000.0,
        "price_change_24h": 12.0,
        "quote_volume_24h": 800000000.0,
        "oi_value_usdt": 300000000.0,
        "C1": 0.065,
        "C2": 22.0,
        "C3": 6.5,
        "C4": 2.8,
        "C5": 0.12,
    }

    msg_long = dispatcher.format_alert_message(sample_long, signal_type="LONG")
    msg_short = dispatcher.format_alert_message(sample_short, signal_type="SHORT")

    # Validate Long Message
    if "[🟢 LONG SQUEEZE]" not in msg_long or "C1 [Cost]" not in msg_long or "C3 [Cost]" not in msg_long:
        print("  -> [FAIL] Long message missing expected [🟢 LONG SQUEEZE] or [Cost] labels.")
        return False
    print("  -> [PASS] Long message header [🟢 LONG SQUEEZE] and [Cost] labels verified.")

    # Validate Short Message
    if "[🔴 SHORT WARNING]" not in msg_short or "C1 [Benefit]" not in msg_short or "C3 [Benefit]" not in msg_short:
        print("  -> [FAIL] Short message missing expected [🔴 SHORT WARNING] or [Benefit] labels.")
        return False
    print("  -> [PASS] Short message header [🔴 SHORT WARNING] and [Benefit] labels verified.")

    print("\n--- Long Alert Preview ---")
    print(msg_long)
    print("\n--- Short Alert Preview ---")
    print(msg_short)

    return True


def test_live_dual_pipeline() -> bool:
    """Test full dual pipeline run via main.run_pipeline."""
    print(f"\n[TEST 4] Live Dual Pipeline Run (Binance Futures Live Data)")
    print("-" * 80)

    try:
        results = run_pipeline(
            scan_limit=15,
            min_ci_threshold=0.50,
            cooldown_hours=4.0,
            delta_bypass=0.15,
            db_path=TEST_DB_PATH,
            dry_run=True,
        )

        long_df = results.get("long", pd.DataFrame())
        short_df = results.get("short", pd.DataFrame())

        if long_df.empty:
            print("  -> [FAIL] Long pipeline returned empty DataFrame.")
            return False

        print(f"  -> [PASS] Long pipeline ranked {len(long_df)} candidates.")
        print(f"  -> [PASS] Short pipeline ranked {len(short_df)} candidates (C1 >= 0 filtered).")
        return True

    except Exception as e:
        print(f"  -> [FAIL] Live pipeline failed: {e}")
        return False
    finally:
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except OSError:
                pass


def main():
    print(SEPARATOR)
    print(" SPK Crypto Anomaly Screener - Short Signal Integration Verification")
    print(f" Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(SEPARATOR)

    t1 = test_topsis_short_logic()
    t2 = test_sqlite_dual_state()
    t3 = test_telegram_formatting()
    t4 = test_live_dual_pipeline()

    print("\n" + SEPARATOR)
    print(" VERIFICATION SUMMARY")
    print(SEPARATOR)
    print(f" 1. TOPSIS Short Logic & Filtering   : {'[PASS] SUCCESS' if t1 else '[FAIL] FAILED'}")
    print(f" 2. SQLite Composite State Cooldown  : {'[PASS] SUCCESS' if t2 else '[FAIL] FAILED'}")
    print(f" 3. Telegram Long/Short Formatter    : {'[PASS] SUCCESS' if t3 else '[FAIL] FAILED'}")
    print(f" 4. Live Dual Pipeline Execution     : {'[PASS] SUCCESS' if t4 else '[FAIL] FAILED'}")
    print(SEPARATOR)

    if t1 and t2 and t3 and t4:
        print(" ALL SHORT INTEGRATION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(" SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
