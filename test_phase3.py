"""Test script for Phase 3: SQLite State Management, Telegram Bot Dispatcher, & Orchestrator.

Validates:
1. SQLite state machine (insert, duplicate cooldown suppression, delta bypass, cooldown expiration).
2. Telegram alert message formatting & dispatcher behavior.
3. End-to-end orchestrator pipeline run.
"""

from datetime import datetime, timedelta, timezone
import os
import sys
import pandas as pd

# UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.database import (
    init_db,
    should_send_alert,
    get_alert_record,
    clear_alerts,
)
from src.telegram_bot import TelegramDispatcher
from main import run_pipeline

TEST_DB_PATH = "test_phase3_screener.db"
SEPARATOR = "=" * 80


def test_sqlite_state_management() -> bool:
    """Test SQLite state machine and cooldown logic."""
    print(f"\n[PART 1] Testing SQLite State Machine & Alert Cooldown Logic")
    print("-" * 80)

    # Initialize fresh test database
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    init_db(TEST_DB_PATH)
    t0 = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)

    # 1. Test First Time Alert -> Expect True
    print(" 1.1 Testing first-time alert insertion for 'SOLUSDT' (Ci=0.68)...")
    res1 = should_send_alert("SOLUSDT", current_ci=0.68, current_price=105.5, db_path=TEST_DB_PATH, current_time=t0)
    rec1 = get_alert_record("SOLUSDT", db_path=TEST_DB_PATH)
    if not res1 or rec1 is None or float(rec1["last_ci"]) != 0.68:
        print("     -> [FAIL] First-time alert failed to insert or returned False.")
        return False
    print("     -> [PASS] First-time alert inserted successfully.")

    # 2. Test Immediate Repeat Alert -> Expect False (Suppressed)
    t_immediate = t0 + timedelta(minutes=15)
    print(" 1.2 Testing repeat alert after 15 mins with same Ci=0.68 (Cooldown: 4h)...")
    res2 = should_send_alert("SOLUSDT", current_ci=0.68, current_price=106.0, db_path=TEST_DB_PATH, current_time=t_immediate)
    if res2:
        print("     -> [FAIL] Repeat alert was not suppressed during cooldown!")
        return False
    print("     -> [PASS] Repeat alert correctly suppressed by cooldown.")

    # 3. Test Delta Bypass Alert -> Expect True (Ci jump >= +0.15)
    t_jump = t0 + timedelta(minutes=30)
    print(" 1.3 Testing delta bypass alert with Ci surge +0.18 (Ci: 0.68 -> 0.86)...")
    res3 = should_send_alert("SOLUSDT", current_ci=0.86, current_price=112.0, delta_bypass=0.15, db_path=TEST_DB_PATH, current_time=t_jump)
    rec3 = get_alert_record("SOLUSDT", db_path=TEST_DB_PATH)
    if not res3 or float(rec3["last_ci"]) != 0.86:
        print("     -> [FAIL] Delta bypass failed to trigger alert or update DB.")
        return False
    print("     -> [PASS] Delta bypass (+0.18 Ci jump) correctly bypassed cooldown and updated state.")

    # 4. Test Cooldown Expiration -> Expect True (After 4.5 hours)
    t_expired = t_jump + timedelta(hours=4.5)
    print(" 1.4 Testing alert after cooldown expiry (4.5h later, Ci=0.72)...")
    res4 = should_send_alert("SOLUSDT", current_ci=0.72, current_price=110.0, cooldown_hours=4.0, db_path=TEST_DB_PATH, current_time=t_expired)
    rec4 = get_alert_record("SOLUSDT", db_path=TEST_DB_PATH)
    if not res4 or float(rec4["last_ci"]) != 0.72:
        print("     -> [FAIL] Expired cooldown failed to trigger alert.")
        return False
    print("     -> [PASS] Alert successfully triggered and state updated after cooldown expiration.")

    # Cleanup test db
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

    return True


def test_telegram_dispatcher() -> bool:
    """Test Telegram message formatting and dispatcher handling."""
    print(f"\n[PART 2] Testing Telegram Message Formatter & Dispatcher")
    print("-" * 80)

    dispatcher = TelegramDispatcher(bot_token="", chat_id="")

    sample_candidate = {
        "symbol": "SOLUSDT",
        "rank": 1,
        "topsis_score": 0.8421,
        "last_price": 105.42,
        "price_change_24h": 6.85,
        "quote_volume_24h": 3450000000.0,
        "oi_value_usdt": 890000000.0,
        "C1": -0.0450,
        "C2": 18.50,
        "C3": 2.15,
        "C4": 2.45,
        "C5": 0.0820,
    }

    print(" 2.1 Formatting sample Telegram alert message...")
    message = dispatcher.format_alert_message(sample_candidate)

    # Validate message contents
    required_keywords = ["SOLUSDT", "0.8421", "#1", "C1", "C2", "C3", "C4", "C5"]
    for kw in required_keywords:
        if kw not in message:
            print(f"     -> [FAIL] Message missing expected keyword: '{kw}'")
            return False

    print("     -> [PASS] Telegram Markdown message formatted properly:\n")
    print("     --- [ PREVIEW MESSAGE START ] ---")
    for line in message.split("\n"):
        print(f"     | {line}")
    print("     --- [ PREVIEW MESSAGE END ] ---\n")

    # 2.2 Test unconfigured dispatcher safe fallback
    print(" 2.2 Verifying unconfigured dispatcher handling...")
    if dispatcher.is_configured:
        print("     -> [FAIL] Empty credentials should indicate unconfigured.")
        return False
    sent_result = dispatcher.send_alert(sample_candidate)
    if sent_result is not False:
        print("     -> [FAIL] Unconfigured dispatcher should return False gracefully.")
        return False
    print("     -> [PASS] Unconfigured dispatcher safely handled without crashing.")

    return True


def test_live_pipeline_orchestrator() -> bool:
    """Test full end-to-end pipeline run via main.run_pipeline."""
    print(f"\n[PART 3] Testing Live End-to-End Orchestrator Pipeline")
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
            print("     -> [FAIL] Orchestrator pipeline returned empty Long DataFrame.")
            return False

        if "rank" not in long_df.columns or "topsis_score" not in long_df.columns:
            print("     -> [FAIL] Output DataFrame missing 'rank' or 'topsis_score' columns.")
            return False

        print(f"     -> [PASS] Live pipeline successfully processed and ranked {len(long_df)} Long pairs and {len(short_df)} Short pairs.")
        return True

    except Exception as e:
        print(f"     -> [FAIL] Pipeline failed with exception: {e}")
        return False
    finally:
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except OSError:
                pass


def main():
    """Main runner for Phase 3 tests."""
    print(SEPARATOR)
    print(" SPK Crypto Anomaly Screener - Phase 3 Verification Test")
    print(" Modules: database.py | telegram_bot.py | main.py")
    print(f" Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(SEPARATOR)

    db_ok = test_sqlite_state_management()
    tg_ok = test_telegram_dispatcher()
    pipeline_ok = test_live_pipeline_orchestrator()

    print("\n" + SEPARATOR)
    print(" FINAL RESULT FOR PHASE 3")
    print(SEPARATOR)
    print(f" Part 1: SQLite State & Cooldown Logic : {'[PASS] SUCCESS' if db_ok else '[FAIL] FAILED'}")
    print(f" Part 2: Telegram Formatter/Dispatcher  : {'[PASS] SUCCESS' if tg_ok else '[FAIL] FAILED'}")
    print(f" Part 3: Live Pipeline Orchestrator    : {'[PASS] SUCCESS' if pipeline_ok else '[FAIL] FAILED'}")
    print(SEPARATOR)

    if db_ok and tg_ok and pipeline_ok:
        print(" ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(" SOME TESTS FAILED. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
