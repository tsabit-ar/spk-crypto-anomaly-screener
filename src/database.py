"""SQLite state management for SPK Crypto Anomaly Screener.

Manages alert cooldowns and score delta bypasses to prevent notification spam
while capturing significant new anomaly spikes.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import sqlite3

DEFAULT_DB_PATH = "topsis_screener.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get SQLite database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database and create alerts_history table if it does not exist.

    Args:
        db_path: Path to SQLite database file.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_history (
                symbol TEXT PRIMARY KEY,
                last_alert_time TIMESTAMP NOT NULL,
                last_ci REAL NOT NULL,
                last_price REAL NOT NULL
            );
        """)
        conn.commit()


def should_send_alert(
    symbol: str,
    current_ci: float,
    current_price: float,
    cooldown_hours: float = 4.0,
    delta_bypass: float = 0.15,
    db_path: str = DEFAULT_DB_PATH,
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if an alert should be sent based on cooldown time or Ci score jump.

    Conditions to send alert:
    1. The coin has never been alerted before (New candidate).
    2. Cooldown period elapsed (>= cooldown_hours since last alert).
    3. Significant anomaly surge (current_ci - last_ci >= delta_bypass).

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').
        current_ci: Current TOPSIS preference score (0.0 to 1.0).
        current_price: Current market price of the asset.
        cooldown_hours: Minimum hours between repeat alerts (default: 4).
        delta_bypass: Score increase threshold to bypass cooldown (default: 0.15).
        db_path: SQLite database path.
        current_time: Optional explicit timestamp (for deterministic testing).

    Returns:
        bool: True if alert should be dispatched, False to suppress.
    """
    init_db(db_path)
    symbol = symbol.upper()
    now = current_time or datetime.now(timezone.utc)
    now_iso = now.isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, last_alert_time, last_ci, last_price FROM alerts_history WHERE symbol = ?",
            (symbol,),
        )
        row = cursor.fetchone()

        if row is None:
            # First time alert for this symbol -> Insert & Send
            cursor.execute(
                """
                INSERT INTO alerts_history (symbol, last_alert_time, last_ci, last_price)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, now_iso, current_ci, current_price),
            )
            conn.commit()
            return True

        # Existing record found -> check cooldown & delta bypass
        last_time_raw = row["last_alert_time"]
        if isinstance(last_time_raw, str):
            try:
                last_time = datetime.fromisoformat(last_time_raw)
            except ValueError:
                last_time = datetime.strptime(last_time_raw, "%Y-%m-%d %H:%M:%S")
        else:
            last_time = last_time_raw

        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - last_time).total_seconds() / 3600.0
        last_ci = float(row["last_ci"])
        ci_diff = current_ci - last_ci

        # Check conditions
        is_cooldown_expired = elapsed_hours >= cooldown_hours
        is_delta_bypassed = ci_diff >= delta_bypass

        if is_cooldown_expired or is_delta_bypassed:
            # Update state with latest alert metrics
            cursor.execute(
                """
                UPDATE alerts_history
                SET last_alert_time = ?, last_ci = ?, last_price = ?
                WHERE symbol = ?
                """,
                (now_iso, current_ci, current_price, symbol),
            )
            conn.commit()
            return True

        # Alert suppressed
        return False


def get_alert_record(symbol: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict]:
    """Retrieve existing alert record for a symbol."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts_history WHERE symbol = ?", (symbol.upper(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_all_alerts(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """List all alerted symbols and their state."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts_history ORDER BY last_alert_time DESC")
        return [dict(row) for row in cursor.fetchall()]


def clear_alerts(db_path: str = DEFAULT_DB_PATH) -> None:
    """Clear all records from alerts_history table (useful for resets/tests)."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alerts_history")
        conn.commit()
