"""SQLite state management for SPK Crypto Anomaly Screener.

Manages alert cooldowns and score delta bypasses to prevent notification spam
while capturing significant new anomaly spikes. Supports dual signal types (LONG / SHORT).
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
    """Initialize SQLite database and ensure schema supports (symbol, signal_type) composite key.

    Args:
        db_path: Path to SQLite database file.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Create table with composite primary key if brand new
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts_history (
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL DEFAULT 'LONG',
                last_alert_time TIMESTAMP NOT NULL,
                last_ci REAL NOT NULL,
                last_price REAL NOT NULL,
                PRIMARY KEY (symbol, signal_type)
            );
        """)

        # Safe schema migration: check if signal_type column exists
        cursor.execute("PRAGMA table_info(alerts_history)")
        columns = [col[1] for col in cursor.fetchall()]
        if "signal_type" not in columns:
            try:
                cursor.execute("ALTER TABLE alerts_history ADD COLUMN signal_type TEXT NOT NULL DEFAULT 'LONG'")
            except Exception:
                pass

        conn.commit()


def should_send_alert(
    symbol: str,
    current_ci: float,
    current_price: float,
    cooldown_hours: float = 4.0,
    delta_bypass: float = 0.15,
    signal_type: str = "LONG",
    db_path: str = DEFAULT_DB_PATH,
    current_time: Optional[datetime] = None,
) -> bool:
    """Check if an alert should be sent based on cooldown time or Ci score jump per (symbol, signal_type).

    Conditions to send alert:
    1. The (symbol, signal_type) pair has never been alerted before (New candidate).
    2. Cooldown period elapsed (>= cooldown_hours since last alert for this direction).
    3. Significant anomaly surge (current_ci - last_ci >= delta_bypass).

    Args:
        symbol: Trading pair symbol (e.g. 'SOLUSDT').
        current_ci: Current TOPSIS preference score (0.0 to 1.0).
        current_price: Current market price of the asset.
        cooldown_hours: Minimum hours between repeat alerts (default: 4.0).
        delta_bypass: Score increase threshold to bypass cooldown (default: 0.15).
        signal_type: Signal direction ('LONG' or 'SHORT', default: 'LONG').
        db_path: SQLite database path.
        current_time: Optional explicit timestamp (for deterministic testing).

    Returns:
        bool: True if alert should be dispatched, False to suppress.
    """
    init_db(db_path)
    symbol = symbol.upper().strip()
    signal_type = signal_type.upper().strip()
    now = current_time or datetime.now(timezone.utc)
    now_iso = now.isoformat()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT symbol, signal_type, last_alert_time, last_ci, last_price
            FROM alerts_history
            WHERE symbol = ? AND signal_type = ?
            """,
            (symbol, signal_type),
        )
        row = cursor.fetchone()

        if row is None:
            # First time alert for this symbol + signal_type -> Insert & Send
            cursor.execute(
                """
                INSERT INTO alerts_history (symbol, signal_type, last_alert_time, last_ci, last_price)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, signal_type, now_iso, current_ci, current_price),
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
            # Update state with latest alert metrics for this symbol + signal_type
            cursor.execute(
                """
                UPDATE alerts_history
                SET last_alert_time = ?, last_ci = ?, last_price = ?
                WHERE symbol = ? AND signal_type = ?
                """,
                (now_iso, current_ci, current_price, symbol, signal_type),
            )
            conn.commit()
            return True

        # Alert suppressed
        return False


def get_alert_record(
    symbol: str,
    signal_type: str = "LONG",
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict]:
    """Retrieve existing alert record for a symbol and signal_type."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM alerts_history WHERE symbol = ? AND signal_type = ?",
            (symbol.upper().strip(), signal_type.upper().strip()),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def list_all_alerts(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """List all alerted symbols, signal types, and their state."""
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
