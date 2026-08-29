"""Telegram Alert Dispatcher module for SPK Crypto Anomaly Screener.

Formats and sends real-time anomaly alerts to Telegram channel/group via Telegram Bot API.
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, Optional
import requests

TELEGRAM_API_BASE_URL = "https://api.telegram.org"
REQUEST_TIMEOUT = 10  # seconds


class TelegramDispatcher:
    """Dispatches formatted cryptocurrency anomaly alerts to Telegram."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize TelegramDispatcher with bot token and target chat ID.

        Args:
            bot_token: Telegram bot token (falls back to TELEGRAM_BOT_TOKEN env var).
            chat_id: Target chat or channel ID (falls back to TELEGRAM_CHAT_ID env var).
        """
        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

    @property
    def is_configured(self) -> bool:
        """Check if both Bot Token and Chat ID are properly configured."""
        return bool(self.bot_token and self.chat_id)

    def format_alert_message(self, row: Dict[str, Any]) -> str:
        """Format an anomaly candidate record into a clean Telegram Markdown message.

        Args:
            row: Dictionary containing coin metadata and TOPSIS metrics.

        Returns:
            str: Formatted Markdown message ready to send.
        """
        symbol = str(row.get("symbol", "UNKNOWN"))
        rank = row.get("rank", "-")
        ci = float(row.get("topsis_score", 0.0))
        last_price = float(row.get("last_price", 0.0))
        price_change_24h = float(row.get("price_change_24h", 0.0))
        quote_vol_24h = float(row.get("quote_volume_24h", 0.0))
        oi_value = float(row.get("oi_value_usdt", 0.0))

        c1_fr = float(row.get("C1", 0.0))
        c2_oi = float(row.get("C2", 0.0))
        c3_bbw = float(row.get("C3", 0.0))
        c4_depth = float(row.get("C4", 0.0))
        c5_vel = float(row.get("C5", 0.0))

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Visual indicator emoji based on score
        badge = "🔥 CRITICAL ANOMALY" if ci >= 0.75 else "⚡ ANOMALY DETECTED"

        message = (
            f"🚨 *SPK CRYPTO SCREENER — {badge}*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🪙 *Pair:* `{symbol}` (Binance Futures)\n"
            f"🏆 *Rank:* `#{rank}`  |  *TOPSIS Score (Ci):* `{ci:.4f}`\n"
            f"💵 *Price:* `${last_price:,.4f}` ({price_change_24h:+.2f}% 24h)\n"
            f"📊 *24h Volume:* `${quote_vol_24h:,.0f} USDT`\n\n"
            f"*🔍 Multi-Criteria Market Signals:*\n"
            f"• *C1 [Cost] Funding Rate:* `{c1_fr:+.4f}%`\n"
            f"• *C2 [Benefit] 4H Delta OI:* `{c2_oi:+.2f}%` (OI: `${oi_value:,.0f}`)\n"
            f"• *C3 [Cost] 1H BBW:* `{c3_bbw:.2f}%` (Compression)\n"
            f"• *C4 [Benefit] 2% Depth Imbalance:* `{c4_depth:.2f}x` (Bid/Ask)\n"
            f"• *C5 [Benefit] Volume/OI Velocity:* `{c5_vel:.4f}`\n\n"
            f"⏰ *Time:* `{now_utc}`\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💡 _$0-Cost Autonomous Decision Support System_"
        )
        return message

    def send_raw_message(self, text: str) -> bool:
        """Send raw text message to Telegram channel.

        Args:
            text: Markdown-formatted message string.

        Returns:
            bool: True if sent successfully, False otherwise.
        """
        if not self.is_configured:
            return False

        url = f"{TELEGRAM_API_BASE_URL}/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return True
            else:
                print(f"[TelegramDispatcher Error] HTTP {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            print(f"[TelegramDispatcher Network Error] {e}")
            return False

    def send_alert(self, row: Dict[str, Any]) -> bool:
        """Format and send an alert for a specific candidate.

        Args:
            row: Dictionary containing candidate metrics.

        Returns:
            bool: True if alert was sent successfully, False otherwise.
        """
        text = self.format_alert_message(row)
        return self.send_raw_message(text)
