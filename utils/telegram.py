"""
Telegram Alert System for Sentinel Autotrader.

Sends real-time notifications for:
- Trade executions (buy/sell/short)
- Circuit breaker triggers
- Errors and warnings
- Daily P&L summaries
- Cycle completion summaries
"""
import requests
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger("sentinel.telegram")


class TelegramAlert:
    """
    Sends formatted messages to a Telegram chat via Bot API.
    Gracefully no-ops if TELEGRAM_BOT_TOKEN is not configured.
    """

    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.info("Telegram alerts disabled (no TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")

    def _send(self, text: str, parse_mode: str = "HTML"):
        """Sends a message to Telegram. Silently fails if not configured."""
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }, timeout=10)
            if resp.status_code != 200:
                log.debug(f"Telegram send failed: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            log.debug(f"Telegram send error: {e}")

    # ─── Alert Types ─────────────────────────────────────────────────

    def trade_executed(self, symbol: str, side: str, notional: float,
                       confidence: float, fill_price: float = 0,
                       trailing_stop_pct: float = 0):
        """Alert when a trade is executed."""
        emoji = "🟢" if side == "BUY" else "🔴"
        self._send(
            f"{emoji} <b>TRADE EXECUTED</b>\n\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Side: <b>{side}</b>\n"
            f"Amount: <b>${notional:,.2f}</b>\n"
            f"Fill Price: ${fill_price:,.2f}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Trailing Stop: {trailing_stop_pct:.1f}%"
        )

    def trade_skipped(self, symbol: str, reason: str):
        """Alert when a trade candidate is skipped."""
        self._send(
            f"⏭️ <b>SKIP</b>: <code>{symbol}</code>\n"
            f"Reason: {reason[:200]}"
        )

    def circuit_breaker(self, drawdown_pct: float, equity: float):
        """Alert when the drawdown circuit breaker triggers."""
        self._send(
            f"🚨 <b>CIRCUIT BREAKER TRIGGERED</b> 🚨\n\n"
            f"Drawdown: <b>{drawdown_pct:.1%}</b>\n"
            f"Current Equity: ${equity:,.2f}\n"
            f"Limit: {config.MAX_DRAWDOWN_PCT:.0%}\n\n"
            f"⚠️ Trading is HALTED until drawdown recovers."
        )

    def error(self, message: str):
        """Alert on critical errors."""
        self._send(f"❌ <b>ERROR</b>\n\n<code>{message[:500]}</code>")

    def cycle_summary(self, cycle_id: str, universe: int, candidates: int,
                      trades: int, skipped: int, errors: int,
                      equity: float, daily_pl: float = 0):
        """Summary at the end of each trading cycle."""
        pl_emoji = "📈" if daily_pl >= 0 else "📉"
        self._send(
            f"📊 <b>CYCLE COMPLETE</b> [{cycle_id}]\n\n"
            f"Universe: {universe} assets\n"
            f"Candidates: {candidates}\n"
            f"Trades: {trades} | Skipped: {skipped} | Errors: {errors}\n\n"
            f"{pl_emoji} Equity: <b>${equity:,.2f}</b>\n"
            f"Session P/L: <b>${daily_pl:+,.2f}</b>"
        )

    def daily_summary(self, equity: float, cash: float,
                      positions: int, total_pl: float,
                      total_pl_pct: float, best_trade: str = "",
                      worst_trade: str = ""):
        """End-of-day portfolio summary."""
        pl_emoji = "📈" if total_pl >= 0 else "📉"
        self._send(
            f"🌅 <b>DAILY SUMMARY</b>\n\n"
            f"Equity: <b>${equity:,.2f}</b>\n"
            f"Cash: ${cash:,.2f}\n"
            f"Open Positions: {positions}\n\n"
            f"{pl_emoji} Total P/L: <b>${total_pl:+,.2f}</b> ({total_pl_pct:+.2f}%)\n\n"
            f"Best: {best_trade}\n"
            f"Worst: {worst_trade}"
        )

    def startup(self):
        """Alert when the bot starts."""
        self._send(
            f"🤖 <b>SENTINEL AUTOTRADER V3.5</b>\n\n"
            f"Status: <b>ONLINE</b>\n"
            f"Mode: Paper Trading\n"
            f"Confidence: {config.CONFIDENCE_THRESHOLD}%\n"
            f"Max Position: {config.MAX_POSITION_PCT:.0%}\n"
            f"Short Selling: {'✅' if config.ENABLE_SHORT_SELLING else '❌'}\n"
            f"Dashboard: {'✅' if config.DASHBOARD_ENABLED else '❌'}\n"
            f"Interval: Every {config.SCAN_INTERVAL_MINUTES} min"
        )

    def shutdown(self):
        """Alert when the bot shuts down."""
        self._send("🔴 <b>SENTINEL AUTOTRADER</b> — Shutting down gracefully.")
