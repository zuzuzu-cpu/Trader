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
                log.warning(f"Telegram send failed: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            log.error(f"Telegram send error: {e}")

    # ─── Alert Types ─────────────────────────────────────────────────

    def trade_card(self, symbol: str, side: str, notional: float,
                   confidence: float, fill_price: float,
                   trailing_stop_pct: float, verdict: dict,
                   q_result: dict = None, sentiment: dict = None,
                   insider: dict = None, social: dict = None,
                   options: dict = None):
        """
        Rich trade explanation card — V5 format.
        Shows all signals, AI reasoning, and exit plan in one structured message.
        """
        if not config.TRADE_CARDS_ENABLED:
            # Fallback to simple alert
            return self.trade_executed(symbol, side, notional, confidence,
                                       fill_price, trailing_stop_pct)

        emoji = "🟢" if side == "BUY" else "🔴"
        direction_label = "LONG ↗" if side == "BUY" else "SHORT ↘"

        # Build signal blocks
        q_block = ""
        if q_result:
            signals = q_result.get("signals", {})
            rsi = signals.get("rsi", "N/A")
            macd = "✅" if signals.get("macd_bullish") else "❌"
            mtf = "✅" if q_result.get("mtf_confirmed") else "❌"
            rs = q_result.get("rs_rating", "N/A")
            rs_fmt = f"{rs:.2f}" if isinstance(rs, float) else str(rs)
            q_block = (
                f"\n📊 <b>Quant Score: {q_result.get('score', 0):.0f}/100</b>\n"
                f"RSI: {rsi:.1f} | MACD: {macd} | MTF: {mtf} | RS: {rs_fmt}"
            )

        sent_block = ""
        if sentiment:
            sent_score = sentiment.get("score", 0)
            sent_conf = sentiment.get("confidence", 0)
            sent_sum = sentiment.get("summary", "")[:100]
            s_emoji = "📈" if sent_score > 0 else ("📉" if sent_score < 0 else "➡️")
            sent_block = (
                f"\n{s_emoji} <b>Sentiment: {sent_score:+d}/10</b> "
                f"(conf: {sent_conf:.0%})\n"
                f"<i>{sent_sum}</i>"
            )

        insider_block = ""
        if insider and insider.get("score", 0) != 0:
            insider_block = (
                f"\n🏢 <b>Insider:</b> {insider.get('summary', '')} "
                f"(score: {insider.get('score', 0):+d})"
            )

        social_block = ""
        if social and social.get("mention_count", 0) > 0:
            vel = social.get("velocity", "LOW")
            mentions = social.get("mention_count", 0)
            social_block = (
                f"\n📱 <b>Social:</b> {mentions} Reddit mentions "
                f"[{vel}] (score: {social.get('score', 0):+d})"
            )

        options_block = ""
        if options and options.get("summary"):
            o_score = options.get("score", 0)
            o_emoji = "⚡" if (options.get("unusual_calls") or options.get("unusual_puts")) else "📋"
            options_block = (
                f"\n{o_emoji} <b>Options:</b> {options.get('summary', '')} "
                f"(score: {o_score:+d})"
            )

        # Exit plan
        tp_price = fill_price * (1 + (verdict.get("take_profit_pct", 5) / 100))
        stop_price = fill_price * (1 - trailing_stop_pct / 100)
        exit_block = (
            f"\n🎯 <b>Exit Plan</b>\n"
            f"Take Profit 50%: ${tp_price:,.2f} (+{verdict.get('take_profit_pct', 5):.0f}%)\n"
            f"Trailing Stop: ${stop_price:,.2f} (-{trailing_stop_pct:.1f}%)"
        )

        # AI reasoning
        reasoning = verdict.get("reasoning", "")[:250]
        ai_block = f"\n🧠 <b>AI:</b> <i>{reasoning}</i>" if reasoning else ""

        message = (
            f"{emoji} <b>TRADE EXECUTED — {symbol}</b>\n"
            f"{'─' * 28}\n"
            f"Direction: <b>{direction_label}</b>\n"
            f"Amount: <b>${notional:,.2f}</b> @ ${fill_price:,.4f}\n"
            f"Confidence: <b>{confidence:.1f}%</b>"
            f"{q_block}"
            f"{sent_block}"
            f"{insider_block}"
            f"{social_block}"
            f"{options_block}"
            f"{ai_block}"
            f"{exit_block}"
        )
        self._send(message)

    def trade_executed(self, symbol: str, side: str, notional: float,
                       confidence: float, fill_price: float = 0,
                       trailing_stop_pct: float = 0):
        """Simple trade alert (fallback when TRADE_CARDS_ENABLED=false)."""
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

    def skip_card(self, symbol: str, confidence: float, verdict: dict,
                  q_result: dict = None):
        """
        Sends a near-miss skip card for candidates that were close but rejected.
        Only fires for candidates above SKIP_CARD_MIN_CONFIDENCE.
        """
        if not config.TRADE_CARDS_ENABLED:
            return
        if confidence < config.SKIP_CARD_MIN_CONFIDENCE:
            return

        quant_score = q_result.get("score", 0) if q_result else 0
        reasoning = verdict.get("reasoning", "")[:200]

        self._send(
            f"⏭️ <b>NEAR-MISS SKIP — {symbol}</b>\n"
            f"{'─' * 28}\n"
            f"Confidence: {confidence:.1f}% (below threshold)\n"
            f"Quant Score: {quant_score:.0f}/100\n\n"
            f"🧠 <b>Bear Case:</b>\n"
            f"<i>{reasoning}</i>"
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

    def closing_agent_alert(self, symbol: str, verdict: str, confidence: float, pnl_pct: float, reasoning: str, action_details: str = ""):
        """Alert when the Closing Agent modifies or closes a position."""
        emoji = "🔒" if verdict == "SELL_ALL" else "✂️" if verdict == "SELL_PARTIAL" else "🛡️"
        self._send(
            f"{emoji} <b>AI EARLY EXIT — {symbol}</b>\n"
            f"Verdict: <b>{verdict}</b> (Conf: {confidence:.1f}%)\n"
            f"Current P/L: <b>{pnl_pct:+.2f}%</b>\n"
            f"Action: {action_details}\n\n"
            f"🧠 <b>Reasoning:</b>\n<i>{reasoning}</i>"
        )
