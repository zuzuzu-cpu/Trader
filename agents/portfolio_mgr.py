"""
The Portfolio Manager (Agent 3) - Final Decision & Execution Authority

The final AI agent that fuses all signals using DeepSeek Reasoner (thinking model)
for high-stakes decisions. Supports both long and short positions.

Features:
- DeepSeek Reasoner for final trade verdicts (chain-of-thought reasoning)
- Weighted confidence scoring (Quant 40% + Sentiment 35% + Risk 25%)
- ATR-based volatility-adjusted position sizing
- Maximum drawdown circuit breaker
- Portfolio-level exposure limits
- Short selling support with separate confidence threshold
- Earnings blackout enforcement
- Full trade reasoning audit trail
- Strict JSON validation with few-shot examples
"""
import re
import json
import time
from typing import Optional

from openai import OpenAI

import config
from agents.schemas import DecisionSchema, DECISION_EXAMPLE
from utils.logger import get_logger
from utils.rate_limiter import deepseek_limiter, retry_on_rate_limit
from utils.json_parser import validate_json

log = get_logger("sentinel.portfolio_mgr")


class PortfolioManager:
    """
    Final decision-maker using DeepSeek Reasoner for high-confidence
    trade verdicts. Supports long and short positions.

    V4 Features:
    - ML Feedback Loop: injects past trade outcomes into Reasoner prompt
    - Kelly Criterion: optimal position sizing based on actual win/loss stats
    """

    def __init__(self, journal=None):
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.short_threshold = config.SHORT_CONFIDENCE_THRESHOLD
        self.journal = journal
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._trade_stats_cache = None
        self._trade_stats_ts = 0  # timestamp-based cache invalidation

    def decide(self, symbol: str, asset_type: str,
               quant_result: dict, sentiment_result: dict,
               risk_result: dict, account_equity: float,
               peak_equity: float,
               insider_result: dict = None,
               options_result: dict = None,
               bear_market: bool = False,
               regime: dict = None,
               macro_context: str = "",
               correlation_result: dict = None) -> dict:
        """
        Final trade decision with position sizing.

        Returns:
        {
            "symbol": str,
            "asset_type": str,
            "should_trade": bool,
            "direction": str ("long" or "short"),
            "confidence": float (0-100),
            "notional": float (dollar amount to trade),
            "trailing_stop_pct": float,
            "reasoning": str,
            "deepseek_reasoning": str,
            "components": dict,
        }
        """
        direction = quant_result.get("direction", "long")

        # ─── 1. Calculate Confidence Score ───────────────────────────
        quant_score = quant_result.get("score", 0)
        sentiment_score = sentiment_result.get("score", 0)
        sentiment_confidence = sentiment_result.get("confidence", 0.5)
        risk_grade = risk_result.get("grade", "HIGH")
        risk_score = risk_result.get("score", 0)

        # Map sentiment (-10 to +10) → (0 to 100), weighted by AI's own confidence
        if direction == "long":
            mapped_sentiment = ((sentiment_score + 10) / 20) * 100
        else:
            # For shorts, invert sentiment: negative = good for shorts
            mapped_sentiment = ((-sentiment_score + 10) / 20) * 100

        # Reduce sentiment weight if the AI itself is unsure
        adjusted_sentiment = mapped_sentiment * sentiment_confidence + 50 * (1 - sentiment_confidence)

        # Combine with configurable weights
        confidence = (
            quant_score * config.WEIGHT_QUANT +
            adjusted_sentiment * config.WEIGHT_SENTIMENT +
            risk_score * config.WEIGHT_RISK
        )

        # ─── V5: Insider + Options Score Adjustments ─────────────────
        insider_score = (insider_result or {}).get("score", 0)
        options_score = (options_result or {}).get("score", 0)
        # Each point of insider/options score = 1.5% confidence boost/penalty
        confidence += (insider_score + options_score) * 1.5
        if insider_score != 0:
            log.info(f"Insider adjustment: {insider_score:+d} → confidence now {confidence:.1f}%")
        if options_score != 0:
            log.info(f"Options adjustment: {options_score:+d} → confidence now {confidence:.1f}%")

        components = {
            "quant": f"{quant_score:.1f} × {config.WEIGHT_QUANT} = {quant_score * config.WEIGHT_QUANT:.1f}",
            "sentiment": f"{adjusted_sentiment:.1f} × {config.WEIGHT_SENTIMENT} = {adjusted_sentiment * config.WEIGHT_SENTIMENT:.1f}",
            "risk": f"{risk_score:.1f} × {config.WEIGHT_RISK} = {risk_score * config.WEIGHT_RISK:.1f}",
            "direction": direction,
            "raw_sentiment_score": sentiment_score,
            "risk_grade": risk_grade,
            "risk_flags": risk_result.get("flags", []),
            "rs_rating": quant_result.get("rs_rating", 1.0),
            "mtf_confirmed": quant_result.get("mtf_confirmed", False),
            "earnings_risk": quant_result.get("earnings_risk", False),
        }

        # ─── 2. Circuit Breakers ─────────────────────────────────────
        threshold = self.short_threshold if direction == "short" else self.confidence_threshold
        
        # Use regime dict for market-aware adjustments
        regime_label = (regime or {}).get("regime", "SIDEWAYS")
        if "BEAR" in regime_label and direction == "long":
            threshold += 10
            log.info(f"{regime_label}: Increasing long threshold to {threshold}%")
            
        should_trade = confidence >= threshold
        block_reason = None

        # Correlation guard veto (overrides everything)
        if correlation_result and not correlation_result.get("can_trade", True):
            should_trade = False
            block_reason = f"CORRELATION_GUARD: {correlation_result.get('reason', 'Too correlated')}"
            log.warning(block_reason)

        # Drawdown circuit breaker
        if peak_equity > 0 and account_equity > 0:
            current_drawdown = (peak_equity - account_equity) / peak_equity
            if current_drawdown > config.MAX_DRAWDOWN_PCT:
                should_trade = False
                block_reason = f"CIRCUIT_BREAKER: Drawdown {current_drawdown:.1%} > {config.MAX_DRAWDOWN_PCT:.0%} limit"
                log.warning(block_reason)

        # Risk grade penalty (softer than hard veto)
        if risk_grade == "HIGH":
            # Apply a confidence penalty instead of a hard veto
            confidence -= 10
            if confidence < threshold:
                should_trade = False
                block_reason = f"HIGH_RISK: Grade={risk_grade}, confidence {confidence:.1f}% after penalty < {threshold}%"

        # Minimum quant score gate
        min_quant = 40 if direction == "long" else 60  # Shorts need higher conviction
        if quant_score < min_quant:
            should_trade = False
            block_reason = f"QUANT_TOO_LOW: {quant_score:.0f} < {min_quant} minimum for {direction}"

        # Earnings blackout (block new long positions near earnings)
        if quant_result.get("earnings_risk", False) and direction == "long":
            should_trade = False
            block_reason = "EARNINGS_BLACKOUT: Too close to earnings date"

        # Short selling disabled check
        if direction == "short" and not config.ENABLE_SHORT_SELLING:
            should_trade = False
            block_reason = "SHORT_SELLING_DISABLED"

        # ─── 3. DeepSeek Reasoner Final Verdict ─────────────────────
        deepseek_reasoning = ""
        if should_trade and confidence >= threshold:
            reasoner_result = self._reasoner_verdict(
                symbol, direction, quant_result, sentiment_result,
                risk_result, confidence, account_equity,
                insider_result=insider_result,
                options_result=options_result,
                regime=regime,
                macro_context=macro_context,
            )
            if reasoner_result:
                deepseek_reasoning = reasoner_result.get("reasoning", "")
                # Reasoner can veto or boost
                if reasoner_result.get("verdict") == "REJECT":
                    should_trade = False
                    block_reason = f"REASONER_VETO: {deepseek_reasoning[:100]}"
                elif reasoner_result.get("confidence_adjustment", 0) != 0:
                    confidence += reasoner_result["confidence_adjustment"]

        # ─── 4. Position Sizing ──────────────────────────────────────
        notional = 0.0
        trailing_stop_pct = config.TRAILING_STOP_PCT

        if should_trade and account_equity > 0:
            notional, trailing_stop_pct = self._calculate_position_size(
                account_equity, quant_result, confidence
            )
            # Regime-based size reduction
            if "BEAR" in regime_label and direction == "long":
                notional *= 0.5
                log.info(f"{regime_label}: Cutting long position size by 50% to ${notional:.2f}")
            if "HIGH_VOL" in regime_label:
                notional *= 0.75
                log.info(f"{regime_label}: High volatility, reducing size 25% to ${notional:.2f}")

        # ─── 5. Build Reasoning ──────────────────────────────────────
        quant_reasons = quant_result.get("reason", "")
        sentiment_summary = sentiment_result.get("summary", "")
        risk_flags = risk_result.get("flags", [])

        side_label = "LONG" if direction == "long" else "SHORT"

        if should_trade:
            reasoning = (
                f"EXECUTE {side_label}: Confidence {confidence:.1f}% | "
                f"Quant[{quant_reasons}] | "
                f"Sentiment[{sentiment_summary}] | "
                f"Risk[{risk_grade}] | "
                f"Size[${notional:.2f}] | "
                f"Stop[{trailing_stop_pct:.1f}%]"
            )
        else:
            reasoning = (
                f"SKIP {side_label}: {block_reason or f'Confidence {confidence:.1f}% < {threshold}%'} | "
                f"Flags: {risk_flags}"
            )

        result = {
            "symbol": symbol,
            "asset_type": asset_type,
            "should_trade": should_trade,
            "direction": direction,
            "confidence": confidence,
            "notional": notional,
            "trailing_stop_pct": trailing_stop_pct,
            "reasoning": reasoning,
            "deepseek_reasoning": deepseek_reasoning,
            "components": components,
        }

        log.info(
            f"Decision for {symbol}: "
            f"{'TRADE' if should_trade else 'SKIP'} {side_label} "
            f"(confidence={confidence:.1f}%, notional=${notional:.2f})"
        )
        return result

    # ─── DeepSeek Reasoner Verdict ───────────────────────────────────

    @retry_on_rate_limit
    def _reasoner_verdict(self, symbol: str, direction: str,
                          quant_result: dict, sentiment_result: dict,
                          risk_result: dict, confidence: float,
                          equity: float,
                          insider_result: dict = None,
                          options_result: dict = None,
                          regime: dict = None,
                          macro_context: str = "") -> Optional[dict]:
        """
        Calls DeepSeek Reasoner (thinking model) for a final high-stakes verdict.
        The reasoner uses chain-of-thought to validate the trade decision.
        """
        deepseek_limiter.acquire()

        signals = quant_result.get("signals", {})

        # Build ML feedback block from trade history
        ml_feedback = self._build_ml_feedback()

        prompt = f"""You are the Portfolio Manager of an autonomous trading system.
The quantitative analysis and risk assessment are complete. You must make the FINAL decision.
IMPORTANT: This is a paper trading account for learning. You should APPROVE trades that have 
reasonable quantitative support, even if conditions aren't perfect. Only REJECT if there is a 
clear, specific danger (e.g. earnings tomorrow, extreme overbought, fraud news).

TRADE PROPOSAL: {direction.upper()} {symbol}
Confidence Score: {confidence:.1f}%
Account Equity: ${equity:,.2f}

QUANTITATIVE ANALYSIS:
- Quant Score: {quant_result.get('score', 0)}/100
- Direction: {direction}
- RSI: {signals.get('rsi', 'N/A')}
- MACD Bullish: {signals.get('macd_bullish', 'N/A')}
- Sharpe Ratio: {signals.get('sharpe', 'N/A')}
- Max Drawdown: {signals.get('max_drawdown', 'N/A')}
- RS Rating (vs SPY): {quant_result.get('rs_rating', 'N/A')}
- Multi-TF Confirmed: {quant_result.get('mtf_confirmed', 'N/A')}
- Earnings Risk: {quant_result.get('earnings_risk', False)}
- Reason: {quant_result.get('reason', '')}

SENTIMENT:
- Score: {sentiment_result.get('score', 0)}/10
- Summary: {sentiment_result.get('summary', '')}

RISK ASSESSMENT:
- Grade: {risk_result.get('grade', 'N/A')}
- Flags: {risk_result.get('flags', [])}
- Reasoning: {risk_result.get('reasoning', '')}

INSIDER TRADING (SEC Form 4, last {config.INSIDER_LOOKBACK_DAYS} days):
- {(insider_result or {}).get('summary', 'No data')} (score: {(insider_result or {}).get('score', 0):+d})

OPTIONS FLOW:
- {(options_result or {}).get('summary', 'No data')} (score: {(options_result or {}).get('score', 0):+d})
- Unusual Calls: {(options_result or {}).get('unusual_calls', False)} | Unusual Puts: {(options_result or {}).get('unusual_puts', False)}
- Put/Call Ratio: {(options_result or {}).get('pc_ratio', 'N/A')}

MARKET REGIME:
- {(regime or {}).get('guidance', 'No regime data')}

MACRO CONTEXT:
- {macro_context or 'No macro data'}
{ml_feedback}
{DECISION_EXAMPLE}

Think step by step:
1. What does the quant score tell us?
2. Does sentiment confirm or contradict?
3. Any risk flags to consider?
4. Final decision reasoning:

Now respond ONLY with this exact JSON structure (no other text):
{{
    "verdict": "APPROVE" or "REJECT",
    "confidence_adjustment": <integer from -5 to +10>,
    "reasoning": "<2-3 sentences>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_REASONER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return validate_json(raw, DecisionSchema, max_retries=2)

        except Exception as e:
            log.warning(f"DeepSeek Reasoner failed for {symbol}: {e}")
            try:
                response = self.client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=350,
                    temperature=0.1,
                )
                raw = response.choices[0].message.content.strip()
                return validate_json(raw, DecisionSchema, max_retries=2)
            except Exception as e2:
                log.warning(f"DeepSeek fallback also failed for {symbol}: {e2}")
                return None

    def _parse_reasoner_response(self, raw: str) -> dict:
        """Parses DeepSeek Reasoner's JSON response."""
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        try:
            data = json.loads(raw)
            # Use Pydantic to validate the core structure
            validated = DecisionSchema(**data)
            
            return {
                "verdict": "APPROVE" if validated.verdict.upper() == "APPROVE" else "REJECT",
                "confidence_adjustment": max(-10, min(10, int(data.get("confidence_adjustment", 0)))),
                "reasoning": validated.reasoning[:300],
            }
        except Exception as e:
            log.warning(f"Portfolio decision validation failed: {e}")
            return {"verdict": "APPROVE", "confidence_adjustment": 0, "reasoning": f"Validation error: {str(e)[:100]}"}

    # ─── Position Sizing ─────────────────────────────────────────────

    def _calculate_position_size(self, equity: float,
                                  quant_result: dict,
                                  confidence: float) -> tuple[float, float]:
        """
        Position sizing with Kelly Criterion override.

        If enough trade history exists, use Kelly Criterion for optimal sizing.
        Otherwise, fall back to ATR-based volatility-adjusted sizing.
        """
        signals = quant_result.get("signals", {})
        atr = signals.get("atr", 0)
        price = signals.get("price", 0)

        # ─── Kelly Criterion (if enabled and enough data) ────────────
        if config.KELLY_ENABLED and self.journal:
            stats = self._get_trade_stats()
            if stats["total_trades"] >= config.KELLY_MIN_TRADES:
                kelly_raw = stats["kelly_pct"]
                # Apply fractional Kelly (e.g. 1/4 Kelly for safety)
                kelly_pct = kelly_raw * config.KELLY_FRACTION
                # Clamp to configured bounds
                kelly_pct = max(config.KELLY_MIN_POSITION_PCT,
                                min(config.KELLY_MAX_POSITION_PCT, kelly_pct))

                # Scale by confidence (higher confidence = closer to full Kelly)
                threshold = self.confidence_threshold
                conf_scale = min(1.0, (confidence - threshold) /
                                 (100 - threshold) * 0.5 + 0.5)

                base_pct = kelly_pct * conf_scale
                log.info(
                    f"Kelly sizing: raw={kelly_raw:.3f}, fractional={kelly_pct:.4f}, "
                    f"conf_scale={conf_scale:.2f}, final={base_pct:.4f} "
                    f"(win_rate={stats['win_rate']:.1%}, trades={stats['total_trades']})"
                )
            else:
                # Not enough trades yet, use default
                base_pct = config.MAX_POSITION_PCT
                log.info(
                    f"Kelly: insufficient data ({stats['total_trades']}/{config.KELLY_MIN_TRADES} trades), "
                    f"using default {base_pct:.1%}"
                )
        else:
            base_pct = config.MAX_POSITION_PCT

            # Scale by confidence (fallback mode)
            threshold = self.confidence_threshold
            confidence_scale = min(1.0, (confidence - threshold) /
                                   (100 - threshold) * 0.5 + 0.5)
            base_pct = base_pct * confidence_scale

        # ─── Volatility adjustment (always applied) ──────────────────
        if atr > 0 and price > 0:
            atr_pct = atr / price
            vol_scale = min(1.0, 0.01 / max(atr_pct, 0.005))
        else:
            vol_scale = 0.5

        # Final notional
        notional = equity * base_pct * vol_scale
        notional = max(config.MIN_POSITION_USD, notional)

        # Trailing stop based on ATR
        if atr > 0 and price > 0:
            trailing_stop_pct = (atr * config.ATR_RISK_MULTIPLIER / price) * 100
            trailing_stop_pct = max(1.5, min(8.0, trailing_stop_pct))
        else:
            trailing_stop_pct = config.TRAILING_STOP_PCT

        return round(notional, 2), round(trailing_stop_pct, 2)

    # ─── ML Feedback Helpers ─────────────────────────────────────────

    def _build_ml_feedback(self) -> str:
        """Builds a text block of recent trade outcomes for the Reasoner prompt."""
        if not self.journal:
            return ""

        try:
            history = self.journal.get_trade_history(limit=10)
            if not history:
                return ""

            stats = self._get_trade_stats()

            lines = ["\nYOUR RECENT TRADE HISTORY (learn from these):"]
            lines.append(
                f"Overall: {stats['total_trades']} trades, "
                f"{stats.get('win_rate', 0):.0%} win rate, "
                f"profit factor {stats.get('profit_factor', 0):.2f}"
            )

            for t in history[:10]:
                pnl = t.get("pnl")
                pnl_str = f"P&L: ${pnl:+.2f} ({t.get('pnl_pct', 0):+.1f}%)" if pnl is not None else "OPEN/PENDING"
                exit_str = f", exit: {t['exit_reason']}" if t.get("exit_reason") else ""
                lines.append(
                    f"  • {t['side']} {t['symbol']} conf={t.get('confidence', 0):.0f}% "
                    f"risk={t.get('risk_grade', '?')} → {pnl_str}{exit_str}"
                )

            lines.append("Use this history to calibrate your decisions. If you've been losing on HIGH_RISK trades, be more cautious. If your win rate is high, be more aggressive.")
            return "\n".join(lines)

        except Exception as e:
            log.debug(f"ML feedback generation failed: {e}")
            return ""

    def _get_trade_stats(self) -> dict:
        """Gets trade stats with timestamp-based per-cycle caching."""
        now = time.time()
        if self._trade_stats_cache and (now - self._trade_stats_ts) < 60:
            return self._trade_stats_cache
        try:
            self._trade_stats_cache = self.journal.get_trade_stats()
            self._trade_stats_ts = now
            return self._trade_stats_cache
        except Exception:
            return {"total_trades": 0, "win_rate": 0, "kelly_pct": 0}
