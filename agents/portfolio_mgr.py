"""
The Portfolio Manager (Agent 3) - Final Decision & Execution Authority

The final AI agent that fuses all signals, calculates confidence scores,
determines ATR-based position sizing, and issues the trade command.

Features:
- Weighted confidence scoring (Quant 40% + Sentiment 35% + Risk 25%)
- ATR-based volatility-adjusted position sizing
- Maximum drawdown circuit breaker
- Portfolio-level exposure limits
- Full trade reasoning audit trail
"""
import os
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger("sentinel.portfolio_mgr")


class PortfolioManager:
    """
    Final decision-maker that combines all agent signals
    and determines position sizing for execution.
    """

    def __init__(self):
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD

    def decide(self, symbol: str, asset_type: str,
               quant_result: dict, sentiment_result: dict,
               risk_result: dict, account_equity: float,
               peak_equity: float) -> dict:
        """
        Final trade decision with position sizing.

        Returns:
        {
            "symbol": str,
            "asset_type": str,
            "should_trade": bool,
            "confidence": float (0-100),
            "notional": float (dollar amount to trade),
            "trailing_stop_pct": float,
            "reasoning": str,
            "components": dict,
        }
        """
        # ─── 1. Calculate Confidence Score ───────────────────────────
        quant_score = quant_result.get("score", 0)
        sentiment_score = sentiment_result.get("score", 0)
        sentiment_confidence = sentiment_result.get("confidence", 0.5)
        risk_grade = risk_result.get("grade", "HIGH")
        risk_score = risk_result.get("score", 0)

        # Map sentiment (-10 to +10) → (0 to 100), weighted by AI's own confidence
        mapped_sentiment = ((sentiment_score + 10) / 20) * 100
        # Reduce sentiment weight if the AI itself is unsure
        adjusted_sentiment = mapped_sentiment * sentiment_confidence + 50 * (1 - sentiment_confidence)

        # Combine with configurable weights
        confidence = (
            quant_score * config.WEIGHT_QUANT +
            adjusted_sentiment * config.WEIGHT_SENTIMENT +
            risk_score * config.WEIGHT_RISK
        )

        components = {
            "quant": f"{quant_score:.1f} × {config.WEIGHT_QUANT} = {quant_score * config.WEIGHT_QUANT:.1f}",
            "sentiment": f"{adjusted_sentiment:.1f} × {config.WEIGHT_SENTIMENT} = {adjusted_sentiment * config.WEIGHT_SENTIMENT:.1f}",
            "risk": f"{risk_score:.1f} × {config.WEIGHT_RISK} = {risk_score * config.WEIGHT_RISK:.1f}",
            "raw_sentiment_score": sentiment_score,
            "risk_grade": risk_grade,
            "risk_flags": risk_result.get("flags", []),
        }

        # ─── 2. Circuit Breakers ─────────────────────────────────────
        should_trade = confidence >= self.confidence_threshold
        block_reason = None

        # Drawdown circuit breaker
        if peak_equity > 0 and account_equity > 0:
            current_drawdown = (peak_equity - account_equity) / peak_equity
            if current_drawdown > config.MAX_DRAWDOWN_PCT:
                should_trade = False
                block_reason = f"CIRCUIT_BREAKER: Drawdown {current_drawdown:.1%} > {config.MAX_DRAWDOWN_PCT:.0%} limit"
                log.warning(block_reason)

        # Risk grade veto
        if risk_grade == "HIGH" and confidence < 90:
            should_trade = False
            block_reason = f"HIGH_RISK_VETO: Grade={risk_grade}, need confidence > 90 but got {confidence:.1f}"

        # Minimum score gate
        if quant_score < 40:
            should_trade = False
            block_reason = f"QUANT_TOO_LOW: {quant_score:.0f} < 40 minimum"

        # ─── 3. Position Sizing ──────────────────────────────────────
        notional = 0.0
        trailing_stop_pct = config.TRAILING_STOP_PCT

        if should_trade and account_equity > 0:
            notional, trailing_stop_pct = self._calculate_position_size(
                account_equity, quant_result, confidence
            )

        # ─── 4. Build Reasoning ──────────────────────────────────────
        quant_reasons = quant_result.get("reason", "")
        sentiment_summary = sentiment_result.get("summary", "")
        risk_reasoning = risk_result.get("reasoning", "")
        risk_flags = risk_result.get("flags", [])

        if should_trade:
            reasoning = (
                f"EXECUTE: Confidence {confidence:.1f}% | "
                f"Quant[{quant_reasons}] | "
                f"Sentiment[{sentiment_summary}] | "
                f"Risk[{risk_grade}] | "
                f"Size[${notional:.2f}] | "
                f"Stop[{trailing_stop_pct:.1f}%]"
            )
        else:
            reasoning = (
                f"SKIP: {block_reason or f'Confidence {confidence:.1f}% < {self.confidence_threshold}%'} | "
                f"Flags: {risk_flags}"
            )

        result = {
            "symbol": symbol,
            "asset_type": asset_type,
            "should_trade": should_trade,
            "confidence": confidence,
            "notional": notional,
            "trailing_stop_pct": trailing_stop_pct,
            "reasoning": reasoning,
            "components": components,
        }

        log.info(
            f"Decision for {symbol}: "
            f"{'TRADE' if should_trade else 'SKIP'} "
            f"(confidence={confidence:.1f}%, notional=${notional:.2f})"
        )
        return result

    def _calculate_position_size(self, equity: float,
                                  quant_result: dict,
                                  confidence: float) -> tuple[float, float]:
        """
        ATR-based volatility-adjusted position sizing.

        Higher volatility → smaller position + wider stop
        Higher confidence → closer to max position size
        """
        signals = quant_result.get("signals", {})
        atr = signals.get("atr", 0)
        price = signals.get("price", 0)

        # Base position size (percentage of equity)
        base_pct = config.MAX_POSITION_PCT

        # Scale by confidence (80% threshold → 100% of max; 100% → 100% of max)
        confidence_scale = min(1.0, (confidence - self.confidence_threshold) /
                               (100 - self.confidence_threshold) * 0.5 + 0.5)

        # Scale by volatility (higher ATR = smaller position)
        if atr > 0 and price > 0:
            atr_pct = atr / price
            # Inverse volatility scaling: 1% ATR → full size, 5% ATR → 20% size
            vol_scale = min(1.0, 0.01 / max(atr_pct, 0.005))
        else:
            vol_scale = 0.5  # Default to half size if no ATR data

        # Final notional
        notional = equity * base_pct * confidence_scale * vol_scale
        notional = max(config.MIN_POSITION_USD, notional)

        # Trailing stop based on ATR
        if atr > 0 and price > 0:
            # Stop = ATR * multiplier, expressed as percentage
            trailing_stop_pct = (atr * config.ATR_RISK_MULTIPLIER / price) * 100
            trailing_stop_pct = max(1.5, min(8.0, trailing_stop_pct))  # Clamp between 1.5% and 8%
        else:
            trailing_stop_pct = config.TRAILING_STOP_PCT

        return round(notional, 2), round(trailing_stop_pct, 2)
