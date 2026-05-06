"""
The Skeptic (Agent 2) - Risk & Liquidity Devil's Advocate

An AI agent programmed to find reasons NOT to trade. It challenges every
trade proposal with hard data on spreads, correlations, and portfolio risk.

Features:
- Bid/ask spread analysis via Alpaca quotes
- Sector concentration checking against current portfolio
- Earnings blackout enforcement
- Short position risk assessment (borrow cost, squeeze risk)
- Portfolio correlation analysis
- Strict JSON validation with few-shot examples
"""
import re
import json
from typing import Optional

from openai import OpenAI

import config
from core.data_fetcher import DataFetcher
from execution.alpaca_broker import AlpacaBroker
from agents.schemas import RiskSchema, RISK_EXAMPLE
from utils.logger import get_logger
from utils.rate_limiter import deepseek_limiter, retry_on_rate_limit
from utils.json_parser import validate_json

log = get_logger("sentinel.skeptic")


class Skeptic:
    """
    AI Risk Manager that acts as Devil's Advocate.
    Checks for red flags that pure math might miss.
    """

    def __init__(self, fetcher: DataFetcher = None, broker: AlpacaBroker = None):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.fetcher = fetcher or DataFetcher()
        self.broker = broker
        self.model = config.DEEPSEEK_MODEL

    def evaluate_risk(self, symbol: str, asset_type: str,
                      quant_result: dict, sentiment_result: dict) -> dict:
        """
        Comprehensive risk evaluation.

        Returns:
        {
            "grade": str ("LOW", "MEDIUM", "HIGH"),
            "score": float (0-100, where 100 = lowest risk),
            "flags": list[str],
            "reasoning": str,
            "spread_pct": float,
        }
        """
        flags = []
        risk_score = 75  # Start optimistic, deduct for issues

        # ─── 1. Spread Analysis ──────────────────────────────────────
        spread_pct = 0.0
        if asset_type != "crypto":
            quotes = self.fetcher.get_latest_quotes([symbol])
            if quotes and symbol in quotes:
                spread_pct = quotes[symbol]["spread_pct"]
                if spread_pct > 1.0:
                    flags.append(f"WIDE_SPREAD({spread_pct:.2f}%)")
                    risk_score -= 20
                elif spread_pct > 0.5:
                    flags.append(f"MOD_SPREAD({spread_pct:.2f}%)")
                    risk_score -= 10
            else:
                flags.append("NO_QUOTE_DATA")

        # ─── 2. Sector Concentration Check ───────────────────────────
        sector = quant_result.get("sector", "Unknown")
        if self.broker and sector != "Unknown":
            sector_exposure = self._check_sector_concentration(sector)
            if sector_exposure > config.MAX_SECTOR_CONCENTRATION:
                flags.append(f"SECTOR_OVERWEIGHT({sector}:{sector_exposure:.0%})")
                risk_score -= 10

        # ─── 3. Position Count Check ─────────────────────────────────
        if self.broker:
            try:
                positions = self.broker.get_positions()
                if len(positions) >= config.MAX_PORTFOLIO_POSITIONS:
                    flags.append(f"MAX_POSITIONS({len(positions)})")
                    risk_score -= 15

                held_symbols = [p.symbol for p in positions]
                if symbol.replace("/", "") in held_symbols:
                    flags.append("ALREADY_HELD")
                    risk_score -= 10
            except Exception:
                pass

        # ─── 4. Volatility Check ─────────────────────────────────────
        signals = quant_result.get("signals", {})
        atr_pct = signals.get("atr_pct", 0)
        if atr_pct > 5:
            flags.append(f"HIGH_VOLATILITY(ATR:{atr_pct:.1f}%)")
            risk_score -= 10
        max_dd = signals.get("max_drawdown", 0)
        if max_dd < -0.25:
            flags.append(f"SEVERE_DRAWDOWN({max_dd:.0%})")
            risk_score -= 10

        # ─── 5. Earnings Risk ────────────────────────────────────────
        direction = quant_result.get("direction", "long")
        earnings_risk = quant_result.get("earnings_risk", False)
        if earnings_risk and direction == "long":
            flags.append("EARNINGS_BLACKOUT")
            risk_score -= 15

        # ─── 6. Short-Specific Risk Checks ──────────────────────────
        if direction == "short":
            if signals.get("above_sma_long", False):
                flags.append("SHORT_AGAINST_UPTREND")
                risk_score -= 12
            if signals.get("volume_spike", False):
                flags.append("SHORT_SQUEEZE_RISK")
                risk_score -= 10
            rs = quant_result.get("rs_rating", 1.0)
            if rs > 1.2:
                flags.append(f"SHORT_RS_LEADER({rs:.2f})")
                risk_score -= 10
            if self.broker:
                try:
                    positions = self.broker.get_positions()
                    short_count = sum(1 for p in positions if float(p.qty) < 0)
                    if short_count >= config.MAX_SHORT_POSITIONS:
                        flags.append(f"MAX_SHORTS({short_count})")
                        risk_score -= 20
                except Exception:
                    pass

        # ─── 7. Sentiment Contradictions ─────────────────────────────
        quant_score = quant_result.get("score", 0)
        sentiment_score = sentiment_result.get("score", 0)
        if direction == "long":
            if quant_score > 70 and sentiment_score < -3:
                flags.append("QUANT_SENTIMENT_DIVERGENCE")
                risk_score -= 10
            if sentiment_score < -5:
                flags.append(f"VERY_NEGATIVE_SENTIMENT({sentiment_score})")
                risk_score -= 10
        else:
            if quant_score > 70 and sentiment_score > 3:
                flags.append("SHORT_POSITIVE_SENTIMENT")
                risk_score -= 10

        # ─── 8. DeepSeek AI Risk Assessment ──────────────────────────
        direction = quant_result.get("direction", "long")
        ai_assessment = self._ai_risk_assessment(
            symbol, asset_type, quant_result, sentiment_result, spread_pct, flags, direction
        )
        if ai_assessment:
            ai_flags = ai_assessment.get("flags", [])
            flags.extend(ai_flags)
            ai_adjustment = ai_assessment.get("score_adjustment", 0)
            risk_score += ai_adjustment

        # ─── Final Grade ─────────────────────────────────────────────
        risk_score = max(0, min(100, risk_score))

        if risk_score >= 60:
            grade = "LOW"
        elif risk_score >= 35:
            grade = "MEDIUM"
        else:
            grade = "HIGH"

        result = {
            "grade": grade,
            "score": risk_score,
            "flags": flags,
            "reasoning": ai_assessment.get("reasoning", "") if ai_assessment else "",
            "spread_pct": spread_pct,
        }

        log.info(
            f"Risk for {symbol}: grade={grade}, score={risk_score}, flags={flags}"
        )
        return result

    @retry_on_rate_limit
    def _ai_risk_assessment(self, symbol: str, asset_type: str,
                             quant_result: dict, sentiment_result: dict,
                             spread_pct: float, existing_flags: list, direction: str = "long") -> Optional[dict]:
        """Calls DeepSeek for nuanced risk assessment with chain-of-thought."""
        deepseek_limiter.acquire()
        
        deep_fund = {}
        if asset_type != "crypto":
            deep_fund = self.fetcher.get_deep_fundamentals(symbol)

        signals = quant_result.get("signals", {})

        system_prompt = """You are 'The Skeptic', an elite AI Risk Manager for a paper trading system.
Your job is to identify genuine risks but also recognize genuine opportunities.
Think carefully before flagging risks - don't manufacture problems where none exist.
You MUST respond in valid JSON format only. No markdown, no explanation outside JSON."""

        user_prompt = f"""Review this trade: {direction.upper()} {symbol} ({asset_type})

QUANTITATIVE DATA:
- Quant Score: {quant_result.get('score', 0)}/100
- RSI: {signals.get('rsi', 'N/A')}
- MACD Bullish: {signals.get('macd_bullish', 'N/A')}
- ADX (trend strength): {signals.get('adx', 'N/A')}
- ATR%: {signals.get('atr_pct', 'N/A')}%
- Sharpe Ratio: {signals.get('sharpe', 'N/A')}
- Max Drawdown: {signals.get('max_drawdown', 'N/A')}
- BB Squeeze: {signals.get('bb_squeeze', 'N/A')}

SENTIMENT:
- Score: {sentiment_result.get('score', 0)}/10
- Confidence: {sentiment_result.get('confidence', 0)}
- Events: {sentiment_result.get('events', [])}
- Summary: {sentiment_result.get('summary', 'N/A')}

DEEP FUNDAMENTALS (Finnhub/FMP):
{json.dumps(deep_fund, indent=2) if deep_fund else 'No data'}

MARKET MICROSTRUCTURE:
- Bid/Ask Spread: {spread_pct:.2f}%

EXISTING FLAGS: {existing_flags}

{RISK_EXAMPLE}

Now analyze and respond ONLY with this exact JSON structure (no other text):
{{
    "flags": [<list of additional risk flags you detect>],
    "score_adjustment": <integer from -20 to +10>,
    "reasoning": "<2-3 sentence explanation of your risk assessment>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=350,  # Increased for better reasoning
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return validate_json(raw, RiskSchema, max_retries=2)

        except Exception as e:
            log.warning(f"DeepSeek risk assessment failed for {symbol}: {e}")
            return None

    def _check_sector_concentration(self, sector: str) -> float:
        """Checks what percentage of current portfolio value is in the given sector."""
        if not self.broker:
            return 0.0

        try:
            positions = self.broker.get_positions()
            if not positions:
                return 0.0

            total_value = sum(float(p.market_value) for p in positions)
            if total_value == 0:
                return 0.0

            sector_value = 0
            for p in positions:
                fund = self.fetcher.get_fundamentals(p.symbol)
                if fund and fund.get("sector") == sector:
                    sector_value += float(p.market_value)

            return sector_value / total_value

        except Exception as e:
            log.debug(f"Sector check failed: {e}")
            return 0.0