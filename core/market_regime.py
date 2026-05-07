"""
Market Regime Detection — Classifies market conditions for every agent.

Regime types:
  - BULL_LOW_VOL:  Trending up, calm (normal long bias)
  - BULL_HIGH_VOL: Trending up, choppy (tight stops, reduce size)
  - BEAR_LOW_VOL:  Trending down, calm (favor shorts, reduce longs)
  - BEAR_HIGH_VOL: Trending down, panicky (raise cash, aggressive stops)
  - SIDEWAYS:      No clear trend, range-bound (mean reversion strategies)

Uses SPY SMA200 for trend direction and SPY ATR% as a VIX proxy.
100% free — no API key needed.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
from core.data_fetcher import DataFetcher
from utils.logger import get_logger

log = get_logger("sentinel.market_regime")


class MarketRegime:
    """
    Detects and classifies the current market regime.
    Passed as context to all agents.
    """

    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
        self._cache = {}           # memoized results per cycle
        self._cycle_id = None

    def get_regime(self, cycle_id: str = "") -> dict:
        """
        Full regime detection pipeline. Returns a dict that every agent receives:
        {
            "regime": str,
            "spy_sma200_distance": float,
            "atr_pct": float,
            "guidance": str,
        }
        """
        if cycle_id and self._cycle_id == cycle_id:
            return self._cache.get("regime", self._default())

        self._cycle_id = cycle_id
        result = self._detect()
        self._cache["regime"] = result
        return result

    def _detect(self) -> dict:
        """Run detection logic."""
        default = self._default()

        try:
            end = datetime.now()
            start = end - timedelta(days=365)

            spy = self.fetcher.get_stock_bars(
                config.REGIME_VIX_PROXY_SYMBOL,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            )

            if spy is None or len(spy) < config.REGIME_SMA_PERIOD:
                log.debug("Regime: insufficient SPY data")
                return default

            close = spy['close'].astype(float)

            # SMA200 for trend direction
            sma200 = close.rolling(window=config.REGIME_SMA_PERIOD).mean().iloc[-1]
            current_price = close.iloc[-1]
            distance_pct = ((current_price - sma200) / sma200) * 100

            # ATR for volatility proxy (VIX substitute)
            high = spy['high'].astype(float)
            low = spy['low'].astype(float)
            atr = self._atr(high, low, close, period=14)
            atr_pct = (atr / current_price) * 100

            # Classify
            if distance_pct > 0:
                trend = "BULL"
            elif distance_pct > -3:
                trend = "SIDEWAYS"
            else:
                trend = "BEAR"

            if atr_pct >= config.REGIME_ATR_VOL_THRESHOLD:
                vol = "HIGH_VOL"
            else:
                vol = "LOW_VOL"

            regime_label = f"{trend}_{vol}"

            # Generate guidance text
            guidance_lines = [
                f"Market regime: {regime_label}",
                f"SPY {distance_pct:.1f}% from SMA200 (trend={trend})",
                f"SPY ATR%: {atr_pct:.1f}% (volatility={vol})",
            ]

            if "BEAR" in regime_label:
                guidance_lines.append("Guidance: DEFENSIVE — reduce long size, tighten stops, favor cash/shorts")
            elif "BULL" in regime_label:
                guidance_lines.append("Guidance: NORMAL — standard risk posture, favor long positions")
            elif "SIDEWAYS" in regime_label:
                guidance_lines.append("Guidance: MEAN REVERSION — avoid trend-following, favor range strategies")

            if "HIGH_VOL" in regime_label:
                guidance_lines.append("Volatility warning: reduce position sizes 25-50%")

            result = {
                "regime": regime_label,
                "spy_sma200_distance": round(distance_pct, 2),
                "atr_pct": round(atr_pct, 2),
                "guidance": " | ".join(guidance_lines),
            }

            log.info(f"Market regime: {regime_label} (SPY SMA200 dist={distance_pct:.1f}%, ATR={atr_pct:.1f}%)")
            return result

        except Exception as e:
            log.debug(f"Regime detection failed: {e}")
            return default

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Average True Range calculation."""
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift()).abs(),
            'lc': (low - close.shift()).abs(),
        }).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1] if len(tr) >= period else 0

    def _default(self) -> dict:
        return {"regime": "SIDEWAYS", "spy_sma200_distance": 0, "atr_pct": 0,
                "guidance": "Regime detection unavailable — using default sideways regime"}


# ─── Global instance ─────────────────────────────────────────────────────
_regime_instance = None

def get_market_regime() -> MarketRegime:
    global _regime_instance
    if _regime_instance is None:
        _regime_instance = MarketRegime()
    return _regime_instance