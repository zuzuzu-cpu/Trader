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
import pandas_ta as ta
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

    # ─── Relative Strength Rating (used by QuantEngine) ─────────────────

    def calculate_relative_strength(self, symbol: str,
                                     start_date: str, end_date: str,
                                     is_crypto: bool = False) -> dict:
        default = {"rs_rating": 1.0, "rs_rank": "NEUTRAL",
                    "symbol_return": 0, "benchmark_return": 0}
        try:
            if is_crypto:
                sym_df = self.fetcher.get_crypto_bars(symbol, start_date, end_date)
            else:
                sym_df = self.fetcher.get_stock_bars(symbol, start_date, end_date)
            if sym_df is None or len(sym_df) < 20:
                return default
            bench_df = self._get_benchmark(start_date, end_date)
            if bench_df is None or len(bench_df) < 20:
                return default
            sym_dates = sym_df.set_index("timestamp")["close"] if "timestamp" in sym_df.columns else sym_df["close"]
            bench_dates = bench_df.set_index("timestamp")["close"] if "timestamp" in bench_df.columns else bench_df["close"]
            if isinstance(sym_dates.index, pd.DatetimeIndex):
                sym_dates.index = sym_dates.index.normalize()
            if isinstance(bench_dates.index, pd.DatetimeIndex):
                bench_dates.index = bench_dates.index.normalize()
            aligned = pd.concat([sym_dates, bench_dates], axis=1, join="inner").dropna()
            aligned.columns = ["sym", "bench"]
            lookback = min(config.RS_LOOKBACK_DAYS, len(aligned) - 1)
            if lookback < 10:
                return default
            sym_return = (float(aligned["sym"].iloc[-1]) / float(aligned["sym"].iloc[-lookback]) - 1)
            bench_return = (float(aligned["bench"].iloc[-1]) / float(aligned["bench"].iloc[-lookback]) - 1)
            rs_rating = (1 + sym_return) / (1 + bench_return) if bench_return != 0 else 1.0 + sym_return
            if rs_rating > 1.3: rs_rank = "LEADER"
            elif rs_rating > 1.1: rs_rank = "STRONG"
            elif rs_rating > 0.9: rs_rank = "NEUTRAL"
            elif rs_rating > 0.7: rs_rank = "WEAK"
            else: rs_rank = "LAGGARD"
            return {"rs_rating": round(rs_rating, 3), "rs_rank": rs_rank,
                    "symbol_return": round(sym_return, 4), "benchmark_return": round(bench_return, 4)}
        except Exception as e:
            log.debug(f"RS calculation failed for {symbol}: {e}")
            return default

    def _get_benchmark(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        self._benchmark_cache = self.fetcher.get_benchmark_bars(start_date, end_date)
        return self._benchmark_cache

    # ─── Multi-Timeframe Confirmation (used by QuantEngine) ────────────

    def confirm_multi_timeframe(self, symbol: str, start_date: str,
                                 end_date: str, direction: str = "long",
                                 is_crypto: bool = False) -> dict:
        if not config.ENABLE_MULTI_TIMEFRAME:
            return {"confirmed": True, "agreeing_timeframes": 3, "details": {}, "mtf_score_bonus": 10}
        try:
            mtf_bars = self.fetcher.get_multi_timeframe_bars(symbol, start_date, end_date, is_crypto=is_crypto)
            agreeing = 0; details = {}
            for tf_name, df in mtf_bars.items():
                if df is None or len(df) < 10:
                    details[tf_name] = "NO_DATA"; continue
                close = df["close"].astype(float)
                signal = self._get_tf_direction(close, direction)
                details[tf_name] = signal
                if signal == "AGREE": agreeing += 1
            confirmed = agreeing >= config.MTF_CONFIRMATION_REQUIRED
            bonus = 15 if agreeing == 3 else (8 if agreeing == 2 else (0 if agreeing == 1 else -10))
            return {"confirmed": confirmed, "agreeing_timeframes": agreeing,
                    "details": details, "mtf_score_bonus": bonus}
        except Exception as e:
            log.debug(f"MTF confirmation failed for {symbol}: {e}")
            return {"confirmed": True, "agreeing_timeframes": 0, "details": {}, "mtf_score_bonus": 0}

    def _get_tf_direction(self, close: pd.Series, direction: str) -> str:
        try:
            ema_fast = ta.ema(close, length=9); ema_slow = ta.ema(close, length=21)
            rsi = ta.rsi(close, length=14)
            if ema_fast is None or ema_slow is None or rsi is None: return "NO_DATA"
            ema_f = float(ema_fast.iloc[-1]); ema_s = float(ema_slow.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            if direction == "long":
                return "AGREE" if (ema_f > ema_s and rsi_val < 75) else "DISAGREE"
            else:
                return "AGREE" if (ema_f < ema_s and rsi_val > 25) else "DISAGREE"
        except Exception:
            return "NO_DATA"


# ─── Global instance ─────────────────────────────────────────────────────
_regime_instance = None

def get_market_regime() -> MarketRegime:
    global _regime_instance
    if _regime_instance is None:
        _regime_instance = MarketRegime()
    return _regime_instance