"""
Market Regime Detection & Relative Strength Calculator.

Features:
- Relative Strength (RS) rating vs benchmark (SPY)
  Measures how a stock performs relative to the market.
  RS > 1.0 = outperforming, RS < 1.0 = underperforming.

- Multi-Timeframe signal confirmation
  Confirms signals across 1H, 1D, 1W to reduce false positives.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional

import config
from core.data_fetcher import DataFetcher
from utils.logger import get_logger

log = get_logger("sentinel.market_regime")


class MarketRegime:
    """
    Calculates relative strength ratings and confirms signals
    across multiple timeframes.
    """

    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
        self._benchmark_cache = None

    # ─── Relative Strength ───────────────────────────────────────────

    def calculate_relative_strength(self, symbol: str,
                                     start_date: str, end_date: str,
                                     is_crypto: bool = False) -> dict:
        """
        Calculates the Relative Strength (RS) rating of a symbol vs SPY.

        RS = (Symbol % Change over N days) / (Benchmark % Change over N days)
        RS > 1.0 = outperforming the market
        RS < 1.0 = underperforming the market

        Returns:
        {
            "rs_rating": float,
            "rs_rank": str ("LEADER", "STRONG", "NEUTRAL", "WEAK", "LAGGARD"),
            "symbol_return": float,
            "benchmark_return": float,
        }
        """
        default = {"rs_rating": 1.0, "rs_rank": "NEUTRAL",
                    "symbol_return": 0, "benchmark_return": 0}

        try:
            # Get symbol bars
            if is_crypto:
                sym_df = self.fetcher.get_crypto_bars(symbol, start_date, end_date)
            else:
                sym_df = self.fetcher.get_stock_bars(symbol, start_date, end_date)

            if sym_df is None or len(sym_df) < 20:
                return default

            # Get benchmark bars (cached)
            bench_df = self._get_benchmark(start_date, end_date)
            if bench_df is None or len(bench_df) < 20:
                return default

            # Calculate returns over the RS lookback period
            # Align by timestamp to ensure we compare exact calendar dates
            sym_dates = sym_df.set_index("timestamp")["close"] if "timestamp" in sym_df.columns else sym_df["close"]
            bench_dates = bench_df.set_index("timestamp")["close"] if "timestamp" in bench_df.columns else bench_df["close"]
            
            # Normalize index to dates to align crypto (00:00 UTC) with stocks (04:00 UTC)
            if isinstance(sym_dates.index, pd.DatetimeIndex):
                sym_dates.index = sym_dates.index.normalize()
            if isinstance(bench_dates.index, pd.DatetimeIndex):
                bench_dates.index = bench_dates.index.normalize()

            # Inner join to get strictly aligned dates
            aligned = pd.concat([sym_dates, bench_dates], axis=1, join="inner").dropna()
            aligned.columns = ["sym", "bench"]
            
            lookback = min(config.RS_LOOKBACK_DAYS, len(aligned) - 1)
            if lookback < 10:
                return default
                
            sym_return = (float(aligned["sym"].iloc[-1]) / float(aligned["sym"].iloc[-lookback]) - 1)
            bench_return = (float(aligned["bench"].iloc[-1]) / float(aligned["bench"].iloc[-lookback]) - 1)

            # RS Rating: how much the symbol outperforms the benchmark
            if bench_return != 0:
                rs_rating = (1 + sym_return) / (1 + bench_return)
            else:
                rs_rating = 1.0 + sym_return

            # Rank the RS rating
            if rs_rating > 1.3:
                rs_rank = "LEADER"
            elif rs_rating > 1.1:
                rs_rank = "STRONG"
            elif rs_rating > 0.9:
                rs_rank = "NEUTRAL"
            elif rs_rating > 0.7:
                rs_rank = "WEAK"
            else:
                rs_rank = "LAGGARD"

            return {
                "rs_rating": round(rs_rating, 3),
                "rs_rank": rs_rank,
                "symbol_return": round(sym_return, 4),
                "benchmark_return": round(bench_return, 4),
            }

        except Exception as e:
            log.debug(f"RS calculation failed for {symbol}: {e}")
            return default

    def _get_benchmark(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Gets benchmark data with simple caching."""
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        self._benchmark_cache = self.fetcher.get_benchmark_bars(start_date, end_date)
        return self._benchmark_cache

    # ─── Multi-Timeframe Confirmation ────────────────────────────────

    def confirm_multi_timeframe(self, symbol: str, start_date: str,
                                 end_date: str, direction: str = "long",
                                 is_crypto: bool = False) -> dict:
        """
        Checks if the signal direction (long/short) is confirmed across
        multiple timeframes (1H, 1D, 1W).

        A "confirmed" signal means at least MTF_CONFIRMATION_REQUIRED
        timeframes agree on the direction.

        Returns:
        {
            "confirmed": bool,
            "agreeing_timeframes": int (out of 3),
            "details": {"1h": str, "1d": str, "1w": str},
            "mtf_score_bonus": int (0 to +15),
        }
        """
        if not config.ENABLE_MULTI_TIMEFRAME:
            return {"confirmed": True, "agreeing_timeframes": 3,
                    "details": {}, "mtf_score_bonus": 10}

        try:
            mtf_bars = self.fetcher.get_multi_timeframe_bars(
                symbol, start_date, end_date, is_crypto=is_crypto
            )

            agreeing = 0
            details = {}

            for tf_name, df in mtf_bars.items():
                if df is None or len(df) < 10:
                    details[tf_name] = "NO_DATA"
                    continue

                close = df["close"].astype(float)
                signal = self._get_tf_direction(close, direction)
                details[tf_name] = signal

                if signal == "AGREE":
                    agreeing += 1

            confirmed = agreeing >= config.MTF_CONFIRMATION_REQUIRED

            # Bonus score for multi-TF confirmation
            if agreeing == 3:
                bonus = 15
            elif agreeing == 2:
                bonus = 8
            elif agreeing == 1:
                bonus = 0
            else:
                bonus = -10  # All disagree = strong penalty

            return {
                "confirmed": confirmed,
                "agreeing_timeframes": agreeing,
                "details": details,
                "mtf_score_bonus": bonus,
            }

        except Exception as e:
            log.debug(f"MTF confirmation failed for {symbol}: {e}")
            return {"confirmed": True, "agreeing_timeframes": 0,
                    "details": {}, "mtf_score_bonus": 0}

    def _get_tf_direction(self, close: pd.Series, direction: str) -> str:
        """
        Determines if a single timeframe agrees with the given direction.
        Uses simple EMA crossover + RSI.
        """
        try:
            ema_fast = ta.ema(close, length=9)
            ema_slow = ta.ema(close, length=21)
            rsi = ta.rsi(close, length=14)

            if ema_fast is None or ema_slow is None or rsi is None:
                return "NO_DATA"

            ema_f = float(ema_fast.iloc[-1])
            ema_s = float(ema_slow.iloc[-1])
            rsi_val = float(rsi.iloc[-1])

            if direction == "long":
                # Bullish: EMA fast > slow AND RSI not overbought
                if ema_f > ema_s and rsi_val < 75:
                    return "AGREE"
                return "DISAGREE"
            else:  # short
                # Bearish: EMA fast < slow AND RSI not oversold
                if ema_f < ema_s and rsi_val > 25:
                    return "AGREE"
                return "DISAGREE"

        except Exception:
            return "NO_DATA"
