"""
Quant Engine - The mathematical core of Sentinel Autotrader.

Performs high-speed quantitative screening using 15+ technical and fundamental
indicators. No AI is used — pure math for 100% numerical accuracy.

Indicator Suite:
──────────────────────────────────────────────────────────────────
TREND/MOMENTUM:     RSI, MACD, EMA crossover, ADX (trend strength)
MEAN REVERSION:     Bollinger Bands, Z-Score
VOLUME:             OBV, VWAP deviation, Volume SMA ratio
VOLATILITY:         ATR, Historical volatility (σ)
RISK:               Sharpe Ratio, Sortino Ratio, Max Drawdown
FUNDAMENTAL:        Piotroski F-Score, Magic Formula, P/E, P/B
NEW IN V3.5:
  RELATIVE STRENGTH:  RS Rating vs SPY benchmark
  MULTI-TIMEFRAME:    1H + 1D + 1W signal confirmation
  EARNINGS AWARENESS: Blackout period before earnings dates
  SHORT SIGNALS:      Identifies bearish candidates for shorting
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional

import config
from core.data_fetcher import DataFetcher
from core.market_regime import MarketRegime
from utils.logger import get_logger

log = get_logger("sentinel.quant_engine")


class QuantEngine:
    """
    Multi-factor quantitative screening engine.
    Each evaluate_* method returns a standardized result dict:
    {
        "symbol": str,
        "asset_type": str,
        "score": float (0-100),
        "direction": str ("long" or "short"),
        "signals": dict,       # Raw indicator values
        "reason": str,         # Human-readable summary
        "sector": str,
        "atr": float,          # For position sizing downstream
        "rs_rating": float,    # Relative strength vs benchmark
        "earnings_risk": bool, # True if within earnings blackout
        "mtf_confirmed": bool, # Multi-timeframe agreement
    }
    """

    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
        self.regime = MarketRegime(fetcher=self.fetcher)

    # ─── Stock Evaluation ────────────────────────────────────────────────

    def evaluate_stock(self, symbol: str, start_date: str, end_date: str) -> dict:
        """
        Full multi-factor evaluation for equities.
        Combines technical indicators (60%) + fundamental analysis (40%).
        Enhanced with RS rating, MTF confirmation, earnings awareness, and short detection.
        """
        result = self._base_result(symbol, "stock")

        # 1. Fetch price data
        df = self.fetcher.get_stock_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < config.SMA_LONG + 10:
            result["reason"] = f"Insufficient data ({len(df) if df is not None else 0} bars)"
            return result

        # 2. Calculate all technical indicators
        signals = self._calculate_technicals(df)
        if signals is None:
            result["reason"] = "Technical calculation failed"
            return result

        # 3. Score technicals (0-60)
        tech_score, tech_reasons = self._score_technicals(signals)

        # GATING: The base tech score is 30. We only want strong longs (>40) or strong shorts (<20)
        # This prevents hitting yfinance/Alpaca rate limits on the 7,000+ asset scan
        if 20 <= tech_score <= 40:
            result["score"] = tech_score
            result["reason"] = f"Neutral technicals ({tech_score:.1f}). Gated to save API budget."
            return result

        # 4. Fetch and score fundamentals (0-40)
        fundamentals = self.fetcher.get_fundamentals(symbol)
        fund_score, fund_reasons = self._score_fundamentals(fundamentals)

        # 5. Combine base score
        total_score = tech_score + fund_score

        # 6. Relative Strength vs Benchmark
        rs = self.regime.calculate_relative_strength(symbol, start_date, end_date)
        result["rs_rating"] = rs["rs_rating"]
        if rs["rs_rank"] == "LEADER":
            total_score += 8
            tech_reasons.append(f"RS_LEADER({rs['rs_rating']:.2f})")
        elif rs["rs_rank"] == "STRONG":
            total_score += 4
            tech_reasons.append(f"RS_STRONG({rs['rs_rating']:.2f})")
        elif rs["rs_rank"] == "LAGGARD":
            total_score -= 8
            tech_reasons.append(f"RS_LAGGARD({rs['rs_rating']:.2f})")

        # 7. Earnings Calendar Check
        earnings = self.fetcher.get_earnings_date(symbol)
        result["earnings_risk"] = earnings["has_upcoming_earnings"]
        if earnings["has_upcoming_earnings"]:
            tech_reasons.append(f"EARNINGS_IN_{earnings['days_until']}D")
            total_score -= 5  # Penalty for earnings uncertainty

        # 8. Multi-Timeframe Confirmation
        direction = "long" if total_score >= 50 else "short"
        mtf = self.regime.confirm_multi_timeframe(symbol, start_date, end_date, direction)
        result["mtf_confirmed"] = mtf["confirmed"]
        total_score += mtf["mtf_score_bonus"]
        if mtf["agreeing_timeframes"] > 0:
            tech_reasons.append(f"MTF_{mtf['agreeing_timeframes']}/3")

        # 9. Determine direction (long vs short)
        result["score"] = max(0, min(100, total_score))
        if result["score"] <= config.QUANT_SHORT_SCORE and config.ENABLE_SHORT_SELLING:
            result["direction"] = "short"
            # For shorts, invert the score: lower quant = stronger short signal
            result["score"] = 100 - result["score"]
            tech_reasons.append("SHORT_SIGNAL")

        result["signals"] = signals
        result["atr"] = signals.get("atr", 0)
        result["sector"] = fundamentals.get("sector", "Unknown") if fundamentals else "Unknown"
        result["reason"] = " | ".join(tech_reasons + fund_reasons)

        return result

    # ─── Crypto Evaluation ───────────────────────────────────────────────

    def evaluate_crypto(self, symbol: str, start_date: str, end_date: str) -> dict:
        """
        Crypto evaluation using pure technical analysis.
        Focus: Bollinger Band squeeze + OBV divergence + momentum.
        """
        result = self._base_result(symbol, "crypto")

        df = self.fetcher.get_crypto_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 30:
            result["reason"] = f"Insufficient data ({len(df) if df is not None else 0} bars)"
            return result

        signals = self._calculate_technicals(df)
        if signals is None:
            result["reason"] = "Technical calculation failed"
            return result

        score = 50  # Base
        reasons = []

        # Bollinger Band Squeeze (high probability breakout)
        if signals.get("bb_squeeze", False):
            score += 15
            reasons.append("BB_SQUEEZE")

        # OBV trending up (smart money accumulation)
        if signals.get("obv_trend_up", False):
            score += 12
            reasons.append("OBV_UP")

        # RSI: oversold = opportunity, overbought = risk
        rsi = signals.get("rsi", 50)
        if rsi < 35:
            score += 12
            reasons.append(f"RSI_OVERSOLD({rsi:.0f})")
        elif rsi > 72:
            score -= 10
            reasons.append(f"RSI_OVERBOUGHT({rsi:.0f})")

        # MACD bullish crossover
        if signals.get("macd_bullish", False):
            score += 10
            reasons.append("MACD_BULL")

        # Strong trend (ADX > 25)
        adx = signals.get("adx", 0)
        if adx > 25:
            score += 8
            reasons.append(f"STRONG_TREND(ADX:{adx:.0f})")

        # Volume spike
        if signals.get("volume_spike", False):
            score += 8
            reasons.append("VOL_SPIKE")

        result["score"] = max(0, min(score, 100))
        result["signals"] = signals
        result["atr"] = signals.get("atr", 0)
        result["reason"] = " | ".join(reasons) if reasons else "Neutral"

        return result

    # ─── ETF Evaluation ──────────────────────────────────────────────────

    def evaluate_etf(self, symbol: str, start_date: str, end_date: str) -> dict:
        """
        ETF evaluation focused on trend-following and momentum.
        """
        result = self._base_result(symbol, "etf")

        df = self.fetcher.get_stock_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < config.SMA_LONG + 10:
            result["reason"] = f"Insufficient data ({len(df) if df is not None else 0} bars)"
            return result

        signals = self._calculate_technicals(df)
        if signals is None:
            result["reason"] = "Technical calculation failed"
            return result

        score = 50
        reasons = []

        # Volume filter (ETFs need high liquidity)
        vol_ratio = signals.get("volume_sma_ratio", 0)
        if vol_ratio < 0.5:
            result["reason"] = "LOW_VOLUME"
            return result

        # Price above SMA50 (uptrend)
        if signals.get("above_sma_long", False):
            score += 15
            reasons.append("UPTREND")

        # EMA crossover (short-term momentum)
        if signals.get("ema_bullish_cross", False):
            score += 10
            reasons.append("EMA_BULL_CROSS")

        # RSI in healthy range
        rsi = signals.get("rsi", 50)
        if 40 < rsi < 60:
            score += 8
            reasons.append(f"HEALTHY_RSI({rsi:.0f})")
        elif rsi < 30:
            score += 12
            reasons.append(f"OVERSOLD({rsi:.0f})")

        # Positive Sharpe
        sharpe = signals.get("sharpe", 0)
        if sharpe > 1.0:
            score += 10
            reasons.append(f"SHARPE({sharpe:.2f})")
        elif sharpe > 0.5:
            score += 5
            reasons.append(f"SHARPE({sharpe:.2f})")

        # Low drawdown
        max_dd = signals.get("max_drawdown", 0)
        if max_dd > -0.05:
            score += 5
            reasons.append(f"LOW_DD({max_dd:.1%})")

        result["score"] = max(0, min(score, 100))
        result["signals"] = signals
        result["atr"] = signals.get("atr", 0)
        result["reason"] = " | ".join(reasons) if reasons else "Neutral"

        return result

    # ─── Core Technical Calculations ─────────────────────────────────────

    def _calculate_technicals(self, df: pd.DataFrame) -> Optional[dict]:
        """
        Calculates 15+ technical indicators on a DataFrame.
        Returns a flat dict of signal values.
        """
        try:
            # Ensure we have the required columns
            required = {"close", "high", "low", "volume"}
            df_cols = {c.lower() for c in df.columns}
            if not required.issubset(df_cols):
                log.warning(f"Missing columns: {required - df_cols}")
                return None

            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            signals = {}

            # ─── RSI ────────────────────────────────────────────────
            rsi_series = ta.rsi(close, length=config.RSI_PERIOD)
            signals["rsi"] = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50

            # ─── MACD ───────────────────────────────────────────────
            macd_df = ta.macd(close, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
            if macd_df is not None and not macd_df.empty:
                macd_cols = macd_df.columns.tolist()
                macd_line = macd_df[macd_cols[0]].iloc[-1]
                signal_line = macd_df[macd_cols[2]].iloc[-1] if len(macd_cols) > 2 else 0
                histogram = macd_df[macd_cols[1]].iloc[-1] if len(macd_cols) > 1 else 0
                signals["macd"] = float(macd_line)
                signals["macd_signal"] = float(signal_line)
                signals["macd_histogram"] = float(histogram)
                signals["macd_bullish"] = macd_line > signal_line and histogram > 0
            else:
                signals["macd_bullish"] = False

            # ─── EMA Crossover ──────────────────────────────────────
            ema_short = ta.ema(close, length=config.EMA_SHORT)
            ema_long = ta.ema(close, length=config.EMA_LONG)
            if ema_short is not None and ema_long is not None:
                signals["ema_short"] = float(ema_short.iloc[-1])
                signals["ema_long"] = float(ema_long.iloc[-1])
                signals["ema_bullish_cross"] = (
                    float(ema_short.iloc[-1]) > float(ema_long.iloc[-1]) and
                    float(ema_short.iloc[-2]) <= float(ema_long.iloc[-2])
                ) if len(ema_short) > 1 else False

            # ─── SMA (trend detection) ─────────────────────────────
            sma_short = ta.sma(close, length=config.SMA_SHORT)
            sma_long = ta.sma(close, length=config.SMA_LONG)
            if sma_long is not None and not sma_long.empty:
                signals["sma_short"] = float(sma_short.iloc[-1]) if sma_short is not None else 0
                signals["sma_long"] = float(sma_long.iloc[-1])
                signals["above_sma_long"] = float(close.iloc[-1]) > float(sma_long.iloc[-1])
            else:
                signals["above_sma_long"] = False

            # ─── ADX (trend strength) ──────────────────────────────
            adx_df = ta.adx(high, low, close, length=14)
            if adx_df is not None and not adx_df.empty:
                adx_col = [c for c in adx_df.columns if "ADX" in c and "DM" not in c]
                signals["adx"] = float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 0
            else:
                signals["adx"] = 0

            # ─── Bollinger Bands ───────────────────────────────────
            bb_df = ta.bbands(close, length=config.BB_PERIOD, std=config.BB_STD)
            if bb_df is not None and not bb_df.empty:
                bbb_col = [c for c in bb_df.columns if "BBB" in c]
                bbp_col = [c for c in bb_df.columns if "BBP" in c]
                if bbb_col:
                    bandwidth = float(bb_df[bbb_col[0]].iloc[-1])
                    signals["bb_bandwidth"] = bandwidth
                    signals["bb_squeeze"] = bandwidth < config.BB_SQUEEZE_THRESHOLD
                if bbp_col:
                    signals["bb_percent_b"] = float(bb_df[bbp_col[0]].iloc[-1])

            # ─── ATR (volatility) ──────────────────────────────────
            atr_series = ta.atr(high, low, close, length=config.ATR_PERIOD)
            if atr_series is not None and not atr_series.empty:
                signals["atr"] = float(atr_series.iloc[-1])
                signals["atr_pct"] = signals["atr"] / float(close.iloc[-1]) * 100
            else:
                signals["atr"] = 0
                signals["atr_pct"] = 0

            # ─── OBV (On-Balance Volume) ───────────────────────────
            obv_series = ta.obv(close, volume)
            if obv_series is not None and not obv_series.empty:
                obv_sma = ta.sma(obv_series, length=10)
                if obv_sma is not None and not obv_sma.empty:
                    signals["obv_trend_up"] = float(obv_series.iloc[-1]) > float(obv_sma.iloc[-1])
                else:
                    signals["obv_trend_up"] = False

            # ─── Volume Analysis ───────────────────────────────────
            vol_sma = ta.sma(volume, length=20)
            if vol_sma is not None and not vol_sma.empty:
                current_vol = float(volume.iloc[-1])
                avg_vol = float(vol_sma.iloc[-1])
                signals["volume_sma_ratio"] = current_vol / avg_vol if avg_vol > 0 else 0
                signals["volume_spike"] = signals["volume_sma_ratio"] > 1.5
            else:
                signals["volume_sma_ratio"] = 0
                signals["volume_spike"] = False

            # ─── Historical Volatility ─────────────────────────────
            returns = close.pct_change().dropna()
            if len(returns) > 10:
                signals["volatility_30d"] = float(returns.tail(30).std() * np.sqrt(252))
            else:
                signals["volatility_30d"] = 0

            # ─── Sharpe Ratio ──────────────────────────────────────
            if len(returns) >= config.SHARPE_WINDOW:
                window_returns = returns.tail(config.SHARPE_WINDOW)
                mean_return = float(window_returns.mean()) * 252  # Annualized
                std_return = float(window_returns.std()) * np.sqrt(252)
                signals["sharpe"] = (mean_return - config.RISK_FREE_RATE) / std_return if std_return > 0 else 0

                # Sortino (only downside deviation)
                downside = window_returns[window_returns < 0]
                downside_std = float(downside.std()) * np.sqrt(252) if len(downside) > 0 else 0.001
                signals["sortino"] = (mean_return - config.RISK_FREE_RATE) / downside_std
            else:
                signals["sharpe"] = 0
                signals["sortino"] = 0

            # ─── Max Drawdown ──────────────────────────────────────
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            signals["max_drawdown"] = float(drawdown.min())

            # ─── Z-Score (mean reversion) ──────────────────────────
            if len(close) >= 20:
                mean_20 = close.tail(20).mean()
                std_20 = close.tail(20).std()
                signals["z_score"] = float((close.iloc[-1] - mean_20) / std_20) if std_20 > 0 else 0
            else:
                signals["z_score"] = 0

            # ─── Current Price ─────────────────────────────────────
            signals["price"] = float(close.iloc[-1])

            return signals

        except Exception as e:
            log.error(f"Technical calculation error: {e}", exc_info=True)
            return None

    # ─── Scoring Functions ───────────────────────────────────────────────

    def _score_technicals(self, signals: dict) -> tuple[float, list[str]]:
        """
        Scores technical indicators on a 0-60 scale.
        Returns (score, list_of_reasons).
        """
        score = 30  # Base (neutral)
        reasons = []

        # RSI (max ±10)
        rsi = signals.get("rsi", 50)
        if rsi < config.RSI_OVERSOLD:
            score += 10
            reasons.append(f"RSI_OVERSOLD({rsi:.0f})")
        elif rsi < 40:
            score += 5
            reasons.append(f"RSI_LOW({rsi:.0f})")
        elif rsi > config.RSI_OVERBOUGHT:
            score -= 10
            reasons.append(f"RSI_OVERBOUGHT({rsi:.0f})")

        # MACD (max +8)
        if signals.get("macd_bullish", False):
            score += 8
            reasons.append("MACD_BULLISH")

        # EMA crossover (max +6)
        if signals.get("ema_bullish_cross", False):
            score += 6
            reasons.append("EMA_CROSS_UP")

        # Trend (max +6)
        if signals.get("above_sma_long", False):
            score += 6
            reasons.append("UPTREND")

        # ADX trend strength (max +5)
        adx = signals.get("adx", 0)
        if adx > 30:
            score += 5
            reasons.append(f"STRONG_TREND")

        # Bollinger squeeze (max +5)
        if signals.get("bb_squeeze", False):
            score += 5
            reasons.append("BB_SQUEEZE")

        # OBV (max +5)
        if signals.get("obv_trend_up", False):
            score += 5
            reasons.append("OBV_RISING")

        # Volume spike (max +3)
        if signals.get("volume_spike", False):
            score += 3
            reasons.append("VOL_SPIKE")

        # Sharpe (max ±5)
        sharpe = signals.get("sharpe", 0)
        if sharpe > 1.5:
            score += 5
            reasons.append(f"HIGH_SHARPE({sharpe:.1f})")
        elif sharpe < -0.5:
            score -= 5
            reasons.append(f"NEG_SHARPE({sharpe:.1f})")

        # Drawdown penalty (max -8)
        max_dd = signals.get("max_drawdown", 0)
        if max_dd < -0.20:
            score -= 8
            reasons.append(f"DEEP_DD({max_dd:.1%})")
        elif max_dd < -0.10:
            score -= 4
            reasons.append(f"MOD_DD({max_dd:.1%})")

        return max(0, min(60, score)), reasons

    def _score_fundamentals(self, fundamentals: Optional[dict]) -> tuple[float, list[str]]:
        """
        Scores fundamental data on a 0-40 scale.
        Returns (score, list_of_reasons).
        """
        if not fundamentals:
            return 15, ["NO_FUNDAMENTALS"]

        score = 15  # Base (neutral)
        reasons = []

        # Piotroski F-Score (max +12)
        f_score = fundamentals.get("piotroski_f_score", 0)
        if f_score >= 7:
            score += 12
            reasons.append(f"F_SCORE({f_score})")
        elif f_score >= 5:
            score += 6
            reasons.append(f"F_SCORE({f_score})")
        elif f_score <= 2:
            score -= 5
            reasons.append(f"WEAK_F({f_score})")

        # Magic Formula (max +8)
        mf = fundamentals.get("magic_formula", 0)
        if mf > 0.25:
            score += 8
            reasons.append(f"MAGIC_HIGH")
        elif mf > 0.10:
            score += 4
            reasons.append(f"MAGIC_OK")

        # Earnings Yield (max +5)
        ey = fundamentals.get("earnings_yield", 0)
        if ey > 0.08:
            score += 5
            reasons.append(f"EY({ey:.1%})")

        # ROE (max +5)
        roe = fundamentals.get("roe", 0)
        if roe > 0.20:
            score += 5
            reasons.append(f"HIGH_ROE({roe:.0%})")

        # Revenue Growth (max +5)
        rg = fundamentals.get("revenue_growth", 0)
        if rg > 0.15:
            score += 5
            reasons.append(f"GROWTH({rg:.0%})")

        # Debt penalty (max -5)
        de = fundamentals.get("debt_to_equity", 0)
        if de > 200:
            score -= 5
            reasons.append(f"HIGH_DEBT(D/E:{de:.0f})")

        return max(0, min(40, score)), reasons

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _base_result(self, symbol: str, asset_type: str) -> dict:
        return {
            "symbol": symbol,
            "asset_type": asset_type,
            "score": 0,
            "direction": "long",
            "signals": {},
            "reason": "",
            "sector": "Unknown",
            "atr": 0,
            "rs_rating": 1.0,
            "earnings_risk": False,
            "mtf_confirmed": False,
        }
