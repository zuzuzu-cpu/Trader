"""
Options Flow Analyzer — Sentinel Autotrader V5

Analyzes options chains to detect unusual institutional positioning.
Smart money often positions in options BEFORE a big move in the stock.

Data Source: Yahoo Finance options chain via yfinance (free, always available)

Signals:
- Put/Call Volume Ratio: <0.5 = bullish, >1.5 = bearish
- IV Rank: High IV vs historical = major move expected
- Unusual Volume: Call/put volume > OPTIONS_VOLUME_THRESHOLD x average OI
- Near-the-money short-dated calls sweeping = directional bet by institution

Scoring:
  +4  Extreme unusual call volume + short-dated (smart money buying upside)
  +2  Elevated call volume, bullish positioning
  +1  Slight bullish options bias
   0  Neutral
  -1  Slight bearish bias
  -2  Elevated put volume
  -4  Extreme unusual put volume (smart money hedging/betting against)

Caching: 4-hour cache (options data refreshes slowly during day)
"""
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf
import numpy as np

import config
from utils.logger import get_logger

log = get_logger("sentinel.options_flow")

_CACHE_TTL_HOURS = 4


class OptionsFlow:
    """
    Analyzes put/call ratios and unusual volume in options chains.
    """

    def __init__(self):
        self._cache_dir = config.DATA_DIR / "options_cache"
        self._cache_dir.mkdir(exist_ok=True)

    def get_options_score(self, symbol: str) -> dict:
        """
        Returns options flow analysis for a symbol.

        Returns:
        {
            "score": int (-4 to +4),
            "pc_ratio": float,         # Put/call volume ratio
            "iv_rank": float,          # Implied volatility rank (0-100)
            "unusual_calls": bool,     # Unusual call volume detected
            "unusual_puts": bool,      # Unusual put volume detected
            "call_volume": int,
            "put_volume": int,
            "summary": str,
        }
        """
        if not config.OPTIONS_FLOW_ENABLED:
            return self._empty_result()

        # Options only available for US stocks/ETFs, not crypto
        if "/" in symbol:  # Crypto pairs use SYMBOL/USD format
            return self._empty_result(summary="Options not available for crypto")

        cached = self._read_cache(symbol)
        if cached is not None:
            return cached

        result = self._fetch_and_score(symbol)
        self._write_cache(symbol, result)
        return result

    # ─── Core Logic ──────────────────────────────────────────────────────────

    def _fetch_and_score(self, symbol: str) -> dict:
        """Fetches options chain from Yahoo Finance and scores the flow."""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return self._empty_result(summary="No options chain available")

            # Focus on near-dated expirations (most liquid, most predictive)
            # Take up to the first 3 expirations (typically 0-45 DTE)
            near_expirations = expirations[:min(3, len(expirations))]

            total_call_vol = 0
            total_put_vol = 0
            total_call_oi = 0
            total_put_oi = 0
            iv_values = []
            unusual_calls = False
            unusual_puts = False

            for exp in near_expirations:
                try:
                    chain = ticker.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts

                    if calls.empty or puts.empty:
                        continue

                    # Aggregate volumes
                    call_vol = int(calls["volume"].fillna(0).sum())
                    put_vol = int(puts["volume"].fillna(0).sum())
                    call_oi = int(calls["openInterest"].fillna(0).sum())
                    put_oi = int(puts["openInterest"].fillna(0).sum())

                    total_call_vol += call_vol
                    total_put_vol += put_vol
                    total_call_oi += call_oi
                    total_put_oi += put_oi

                    # Collect IV values
                    call_ivs = calls["impliedVolatility"].dropna().tolist()
                    put_ivs = puts["impliedVolatility"].dropna().tolist()
                    iv_values.extend(call_ivs + put_ivs)

                    # Unusual volume detection: volume >> open interest
                    threshold = config.OPTIONS_VOLUME_THRESHOLD
                    if call_oi > 0 and call_vol > call_oi * threshold:
                        unusual_calls = True
                        log.info(f"Unusual call volume: {symbol} exp={exp} vol={call_vol:,} oi={call_oi:,}")
                    if put_oi > 0 and put_vol > put_oi * threshold:
                        unusual_puts = True
                        log.info(f"Unusual put volume: {symbol} exp={exp} vol={put_vol:,} oi={put_oi:,}")

                except Exception as e:
                    log.debug(f"Options chain parse error for {symbol}/{exp}: {e}")
                    continue

            # Put/Call ratio
            if total_call_vol > 0:
                pc_ratio = round(total_put_vol / total_call_vol, 3)
            else:
                pc_ratio = 1.0

            # IV Rank (simple: current avg IV as percentile signal)
            avg_iv = float(np.mean(iv_values)) * 100 if iv_values else 25.0
            # Rough IV rank: normalize to 0-100 assuming typical range 10-80%
            iv_rank = min(100, max(0, (avg_iv - 10) / 70 * 100))

            # Scoring
            score = 0
            reasons = []

            if unusual_calls and not unusual_puts:
                score += 4
                reasons.append("UNUSUAL_CALL_SWEEP")
            elif unusual_puts and not unusual_calls:
                score -= 4
                reasons.append("UNUSUAL_PUT_SWEEP")
            elif unusual_calls and unusual_puts:
                reasons.append("UNUSUAL_BOTH_SIDES (strangle/straddle)")

            # PC ratio adjustment
            if pc_ratio < 0.5:
                score += 2
                reasons.append(f"BULLISH_PC_RATIO({pc_ratio:.2f})")
            elif pc_ratio < 0.8:
                score += 1
                reasons.append(f"SLIGHT_BULLISH_PC({pc_ratio:.2f})")
            elif pc_ratio > 1.5:
                score -= 2
                reasons.append(f"BEARISH_PC_RATIO({pc_ratio:.2f})")
            elif pc_ratio > 1.2:
                score -= 1
                reasons.append(f"SLIGHT_BEARISH_PC({pc_ratio:.2f})")

            # Clamp score
            score = max(-4, min(4, score))

            summary_parts = [
                f"P/C={pc_ratio:.2f}",
                f"calls={total_call_vol:,}",
                f"puts={total_put_vol:,}",
                f"IV≈{avg_iv:.0f}%",
            ]
            if unusual_calls:
                summary_parts.append("⚡UNUSUAL CALLS")
            if unusual_puts:
                summary_parts.append("⚡UNUSUAL PUTS")

            summary = " | ".join(summary_parts)

            result = {
                "score": score,
                "pc_ratio": pc_ratio,
                "iv_rank": round(iv_rank, 1),
                "iv_avg": round(avg_iv, 1),
                "unusual_calls": unusual_calls,
                "unusual_puts": unusual_puts,
                "call_volume": total_call_vol,
                "put_volume": total_put_vol,
                "reasons": reasons,
                "summary": summary,
            }

            log.info(f"Options [{symbol}]: {summary} → score={score}")
            return result

        except Exception as e:
            log.debug(f"Options flow failed for {symbol}: {e}")
            return self._empty_result(summary=f"Options fetch error: {str(e)[:50]}")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(summary: str = "No options data") -> dict:
        return {
            "score": 0,
            "pc_ratio": 1.0,
            "iv_rank": 0.0,
            "iv_avg": 0.0,
            "unusual_calls": False,
            "unusual_puts": False,
            "call_volume": 0,
            "put_volume": 0,
            "reasons": [],
            "summary": summary,
        }

    def _cache_path(self, symbol: str) -> Path:
        safe = hashlib.md5(symbol.encode()).hexdigest()
        return self._cache_dir / f"options_{safe}.json"

    def _read_cache(self, symbol: str) -> Optional[dict]:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > _CACHE_TTL_HOURS:
            path.unlink(missing_ok=True)
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, symbol: str, data: dict):
        try:
            with open(self._cache_path(symbol), "w") as f:
                json.dump(data, f)
        except Exception as e:
            log.debug(f"Options cache write failed: {e}")
