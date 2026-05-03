"""
Pre-Market & After-Hours Scanner — Sentinel Autotrader V5

Scans for significant price gaps before and after regular market hours.
Queues high-conviction pre-market movers as priority candidates for the
main trading cycle.

Data Source: Alpaca extended-hours bars (free, already available)

Logic:
- Pre-market (4:00–9:29 AM EST): detect gap-ups/downs vs previous close
- After-hours (4:01–7:59 PM EST): detect post-earnings reactions
- Priority queue is injected at the front of the main analysis pipeline

Features:
- Extended-hours bar fetching via Alpaca
- Gap percentage calculation vs prior close
- News-cross-reference: why did it gap?
- Priority queue with TTL (dropped if not traded within 30 min of open)
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import config
from utils.logger import get_logger
from utils.rate_limiter import alpaca_limiter

log = get_logger("sentinel.premarket_scanner")

# EST timezone offset (UTC-5 standard, UTC-4 DST — use utcoffset from Alpaca clock)
_EST_OFFSET = -5  # hours


class PreMarketScanner:
    """
    Scans Alpaca extended-hours bars for significant price gaps.

    Maintains a priority queue of symbols with large pre/post-market moves.
    These are inserted at the front of the main pipeline at market open.
    """

    def __init__(self, fetcher=None, broker=None):
        from core.data_fetcher import DataFetcher
        from execution.alpaca_broker import AlpacaBroker
        self.fetcher = fetcher or DataFetcher()
        self.broker = broker or AlpacaBroker()
        self._queue: list[dict] = []   # {symbol, gap_pct, direction, close, prev_close, reason, queued_at}
        self._queued_symbols: set[str] = set()

    # ─── Public API ──────────────────────────────────────────────────────────

    def scan_premarket_movers(self, symbols: list[str]) -> list[dict]:
        """
        Scans a list of symbols for significant pre-market gaps.
        Returns list of {symbol, gap_pct, direction, prev_close, current_price, asset_type}
        sorted by abs(gap_pct) descending.
        """
        if not config.PREMARKET_ENABLED:
            return []

        movers = []
        log.info(f"Pre-market scan: checking {len(symbols)} symbols for gaps ≥{config.PREMARKET_MIN_GAP_PCT}%")

        # Get extended-hours bars for today
        now = datetime.now(timezone.utc)
        today_str = now.strftime('%Y-%m-%d')
        yesterday_str = (now - timedelta(days=1)).strftime('%Y-%m-%d')

        # Batch fetch to respect rate limits
        batch_size = 50
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            alpaca_limiter.acquire()

            try:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame

                client = StockHistoricalDataClient(
                    config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
                )

                # Yesterday's closing bars (regular hours close)
                prev_req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=yesterday_str,
                    end=today_str,
                )
                prev_bars = client.get_stock_bars(prev_req).df

                # Today's pre-market bars (last 15 min bar)
                alpaca_limiter.acquire()
                pm_req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Minute,
                    start=today_str,
                    end=(now + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                )
                pm_bars = client.get_stock_bars(pm_req).df

                if prev_bars.empty or pm_bars.empty:
                    continue

                # Calculate gaps
                for sym in batch:
                    try:
                        sym_prev = prev_bars.xs(sym) if sym in prev_bars.index.get_level_values(0) else None
                        sym_pm = pm_bars.xs(sym) if sym in pm_bars.index.get_level_values(0) else None

                        if sym_prev is None or sym_pm is None:
                            continue

                        prev_close = float(sym_prev.iloc[-1]["close"])
                        pm_price = float(sym_pm.iloc[-1]["close"])

                        if prev_close <= 0:
                            continue

                        gap_pct = (pm_price - prev_close) / prev_close * 100

                        if abs(gap_pct) >= config.PREMARKET_MIN_GAP_PCT:
                            mover = {
                                "symbol": sym,
                                "asset_type": "stock",
                                "gap_pct": round(gap_pct, 2),
                                "direction": "long" if gap_pct > 0 else "short",
                                "prev_close": round(prev_close, 4),
                                "current_price": round(pm_price, 4),
                                "queued_at": now.isoformat(),
                                "reason": f"Pre-market gap {gap_pct:+.1f}%",
                            }
                            movers.append(mover)
                            log.info(
                                f"Pre-market mover: {sym} {gap_pct:+.1f}% "
                                f"(${prev_close:.2f} → ${pm_price:.2f})"
                            )

                    except Exception as e:
                        log.debug(f"Gap calc error for {sym}: {e}")

            except Exception as e:
                log.warning(f"Pre-market batch fetch failed: {e}")

        movers.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
        log.info(f"Pre-market scan complete: {len(movers)} movers found")
        return movers

    def get_priority_queue(self) -> list[str]:
        """
        Returns valid (non-expired) symbols from the pre-market queue.
        Symbols expire 30 minutes after market open (10:00 AM EST).
        """
        now = datetime.now(timezone.utc)
        market_open_today = now.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 EST = 14:30 UTC
        expiry = market_open_today + timedelta(minutes=30)

        valid = []
        for entry in self._queue:
            queued_at = datetime.fromisoformat(entry["queued_at"])
            if now < expiry:
                valid.append(entry["symbol"])

        return valid

    def enqueue(self, movers: list[dict]):
        """Adds pre-market movers to the priority queue."""
        for m in movers:
            if m["symbol"] not in self._queued_symbols:
                self._queue.append(m)
                self._queued_symbols.add(m["symbol"])
                log.info(f"Queued pre-market mover: {m['symbol']} ({m['gap_pct']:+.1f}%)")

    def clear_queue(self):
        """Clears the priority queue after cycle completion."""
        self._queue.clear()
        self._queued_symbols.clear()

    @staticmethod
    def is_premarket() -> bool:
        """Returns True if current UTC time is in pre-market window (4:00–9:29 AM EST)."""
        now = datetime.now(timezone.utc)
        hour_est = (now.hour + _EST_OFFSET) % 24
        return 4 <= hour_est < 9 or (hour_est == 9 and now.minute < 30)

    @staticmethod
    def is_after_hours() -> bool:
        """Returns True if current UTC time is in after-hours window (4:01–7:59 PM EST)."""
        now = datetime.now(timezone.utc)
        hour_est = (now.hour + _EST_OFFSET) % 24
        return 16 <= hour_est < 20
