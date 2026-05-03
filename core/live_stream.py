"""
Live Stream Manager - WebSocket real-time price feed from Alpaca.

Replaces 30-minute REST polling with real-time bar streaming.
Maintains a thread-safe in-memory price cache that the main pipeline reads.

Features:
- Alpaca StockDataStream + CryptoDataStream WebSocket connections
- Thread-safe price cache (latest bar per symbol)
- Spike detection: logs alert when a bar moves ≥ SPIKE_ALERT_PCT
- Auto-reconnect on disconnect
- Graceful shutdown
"""
import threading
import time
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger("sentinel.live_stream")


class LiveStreamManager:
    """
    Manages WebSocket connections to Alpaca for real-time bar data.

    Usage:
        stream = LiveStreamManager()
        stream.start(stock_symbols, crypto_symbols)
        # Later:
        bar = stream.get_latest_bar("NVDA")
        stream.stop()
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}   # {symbol: {open, high, low, close, volume, ts}}
        self._lock = threading.Lock()
        self._running = False
        self._stock_thread: Optional[threading.Thread] = None
        self._crypto_thread: Optional[threading.Thread] = None
        self._spike_callbacks: list = []    # Registered callbacks for spike events

    # ─── Public API ──────────────────────────────────────────────────────────

    def start(self, stock_symbols: list[str], crypto_symbols: list[str]):
        """Starts background WebSocket threads for stocks and crypto."""
        if not config.LIVE_STREAM_ENABLED:
            log.info("Live streaming disabled (LIVE_STREAM_ENABLED=false)")
            return

        self._running = True
        log.info(
            f"Starting live stream: {len(stock_symbols)} stocks, "
            f"{len(crypto_symbols)} crypto pairs"
        )

        if stock_symbols:
            self._stock_thread = threading.Thread(
                target=self._run_stock_stream,
                args=(stock_symbols,),
                daemon=True,
                name="SentinelStockStream"
            )
            self._stock_thread.start()

        if crypto_symbols:
            self._crypto_thread = threading.Thread(
                target=self._run_crypto_stream,
                args=(crypto_symbols,),
                daemon=True,
                name="SentinelCryptoStream"
            )
            self._crypto_thread.start()

        log.info("Live stream threads started.")

    def stop(self):
        """Signals both stream threads to stop."""
        self._running = False
        log.info("Live stream manager stopping.")

    def get_latest_bar(self, symbol: str) -> Optional[dict]:
        """
        Returns the latest live bar for a symbol, or None if not yet received.
        Thread-safe.
        """
        with self._lock:
            return self._cache.get(symbol)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Returns just the latest close price for a symbol."""
        bar = self.get_latest_bar(symbol)
        return bar["close"] if bar else None

    def on_spike(self, callback):
        """Register a callback for price spikes: callback(symbol, bar, pct_change)"""
        self._spike_callbacks.append(callback)

    @property
    def cache_size(self) -> int:
        """Number of symbols currently in the live cache."""
        with self._lock:
            return len(self._cache)

    # ─── Internal Handlers ───────────────────────────────────────────────────

    def _update_cache(self, symbol: str, bar: dict):
        """Updates the price cache and checks for spikes. Thread-safe."""
        with self._lock:
            prev = self._cache.get(symbol)
            self._cache[symbol] = bar

        # Spike detection (outside lock to avoid blocking)
        if prev and prev.get("close", 0) > 0:
            pct_change = abs(bar["close"] - prev["close"]) / prev["close"] * 100
            if pct_change >= config.SPIKE_ALERT_PCT:
                direction = "↑" if bar["close"] > prev["close"] else "↓"
                log.warning(
                    f"🚨 PRICE SPIKE: {symbol} {direction}{pct_change:.1f}% "
                    f"(${prev['close']:.2f} → ${bar['close']:.2f})"
                )
                for cb in self._spike_callbacks:
                    try:
                        cb(symbol, bar, pct_change)
                    except Exception as e:
                        log.debug(f"Spike callback error: {e}")

    # ─── Stock WebSocket Thread ───────────────────────────────────────────────

    def _run_stock_stream(self, symbols: list[str]):
        """Runs the Alpaca stock WebSocket stream in a loop with auto-reconnect."""
        while self._running:
            try:
                from alpaca.data.live import StockDataStream
                from alpaca.data.enums import DataFeed

                stream = StockDataStream(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                    feed=DataFeed.IEX,  # IEX is free; switch to SIP for paid plans
                )

                async def handle_bar(bar):
                    self._update_cache(bar.symbol, {
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "ts": str(bar.timestamp),
                    })

                # Subscribe in batches of 100 (Alpaca limit per subscription)
                batch_size = 100
                for i in range(0, len(symbols), batch_size):
                    batch = symbols[i:i + batch_size]
                    stream.subscribe_bars(handle_bar, *batch)

                log.info(f"Stock stream connected. Subscribed to {len(symbols)} symbols.")
                stream.run()  # Blocks until disconnect

            except Exception as e:
                if self._running:
                    log.warning(f"Stock stream disconnected: {e}. Reconnecting in 10s...")
                    time.sleep(10)
                else:
                    break

        log.info("Stock stream thread exited.")

    # ─── Crypto WebSocket Thread ──────────────────────────────────────────────

    def _run_crypto_stream(self, symbols: list[str]):
        """Runs the Alpaca crypto WebSocket stream with auto-reconnect."""
        while self._running:
            try:
                from alpaca.data.live import CryptoDataStream

                stream = CryptoDataStream(
                    config.ALPACA_API_KEY,
                    config.ALPACA_SECRET_KEY,
                )

                async def handle_crypto_bar(bar):
                    self._update_cache(bar.symbol, {
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "ts": str(bar.timestamp),
                    })

                stream.subscribe_bars(handle_crypto_bar, *symbols)
                log.info(f"Crypto stream connected. Subscribed to {len(symbols)} pairs.")
                stream.run()

            except Exception as e:
                if self._running:
                    log.warning(f"Crypto stream disconnected: {e}. Reconnecting in 10s...")
                    time.sleep(10)
                else:
                    break

        log.info("Crypto stream thread exited.")
