"""
Alpaca Broker - Paper Trading Execution Engine V4

Handles all interaction with Alpaca's Trading API:
- Market orders with fractional share support (notional)
- Short selling via market orders
- Trailing stop orders with fill monitoring
- Partial profit taking (scale-out at target)
- Market hours awareness (prevents "not_filled" on closed markets)
- Extended hours trading
- Position management and portfolio queries
- Order status tracking
"""
import os
import time
import math
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TrailingStopOrderRequest,
    LimitOrderRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError

import config
from utils.logger import get_logger
from utils.rate_limiter import alpaca_limiter, retry_on_rate_limit

log = get_logger("sentinel.alpaca_broker")


class AlpacaBroker:
    """
    Production-grade Alpaca paper trading broker with V4 features:
    - Market hours check to avoid "not_filled" errors
    - Partial profit taking (sell 50% at +5%, trail the rest)
    - Automatic order cancellation on timeout
    """

    def __init__(self):
        self.trading_client = TradingClient(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=True,
        )

    # ─── Account ─────────────────────────────────────────────────────

    @retry_on_rate_limit
    def get_account(self) -> dict:
        """Returns account details as a clean dict."""
        alpaca_limiter.acquire()
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "daytrade_count": account.daytrade_count,
            "status": account.status,
        }

    # ─── Market Hours Check ──────────────────────────────────────────

    @retry_on_rate_limit
    def is_market_open(self) -> bool:
        """Checks if the US stock market is currently open using Alpaca's clock API."""
        # Safety net: Manually check for weekends (Sat=5, Sun=6)
        from datetime import datetime
        import pytz
        ny_time = datetime.now(pytz.timezone('America/New_York'))
        if ny_time.weekday() >= 5:
            return False

        alpaca_limiter.acquire()
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            log.warning(f"Failed to check market clock: {e}")
            return False  # Assume closed if we can't check

    # ─── Order Execution ─────────────────────────────────────────────

    @retry_on_rate_limit
    def place_buy_order(self, symbol: str, notional: float,
                        extended_hours: bool = False) -> Optional[str]:
        """
        Places a market buy order using notional (dollar amount).
        Supports fractional shares automatically.
        Returns the order ID or None on failure.
        """
        alpaca_limiter.acquire()

        # Determine time in force
        tif = TimeInForce.DAY
        if extended_hours and config.TRADE_EXTENDED_HOURS:
            # For extended hours, we need to use limit orders (Alpaca requirement)
            # For simplicity, we'll use DAY orders during market hours
            tif = TimeInForce.DAY

        try:
            order_request = MarketOrderRequest(
                symbol=symbol.replace("/", ""),  # Alpaca crypto uses BTCUSD not BTC/USD
                notional=round(notional, 2),
                side=OrderSide.BUY,
                time_in_force=tif,
            )
            order = self.trading_client.submit_order(order_request)
            log.info(
                f"BUY order submitted: {symbol} ${notional:.2f} → order_id={order.id}",
                extra={"symbol": symbol, "notional": notional, "order_id": str(order.id), "action": "BUY"}
            )
            return str(order.id)

        except APIError as e:
            log.error(f"Alpaca API error placing buy for {symbol}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error placing buy for {symbol}: {e}")
            return None

    @retry_on_rate_limit
    def place_short_order(self, symbol: str, notional: float) -> Optional[str]:
        """
        Places a market short sell order using notional (dollar amount).
        Returns the order ID or None on failure.
        """
        alpaca_limiter.acquire()
        try:
            order_request = MarketOrderRequest(
                symbol=symbol.replace("/", ""),
                notional=round(notional, 2),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            log.info(
                f"SHORT order submitted: {symbol} ${notional:.2f} → order_id={order.id}",
                extra={"symbol": symbol, "notional": notional, "order_id": str(order.id), "action": "SHORT"}
            )
            return str(order.id)
        except APIError as e:
            log.error(f"Alpaca API error placing short for {symbol}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error placing short for {symbol}: {e}")
            return None

    @retry_on_rate_limit
    def place_trailing_stop(self, symbol: str, qty: float,
                            trail_percent: float,
                            side: str = "sell") -> Optional[str]:
        """
        Places a trailing stop order to protect a position.
        side='sell' for long protection, side='buy' for short protection.
        Returns the order ID or None on failure.
        """
        alpaca_limiter.acquire()
        order_side = OrderSide.SELL if side == "sell" else OrderSide.BUY
        try:
            stop_request = TrailingStopOrderRequest(
                symbol=symbol.replace("/", ""),
                qty=abs(qty),
                side=order_side,
                time_in_force=TimeInForce.GTC,
                trail_percent=trail_percent,
            )
            order = self.trading_client.submit_order(stop_request)
            log.info(
                f"TRAILING STOP set: {symbol} qty={qty} trail={trail_percent}% side={side} → order_id={order.id}",
                extra={"symbol": symbol, "action": "TRAILING_STOP", "order_id": str(order.id)}
            )
            return str(order.id)

        except APIError as e:
            log.error(f"Alpaca API error placing trailing stop for {symbol}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error placing trailing stop for {symbol}: {e}")
            return None

    @retry_on_rate_limit
    def place_limit_order(self, symbol: str, qty: float, limit_price: float,
                          side: str = "sell") -> Optional[str]:
        """
        Places a limit order for partial profit taking.
        Returns order ID or None on failure.
        """
        alpaca_limiter.acquire()
        order_side = OrderSide.SELL if side == "sell" else OrderSide.BUY
        try:
            limit_request = LimitOrderRequest(
                symbol=symbol.replace("/", ""),
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
                limit_price=round(limit_price, 2),
            )
            order = self.trading_client.submit_order(limit_request)
            log.info(
                f"LIMIT ORDER set: {side.upper()} {symbol} qty={qty:.4f} @ ${limit_price:.2f} → order_id={order.id}",
                extra={"symbol": symbol, "action": "LIMIT", "order_id": str(order.id)}
            )
            return str(order.id)
        except APIError as e:
            log.error(f"Alpaca API error placing limit order for {symbol}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error placing limit order for {symbol}: {e}")
            return None

    @retry_on_rate_limit
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an open order by ID."""
        alpaca_limiter.acquire()
        try:
            self.trading_client.cancel_order_by_id(order_id)
            log.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            log.debug(f"Failed to cancel order {order_id}: {e}")
            return False

    def execute_trade(self, symbol: str, notional: float,
                      trailing_stop_pct: float,
                      direction: str = "long",
                      asset_type: str = "stock",
                      max_wait_seconds: int = 30) -> dict:
        """
        Full trade execution pipeline V4:
        1. Check if market is open (for stocks/ETFs)
        2. Place market order (buy for long, sell for short)
        3. Wait for fill
        4. If partial profit enabled: split into take-profit + trailing stop
        5. Otherwise: place single trailing stop

        Returns execution result dict.
        """
        result = {
            "symbol": symbol,
            "direction": direction,
            "order_id": None,
            "stop_order_id": None,
            "take_profit_order_id": None,
            "fill_price": None,
            "fill_qty": None,
            "status": "failed",
        }

        # ─── Step 0: Market hours check (stocks/ETFs only) ───────────
        is_crypto = asset_type == "crypto" or "/" in symbol
        if not is_crypto and config.SKIP_CLOSED_MARKET:
            if not self.is_market_open():
                log.info(f"Market closed — skipping {direction.upper()} {symbol} (stock/ETF orders won't fill)")
                result["status"] = "market_closed"
                return result

        # ─── Step 1: Place order ─────────────────────────────────────
        if direction == "long":
            order_id = self.place_buy_order(symbol, notional)
        else:
            order_id = self.place_short_order(symbol, notional)

        if not order_id:
            result["status"] = f"{direction}_order_failed"
            return result
        result["order_id"] = order_id

        # ─── Step 2: Wait for fill ───────────────────────────────────
        fill_info = self._wait_for_fill(order_id, max_wait_seconds)
        if not fill_info:
            log.warning(f"{direction.upper()} order {order_id} for {symbol} did not fill within {max_wait_seconds}s")
            # Cancel the unfilled order so it doesn't execute later unexpectedly
            self.cancel_order(order_id)
            result["status"] = "not_filled"
            return result

        result["fill_price"] = fill_info["filled_avg_price"]
        result["fill_qty"] = fill_info["filled_qty"]
        result["status"] = "filled"

        # ─── Step 3: Exit strategy ───────────────────────────────────
        if fill_info["filled_qty"] > 0:
            stop_side = "sell" if direction == "long" else "buy"
            total_qty = fill_info["filled_qty"]

            # ─── Partial profit taking ───────────────────────────
            if config.PARTIAL_PROFIT_ENABLED and total_qty > 0.01:
                take_profit_qty = round(total_qty * config.TAKE_PROFIT_RATIO, 4)
                remainder_qty = round(total_qty - take_profit_qty, 4)

                # Ensure we have valid quantities
                if take_profit_qty < 0.001:
                    take_profit_qty = 0
                    remainder_qty = total_qty
                if remainder_qty < 0.001:
                    take_profit_qty = total_qty
                    remainder_qty = 0

                # Calculate take-profit price
                fill_price = fill_info["filled_avg_price"]
                if direction == "long":
                    tp_price = fill_price * (1 + config.PROFIT_TARGET_PCT / 100)
                else:
                    tp_price = fill_price * (1 - config.PROFIT_TARGET_PCT / 100)

                # Place take-profit limit order on first half
                tp_order_id = None
                if take_profit_qty > 0:
                    tp_order_id = self.place_limit_order(
                        symbol, take_profit_qty, tp_price, side=stop_side
                    )
                    result["take_profit_order_id"] = tp_order_id
                    if tp_order_id:
                        log.info(
                            f"Take-profit set: {stop_side.upper()} {take_profit_qty:.4f} "
                            f"@ ${tp_price:.2f} (+{config.PROFIT_TARGET_PCT}%)"
                        )

                # Place wider trailing stop on remainder
                if remainder_qty > 0:
                    stop_order_id = self.place_trailing_stop(
                        symbol, remainder_qty, config.REMAINDER_TRAIL_PCT,
                        side=stop_side
                    )
                    result["stop_order_id"] = stop_order_id
                    if stop_order_id:
                        log.info(
                            f"Remainder trailing stop: {remainder_qty:.4f} shares "
                            f"@ {config.REMAINDER_TRAIL_PCT}% trail"
                        )

                if (tp_order_id or take_profit_qty == 0) and \
                   (result.get("stop_order_id") or remainder_qty == 0):
                    result["status"] = "complete"
                    log.info(
                        f"Trade complete (partial profit): {direction.upper()} {symbol} "
                        f"filled @ ${fill_price:.2f}, qty={total_qty:.4f} "
                        f"(TP: {take_profit_qty:.4f} @ ${tp_price:.2f}, "
                        f"Trail: {remainder_qty:.4f} @ {config.REMAINDER_TRAIL_PCT}%)"
                    )
            else:
                # ─── Standard single trailing stop ───────────────
                stop_order_id = self.place_trailing_stop(
                    symbol, total_qty, trailing_stop_pct, side=stop_side
                )
                result["stop_order_id"] = stop_order_id
                if stop_order_id:
                    result["status"] = "complete"
                    log.info(
                        f"Trade complete: {direction.upper()} {symbol} filled @ "
                        f"${fill_info['filled_avg_price']:.2f} "
                        f"qty={total_qty:.4f}, trailing stop @ {trailing_stop_pct}%"
                    )

        return result

    # ─── Order Monitoring ────────────────────────────────────────────

    @retry_on_rate_limit
    def _wait_for_fill(self, order_id: str, max_wait: int = 30) -> Optional[dict]:
        """
        Polls order status until filled or timeout.
        Returns {filled_avg_price, filled_qty} or None.
        """
        start = time.time()
        while time.time() - start < max_wait:
            alpaca_limiter.acquire()
            try:
                order = self.trading_client.get_order_by_id(order_id)
                if order.status.value in ("filled", "partially_filled"):
                    return {
                        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                        "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    }
                if order.status.value in ("canceled", "expired", "rejected"):
                    log.warning(f"Order {order_id} status: {order.status.value}")
                    return None
            except Exception as e:
                log.debug(f"Error checking order {order_id}: {e}")

            time.sleep(1)

        return None

    # ─── Position Management ─────────────────────────────────────────

    @retry_on_rate_limit
    def get_positions(self) -> list:
        """Returns all current open positions."""
        alpaca_limiter.acquire()
        return self.trading_client.get_all_positions()

    @retry_on_rate_limit
    def get_position(self, symbol: str) -> Optional[object]:
        """Returns position for a specific symbol."""
        alpaca_limiter.acquire()
        try:
            return self.trading_client.get_open_position(symbol.replace("/", ""))
        except Exception:
            return None

    @retry_on_rate_limit
    def close_position(self, symbol: str) -> bool:
        """Closes a specific position."""
        alpaca_limiter.acquire()
        try:
            self.trading_client.close_position(symbol.replace("/", ""))
            log.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            log.error(f"Failed to close position {symbol}: {e}")
            return False

    @retry_on_rate_limit
    def close_all_positions(self) -> bool:
        """Emergency: closes all positions and cancels all orders."""
        alpaca_limiter.acquire()
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            log.warning("ALL POSITIONS CLOSED (emergency)")
            return True
        except Exception as e:
            log.error(f"Failed to close all positions: {e}")
            return False

    # ─── Open Orders ─────────────────────────────────────────────────

    @retry_on_rate_limit
    def get_open_orders(self) -> list:
        """Returns all open (pending) orders."""
        alpaca_limiter.acquire()
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return self.trading_client.get_orders(request)
        except Exception as e:
            log.error(f"Failed to fetch open orders: {e}")
            return []
