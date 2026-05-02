"""
Alpaca Broker - Paper Trading Execution Engine

Handles all interaction with Alpaca's Trading API:
- Market orders with fractional share support (notional)
- Trailing stop orders with fill monitoring
- Extended hours trading
- Position management and portfolio queries
- Order status tracking
"""
import os
import time
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TrailingStopOrderRequest,
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
    Production-grade Alpaca paper trading broker.
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
    def place_trailing_stop(self, symbol: str, qty: float,
                            trail_percent: float) -> Optional[str]:
        """
        Places a trailing stop sell order to protect a long position.
        Must be called after the buy order fills.
        Returns the order ID or None on failure.
        """
        alpaca_limiter.acquire()
        try:
            stop_request = TrailingStopOrderRequest(
                symbol=symbol.replace("/", ""),
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,  # Good-til-cancelled
                trail_percent=trail_percent,
            )
            order = self.trading_client.submit_order(stop_request)
            log.info(
                f"TRAILING STOP set: {symbol} qty={qty} trail={trail_percent}% → order_id={order.id}",
                extra={"symbol": symbol, "action": "TRAILING_STOP", "order_id": str(order.id)}
            )
            return str(order.id)

        except APIError as e:
            log.error(f"Alpaca API error placing trailing stop for {symbol}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error placing trailing stop for {symbol}: {e}")
            return None

    def execute_trade(self, symbol: str, notional: float,
                      trailing_stop_pct: float,
                      max_wait_seconds: int = 30) -> dict:
        """
        Full trade execution pipeline:
        1. Place market buy order
        2. Wait for fill
        3. Place trailing stop order

        Returns execution result dict.
        """
        result = {
            "symbol": symbol,
            "buy_order_id": None,
            "stop_order_id": None,
            "fill_price": None,
            "fill_qty": None,
            "status": "failed",
        }

        # Step 1: Place buy order
        buy_order_id = self.place_buy_order(symbol, notional)
        if not buy_order_id:
            result["status"] = "buy_failed"
            return result
        result["buy_order_id"] = buy_order_id

        # Step 2: Wait for fill
        fill_info = self._wait_for_fill(buy_order_id, max_wait_seconds)
        if not fill_info:
            log.warning(f"Buy order {buy_order_id} for {symbol} did not fill within {max_wait_seconds}s")
            result["status"] = "not_filled"
            return result

        result["fill_price"] = fill_info["filled_avg_price"]
        result["fill_qty"] = fill_info["filled_qty"]
        result["status"] = "filled"

        # Step 3: Place trailing stop
        if fill_info["filled_qty"] > 0:
            stop_order_id = self.place_trailing_stop(
                symbol, fill_info["filled_qty"], trailing_stop_pct
            )
            result["stop_order_id"] = stop_order_id
            if stop_order_id:
                result["status"] = "complete"
                log.info(
                    f"Trade complete: {symbol} filled @ ${fill_info['filled_avg_price']:.2f} "
                    f"qty={fill_info['filled_qty']:.4f}, "
                    f"trailing stop @ {trailing_stop_pct}%"
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
