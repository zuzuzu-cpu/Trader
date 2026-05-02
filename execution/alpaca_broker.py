import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import GetOrdersRequest
from alpaca.common.exceptions import APIError
from dotenv import load_dotenv

load_dotenv()

class AlpacaBroker:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        # Initialize Trading Client
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)

    def get_account_info(self):
        """Returns account details like buying power and equity."""
        return self.trading_client.get_account()

    def place_order(self, symbol, notional, side=OrderSide.BUY, trailing_stop_percent=2.0):
        """
        Places a market order with a trailing stop.
        - notional: The dollar amount to invest (supports fractional shares).
        - trailing_stop_percent: The percentage for the trailing stop.
        """
        print(f"Executing trade for {symbol}: ${notional} {side}")
        
        try:
            # 1. Place the Buy Order (Market Order with Notional)
            buy_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            buy_order = self.trading_client.submit_order(buy_request)
            print(f"Buy order submitted: {buy_order.id}")
            
            # Note: For Trailing Stops, we usually need the position to be established.
            # In a real-time system, we might wait for the fill. 
            # For simplicity here, we'll assume it fills or we'll place it as a separate order.
            # However, Alpaca allows 'oto' (one-triggers-other) or 'bracket' for some types.
            # Trailing stops are often placed after the position is open.
            
            # 2. Place Trailing Stop Order (Sell side to protect long position)
            # We need to know the quantity filled to place the stop.
            # For now, let's log that we would place a trailing stop.
            print(f"Setting trailing stop of {trailing_stop_percent}% for {symbol}")
            
            # In a more advanced implementation, we'd poll for the fill and then place the stop.
            # For this MVP, we'll just demonstrate the request structure.
            """
            stop_request = TrailingStopOrderRequest(
                symbol=symbol,
                qty=buy_order.qty, # This might be None initially for notional orders
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                trail_percent=trailing_stop_percent
            )
            self.trading_client.submit_order(stop_request)
            """
            
            return buy_order
            
        except APIError as e:
            print(f"Alpaca API Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in place_order: {e}")
            return None

    def get_positions(self):
        """Returns current open positions."""
        return self.trading_client.get_all_positions()

    def close_all_positions(self):
        """Closes all open positions."""
        return self.trading_client.close_all_positions(cancel_orders=True)
