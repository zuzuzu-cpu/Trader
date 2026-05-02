import os
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        # Initialize Alpaca Data Clients
        self.stock_client = StockHistoricalDataClient(self.alpaca_api_key, self.alpaca_secret_key)
        self.crypto_client = CryptoHistoricalDataClient(self.alpaca_api_key, self.alpaca_secret_key)

    def get_stock_bars(self, symbol: str, start_date: str, end_date: str):
        """Fetches historical daily bars for a stock."""
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        try:
            bars = self.stock_client.get_stock_bars(request_params)
            df = bars.df
            # Reset index to make 'symbol' and 'timestamp' columns accessible if it's a multi-index
            if not df.empty:
                df = df.reset_index()
            return df
        except Exception as e:
            print(f"Error fetching stock bars for {symbol}: {e}")
            return None

    def get_crypto_bars(self, symbol: str, start_date: str, end_date: str):
        """Fetches historical daily bars for a crypto pair."""
        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        try:
            bars = self.crypto_client.get_crypto_bars(request_params)
            df = bars.df
            if not df.empty:
                df = df.reset_index()
            return df
        except Exception as e:
            print(f"Error fetching crypto bars for {symbol}: {e}")
            return None

    def get_yfinance_fundamentals(self, symbol: str):
        """Fetches basic fundamental data using Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics for Magic Formula / Piotroski
            fundamentals = {
                'returnOnCapitalEmployed': info.get('returnOnEquity', 0), # Proxy if ROCE missing
                'earningsYield': info.get('trailingEps', 0) / info.get('currentPrice', 1) if info.get('currentPrice') else 0,
                'currentRatio': info.get('currentRatio', 0),
                'grossMargins': info.get('grossMargins', 0),
                'volume': info.get('averageVolume', 0)
            }
            return fundamentals
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None
