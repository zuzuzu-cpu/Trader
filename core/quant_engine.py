import pandas as pd
import pandas_ta as ta
from .data_fetcher import DataFetcher

class QuantEngine:
    def __init__(self):
        self.fetcher = DataFetcher()

    def evaluate_stock(self, symbol: str, start_date: str, end_date: str):
        """
        Evaluate a stock using a mix of fundamental and technical data.
        Returns a score from 0 to 100.
        """
        # 1. Fetch Technicals
        df = self.fetcher.get_stock_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 50:
            return {"symbol": symbol, "score": 0, "reason": "Not enough data"}

        # Calculate RSI
        df.ta.rsi(length=14, append=True)
        # Calculate MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        latest = df.iloc[-1]
        rsi = latest.get('RSI_14', 50)
        
        # 2. Fetch Fundamentals
        fundamentals = self.fetcher.get_yfinance_fundamentals(symbol)
        if not fundamentals:
            return {"symbol": symbol, "score": 0, "reason": "No fundamental data"}
            
        score = 50 # Base score
        
        # Simple Magic Formula proxy scoring
        if fundamentals['earningsYield'] > 0.05:
            score += 10
        if fundamentals['returnOnCapitalEmployed'] > 0.15:
            score += 10
            
        # Technical score
        if rsi < 40: # Oversold, potential buy
            score += 15
        elif rsi > 70: # Overbought
            score -= 15
            
        return {"symbol": symbol, "score": min(score, 100), "reason": f"RSI: {rsi:.2f}, EY: {fundamentals['earningsYield']:.2f}"}

    def evaluate_crypto(self, symbol: str, start_date: str, end_date: str):
        """
        Evaluate crypto using Bollinger Bands Squeeze and OBV.
        """
        df = self.fetcher.get_crypto_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 20:
            return {"symbol": symbol, "score": 0, "reason": "Not enough data"}

        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        # OBV
        df.ta.obv(append=True)
        
        latest = df.iloc[-1]
        
        # Checking for squeeze: if Bandwidth is narrow
        # Columns usually generated: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        bbb_col = [c for c in df.columns if 'BBB' in c]
        if not bbb_col:
            return {"symbol": symbol, "score": 0, "reason": "BB calculation failed"}
            
        bandwidth = latest[bbb_col[0]]
        
        score = 50
        if bandwidth < 5: # Arbitrary threshold for "squeeze" depending on asset
            score += 20
            
        # OBV trend (compare current OBV to 5-day SMA of OBV)
        obv_col = [c for c in df.columns if 'OBV' in c][0]
        df['OBV_SMA5'] = ta.sma(df[obv_col], length=5)
        
        if latest[obv_col] > df['OBV_SMA5'].iloc[-1]:
            score += 20
            
        return {"symbol": symbol, "score": min(score, 100), "reason": f"Bandwidth: {bandwidth:.2f}"}

    def evaluate_etf(self, symbol: str, start_date: str, end_date: str):
        """
        Evaluate ETF based on high volume and momentum.
        """
        df = self.fetcher.get_stock_bars(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 50:
            return {"symbol": symbol, "score": 0, "reason": "Not enough data"}

        df.ta.rsi(length=14, append=True)
        df.ta.sma(length=50, append=True)
        
        latest = df.iloc[-1]
        rsi = latest.get('RSI_14', 50)
        sma50_col = [c for c in df.columns if 'SMA_50' in c][0]
        
        score = 50
        
        # Must have good volume
        vol_sma20 = ta.sma(df['volume'], length=20).iloc[-1]
        if vol_sma20 < 1000000: # Less than 1M volume
            return {"symbol": symbol, "score": 0, "reason": "Low volume"}
            
        if latest['close'] > latest[sma50_col]:
            score += 20 # Uptrend
        if 40 < rsi < 60:
            score += 10 # Healthy momentum
            
        return {"symbol": symbol, "score": min(score, 100), "reason": f"Uptrend, RSI: {rsi:.2f}"}
