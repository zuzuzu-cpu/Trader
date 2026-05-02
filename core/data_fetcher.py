"""
Data Fetcher - The data backbone of Sentinel Autotrader.

Handles all external data ingestion with:
- Rate-limited Alpaca Data V2 calls (bars, quotes, snapshots)
- Yahoo Finance fundamentals with caching
- NewsAPI headline fetching
- Batch request support for efficiency
- Automatic retry on transient failures
"""
import os
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, CryptoBarsRequest,
    StockLatestQuoteRequest, StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame

import config
from utils.logger import get_logger
from utils.rate_limiter import alpaca_limiter, newsapi_limiter, retry_on_rate_limit

log = get_logger("sentinel.data_fetcher")


class DataFetcher:
    """
    Centralized data fetcher with caching, rate limiting, and batch support.
    """

    def __init__(self):
        self.stock_client = StockHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        self.crypto_client = CryptoHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY
        )
        self._cache_dir = config.DATA_DIR / "cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._fundamentals_cache = {}  # In-memory cache for fundamentals

    # ─── Alpaca Stock Bars ───────────────────────────────────────────────

    @retry_on_rate_limit
    def get_stock_bars(self, symbol: str, start_date: str, end_date: str,
                       timeframe: TimeFrame = TimeFrame.Day) -> Optional[pd.DataFrame]:
        """Fetches historical bars for a stock with rate limiting and caching."""
        cache_key = f"bars_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached = self._read_cache(cache_key, max_age_hours=4)
        if cached is not None:
            return cached

        alpaca_limiter.acquire()
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
            bars = self.stock_client.get_stock_bars(request_params)
            df = bars.df
            if not df.empty:
                df = df.reset_index()
                # Normalize column names to lowercase
                df.columns = [c.lower() for c in df.columns]
                self._write_cache(cache_key, df)
            return df
        except Exception as e:
            log.warning(f"Failed to fetch stock bars for {symbol}: {e}")
            return None

    # ─── Alpaca Crypto Bars ──────────────────────────────────────────────

    @retry_on_rate_limit
    def get_crypto_bars(self, symbol: str, start_date: str, end_date: str,
                        timeframe: TimeFrame = TimeFrame.Day) -> Optional[pd.DataFrame]:
        """Fetches historical bars for a crypto pair."""
        cache_key = f"crypto_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached = self._read_cache(cache_key, max_age_hours=1)
        if cached is not None:
            return cached

        alpaca_limiter.acquire()
        try:
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
            bars = self.crypto_client.get_crypto_bars(request_params)
            df = bars.df
            if not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                self._write_cache(cache_key, df)
            return df
        except Exception as e:
            log.warning(f"Failed to fetch crypto bars for {symbol}: {e}")
            return None

    # ─── Alpaca Latest Quotes (batch) ────────────────────────────────────

    @retry_on_rate_limit
    def get_latest_quotes(self, symbols: list[str]) -> dict:
        """
        Fetches latest quotes for multiple symbols in a single batch request.
        Returns dict of {symbol: {bid, ask, spread, spread_pct}}
        """
        alpaca_limiter.acquire()
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.stock_client.get_stock_latest_quote(req)
            result = {}
            for sym, quote in quotes.items():
                bid = float(quote.bid_price) if quote.bid_price else 0
                ask = float(quote.ask_price) if quote.ask_price else 0
                spread = ask - bid
                spread_pct = (spread / ask * 100) if ask > 0 else 0
                result[sym] = {
                    "bid": bid, "ask": ask,
                    "spread": spread, "spread_pct": spread_pct,
                }
            return result
        except Exception as e:
            log.warning(f"Failed to fetch quotes for {symbols}: {e}")
            return {}

    # ─── Yahoo Finance Fundamentals ──────────────────────────────────────

    def get_fundamentals(self, symbol: str) -> Optional[dict]:
        """
        Fetches comprehensive fundamental data from Yahoo Finance.
        Implements in-memory caching (fundamentals don't change intraday).
        """
        if symbol in self._fundamentals_cache:
            return self._fundamentals_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            price = info.get('currentPrice') or info.get('regularMarketPrice') or 1
            eps = info.get('trailingEps', 0) or 0
            book_value = info.get('bookValue', 0) or 0

            fundamentals = {
                # Valuation
                'price': price,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'pb_ratio': price / book_value if book_value > 0 else 0,
                'earnings_yield': eps / price if price > 0 else 0,

                # Profitability
                'roe': info.get('returnOnEquity', 0) or 0,
                'roa': info.get('returnOnAssets', 0) or 0,
                'gross_margin': info.get('grossMargins', 0) or 0,
                'operating_margin': info.get('operatingMargins', 0) or 0,
                'profit_margin': info.get('profitMargins', 0) or 0,

                # Financial health
                'current_ratio': info.get('currentRatio', 0) or 0,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,

                # Growth
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'earnings_growth': info.get('earningsGrowth', 0) or 0,

                # Dividends
                'dividend_yield': info.get('dividendYield', 0) or 0,

                # Market data
                'market_cap': info.get('marketCap', 0) or 0,
                'avg_volume': info.get('averageVolume', 0) or 0,
                'beta': info.get('beta', 1) or 1,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),

                # Shares
                'shares_outstanding': info.get('sharesOutstanding', 0) or 0,
                'float_shares': info.get('floatShares', 0) or 0,
            }

            # ─── Piotroski F-Score (9 criteria) ─────────────────────────
            f_score = 0
            # 1. Positive net income
            if eps > 0:
                f_score += 1
            # 2. Positive ROA
            if fundamentals['roa'] > 0:
                f_score += 1
            # 3. Positive operating cash flow (proxy: operating margin > 0)
            if fundamentals['operating_margin'] > 0:
                f_score += 1
            # 4. Cash flow > net income (proxy: operating margin > profit margin)
            if fundamentals['operating_margin'] > fundamentals['profit_margin']:
                f_score += 1
            # 5. Lower debt ratio (D/E < 50)
            if 0 < fundamentals['debt_to_equity'] < 50:
                f_score += 1
            # 6. Higher current ratio (> 1)
            if fundamentals['current_ratio'] > 1:
                f_score += 1
            # 7. No dilution (placeholder — would need historical share count)
            if fundamentals['shares_outstanding'] > 0:
                f_score += 1
            # 8. Higher gross margin (> 20%)
            if fundamentals['gross_margin'] > 0.20:
                f_score += 1
            # 9. Positive revenue growth
            if fundamentals['revenue_growth'] > 0:
                f_score += 1

            fundamentals['piotroski_f_score'] = f_score

            # ─── Magic Formula Score ────────────────────────────────────
            # Earnings Yield + Return on Capital (higher = better)
            magic_formula = fundamentals['earnings_yield'] + fundamentals['roe']
            fundamentals['magic_formula'] = magic_formula

            self._fundamentals_cache[symbol] = fundamentals
            return fundamentals

        except Exception as e:
            log.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return None

    # ─── News Fetching ───────────────────────────────────────────────────

    def get_news_headlines(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Fetches news headlines from multiple sources.
        Returns list of {title, source, published_at, description}
        """
        headlines = []

        # Source 1: Yahoo Finance (free, no API key needed)
        try:
            ticker = yf.Ticker(query.replace("/", "-"))
            news = ticker.news or []
            for article in news[:max_results]:
                headlines.append({
                    "title": article.get("title", ""),
                    "source": article.get("publisher", "Yahoo Finance"),
                    "published_at": article.get("providerPublishTime", ""),
                    "description": article.get("title", ""),  # YF doesn't always have description
                })
        except Exception as e:
            log.debug(f"Yahoo Finance news fetch failed for {query}: {e}")

        # Source 2: NewsAPI (if key is available)
        if config.NEWSAPI_KEY and len(headlines) < max_results:
            try:
                newsapi_limiter.acquire()
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "sortBy": "publishedAt",
                        "pageSize": max_results,
                        "language": "en",
                        "apiKey": config.NEWSAPI_KEY,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
                    for article in articles:
                        headlines.append({
                            "title": article.get("title", ""),
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                            "published_at": article.get("publishedAt", ""),
                            "description": article.get("description", ""),
                        })
            except Exception as e:
                log.debug(f"NewsAPI fetch failed for {query}: {e}")

        return headlines[:max_results]

    # ─── Cache Helpers ───────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{safe_key}.parquet"

    def _read_cache(self, key: str, max_age_hours: int = 4) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > max_age_hours * 3600:
            path.unlink(missing_ok=True)
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, df: pd.DataFrame):
        try:
            path = self._cache_path(key)
            df.to_parquet(path, index=False)
        except Exception as e:
            log.debug(f"Cache write failed: {e}")

    def clear_cache(self):
        """Clears all cached data."""
        for f in self._cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)
        self._fundamentals_cache.clear()
        log.info("Cache cleared")
