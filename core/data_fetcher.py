"""
Data Fetcher - The data backbone of Sentinel Autotrader.

Handles all external data ingestion with:
- Rate-limited Alpaca Data V2 calls (bars, quotes, snapshots)
- Multi-timeframe bars (1H, 1D, 1W) for signal confirmation
- Yahoo Finance fundamentals with caching
- Earnings calendar awareness (avoid pre-earnings trades)
- Benchmark data for relative strength calculation
- NewsAPI headline fetching
- Batch request support for efficiency
- Automatic retry on transient failures
"""
import os
import time
import json
import sqlite3
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
from utils.rate_limiter import (
    alpaca_limiter, newsapi_limiter, retry_on_rate_limit,
    finnhub_limiter, fmp_limiter, sec_limiter
)

log = get_logger("sentinel.data_fetcher")


class DataFetcher:
    """
    Centralized data fetcher with caching, rate limiting, and batch support.
    """

    def __init__(self):
        self.stock_client = StockHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            url_override=config.ALPACA_DATA_URL
        )
        self.crypto_client = CryptoHistoricalDataClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
            url_override=config.ALPACA_DATA_URL
        )

        self._cache_dir = config.DATA_DIR / "cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._fundamentals_cache = {}  # In-memory cache for fundamentals

        # V5: SQLite Deep Fundamentals Cache
        self._db_path = config.DATA_DIR / "fundamentals.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deep_fundamentals (
                    symbol TEXT PRIMARY KEY,
                    data TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

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
            formatted_symbol = f"{symbol[:-3]}/{symbol[-3:]}" if "USD" in symbol and "/" not in symbol else symbol
            request_params = CryptoBarsRequest(
                symbol_or_symbols=formatted_symbol,
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
        if symbol in config.ETF_SYMBOLS or symbol in config.CRYPTO_SYMBOLS:
            return {}  # Skip fundamentals for ETFs and Crypto to avoid 404s

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
            err_msg = str(e)
            if "No fundamentals data found" in err_msg or "Quote not found" in err_msg:
                # Silently skip ETFs and obscure tickers that don't have company fundamentals
                return {}
            log.warning(f"Failed to fetch fundamentals for {symbol}: {err_msg}")
            return None

    # ─── V5: Deep Fundamentals (FMP / Finnhub / SEC) ─────────────────────

    def get_deep_fundamentals(self, symbol: str) -> dict:
        """
        Fetches deep fundamentals for high-conviction candidates.
        Tries SQLite Cache -> Finnhub -> FMP -> SEC EDGAR.
        """
        # 1. Check SQLite Cache
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data, updated_at FROM deep_fundamentals WHERE symbol=?",
                    (symbol,)
                )
                row = cursor.fetchone()
                if row:
                    updated_at = datetime.fromisoformat(row[1])
                    if datetime.now() - updated_at < timedelta(days=config.FUNDAMENTALS_CACHE_DAYS):
                        return json.loads(row[0])
        except Exception as e:
            log.error(f"SQLite cache error for {symbol}: {e}")

        deep_data = {}

        # 2. Try Finnhub (Fast, generous limit)
        if config.FINNHUB_API_KEY:
            try:
                if finnhub_limiter.acquire(timeout=2.0):
                    res = requests.get(
                        f"https://finnhub.io/api/v1/stock/metric",
                        params={"symbol": symbol, "metric": "all", "token": config.FINNHUB_API_KEY},
                        timeout=5
                    )
                    if res.status_code == 200:
                        metrics = res.json().get("metric", {})
                        if metrics:
                            deep_data["finnhub_pe"] = metrics.get("peBasicExclExtraTTM")
                            deep_data["finnhub_roe"] = metrics.get("roeTTM")
                            deep_data["finnhub_debt_equity"] = metrics.get("totalDebtToEquityAnnual")
                else:
                    log.warning(f"Finnhub rate limit exhausted, skipping {symbol}")
            except Exception as e:
                log.warning(f"Finnhub fetch failed for {symbol}: {e}")

        # 3. Try FMP (Deep learning base, strict 250/day limit)
        if config.FMP_API_KEY and not deep_data:
            try:
                if fmp_limiter.acquire(timeout=2.0):
                    res = requests.get(
                        f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}",
                        params={"apikey": config.FMP_API_KEY},
                        timeout=5
                    )
                    if res.status_code == 200 and res.json():
                        metrics = res.json()[0]
                        deep_data["fmp_pe"] = metrics.get("peRatioTTM")
                        deep_data["fmp_roe"] = metrics.get("roeTTM")
                        deep_data["fmp_debt_equity"] = metrics.get("debtToEquityTTM")
                        deep_data["fmp_pb"] = metrics.get("pbRatioTTM")
                else:
                    log.warning(f"FMP rate limit exhausted, skipping {symbol}")
            except Exception as e:
                log.warning(f"FMP fetch failed for {symbol}: {e}")

        # 4. Save to Cache
        if deep_data:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO deep_fundamentals (symbol, data, updated_at) VALUES (?, ?, ?)",
                        (symbol, json.dumps(deep_data), datetime.now().isoformat())
                    )
            except Exception as e:
                log.error(f"SQLite save error for {symbol}: {e}")

        return deep_data

    # ─── News Fetching ───────────────────────────────────────────────────

    def get_news_headlines(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Fetches news headlines from multiple sources.
        Priority: Yahoo Finance (primary, unlimited) -> NewsAPI (backup)
        """
        from utils.rate_limiter import yahoo_limiter, newsapi_limiter
        
        headlines = []
        # Clean query (e.g. BTC/USD -> BTC)
        clean_query = query.split("/")[0].replace("-", " ")

        # Source 1: Yahoo Finance (PRIMARY - unlimited, free)
        try:
            yahoo_limiter.acquire()
            ticker = yf.Ticker(query.replace("/", "-"))
            news = ticker.news or []
            for article in news[:max_results]:
                headlines.append({
                    "title": article.get("title", ""),
                    "source": article.get("publisher", "Yahoo Finance"),
                    "published_at": article.get("providerPublishTime", ""),
                    "description": article.get("title", ""),
                })
        except Exception as e:
            log.debug(f"Yahoo Finance news fetch failed for {query}: {e}")

        # Source 2: NewsAPI (BACKUP - only 100/day, reserve for gaps)
        if len(headlines) < max_results and config.NEWSAPI_KEY:
            try:
                newsapi_limiter.acquire()
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": clean_query,
                        "sortBy": "publishedAt",
                        "pageSize": max_results - len(headlines),
                        "language": "en",
                        "apiKey": config.NEWSAPI_KEY,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
                    for article in articles:
                        # Prevent duplicates
                        if any(h["title"] == article.get("title", "") for h in headlines):
                            continue
                        headlines.append({
                            "title": article.get("title", ""),
                            "source": article.get("source", {}).get("name", "NewsAPI"),
                            "published_at": article.get("publishedAt", ""),
                            "description": article.get("description", ""),
                        })
            except Exception as e:
                log.debug(f"NewsAPI fetch failed for {clean_query}: {e}")

        return headlines[:max_results]

    # ─── Multi-Timeframe Bars ────────────────────────────────────────────

    def get_multi_timeframe_bars(self, symbol: str, start_date: str,
                                  end_date: str, is_crypto: bool = False) -> dict:
        """
        Fetches bars at 3 timeframes: 1H, 1D, 1W.
        Returns {"1h": df, "1d": df, "1w": df} (any may be None).
        """
        fetch_fn = self.get_crypto_bars if is_crypto else self.get_stock_bars

        result = {}
        # Daily (already the default)
        result["1d"] = fetch_fn(symbol, start_date, end_date, TimeFrame.Day)

        # Hourly (last 14 days to ensure at least 21 trading hours for EMA calculations)
        from datetime import datetime, timedelta
        hourly_start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        result["1h"] = fetch_fn(symbol, hourly_start, end_date, TimeFrame.Hour)

        # Weekly
        result["1w"] = fetch_fn(symbol, start_date, end_date, TimeFrame.Week)

        return result

    # ─── Earnings Calendar ───────────────────────────────────────────────

    def get_earnings_date(self, symbol: str) -> dict:
        """
        Checks if a stock has upcoming earnings within the blackout window.
        Returns {"has_upcoming_earnings": bool, "days_until": int, "date": str}
        """
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is None or cal.empty:
                return {"has_upcoming_earnings": False, "days_until": 999, "date": ""}

            # calendar can be a DataFrame with 'Earnings Date' row or columns
            from datetime import datetime
            today = datetime.now().date()

            # Try to extract earnings date
            earnings_date = None
            if isinstance(cal, pd.DataFrame):
                # Some versions return a DataFrame with dates as columns
                for col in cal.columns:
                    try:
                        d = pd.to_datetime(col).date()
                        if d >= today:
                            earnings_date = d
                            break
                    except Exception:
                        pass
                # Or it might have 'Earnings Date' as a row
                if earnings_date is None and 'Earnings Date' in cal.index:
                    val = cal.loc['Earnings Date'].iloc[0]
                    try:
                        earnings_date = pd.to_datetime(val).date()
                    except Exception:
                        pass

            if earnings_date and earnings_date >= today:
                days_until = (earnings_date - today).days
                return {
                    "has_upcoming_earnings": days_until <= config.EARNINGS_BLACKOUT_DAYS,
                    "days_until": days_until,
                    "date": str(earnings_date),
                }

            return {"has_upcoming_earnings": False, "days_until": 999, "date": ""}

        except Exception as e:
            log.debug(f"Earnings calendar fetch failed for {symbol}: {e}")
            return {"has_upcoming_earnings": False, "days_until": 999, "date": ""}

    # ─── Benchmark Data (for Relative Strength) ─────────────────────────

    def get_benchmark_bars(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches daily bars for the benchmark (SPY by default).
        Used to calculate relative strength ratings.
        """
        return self.get_stock_bars(config.BENCHMARK_SYMBOL, start_date, end_date)

    # ─── Batch Prefetching (Performance) ───────────────────────────────
    
    def prefetch_stock_bars(self, symbols: list[str], start_date: str, end_date: str):
        """Fetches bars for hundreds of stocks in optimized batches to respect rate limits."""
        if not symbols: return
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            log.info(f"Prefetching stock bars: batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            alpaca_limiter.acquire()
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                bars = self.stock_client.get_stock_bars(req)
                df_all = bars.df
                
                if not df_all.empty:
                    # Split multi-index DF by symbol and cache individually
                    for symbol in batch:
                        if symbol in df_all.index.get_level_values(0):
                            df_sym = df_all.xs(symbol).reset_index()
                            df_sym.columns = [c.lower() for c in df_sym.columns]
                            cache_key = f"bars_{symbol}_{start_date}_{end_date}_{TimeFrame.Day}"
                            self._write_cache(cache_key, df_sym)
            except Exception as e:
                log.debug(f"Batch prefetch failed for {len(batch)} symbols: {e}")

    def prefetch_crypto_bars(self, symbols: list[str], start_date: str, end_date: str):
        """Fetches bars for all crypto pairs in batches."""
        if not symbols: return
        batch_size = 50 # Crypto requests are heavier
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            formatted_batch = [f"{s[:-3]}/{s[-3:]}" if "USD" in s and "/" not in s else s for s in batch]
            alpaca_limiter.acquire()
            try:
                req = CryptoBarsRequest(
                    symbol_or_symbols=formatted_batch,
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                )
                bars = self.crypto_client.get_crypto_bars(req)
                df_all = bars.df
                if not df_all.empty:
                    for symbol, formatted_sym in zip(batch, formatted_batch):
                        if formatted_sym in df_all.index.get_level_values(0):
                            df_sym = df_all.xs(formatted_sym).reset_index()
                            df_sym.columns = [c.lower() for c in df_sym.columns]
                            cache_key = f"crypto_{symbol}_{start_date}_{end_date}_{TimeFrame.Day}"
                            self._write_cache(cache_key, df_sym)
            except Exception as e:
                log.debug(f"Crypto prefetch failed: {e}")

    def prefetch_fundamentals(self, symbols: list[str]):
        """Fetches fundamentals for a list of symbols using yfinance batching."""
        if not symbols: return
        # Limit to first 1000 for fundamentals to avoid IP blocks (yfinance is sensitive)
        target_symbols = symbols[:1000]
        log.info(f"Prefetching fundamentals for top {len(target_symbols)} candidates...")
        
        try:
            tickers = yf.Tickers(" ".join(target_symbols))
            for sym in target_symbols:
                try:
                    # Accessing .info triggers a fetch if not already done by yf.Tickers
                    info = tickers.tickers[sym].info
                    if info:
                        # Process and cache similarly to get_fundamentals
                        # For brevity, we just populate the in-memory cache
                        # get_fundamentals will then return this immediately
                        self.get_fundamentals(sym) 
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"Fundamentals prefetch failed: {e}")

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
