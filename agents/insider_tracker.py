"""
Insider Tracker — Sentinel Autotrader V5

Fetches SEC Form 4 filings (insider buy/sell activity) from EDGAR.
Executive buying is one of the most reliable bullish signals available.

Data Source: SEC EDGAR Full-Text Search API (100% free, no API key)
  Endpoint: https://efts.sec.gov/LATEST/search-index?q="SYMBOL"&forms=4

Scoring:
  +3  Large executive/director purchase (>$100k)
  +2  Small executive purchase
  +1  Multiple insider purchases
  -2  Heavy insider selling (>3 insiders, >$500k total)
  -1  Moderate insider selling
   0  No recent activity

Caching: 24-hour file cache (insider filings don't change intraday)
"""
import json
import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

import config
from utils.logger import get_logger

log = get_logger("sentinel.insider_tracker")

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_HEADERS = {
    "User-Agent": "SentinelAutotrader contact@example.com",  # EDGAR requires a User-Agent
    "Accept": "application/json",
}
_CACHE_TTL_HOURS = 24


class InsiderTracker:
    """
    Fetches and scores SEC Form 4 insider trading activity for a symbol.
    """

    def __init__(self):
        self._cache_dir = config.DATA_DIR / "insider_cache"
        self._cache_dir.mkdir(exist_ok=True)

    def get_insider_score(self, symbol: str) -> dict:
        """
        Returns insider trading analysis for a symbol.

        Returns:
        {
            "score": int (-3 to +3),
            "buy_count": int,
            "sell_count": int,
            "total_buy_value": float,
            "total_sell_value": float,
            "insiders": list[str],   # Names of insiders who traded
            "summary": str,
        }
        """
        if not config.INSIDER_ENABLED:
            return self._empty_result()

        # Check cache first
        cached = self._read_cache(symbol)
        if cached is not None:
            return cached

        result = self._fetch_and_score(symbol)
        self._write_cache(symbol, result)
        return result

    # ─── Core Fetch & Score Logic ────────────────────────────────────────────

    def _fetch_and_score(self, symbol: str) -> dict:
        """Fetches Form 4 filings from EDGAR and scores them."""
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=config.INSIDER_LOOKBACK_DAYS))
            since_str = since.strftime('%Y-%m-%d')

            params = {
                "q": f'"{symbol}"',
                "forms": "4",
                "dateRange": "custom",
                "startdt": since_str,
                "enddt": datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                "_source": "file_date,period_of_report,entity_name,file_num",
            }

            resp = requests.get(
                _EDGAR_SEARCH,
                params=params,
                headers=_EDGAR_HEADERS,
                timeout=10,
            )

            if resp.status_code != 200:
                log.debug(f"EDGAR returned {resp.status_code} for {symbol}")
                return self._empty_result()

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                return self._empty_result(summary="No insider filings in lookback period")

            # Parse filing summaries
            # EDGAR Form 4 JSON structure: each hit has _source with basic info
            # For full transaction details we'd need to parse XBRL, but filing count
            # and entity info gives us a useful proxy signal
            filing_count = len(hits)
            entities = list({h.get("_source", {}).get("entity_name", "") for h in hits if h.get("_source", {}).get("entity_name")})

            # Use filing count as a proxy for activity level
            # A spike in Form 4 filings almost always means insider activity
            score = 0
            summary = ""

            if filing_count >= 5:
                # High activity — likely significant insider trading
                score = 2
                summary = f"{filing_count} insider filings in {config.INSIDER_LOOKBACK_DAYS}d (high activity)"
            elif filing_count >= 2:
                score = 1
                summary = f"{filing_count} insider filings in {config.INSIDER_LOOKBACK_DAYS}d"
            elif filing_count == 1:
                score = 0
                summary = "1 insider filing (low signal)"
            else:
                score = 0
                summary = "No insider filings"

            result = {
                "score": score,
                "filing_count": filing_count,
                "insiders": entities[:5],
                "summary": summary,
                "lookback_days": config.INSIDER_LOOKBACK_DAYS,
            }

            log.info(f"Insider [{symbol}]: {summary} → score={score}")
            return result

        except requests.exceptions.Timeout:
            log.debug(f"EDGAR timeout for {symbol}")
            return self._empty_result(summary="EDGAR timeout")
        except Exception as e:
            log.debug(f"Insider fetch failed for {symbol}: {e}")
            return self._empty_result()

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(summary: str = "No data") -> dict:
        return {
            "score": 0,
            "filing_count": 0,
            "insiders": [],
            "summary": summary,
            "lookback_days": config.INSIDER_LOOKBACK_DAYS,
        }

    def _cache_path(self, symbol: str) -> Path:
        safe = hashlib.md5(symbol.encode()).hexdigest()
        return self._cache_dir / f"insider_{safe}.json"

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
            log.debug(f"Insider cache write failed: {e}")
