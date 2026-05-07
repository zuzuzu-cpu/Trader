"""
Macro Agent — Fetches FRED economic data once daily and produces a macro
context summary that every other agent receives in their prompts.

Data Sources:
  - Federal Reserve FRED API (free key required)
  - Series: Fed Funds Rate, CPI, Unemployment, 10Y Yield

Designed to be called at the start of each day's first cycle only.
Results are cached in SQLite and reused across all cycles that day.
"""
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger("sentinel.macro_agent")

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


class MacroAgent:
    """
    Fetches macro economic data from FRED and caches it for the day.
    Passes a summary string to all other agents.
    """

    def __init__(self):
        self._cache = None
        self._cache_ts = 0

    def get_macro_context(self) -> str:
        """
        Returns a macro context string for injection into agent prompts.
        Cached for MACRO_REFRESH_HOURS hours.
        """
        now = time.time()
        if self._cache and (now - self._cache_ts) < config.MACRO_REFRESH_HOURS * 3600:
            return self._cache

        if not config.FRED_API_KEY:
            self._cache = "No FRED API key configured — macro context unavailable"
            self._cache_ts = now
            return self._cache

        data = self._fetch_fred_data()
        context = self._build_context(data)
        self._cache = context
        self._cache_ts = now
        return context

    def _fetch_fred_data(self) -> dict:
        """Fetch all configured FRED series."""
        import requests

        series_ids = [s.strip() for s in config.MACRO_FRED_SERIES.split(",")]
        results = {}

        for series_id in series_ids:
            try:
                resp = requests.get(
                    _FRED_BASE,
                    params={
                        "series_id": series_id,
                        "api_key": config.FRED_API_KEY,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 2,
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    if obs:
                        latest = obs[0]
                        results[series_id] = {
                            "value": latest.get("value"),
                            "date": latest.get("date"),
                        }
            except Exception as e:
                log.debug(f"FRED fetch failed for {series_id}: {e}")

        return results

    def _build_context(self, data: dict) -> str:
        """Build a human-readable macro context string."""
        if not data:
            return "Macro data unavailable"

        parts = []

        # Fed Funds Rate
        fed = data.get("FEDFUNDS", {}).get("value")
        if fed:
            parts.append(f"Fed Funds Rate: {fed}%")

        # CPI (inflation)
        cpi = data.get("CPIAUCSL", {}).get("value")
        if cpi:
            try:
                cpi_float = float(cpi)
                cpi_pct = (cpi_float - 100)  # approximate inflation since base
                parts.append(f"CPI Index: {cpi_float:.1f} (≈{cpi_pct:.1f}% since base)")
            except:
                parts.append(f"CPI: {cpi}")

        # Unemployment
        unemp = data.get("UNRATE", {}).get("value")
        if unemp:
            parts.append(f"Unemployment: {unemp}%")

        # 10Y Yield
        yield10 = data.get("DGS10", {}).get("value")
        if yield10:
            parts.append(f"10Y Yield: {yield10}%")

        # Build summary via rules (not DeepSeek — costs unnecessary tokens)
        summary = " | ".join(parts)

        # Add macro guidance
        guidance = []
        if fed:
            try:
                if float(fed) > 5:
                    guidance.append("RESTRICTIVE: High rates pressure growth stocks")
                elif float(fed) > 3:
                    guidance.append("TIGHTENING: Rates are elevated, favor value/defensives")
            except:
                pass
        if yield10:
            try:
                if float(yield10) > 4.5:
                    guidance.append("Bond yields high: growth stocks face headwinds")
                elif float(yield10) < 2:
                    guidance.append("Low bond yields: risk-on environment")
            except:
                pass

        if guidance:
            summary += " | " + " | ".join(guidance)

        return summary or "No macro data available"


macro_agent = MacroAgent()