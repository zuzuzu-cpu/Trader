"""
Social Sentinel — Reddit/WallStreetBets Sentiment Agent V5

Monitors Reddit for unusual mention velocity around stock symbols.
A spike in WSB/r/stocks mentions before a price move is a leading indicator.

Data Sources (all free, no API key needed):
  - r/wallstreetbets (new + hot)
  - r/stocks
  - r/investing
  - r/options

Uses Reddit's public JSON API (no OAuth required for read-only access).

Scoring:
  +3  Extreme viral (>50 mentions in last 24h, trending)
  +2  High velocity (>20 mentions, multiple subs)
  +1  Moderate (5-20 mentions)
   0  Low/no signal (<5 mentions)
  -1  Bearish language dominates (DD saying "puts", "short", "avoid")

Rate Limiting: Very conservative — 2 req/min max to avoid IP blocks.
Caching: 2-hour cache (Reddit mentions are semi-stable intraday).
"""
import re
import json
import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

import config
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

log = get_logger("sentinel.social_sentinel")

# Very conservative rate limit for Reddit to avoid IP bans
_reddit_limiter = RateLimiter("reddit", max_calls=2, period_seconds=60)

_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]
_REDDIT_BASE = "https://www.reddit.com"
_CACHE_TTL_HOURS = 2

# Keyword scoring weights
_BULLISH_WORDS = {"moon", "rocket", "bull", "calls", "buy", "long", "squeeze", "breakout",
                  "yolo", "beat", "upgrade", "catalyst", "undervalued", "gem"}
_BEARISH_WORDS = {"puts", "short", "bear", "crash", "overvalued", "avoid", "dump",
                  "fraud", "bankrupt", "lawsuit", "downgrade", "sell", "exit"}


class SocialSentinel:
    """
    Monitors Reddit for unusual mention velocity and sentiment.
    """

    def __init__(self):
        self._cache_dir = config.DATA_DIR / "social_cache"
        self._cache_dir.mkdir(exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": config.REDDIT_USER_AGENT,
            "Accept": "application/json",
        })

    def get_social_score(self, symbol: str) -> dict:
        """
        Returns social sentiment analysis for a ticker symbol.

        Returns:
        {
            "score": int (0 to +3 or -1),
            "mention_count": int,
            "bullish_count": int,
            "bearish_count": int,
            "velocity": str ("HIGH" | "MODERATE" | "LOW" | "NONE"),
            "summary": str,
            "top_posts": list[str],
        }
        """
        if not config.SOCIAL_SENTIMENT_ENABLED:
            return self._empty_result()

        cached = self._read_cache(symbol)
        if cached is not None:
            return cached

        result = self._fetch_and_score(symbol)
        self._write_cache(symbol, result)
        return result

    # ─── Core Logic ──────────────────────────────────────────────────────────

    def _fetch_and_score(self, symbol: str) -> dict:
        """Fetches posts from multiple subreddits and scores mention velocity."""
        all_posts = []
        clean_symbol = symbol.replace("/", "").upper()

        for sub in _SUBREDDITS:
            posts = self._search_subreddit(sub, clean_symbol)
            all_posts.extend(posts)
            time.sleep(0.5)  # Small delay between subreddit calls

        if not all_posts:
            return self._empty_result(summary=f"No Reddit mentions of ${clean_symbol}")

        # Filter to last 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_posts = []
        for p in all_posts:
            try:
                created = datetime.fromtimestamp(p.get("created_utc", 0), tz=timezone.utc)
                if created > cutoff:
                    recent_posts.append(p)
            except Exception:
                continue

        mention_count = len(recent_posts)

        if mention_count < config.WSB_MIN_MENTIONS:
            return self._empty_result(
                summary=f"Low Reddit activity: {mention_count} mentions in 24h"
            )

        # Keyword sentiment scan
        bullish_count = 0
        bearish_count = 0
        top_posts = []

        for p in recent_posts[:20]:
            title = (p.get("title", "") + " " + p.get("selftext", "")).lower()
            words = set(re.findall(r'\b\w+\b', title))
            b_hits = words & _BULLISH_WORDS
            s_hits = words & _BEARISH_WORDS
            bullish_count += len(b_hits)
            bearish_count += len(s_hits)
            if p.get("title"):
                top_posts.append(p["title"][:100])

        # Scoring
        if mention_count >= 50:
            velocity = "HIGH"
            base_score = 3
        elif mention_count >= 20:
            velocity = "MODERATE"
            base_score = 2
        elif mention_count >= config.WSB_MIN_MENTIONS:
            velocity = "LOW"
            base_score = 1
        else:
            velocity = "NONE"
            base_score = 0

        # Sentiment adjustment
        if bearish_count > bullish_count * 2:
            base_score = min(base_score, -1)  # Bearish dominates
            summary = f"${clean_symbol}: {mention_count} mentions (BEARISH tone)"
        elif bullish_count > bearish_count:
            summary = f"${clean_symbol}: {mention_count} mentions (BULLISH tone)"
        else:
            summary = f"${clean_symbol}: {mention_count} mentions (NEUTRAL tone)"

        result = {
            "score": base_score,
            "mention_count": mention_count,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "velocity": velocity,
            "summary": summary,
            "top_posts": top_posts[:3],
        }

        log.info(
            f"Social [{clean_symbol}]: {mention_count} mentions, "
            f"velocity={velocity}, score={base_score}"
        )
        return result

    def _search_subreddit(self, subreddit: str, symbol: str) -> list[dict]:
        """Searches a subreddit for recent mentions of a ticker symbol."""
        _reddit_limiter.acquire()
        try:
            url = f"{_REDDIT_BASE}/r/{subreddit}/search.json"
            params = {
                "q": f"${symbol}",      # Use $ prefix for ticker searches
                "sort": "new",
                "limit": 25,
                "t": "day",             # Past 24 hours
                "restrict_sr": "on",    # Only in this subreddit
            }
            resp = self._session.get(url, params=params, timeout=10)

            if resp.status_code == 429:
                log.debug(f"Reddit rate limited on r/{subreddit}. Backing off 30s.")
                time.sleep(30)
                return []

            if resp.status_code != 200:
                return []

            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            return [p.get("data", {}) for p in posts]

        except Exception as e:
            log.debug(f"Reddit search failed for r/{subreddit}/{symbol}: {e}")
            return []

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(summary: str = "No social signal") -> dict:
        return {
            "score": 0,
            "mention_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "velocity": "NONE",
            "summary": summary,
            "top_posts": [],
        }

    def _cache_path(self, symbol: str) -> Path:
        safe = hashlib.md5(symbol.encode()).hexdigest()
        return self._cache_dir / f"social_{safe}.json"

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
            log.debug(f"Social cache write failed: {e}")
