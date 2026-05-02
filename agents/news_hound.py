"""
The News Hound (Agent 1) - Multi-Source Sentiment Intelligence

Analyzes news from multiple sources for filtered assets using DeepSeek AI.
Returns structured sentiment analysis with confidence scoring.

Features:
- Multi-source news aggregation (Yahoo Finance + NewsAPI)
- Structured DeepSeek prompts with chain-of-thought
- Event classification (earnings, FDA, mergers, etc.)
- Source credibility weighting
- Rate-limited API calls
"""
import os
import re
import json
from typing import Optional

from openai import OpenAI

import config
from core.data_fetcher import DataFetcher
from utils.logger import get_logger
from utils.rate_limiter import deepseek_limiter, retry_on_rate_limit

log = get_logger("sentinel.news_hound")


class NewsHound:
    """
    AI-powered sentiment analyzer using DeepSeek.
    Ingests news headlines and returns a structured sentiment assessment.
    """

    def __init__(self, fetcher: DataFetcher = None):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.fetcher = fetcher or DataFetcher()
        self.model = config.DEEPSEEK_MODEL

    def analyze_sentiment(self, symbol: str, asset_type: str = "stock") -> dict:
        """
        Full sentiment analysis pipeline:
        1. Fetch news from multiple sources
        2. Send structured prompt to DeepSeek
        3. Parse and validate the response

        Returns:
        {
            "score": int (-10 to +10),
            "confidence": float (0-1),
            "events": list[str],
            "summary": str,
            "headline_count": int,
        }
        """
        default_result = {
            "score": 0, "confidence": 0.3,
            "events": [], "summary": "No data available",
            "headline_count": 0,
        }

        # 1. Fetch news
        headlines = self.fetcher.get_news_headlines(
            symbol, max_results=config.NEWS_HEADLINE_COUNT
        )

        if not headlines:
            log.info(f"No news found for {symbol}")
            return default_result

        # 2. Format news for the prompt
        news_block = self._format_headlines(headlines)

        # 3. Call DeepSeek
        result = self._call_deepseek(symbol, asset_type, news_block, len(headlines))
        result["headline_count"] = len(headlines)

        log.info(
            f"Sentiment for {symbol}: score={result['score']}, "
            f"confidence={result['confidence']:.2f}, events={result['events']}"
        )
        return result

    @retry_on_rate_limit
    def _call_deepseek(self, symbol: str, asset_type: str,
                       news_block: str, headline_count: int) -> dict:
        """Sends structured prompt to DeepSeek and parses the response."""
        deepseek_limiter.acquire()

        system_prompt = """You are 'The News Hound', an elite financial sentiment analyst.
You analyze news to determine market sentiment for trading decisions.
You MUST respond in valid JSON format only. No markdown, no explanation outside JSON."""

        user_prompt = f"""Analyze the following {headline_count} recent news headlines for {symbol} ({asset_type}).

NEWS HEADLINES:
{news_block}

Respond with ONLY this JSON structure:
{{
    "score": <integer from -10 to +10>,
    "confidence": <float from 0.0 to 1.0>,
    "events": [<list of detected event types like "earnings_beat", "fda_approval", "merger", "lawsuit", "analyst_upgrade", "insider_selling", "whale_movement", "regulatory_risk">],
    "summary": "<one sentence summary of overall sentiment>"
}}

Scoring guide:
- -10: Catastrophic (fraud, bankruptcy, delisting)
- -7 to -5: Very bearish (major lawsuit, earnings miss, downgrade)
- -4 to -2: Bearish (negative outlook, sector weakness)
- -1 to +1: Neutral (no significant signal)
- +2 to +4: Bullish (positive outlook, sector strength)
- +5 to +7: Very bullish (earnings beat, upgrade, FDA approval)
- +8 to +10: Euphoric (breakthrough, major acquisition target)

Weight signals by source credibility:
- Official press releases, SEC filings: HIGH weight
- Major financial news (Reuters, Bloomberg, WSJ): HIGH weight
- Analyst reports: MEDIUM weight
- Social media, blogs: LOW weight"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)

        except Exception as e:
            log.error(f"DeepSeek sentiment call failed for {symbol}: {e}")
            return {
                "score": 0, "confidence": 0.2,
                "events": [], "summary": f"AI analysis failed: {str(e)[:50]}",
            }

    def _parse_response(self, raw: str) -> dict:
        """Robustly parses DeepSeek's JSON response."""
        # Try to extract JSON from the response (in case it has markdown wrapping)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)

        try:
            data = json.loads(raw)
            return {
                "score": max(-10, min(10, int(data.get("score", 0)))),
                "confidence": max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                "events": list(data.get("events", [])),
                "summary": str(data.get("summary", ""))[:200],
            }
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Failed to parse sentiment response: {e}")
            # Fallback: try to extract just a number
            numbers = re.findall(r'-?\d+', raw)
            score = int(numbers[0]) if numbers else 0
            return {
                "score": max(-10, min(10, score)),
                "confidence": 0.3,
                "events": [],
                "summary": raw[:100],
            }

    def _format_headlines(self, headlines: list[dict]) -> str:
        """Formats headlines into a clean block for the prompt."""
        lines = []
        for i, h in enumerate(headlines, 1):
            source = h.get("source", "Unknown")
            title = h.get("title", "No title")
            desc = h.get("description", "")
            line = f"{i}. [{source}] {title}"
            if desc and desc != title:
                line += f"\n   > {desc[:150]}"
            lines.append(line)
        return "\n".join(lines)
