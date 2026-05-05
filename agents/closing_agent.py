"""
Closing Agent (Agent 6) - Proactive Exit Manager

Monitors all open positions and uses DeepSeek Reasoner to decide if a position 
should be closed early, partially closed, or have its trailing stop tightened 
based on changing market conditions.
"""
import re
import json
from typing import Optional

from openai import OpenAI

import config
from utils.logger import get_logger
from utils.rate_limiter import deepseek_limiter, retry_on_rate_limit

log = get_logger("sentinel.closing_agent")

class ClosingAgent:
    """
    AI Exit Manager that proactively manages open positions.
    Can recommend:
    - SELL_ALL: Close the entire position immediately
    - SELL_PARTIAL: Close a percentage of the position
    - ADJUST_STOP: Tighten the trailing stop
    - HOLD: Do nothing
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self.model = config.DEEPSEEK_REASONER_MODEL

    def evaluate_exit(self, symbol: str, asset_type: str, direction: str,
                      qty: float, current_price: float, avg_entry_price: float,
                      current_pnl_pct: float, hold_time_minutes: int,
                      quant_result: dict, sentiment_result: dict,
                      risk_result: dict) -> dict:
        """
        Evaluates an open position for early exit.
        
        Returns:
        {
            "verdict": "SELL_ALL" | "SELL_PARTIAL" | "ADJUST_STOP" | "HOLD",
            "confidence": float (0-100),
            "sell_pct": float (0.0 to 1.0),
            "new_trail_pct": float,
            "reasoning": str,
            "deepseek_reasoning": str
        }
        """
        # Minimum hurdle: if we are in profit > 1%, or loss > 3%, or holding > 2 days,
        # we consider asking the AI. But here we let the AI decide entirely.

        assessment = self._ai_exit_assessment(
            symbol, asset_type, direction, qty, current_price, avg_entry_price,
            current_pnl_pct, hold_time_minutes, quant_result, sentiment_result, risk_result
        )

        if not assessment:
            return {"verdict": "HOLD", "confidence": 0, "sell_pct": 0, "new_trail_pct": 0, "reasoning": "AI error"}

        log.info(
            f"Closing Agent [{symbol}]: Verdict={assessment['verdict']} "
            f"(Conf: {assessment['confidence']:.1f}%) P&L: {current_pnl_pct:+.2f}% "
            f"Reason: {assessment['reasoning']}"
        )

        return assessment

    @retry_on_rate_limit
    def _ai_exit_assessment(self, symbol: str, asset_type: str, direction: str,
                            qty: float, current_price: float, avg_entry_price: float,
                            current_pnl_pct: float, hold_time_minutes: int,
                            quant_result: dict, sentiment_result: dict,
                            risk_result: dict) -> Optional[dict]:
        
        deepseek_limiter.acquire()

        signals = quant_result.get("signals", {})
        hold_days = hold_time_minutes / 1440

        prompt = f"""You are the 'Closing Agent', an AI Exit Manager for an autonomous trading system.
Your job is to manage the open position below. You must decide whether to HOLD, SELL_ALL, SELL_PARTIAL, or ADJUST_STOP.

CURRENT POSITION:
- Symbol: {symbol} ({asset_type})
- Direction: {direction.upper()}
- Quantity: {qty}
- Entry Price: ${avg_entry_price:.2f}
- Current Price: ${current_price:.2f}
- Current P&L: {current_pnl_pct:+.2f}%
- Hold Time: {hold_days:.1f} days ({hold_time_minutes} minutes)

LATEST MARKET DATA:
- Quant Score: {quant_result.get('score', 0)}/100 (Direction: {quant_result.get('direction', 'long')})
- RSI: {signals.get('rsi', 'N/A')}
- MACD Bullish: {signals.get('macd_bullish', 'N/A')}
- Reason: {quant_result.get('reason', '')}
- Sentiment Score: {sentiment_result.get('score', 0)}/10
- Risk Grade: {risk_result.get('grade', 'N/A')}
- Risk Flags: {risk_result.get('flags', [])}

ACTIONS AVAILABLE:
1. "HOLD": Let the existing trailing stops handle it. (Default choice if thesis is still intact)
2. "SELL_ALL": Liquidate immediately. Use if thesis is broken, sudden extreme risk, or extreme overbought/oversold.
3. "SELL_PARTIAL": Scale out to lock in profits or reduce exposure.
4. "ADJUST_STOP": Keep the position but tighten the trailing stop (e.g., to 1.5%).

Respond with ONLY valid JSON:
{{
    "verdict": "HOLD", "SELL_ALL", "SELL_PARTIAL", or "ADJUST_STOP",
    "confidence": <integer 0-100>,
    "sell_pct": <float 0.1 to 0.9, only if SELL_PARTIAL>,
    "new_trail_pct": <float e.g. 1.5, only if ADJUST_STOP>,
    "reasoning": "<2-3 sentences explaining why>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_response(raw)

        except Exception as e:
            log.warning(f"DeepSeek Closing Agent failed for {symbol}: {e}")
            return None

    def _parse_response(self, raw: str) -> dict:
        # 1. Clean markdown code blocks if present
        clean_raw = raw
        if "```json" in raw:
            json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
            if json_match:
                clean_raw = json_match.group(1)
        elif "```" in raw:
            json_match = re.search(r'```\s*(.*?)\s*```', raw, re.DOTALL)
            if json_match:
                clean_raw = json_match.group(1)
        
        # 2. Extract first curly brace block if still not clean
        if not clean_raw.strip().startswith("{"):
            json_match = re.search(r'\{.*\}', clean_raw, re.DOTALL)
            if json_match:
                clean_raw = json_match.group(0)

        try:
            data = json.loads(clean_raw)
            return {
                "verdict": str(data.get("verdict") or "HOLD"),
                "confidence": float(data.get("confidence") or 0.0),
                "sell_pct": float(data.get("sell_pct") or 0.5),
                "new_trail_pct": float(data.get("new_trail_pct") or 1.5),
                "reasoning": str(data.get("reasoning") or "")[:300],
                "deepseek_reasoning": "", 
            }
        except Exception as e:
            log.warning(f"Failed to parse closing agent JSON: {e} | Raw starts with: {raw[:100]}...")
            
            # Fallback regex parsing with improved reasoning extraction
            v_match = re.search(r'"verdict"\s*:\s*"([A-Z_]+)"', raw)
            c_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
            r_match = re.search(r'"reasoning"\s*:\s*"(.*?)"', raw)
            
            verdict = v_match.group(1) if v_match else "HOLD"
            conf = float(c_match.group(1)) if c_match else 0.0
            reason = r_match.group(1) if r_match else "Parsed via fallback regex due to invalid JSON formatting"
            
            return {
                "verdict": verdict,
                "confidence": conf,
                "sell_pct": 0.5,
                "new_trail_pct": 1.5,
                "reasoning": f"(REGEX) {reason}"[:300]
            }

