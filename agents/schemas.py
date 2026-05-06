"""
Strict JSON schemas for AI response validation.
Uses Pydantic for type-safe validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any
import json
import re


class SentimentSchema(BaseModel):
    """Strict schema for news sentiment analysis."""
    score: int = Field(..., ge=-10, le=10, description="Sentiment score -10 to +10")
    confidence: float = Field(..., ge=0, le=1, description="AI confidence 0-1")
    events: List[str] = Field(default_factory=list, description="Detected event types")
    summary: str = Field(..., max_length=200, description="One sentence summary")
    
    @field_validator('events', mode='before')
    @classmethod
    def parse_events(cls, v):
        if isinstance(v, str):
            # Try to parse string as JSON
            try:
                return json.loads(v)
            except:
                return [v] if v else []
        return v or []


class RiskSchema(BaseModel):
    """Strict schema for risk assessment."""
    grade: str = Field(..., description="Risk grade: LOW/MEDIUM/HIGH")
    score: int = Field(..., ge=0, le=100, description="Risk score 0-100")
    flags: List[str] = Field(default_factory=list, description="Risk flags detected")
    reasoning: str = Field(..., max_length=300, description="Explanation")
    spread_pct: Optional[float] = None
    
    @field_validator('grade', mode='before')
    @classmethod
    def normalize_grade(cls, v):
        if isinstance(v, str):
            return v.upper().strip()
        return "MEDIUM"


class DecisionSchema(BaseModel):
    """Strict schema for trade decisions."""
    verdict: str = Field(..., description="APPROVE or REJECT")
    confidence_adjustment: int = Field(default=0, ge=-10, le=10, description="Confidence delta")
    reasoning: str = Field(..., max_length=300, description="Decision explanation")
    suggested_size_pct: Optional[float] = Field(default=0.02, ge=0, le=0.1)
    
    @field_validator('verdict', mode='before')
    @classmethod
    def normalize_verdict(cls, v):
        if isinstance(v, str):
            v = v.upper().strip()
            if v in ['BUY', 'LONG', 'ENTER']:
                return 'APPROVE'
            elif v in ['SELL', 'SHORT', 'EXIT', 'HOLD']:
                return 'REJECT'
        return v


class ClosingSchema(BaseModel):
    """Strict schema for closing agent decisions."""
    verdict: str = Field(..., description="SELL_ALL, SELL_PARTIAL, ADJUST_STOP, HOLD")
    confidence: float = Field(..., ge=0, le=100, description="Confidence 0-100")
    reasoning: str = Field(..., max_length=300, description="Decision explanation")
    sell_pct: Optional[float] = Field(default=0.5, ge=0, le=1)
    new_trail_pct: Optional[float] = Field(default=3.0, ge=0, le=10)
    
    @field_validator('verdict', mode='before')
    @classmethod
    def normalize_verdict(cls, v):
        valid = ['SELL_ALL', 'SELL_PARTIAL', 'ADJUST_STOP', 'HOLD']
        if isinstance(v, str):
            v = v.upper().strip()
            if v not in valid:
                return 'HOLD'
        return v


# ─── JSON Parsing Utility ──────────────────────────────────────────────

def parse_ai_json(raw: str, schema_class, max_retries: int = 3) -> dict:
    """
    Strict JSON parsing with validation.
    Tries multiple strategies to extract valid JSON.
    """
    # Strategy 1: Direct parse
    for attempt in range(max_retries):
        try:
            # Try to extract JSON from markdown or plain text
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            
            data = json.loads(raw)
            validated = schema_class(**data)
            return validated.model_dump(exclude_none=True)
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            if attempt < max_retries - 1:
                # Try to fix common issues
                raw = raw.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
            continue
    
    # Fallback: return default valid response
    return _get_default_response(schema_class)


def _get_default_response(schema_class) -> dict:
    """Return default response for schema on parse failure."""
    defaults = {
        SentimentSchema: {"score": 0, "confidence": 0.3, "events": [], "summary": "Parse failed - default neutral"},
        RiskSchema: {"grade": "MEDIUM", "score": 50, "flags": [], "reasoning": "Parse failed - default medium risk"},
        DecisionSchema: {"verdict": "REJECT", "confidence_adjustment": 0, "reasoning": "Parse failed"},
        ClosingSchema: {"verdict": "HOLD", "confidence": 50, "reasoning": "Parse failed - hold"},
    }
    return defaults.get(schema_class, {})


# ─── Few-Shot Examples ──────────────────────────────────────────────

SENTIMENT_EXAMPLE = """Example response:
```json
{"score": 6, "confidence": 0.85, "events": ["earnings_beat", "analyst_upgrade"], "summary": "Strong Q3 earnings beat with raised guidance"}
```"""

RISK_EXAMPLE = """Example response:
```json
{"flags": ["HIGH_VOLATILITY", "EARNINGS_NEXT_WEEK"], "score_adjustment": -10, "reasoning": "High volatility and upcoming earnings increase risk"}
```"""

DECISION_EXAMPLE = """Example response:
```json
{"verdict": "APPROVE", "confidence_adjustment": 5, "reasoning": "Strong quant score and positive sentiment support the trade despite moderate risk"}
```"""

CLOSING_EXAMPLE = """Example response:
```json
{"verdict": "SELL_ALL", "confidence": 80, "reasoning": "Profit target hit with strong momentum - take profits", "sell_pct": 1.0}
```"""