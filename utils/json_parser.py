"""
Strict JSON parsing utility for AI responses.
Provides robust parsing with validation and retry logic.
"""
import json
import re
from typing import Type, TypeVar, Any, Optional

from pydantic import BaseModel, ValidationError
from utils.logger import get_logger

log = get_logger("sentinel.json_parser")


T = TypeVar('T', bound=BaseModel)


def extract_json(raw: str) -> Optional[dict]:
    """
    Extract JSON from AI response using multiple strategies.
    Returns None if no valid JSON found.
    """
    if not raw:
        return None
    
    # Strategy 1: Direct parse attempt
    try:
        return json.loads(raw)
    except:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Strategy 3: Extract any JSON-like object
    json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Strategy 4: Try to fix common issues
    fixed = raw.replace("'", '"').replace("None", "null")
    fixed = fixed.replace("True", "true").replace("False", "false")
    fixed = re.sub(r'(\w+):', r'"\1":', fixed)  # Quote unquoted keys
    try:
        return json.loads(fixed)
    except:
        pass
    
    return None


def validate_json(
    raw: str, 
    schema_class: Type[BaseModel], 
    max_retries: int = 2,
    default_on_fail: bool = True
) -> dict:
    """
    Parse and validate AI JSON response with strict schema.
    
    Args:
        raw: Raw AI response string
        schema_class: Pydantic schema to validate against
        max_retries: Number of parsing attempts
        default_on_fail: Return default on validation failure
    
    Returns:
        Validated dict or default response
    """
    data = extract_json(raw)
    
    if data is None:
        log.warning(f"Failed to extract JSON from response")
        if default_on_fail:
            return _get_default(schema_class)
        return {}
    
    # Validate with schema
    try:
        validated = schema_class(**data)
        return validated.model_dump(exclude_none=True)
    except ValidationError as e:
        log.warning(f"Validation error: {e}")
        
        # Try with fixed data
        for attempt in range(max_retries):
            try:
                # Fix common issues
                fixed_data = _fix_common_issues(data)
                validated = schema_class(**fixed_data)
                return validated.model_dump(exclude_none=True)
            except ValidationError:
                continue
        
        if default_on_fail:
            return _get_default(schema_class)
        return data


def _fix_common_issues(data: dict) -> dict:
    """Fix common JSON/validation issues."""
    fixed = {}
    
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, str):
            # Handle string that should be other types
            if key in ['events', 'flags'] and value:
                if value.startswith('['):
                    try:
                        fixed[key] = json.loads(value)
                        continue
                    except:
                        fixed[key] = [value]
                        continue
            fixed[key] = value
        elif isinstance(value, (int, float, bool)):
            fixed[key] = value
        elif isinstance(value, list):
            fixed[key] = value
        elif isinstance(value, dict):
            fixed[key] = value
        else:
            fixed[key] = str(value)
    
    return fixed


def _get_default(schema_class: Type[BaseModel]) -> dict:
    """Get default response for schema on complete failure."""
    defaults = {
        'SentimentSchema': {
            'score': 0, 
            'confidence': 0.3, 
            'events': [], 
            'summary': 'Parse failed - default neutral'
        },
        'RiskSchema': {
            'grade': 'MEDIUM', 
            'score': 50, 
            'flags': [], 
            'reasoning': 'Parse failed - default medium risk'
        },
        'DecisionSchema': {
            'verdict': 'REJECT', 
            'confidence_adjustment': 0, 
            'reasoning': 'Parse failed'
        },
        'ClosingSchema': {
            'verdict': 'HOLD', 
            'confidence': 50, 
            'reasoning': 'Parse failed - hold'
        },
    }
    
    schema_name = schema_class.__name__
    return defaults.get(schema_name, {})


def validate_with_retry(
    raw: str,
    schema_class: Type[BaseModel],
    max_retries: int = 3,
    retry_callback: Optional[callable] = None
) -> dict:
    """
    Validate JSON with retry callback for regeneration.
    
    Args:
        raw: Raw AI response
        schema_class: Pydantic schema  
        max_retries: Max validation attempts
        retry_callback: Called on failure with attempt number, should return new raw string
    """
    for attempt in range(max_retries):
        result = validate_json(raw, schema_class, default_on_fail=False)
        
        # Check if we got valid data (not just defaults)
        if result and 'Parse failed' not in str(result.get('summary', '')):
            return result
        
        if retry_callback and attempt < max_retries - 1:
            raw = retry_callback(attempt)
            continue
    
    # Return default on all failures
    return _get_default(schema_class)