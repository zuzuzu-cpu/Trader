"""
Rate limiter for external API calls.
Implements a token-bucket algorithm to stay within rate limits for
Alpaca (200 req/min) and DeepSeek (dynamic concurrency).
"""
import time
import threading
from functools import wraps

from utils.logger import get_logger

log = get_logger("sentinel.rate_limiter")


class RateLimiter:
    """
    Thread-safe token bucket rate limiter.
    Tokens are refilled at a fixed rate. Each API call consumes one token.
    If no tokens are available, the caller blocks until one is refilled.
    """

    def __init__(self, name: str, max_calls: int, period_seconds: float = 60.0):
        self.name = name
        self.max_calls = max_calls
        self.period = period_seconds
        self.tokens = max_calls
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * (self.max_calls / self.period)
        self.tokens = min(self.max_calls, self.tokens + new_tokens)
        self.last_refill = now

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """
        Block until a token is available, then consume it.
        If blocking=False, returns False immediately if no tokens.
        If timeout is set, returns False after waiting `timeout` seconds.
        """
        start_time = time.monotonic()
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                    
            if not blocking:
                return False
                
            if timeout is not None and (time.monotonic() - start_time) >= timeout:
                return False

            # Wait a small interval before retrying
            time.sleep(0.1)

    def __call__(self, func):
        """Use as a decorator: @rate_limiter"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.acquire()
            return func(*args, **kwargs)
        return wrapper


# ─── Global rate limiter instances ───────────────────────────────────────────
# These are shared across all modules that import them.

alpaca_limiter = RateLimiter("alpaca", max_calls=180, period_seconds=60)
deepseek_limiter = RateLimiter("deepseek", max_calls=30, period_seconds=60)
newsapi_limiter = RateLimiter("newsapi", max_calls=5, period_seconds=60)  # Very conservative
finnhub_limiter = RateLimiter("finnhub", max_calls=55, period_seconds=60) # 60/min limit
fmp_limiter = RateLimiter("fmp", max_calls=240, period_seconds=86400) # 250/day limit
sec_limiter = RateLimiter("sec", max_calls=10, period_seconds=1) # 10/sec limit


def retry_on_rate_limit(func, max_retries=3, base_delay=2.0):
    """
    Decorator that retries a function call if it raises a rate-limit error (HTTP 429).
    Uses exponential backoff.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        log.warning(f"Rate limited on {func.__name__}, retry {attempt+1}/{max_retries} in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        log.error(f"Rate limit exhausted for {func.__name__} after {max_retries} retries")
                        raise
                else:
                    raise
    return wrapper
