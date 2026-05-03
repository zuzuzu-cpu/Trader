"""
Sentinel Autotrader V3.5 - Centralized Configuration
All tunable parameters in one place. Override via .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


# ─── API Keys ───────────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ─── Rate Limiting ──────────────────────────────────────────────────────────
ALPACA_RATE_LIMIT = int(os.getenv("ALPACA_RATE_LIMIT", 180))        # requests per minute (200 max, keep buffer)
DEEPSEEK_RATE_LIMIT = int(os.getenv("DEEPSEEK_RATE_LIMIT", 30))     # conservative concurrent limit
NEWSAPI_RATE_LIMIT = int(os.getenv("NEWSAPI_RATE_LIMIT", 90))       # 100/day free tier, keep buffer


# ─── Async Processing ──────────────────────────────────────────────────────
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 15))                      # ThreadPoolExecutor workers
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))                       # Symbols per batch


# ─── Universe Scanning ──────────────────────────────────────────────────────
MIN_PRICE = float(os.getenv("MIN_PRICE", 5.0))                      # Skip penny stocks
MAX_PRICE = float(os.getenv("MAX_PRICE", 10000.0))
MIN_DAILY_VOLUME = int(os.getenv("MIN_DAILY_VOLUME", 500_000))      # Minimum avg daily volume
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", 500_000_000))    # $500M minimum market cap
MAX_UNIVERSE_SIZE = int(os.getenv("MAX_UNIVERSE_SIZE", 10000))     # Max assets to scan

# Static watchlists (augment dynamic scanning)
CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD",
    "DOGE/USD", "ADA/USD", "DOT/USD", "MATIC/USD", "UNI/USD",
]
ETF_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "XLF",
    "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE",
    "GLD", "SLV", "TLT", "HYG", "LQD", "EMB", "VWO", "EFA",
    "SOXX", "SMH", "IBB", "XBI", "KWEB", "FXI",
]

# Benchmark for relative strength
BENCHMARK_SYMBOL = os.getenv("BENCHMARK_SYMBOL", "SPY")


# ─── Quant Engine Thresholds ────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 120))                 # Days of historical data
QUANT_PASS_SCORE = float(os.getenv("QUANT_PASS_SCORE", 55))         # Minimum score to pass quant filter (long)
QUANT_SHORT_SCORE = float(os.getenv("QUANT_SHORT_SCORE", 25))       # Below this = short candidate

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 4.0       # Bandwidth below this = squeeze

# Moving Averages
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 9
EMA_LONG = 21

# Volatility / Risk
ATR_PERIOD = 14
SHARPE_WINDOW = 60               # Days for rolling Sharpe
RISK_FREE_RATE = 0.05            # Annual risk-free rate for Sharpe/Sortino

# Multi-Timeframe
ENABLE_MULTI_TIMEFRAME = os.getenv("ENABLE_MULTI_TIMEFRAME", "true").lower() == "true"
MTF_CONFIRMATION_REQUIRED = int(os.getenv("MTF_CONFIRMATION_REQUIRED", 2))  # out of 3 timeframes must agree

# Relative Strength
RS_LOOKBACK_DAYS = int(os.getenv("RS_LOOKBACK_DAYS", 63))           # ~3 months for RS calculation

# Earnings Calendar
EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", 3)) # Days before earnings to avoid


# ─── AI Agent Configuration ─────────────────────────────────────────────────
SENTIMENT_RANGE = (-10, 10)
NEWS_HEADLINE_COUNT = 10          # Headlines to send to DeepSeek


# ─── Short Selling ──────────────────────────────────────────────────────────
ENABLE_SHORT_SELLING = os.getenv("ENABLE_SHORT_SELLING", "true").lower() == "true"
SHORT_CONFIDENCE_THRESHOLD = float(os.getenv("SHORT_CONFIDENCE_THRESHOLD", 50.0))
MAX_SHORT_POSITIONS = int(os.getenv("MAX_SHORT_POSITIONS", 10))


# ─── Portfolio & Risk Management ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 50.0))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.02))       # 2% max per trade
MIN_POSITION_USD = float(os.getenv("MIN_POSITION_USD", 10.0))       # Minimum $10 per trade
MAX_PORTFOLIO_POSITIONS = int(os.getenv("MAX_PORTFOLIO_POSITIONS", 25))
MAX_SECTOR_CONCENTRATION = float(os.getenv("MAX_SECTOR_CONCENTRATION", 0.30))  # 30% max in one sector
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", 0.15))       # 15% max drawdown circuit breaker
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", 3.0))      # 3% trailing stop
ATR_RISK_MULTIPLIER = float(os.getenv("ATR_RISK_MULTIPLIER", 1.5))  # Stop = ATR * multiplier


# ─── Kelly Criterion Position Sizing ────────────────────────────────────────
KELLY_ENABLED = os.getenv("KELLY_ENABLED", "true").lower() == "true"
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", 0.25))            # Use 1/4 Kelly (conservative)
KELLY_MIN_TRADES = int(os.getenv("KELLY_MIN_TRADES", 10))             # Need at least 10 trades for stats
KELLY_MAX_POSITION_PCT = float(os.getenv("KELLY_MAX_POSITION_PCT", 0.05))  # Cap at 5% even if Kelly says more
KELLY_MIN_POSITION_PCT = float(os.getenv("KELLY_MIN_POSITION_PCT", 0.005)) # Floor at 0.5%


# ─── Partial Profit Taking ──────────────────────────────────────────────────
PARTIAL_PROFIT_ENABLED = os.getenv("PARTIAL_PROFIT_ENABLED", "true").lower() == "true"
PROFIT_TARGET_PCT = float(os.getenv("PROFIT_TARGET_PCT", 5.0))       # Take profit at +5%
TAKE_PROFIT_RATIO = float(os.getenv("TAKE_PROFIT_RATIO", 0.5))      # Sell 50% at target
REMAINDER_TRAIL_PCT = float(os.getenv("REMAINDER_TRAIL_PCT", 4.0))   # Wider stop on remainder


# ─── Market Hours Awareness ─────────────────────────────────────────────────
SKIP_CLOSED_MARKET = os.getenv("SKIP_CLOSED_MARKET", "true").lower() == "true"


# ─── Scoring Weights ────────────────────────────────────────────────────────
WEIGHT_QUANT = float(os.getenv("WEIGHT_QUANT", 0.40))
WEIGHT_SENTIMENT = float(os.getenv("WEIGHT_SENTIMENT", 0.35))
WEIGHT_RISK = float(os.getenv("WEIGHT_RISK", 0.25))


# ─── Scheduler ──────────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", 30))  # Run every 30 min
MARKET_OPEN_HOUR = 9                                                  # EST
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
TRADE_EXTENDED_HOURS = os.getenv("TRADE_EXTENDED_HOURS", "true").lower() == "true"


# ─── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8080))


# ─── V5: WebSocket Live Streaming ───────────────────────────────────────────
LIVE_STREAM_ENABLED = os.getenv("LIVE_STREAM_ENABLED", "true").lower() == "true"
SPIKE_ALERT_PCT = float(os.getenv("SPIKE_ALERT_PCT", 2.0))        # Alert on ≥2% bar move

# ─── V5: Pre-Market / After-Hours Scanning ──────────────────────────────────
PREMARKET_ENABLED = os.getenv("PREMARKET_ENABLED", "true").lower() == "true"
AFTER_HOURS_ENABLED = os.getenv("AFTER_HOURS_ENABLED", "true").lower() == "true"
PREMARKET_MIN_GAP_PCT = float(os.getenv("PREMARKET_MIN_GAP_PCT", 2.0))   # Min gap % to queue

# ─── V5: Insider Trading Tracker ────────────────────────────────────────────
INSIDER_ENABLED = os.getenv("INSIDER_ENABLED", "true").lower() == "true"
INSIDER_LOOKBACK_DAYS = int(os.getenv("INSIDER_LOOKBACK_DAYS", 14))       # Look back 14 days

# ─── V5: Social Sentiment (Reddit/WSB) ──────────────────────────────────────
SOCIAL_SENTIMENT_ENABLED = os.getenv("SOCIAL_SENTIMENT_ENABLED", "true").lower() == "true"
WSB_MIN_MENTIONS = int(os.getenv("WSB_MIN_MENTIONS", 5))          # Minimum mentions to score
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SentinelAutotrader/5.0")

# ─── V5: Options Flow ────────────────────────────────────────────────────────
OPTIONS_FLOW_ENABLED = os.getenv("OPTIONS_FLOW_ENABLED", "true").lower() == "true"
OPTIONS_VOLUME_THRESHOLD = float(os.getenv("OPTIONS_VOLUME_THRESHOLD", 3.0))  # 3x avg = unusual

# ─── V5: Trade Explanation Cards ─────────────────────────────────────────────
TRADE_CARDS_ENABLED = os.getenv("TRADE_CARDS_ENABLED", "true").lower() == "true"
SKIP_CARD_MIN_CONFIDENCE = float(os.getenv("SKIP_CARD_MIN_CONFIDENCE", 45.0)) # Only card near-misses


# ─── Paths ──────────────────────────────────────────────────────────────────
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sentinel.db"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
