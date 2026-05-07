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
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

# Portfolio tracking
INITIAL_EQUITY = float(os.getenv("INITIAL_EQUITY", "100000"))

# DeepSeek (AI Reasoner / News Sentiment)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ─── V5: Deep Fundamentals API Keys ─────────────────────────────────────────
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FUNDAMENTALS_CACHE_DAYS = int(os.getenv("FUNDAMENTALS_CACHE_DAYS", "30"))
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 70.0))
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


# ─── Scaled Partial Exits ────────────────────────────────────────────────
EXIT_SCALE_ENABLED = os.getenv("EXIT_SCALE_ENABLED", "true").lower() == "true"
EXIT_SCALE_1_TRIGGER_PCT = float(os.getenv("EXIT_SCALE_1_TRIGGER_PCT", "4.0"))
EXIT_SCALE_1_PCT = float(os.getenv("EXIT_SCALE_1_PCT", "25.0"))
EXIT_SCALE_2_TRIGGER_PCT = float(os.getenv("EXIT_SCALE_2_TRIGGER_PCT", "6.0"))
EXIT_SCALE_2_PCT = float(os.getenv("EXIT_SCALE_2_PCT", "25.0"))
EXIT_SCALE_3_TRIGGER_PCT = float(os.getenv("EXIT_SCALE_3_TRIGGER_PCT", "8.0"))
EXIT_SCALE_3_PCT = float(os.getenv("EXIT_SCALE_3_PCT", "50.0"))

# ─── Closing Agent Tiered Exit Hierarchy ──────────────────────────────────
CLOSING_AGENT_ENABLED = os.getenv("CLOSING_AGENT_ENABLED", "true").lower() == "true"
CLOSING_CONFIDENCE_THRESHOLD = float(os.getenv("CLOSING_CONFIDENCE_THRESHOLD", 75.0))
HARD_STOP_LOSS_PCT = float(os.getenv("HARD_STOP_LOSS_PCT", "8.0"))            # Tier 1: hard stop
HARD_TAKE_PROFIT_PCT = float(os.getenv("HARD_TAKE_PROFIT_PCT", "12.0"))       # Tier 1: hard take-profit
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "14"))                         # Tier 1: max days per position
TRAIL_STOP_REFRESH_MINUTES = int(os.getenv("TRAIL_STOP_REFRESH_MINUTES", "30"))  # Tier 2: recalc trail freq
VOLUME_COLLAPSE_RATIO = float(os.getenv("VOLUME_COLLAPSE_RATIO", "0.3"))      # Tier 2: volume < 30% of avg
RSI_FAIL_THRESHOLD = float(os.getenv("RSI_FAIL_THRESHOLD", "50.0"))           # Tier 2: RSI cross below
CLOSING_AI_TRIGGER_PNL_MOVE = float(os.getenv("CLOSING_AI_TRIGGER_PNL_MOVE", "3.0"))  # Tier 3: P&L move % since last eval
CLOSING_AI_TRIGGER_HOURS = int(os.getenv("CLOSING_AI_TRIGGER_HOURS", "24"))            # Tier 3: eval if held > N hours
CLOSING_PORTFOLIO_HEAT_MAX = float(os.getenv("CLOSING_PORTFOLIO_HEAT_MAX", "0.25"))     # Tier 4: max portfolio risk heat

# ─── Re-entry Cooldown ────────────────────────────────────────────────────
COOLDOWN_ENABLED = os.getenv("COOLDOWN_ENABLED", "true").lower() == "true"
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "24"))

# ─── Correlation Guard ────────────────────────────────────────────────────
CORRELATION_GUARD_ENABLED = os.getenv("CORRELATION_GUARD_ENABLED", "true").lower() == "true"
CORRELATION_THRESHOLD = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
CORRELATION_LOOKBACK_DAYS = int(os.getenv("CORRELATION_LOOKBACK_DAYS", "60"))

# ─── Macro Agent ─────────────────────────────────────────────────────────
MACRO_AGENT_ENABLED = os.getenv("MACRO_AGENT_ENABLED", "true").lower() == "true"
MACRO_REFRESH_HOURS = int(os.getenv("MACRO_REFRESH_HOURS", "24"))
MACRO_FRED_SERIES = os.getenv("MACRO_FRED_SERIES", "FEDFUNDS,CPIAUCSL,UNRATE,DGS10")

# ─── Regime Detection ────────────────────────────────────────────────────
REGIME_VIX_PROXY_SYMBOL = os.getenv("REGIME_VIX_PROXY_SYMBOL", "SPY")      # VIX substitute
REGIME_SMA_PERIOD = int(os.getenv("REGIME_SMA_PERIOD", "200"))
REGIME_ATR_VOL_THRESHOLD = float(os.getenv("REGIME_ATR_VOL_THRESHOLD", "2.0"))  # ATR% > 2% = high vol

# ─── Paths ──────────────────────────────────────────────────────────────────
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sentinel.db"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
