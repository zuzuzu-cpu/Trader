"""
Sentinel Autotrader V3.0 - Centralized Configuration
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

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY", "")


# ─── Rate Limiting ──────────────────────────────────────────────────────────
ALPACA_RATE_LIMIT = int(os.getenv("ALPACA_RATE_LIMIT", 180))        # requests per minute (200 max, keep buffer)
DEEPSEEK_RATE_LIMIT = int(os.getenv("DEEPSEEK_RATE_LIMIT", 30))     # conservative concurrent limit
NEWSAPI_RATE_LIMIT = int(os.getenv("NEWSAPI_RATE_LIMIT", 90))       # 100/day free tier, keep buffer


# ─── Universe Scanning ──────────────────────────────────────────────────────
MIN_PRICE = float(os.getenv("MIN_PRICE", 5.0))                      # Skip penny stocks
MAX_PRICE = float(os.getenv("MAX_PRICE", 10000.0))
MIN_DAILY_VOLUME = int(os.getenv("MIN_DAILY_VOLUME", 500_000))      # Minimum avg daily volume
MIN_MARKET_CAP = float(os.getenv("MIN_MARKET_CAP", 500_000_000))    # $500M minimum market cap

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


# ─── Quant Engine Thresholds ────────────────────────────────────────────────
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 120))                 # Days of historical data
QUANT_PASS_SCORE = float(os.getenv("QUANT_PASS_SCORE", 55))         # Minimum score to pass quant filter

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


# ─── AI Agent Configuration ─────────────────────────────────────────────────
SENTIMENT_RANGE = (-10, 10)
NEWS_HEADLINE_COUNT = 10          # Headlines to send to DeepSeek


# ─── Portfolio & Risk Management ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 80))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", 0.02))       # 2% max per trade
MIN_POSITION_USD = float(os.getenv("MIN_POSITION_USD", 10.0))       # Minimum $10 per trade
MAX_PORTFOLIO_POSITIONS = int(os.getenv("MAX_PORTFOLIO_POSITIONS", 25))
MAX_SECTOR_CONCENTRATION = float(os.getenv("MAX_SECTOR_CONCENTRATION", 0.30))  # 30% max in one sector
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", 0.15))       # 15% max drawdown circuit breaker
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", 3.0))      # 3% trailing stop
ATR_RISK_MULTIPLIER = float(os.getenv("ATR_RISK_MULTIPLIER", 1.5))  # Stop = ATR * multiplier


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


# ─── Paths ──────────────────────────────────────────────────────────────────
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "sentinel.db"
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
