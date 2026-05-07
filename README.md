# Sentinel Autotrader V6 🔥

Sentinel is a production-grade, autonomous trading bot for 24/7 market execution. It uses a multi-agent architecture combining quantitative analysis with DeepSeek AI-powered decision making.

## Architecture: The V6 Pipeline

### Entry Pipeline (every cycle)
1. **Universe Scanner** — discovers 10,000+ tradable assets from Alpaca
2. **Quant Engine** — 15+ indicators, RS rating vs SPY, multi-timeframe confirmation
3. **News Hound** — Yahoo Finance + NewsAPI → DeepSeek sentiment analysis
4. **Skeptic** — AI risk assessment with spread, concentration, volatility checks
5. **Insider Tracker** — SEC Form 4 filings (insider buys/sells)
6. **Options Flow** — unusual options volume + put/call ratio
7. **Portfolio Manager** — Kelly Criterion sizing + DeepSeek Reasoner final verdict

### Exit Pipeline (4-tier hierarchy)
| Tier | Type | Examples |
|------|------|----------|
| **Tier 1** | Hard rules (sub-ms) | Stop-loss, take-profit, max hold days, regime flip |
| **Tier 2** | Technical rules | Trailing stop, volume collapse, RSI/EMA failure |
| **Tier 3** | AI-assisted | DeepSeek only when P&L moves >3% or held >24h |
| **Tier 4** | Portfolio-level (daily) | Correlation reduction, heat check, cooldown list |

### Context Injection (every prompt)
- **Market Regime** — SPY SMA200 + ATR% (VIX proxy) → BULL/BEAR/SIDEWAYS × HIGH/LOW volatility
- **Macro Agent** — FRED data (fed rate, CPI, unemployment, 10Y yield) cached daily
- **Correlation Guard** — blocks trades >0.7 correlated with existing positions

## Quick Start

```bash
git clone https://github.com/zuzuzu-cpu/Trader.git
cd Trader
cp .env.example .env  # add your keys
docker compose up -d --build
```

### Required .env keys
```
ALPACA_API_KEY=pk_...
ALPACA_SECRET_KEY=sk_...
DEEPSEEK_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=123:abc
TELEGRAM_CHAT_ID=123456789
INITIAL_EQUITY=100000
FRED_API_KEY=...          # optional: macro context
```

## Tech Stack
- **Language**: Python 3.12+
- **AI**: DeepSeek-Chat (fast) + DeepSeek-Reasoner (complex decisions)
- **Data**: Alpaca V2, Yahoo Finance, SEC EDGAR, Finnhub, FMP, FRED
- **Database**: SQLite (WAL mode, busy timeout)
- **Dashboard**: Flask, Chart.js, Tailwind
- **Infra**: Docker & Docker Compose

## Disclaimer
This software is for educational purposes only. Trading involves significant risk of loss. Use at your own risk.
