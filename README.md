# Sentinel Autotrader V4 🚀

Sentinel is a production-grade, autonomous trading bot designed for 24/7 market execution. It leverages a multi-agent architecture to combine deep quantitative analysis with AI-driven sentiment intelligence.

## 🧠 Architecture: The Multi-Agent Pipeline

Sentinel operates using a specialized pipeline of AI agents:

1.  **The News Hound (`Agent 1`)**: Scrapes multi-source news (Yahoo Finance, NewsAPI) and uses **DeepSeek AI** to classify market sentiment and detect high-impact events.
2.  **The Quant Engine**: Calculates 15+ technical indicators (RSI, MACD, Bollinger Bands, ATR) using `pandas-ta` to identify momentum and volatility regimes.
3.  **The Portfolio Manager (`Agent 3`)**: Synthesizes AI sentiment and technical signals. It uses the **Kelly Criterion** for optimal position sizing and ensures portfolio diversification.
4.  **The Closing Agent**: Monitors existing positions in real-time, adjusting trailing stops and clearing orphaned orders to protect capital.

## ✨ Key Features

*   **Multi-Asset Support**: Seamlessly trades US Equities, ETFs, and Crypto via the Alpaca API.
*   **Robust Stability**: Built-in rate limiting with smart timeouts, automatic retry logic, and SQLite connection pooling for high-reliability 24/7 uptime.
*   **Mathematical Precision**: Temporal alignment of asset data with benchmarks (SPY) for accurate Relative Strength (RS) ranking.
*   **Real-time Dashboard**: A sleek Flask-based dashboard for monitoring live trades, portfolio metrics, and bot health.
*   **Dockerized Deployment**: Fully containerized for one-command deployment on any VPS (Hostinger, AWS, GCP).

## 🛠 Tech Stack

*   **Language**: Python 3.10+
*   **Data**: Alpaca V2, NewsAPI, Finnhub, FMP
*   **AI**: DeepSeek-V3 / DeepSeek-Reasoner
*   **Database**: SQLite (optimized for concurrent access)
*   **Interface**: Flask, Chart.js, Tailwind CSS
*   **Infrastructure**: Docker & Docker Compose

## 🚀 Quick Start

### 1. Clone & Configure
```bash
git clone https://github.com/zuzuzu-cpu/Trader.git
cd Trader
cp .env.example .env
```
Edit the `.env` file with your API keys (Alpaca, DeepSeek, NewsAPI).

### 2. Deploy with Docker
```bash
docker compose up -d --build
```

### 3. Monitor
Access your live trading dashboard at `http://localhost:5000` (or your VPS IP).

## ⚠️ Disclaimer
This software is for educational purposes only. Trading involves significant risk of loss. Use at your own risk.
