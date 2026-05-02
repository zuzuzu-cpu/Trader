# 🚀 Sentinel Autotrader V3.5 - Deployment Guide

This guide covers deploying the **Sentinel Autotrader** to a **Hostinger KVM2 VPS** using GitHub and Docker Compose. 

The architecture consists of two containers running together:
1. `sentinel-bot`: The core Python trading engine (running `main.py`).
2. `sentinel-dashboard`: The Flask web UI (running `dashboard/app.py` on port 5000).

---

## Prerequisites

1. **Hostinger KVM2 VPS** running **Ubuntu 22.04 or 24.04**.
2. A **GitHub Repository** where your Sentinel code is hosted (Private repository recommended).
3. API Keys for Alpaca, DeepSeek, Telegram, NewsAPI, and CoinMarketCap.

---

## Step 1: VPS Initial Setup

SSH into your Hostinger VPS using your terminal:

```bash
ssh root@YOUR_VPS_IP_ADDRESS
```

Once logged in, update your system and install necessary dependencies (Git, Docker, and Docker Compose):

```bash
# Update package lists
apt update && apt upgrade -y

# Install Git
apt install git -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose plugin
apt-get install docker-compose-plugin -y

# Verify installations
docker --version
docker compose version
```

---

## Step 2: Clone the Repository

Since you are likely using a Private GitHub repository, you will need to authenticate. The easiest way on a VPS is using a Personal Access Token (PAT) or setting up an SSH key.

If using HTTPS with a PAT:
```bash
git clone https://github.com/zuzuzu-cpu/Trader.git sentinel
cd sentinel
```
*(When prompted for a password, paste your GitHub Personal Access Token).*

---

## Step 3: Configure Environment Variables

The `.env` file contains highly sensitive API keys and **must not** be committed to GitHub. You need to create this file manually on your VPS.

```bash
nano .env
```

Paste the following template and fill in your actual keys:

```ini
# Alpaca Paper Trading
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# DeepSeek AI
DEEPSEEK_API_KEY=your_deepseek_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_REASONER_MODEL=deepseek-reasoner

# Other Data Sources
NEWSAPI_KEY=your_newsapi_key_here
COINMARKETCAP_API_KEY=your_cmc_key_here

# Telegram Alerts (Optional but recommended)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Initial Equity (For P/L calculation baseline)
INITIAL_EQUITY=100000.00
```

Press `Ctrl+O`, `Enter` to save, and `Ctrl+X` to exit Nano.

---

## Step 4: Build and Launch with Docker Compose

Ensure you are inside the `sentinel` directory where `docker-compose.yml` is located.

Run the following command to build the containers and launch them in the background (detached mode):

```bash
docker compose up -d --build
```

### Verify the deployment:

Check if both containers are running:
```bash
docker ps
```
You should see `sentinel-bot` and `sentinel-dashboard`.

View the live logs of the trading engine:
```bash
docker compose logs -f bot
```
*(Press `Ctrl+C` to exit the log view).*

---

## Step 5: Access the Live Dashboard

The web dashboard is now running on port `5000`.

To access it, open your web browser and navigate to:
`http://YOUR_VPS_IP_ADDRESS:5000`

> **Security Note:** By default, port 5000 is exposed to the internet via HTTP. For a production environment, it is highly recommended to set up an Nginx reverse proxy with a free Let's Encrypt SSL certificate and basic authentication to secure the dashboard.

---

## Maintenance Commands

**To pull the latest code from GitHub and restart:**
```bash
cd /root/sentinel
git pull origin main
docker compose up -d --build
```

**To completely stop the bot:**
```bash
docker compose down
```

**To view the database file:**
The SQLite database is stored in the `data/` directory mapped to the host. You can download `/root/sentinel/data/sentinel.db` via SFTP to analyze the trade history locally.
