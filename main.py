import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

from core.quant_engine import QuantEngine
from agents.news_hound import NewsHound
from agents.skeptic import Skeptic
from agents.portfolio_mgr import PortfolioManager
from execution.alpaca_broker import AlpacaBroker

load_dotenv()

def run_sentinel():
    print(f"--- Sentinel Autotrader V3.0 Cycle Started: {datetime.now()} ---")
    
    # 1. Initialize Components
    quant = QuantEngine()
    news_hound = NewsHound()
    skeptic = Skeptic()
    portfolio_mgr = PortfolioManager()
    broker = AlpacaBroker()
    
    # 2. Define Assets to Screen (The "Hot List" candidates)
    # In a production system, this list could be fetched from a scanner or a static list.
    stocks = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL"]
    cryptos = ["BTCUSD", "ETHUSD", "SOLUSD"]
    etfs = ["SPY", "QQQ", "IWM"]
    
    # 3. Step 1: Local Quantitative Filtering
    print("Step 1: Running Local Quant Core...")
    candidates = []
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    for s in stocks:
        result = quant.evaluate_stock(s, start_str, end_str)
        if result['score'] > 60: # Initial filter threshold
            candidates.append(('stock', result))

    for c in cryptos:
        result = quant.evaluate_crypto(c, start_str, end_str)
        if result['score'] > 60:
            candidates.append(('crypto', result))
            
    for e in etfs:
        result = quant.evaluate_etf(e, start_str, end_str)
        if result['score'] > 60:
            candidates.append(('etf', result))

    print(f"Found {len(candidates)} candidates passing local quant filter.")

    # 4. Step 2, 3, 4 & 5: AI Swarm and Execution
    for asset_type, q_result in candidates:
        symbol = q_result['symbol']
        q_score = q_result['score']
        
        print(f"\nAnalyzing Candidate: {symbol} (Quant Score: {q_score})")
        
        # Step 2: News Hound
        print(f"Agent 1: News Hound analyzing {symbol}...")
        sentiment_score = news_hound.analyze_sentiment(symbol)
        print(f"Sentiment Score: {sentiment_score}")
        
        # Step 3: The Skeptic
        print(f"Agent 2: The Skeptic debating {symbol}...")
        risk_grade = skeptic.evaluate_risk(symbol, q_score, sentiment_score)
        print(f"Risk Grade: {risk_grade}")
        
        # Step 4: Portfolio Manager Verdict
        print(f"Agent 3: Portfolio Manager deciding...")
        verdict = portfolio_mgr.decide(symbol, q_score, sentiment_score, risk_grade)
        print(f"Verdict: {verdict['reason']} -> Confidence: {verdict['confidence']:.2f}%")
        
        # Step 5: Execution
        if verdict['should_trade']:
            print(f"*** CONFIDENCE > THRESHOLD! Executing Trade for {symbol} ***")
            
            # Calculate position size (e.g., 1.5% of equity)
            account = broker.get_account_info()
            equity = float(account.equity)
            notional = equity * float(os.getenv("TRADE_FRACTION_PERCENT", 0.015))
            
            # Place order
            broker.place_order(symbol, notional=notional)
        else:
            print(f"Trade skipped for {symbol}. Confidence below threshold.")

    print("\n--- Cycle Complete ---")

if __name__ == "__main__":
    # Run once for demonstration. In production, use a scheduler.
    run_sentinel()
