import os
from openai import OpenAI
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

class Skeptic:
    def __init__(self):
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.stock_client = StockHistoricalDataClient(self.alpaca_api_key, self.alpaca_secret_key)
        
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )

    def get_spread(self, symbol: str):
        """Fetches the latest quote to determine the bid/ask spread."""
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_client.get_stock_latest_quote(req)
            quote = quotes[symbol]
            spread = quote.ask_price - quote.bid_price
            spread_pct = (spread / quote.ask_price) * 100 if quote.ask_price else 0
            return spread, spread_pct
        except Exception as e:
            print(f"Error fetching spread for {symbol}: {e}")
            return None, None

    def evaluate_risk(self, symbol: str, quant_score: float, sentiment_score: int):
        """
        Acts as the Devil's Advocate to find reasons NOT to trade.
        Returns a risk grade: 'Low', 'Medium', or 'High'
        """
        spread, spread_pct = self.get_spread(symbol)
        
        prompt = f"""
        You are 'The Skeptic', an AI Risk Manager.
        You are reviewing a potential long trade for {symbol}.
        
        Data points:
        - Quant Engine Score (0-100): {quant_score}
        - Sentiment Score (-10 to +10): {sentiment_score}
        - Bid/Ask Spread: {f'{spread_pct:.2f}%' if spread_pct is not None else 'Unknown'}
        
        Consider the risks of false breakouts, wide spreads, and poor sentiment.
        Is this trade a High, Medium, or Low risk?
        Output ONLY one word: LOW, MEDIUM, or HIGH.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            grade = response.choices[0].message.content.strip().upper()
            if grade not in ["LOW", "MEDIUM", "HIGH"]:
                return "HIGH" # Default to high risk if confused
            return grade
        except Exception as e:
            print(f"Error evaluating risk for {symbol}: {e}")
            return "HIGH"
