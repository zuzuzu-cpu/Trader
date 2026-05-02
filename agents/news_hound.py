import os
import yfinance as yf
from openai import OpenAI

class NewsHound:
    def __init__(self):
        # Configure OpenAI client to use DeepSeek
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        )

    def fetch_news(self, symbol: str):
        """Fetches recent news using Yahoo Finance as a proxy for news."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                return "No recent news found."
            
            # Combine the titles and publishers of the latest 5 articles
            news_text = "\n".join([f"- {n.get('title', '')} ({n.get('publisher', '')})" for n in news[:5]])
            return news_text
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return "Error fetching news."

    def analyze_sentiment(self, symbol: str):
        """
        Analyzes news and returns a sentiment score between -10 and +10.
        """
        news_text = self.fetch_news(symbol)
        
        prompt = f"""
        You are 'The News Hound', an expert financial sentiment analyzer.
        Read the following recent news headlines for the asset {symbol}:
        
        {news_text}
        
        Analyze the sentiment and return ONLY a single integer score between -10 and +10.
        -10 is extremely bearish, 0 is neutral, and +10 is extremely bullish.
        Do not explain your reasoning, just output the number.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            score_str = response.choices[0].message.content.strip()
            # Parse the score
            score = int(score_str)
            # Bound the score
            score = max(-10, min(10, score))
            return score
        except Exception as e:
            print(f"Error analyzing sentiment for {symbol}: {e}")
            return 0 # Neutral on error
