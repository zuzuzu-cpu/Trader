import os

class PortfolioManager:
    def __init__(self):
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 85))

    def calculate_confidence(self, quant_score: float, sentiment_score: int, risk_grade: str):
        """
        Calculates a weighted Confidence Score.
        Quant: 40%
        Sentiment: 40%
        Risk: 20%
        """
        # Map sentiment (-10 to 10) to a 0-100 scale
        mapped_sentiment = ((sentiment_score + 10) / 20) * 100
        
        # Map risk grade to a score
        risk_scores = {
            "LOW": 100,
            "MEDIUM": 50,
            "HIGH": 0
        }
        mapped_risk = risk_scores.get(risk_grade, 0)
        
        confidence = (quant_score * 0.40) + (mapped_sentiment * 0.40) + (mapped_risk * 0.20)
        return confidence

    def decide(self, symbol: str, quant_score: float, sentiment_score: int, risk_grade: str):
        """
        Makes the final decision to trade or not.
        """
        confidence = self.calculate_confidence(quant_score, sentiment_score, risk_grade)
        
        should_trade = confidence >= self.confidence_threshold
        
        decision = {
            "symbol": symbol,
            "confidence": confidence,
            "should_trade": should_trade,
            "reason": f"Q:{quant_score:.1f} S:{sentiment_score} R:{risk_grade}"
        }
        
        return decision
