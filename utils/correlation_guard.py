"""
Correlation Guard — Pre-trade correlation check.

Before any new position is opened, this calculates the price correlation
of the new asset against every existing position. If correlation exceeds
the threshold, the trade is rejected — those correlated positions would
amplify risk in a downturn instead of diversifying it.

Pure Python, no AI needed.
"""
import numpy as np
import pandas as pd
from typing import Optional

import config
from core.data_fetcher import DataFetcher
from utils.logger import get_logger

log = get_logger("sentinel.correlation_guard")


class CorrelationGuard:
    """
    Checks price correlation between candidate symbols and existing positions.
    """
    
    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
        self._cache = {}
    
    def check(self, symbol: str, held_symbols: list[str],
              direction: str = "long") -> dict:
        """
        Check if a symbol is too correlated with existing positions.
        
        Returns:
        {
            "can_trade": bool,
            "max_correlation": float,
            "conflicting_position": str | None,
            "reason": str
        }
        """
        if not config.CORRELATION_GUARD_ENABLED:
            return {"can_trade": True, "max_correlation": 0, 
                    "conflicting_position": None, "reason": ""}
        
        if not held_symbols:
            return {"can_trade": True, "max_correlation": 0,
                    "conflicting_position": None, "reason": "No existing positions"}
        
        # Skip crypto pairs
        clean_symbol = symbol.replace("/", "")
        if any(x in symbol for x in ["USD", "BTC", "ETH"]) and len(symbol) > 5:
            return {"can_trade": True, "max_correlation": 0,
                    "conflicting_position": None, "reason": "Crypto excluded from correlation check"}
        
        try:
            end = pd.Timestamp.now()
            start = end - pd.Timedelta(days=config.CORRELATION_LOOKBACK_DAYS)
            
            # Fetch candidate price history
            candidate_bars = self._get_price_series(symbol, start, end)
            if candidate_bars is None or len(candidate_bars) < 20:
                return {"can_trade": True, "max_correlation": 0,
                        "conflicting_position": None, "reason": "Insufficient price history"}
            
            max_corr = 0
            worst_symbol = None
            
            for held in held_symbols:
                clean_held = held.replace("/", "")
                if clean_held == clean_symbol:
                    continue
                
                held_bars = self._get_price_series(clean_held, start, end)
                if held_bars is None or len(held_bars) < 20:
                    continue
                
                # Align on date index & compute correlation
                aligned = pd.concat([candidate_bars, held_bars], axis=1, join="inner").dropna()
                if len(aligned) < 15:
                    continue
                
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                
                if abs(corr) > max_corr:
                    max_corr = abs(corr)
                    worst_symbol = clean_held
            
            if max_corr >= config.CORRELATION_THRESHOLD and worst_symbol:
                log.info(f"Correlation guard: {symbol} correlated {max_corr:.2f} with {worst_symbol} (threshold={config.CORRELATION_THRESHOLD})")
                return {
                    "can_trade": False,
                    "max_correlation": round(max_corr, 3),
                    "conflicting_position": worst_symbol,
                    "reason": f"Correlated {max_corr:.2f} with {worst_symbol} (limit {config.CORRELATION_THRESHOLD})"
                }
            
            return {
                "can_trade": True,
                "max_correlation": round(max_corr, 3),
                "conflicting_position": worst_symbol,
                "reason": f"Max correlation {max_corr:.2f} (threshold {config.CORRELATION_THRESHOLD})"
            }
            
        except Exception as e:
            log.debug(f"Correlation check failed for {symbol}: {e}")
            return {"can_trade": True, "max_correlation": 0,
                    "conflicting_position": None, "reason": f"Check error: {e}"}
    
    def _get_price_series(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> Optional[pd.Series]:
        """Get daily close price series for a symbol, cached."""
        cache_key = f"{symbol}_{start.date()}_{end.date()}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            is_crypto = "/" in symbol or (any(c in symbol for c in ["USD", "BTC", "ETH"]) and len(symbol) > 5)
            if is_crypto:
                df = self.fetcher.get_crypto_bars(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            else:
                df = self.fetcher.get_stock_bars(symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            
            if df is None or df.empty:
                return None
            
            close = df.set_index("timestamp")["close"] if "timestamp" in df.columns else df["close"]
            close = close.astype(float)
            self._cache[cache_key] = close
            return close
            
        except Exception as e:
            log.debug(f"Price fetch for correlation {symbol}: {e}")
            return None
    
    def portfolio_heat(self, positions: list) -> dict:
        """
        Calculate total portfolio heat — sum of all position risk as % of equity.
        Returns {"total_pct": float, "count": int, "over_heat": bool}
        """
        try:
            total_market_value = sum(abs(float(p.market_value)) for p in positions) if positions else 0
            account = None
            total_equity = 0
            
            # Try to get account for equity
            try:
                from execution.alpaca_broker import AlpacaBroker
                broker = AlpacaBroker()
                account = broker.get_account()
                total_equity = account.get("equity", 0)
            except:
                total_equity = max(total_market_value * 2, 100000)
            
            heat_pct = total_market_value / total_equity if total_equity > 0 else 0
            
            return {
                "total_market_value": total_market_value,
                "total_equity": total_equity,
                "heat_pct": round(heat_pct, 4),
                "count": len(positions) if positions else 0,
                "over_heat": heat_pct > config.CLOSING_PORTFOLIO_HEAT_MAX
            }
        except Exception as e:
            log.debug(f"Portfolio heat calc failed: {e}")
            return {"heat_pct": 0, "count": 0, "over_heat": False}


correlation_guard = CorrelationGuard()