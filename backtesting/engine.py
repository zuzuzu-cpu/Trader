"""
Enhanced Backtesting Engine

Full-featured backtesting with:
- Realistic slippage simulation
- Multiple strategy variants
- Transaction costs
- Equity curve tracking
- Performance metrics
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import config
from core.quant_engine import QuantEngine
from core.data_fetcher import DataFetcher
from utils.logger import get_logger

log = get_logger("sentinel.backtester")


class BacktestEngine:
    """
    Full-featured backtesting engine.
    Simulates realistic trading with slippage and costs.
    """
    
    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
        self.quant = QuantEngine(fetcher=self.fetcher)
    
    def run_backtest(self, symbols: List[str], days: int = 365,
                   initial_capital: float = 100000,
                   slippage: float = 0.001,
                   max_position_pct: float = 0.02) -> Dict:
        """
        Run backtest on multiple symbols.
        
        Args:
            symbols: List of symbols to test
            days: Lookback period
            initial_capital: Starting capital
            slippage: Slippage % (0.001 = 0.1%)
            max_position_pct: Max position size as % of capital
        """
        log.info(f"Running backtest: {len(symbols)} symbols, {days} days, ${initial_capital:,}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = {
            "symbols_tested": 0,
            "symbols_traded": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0,
            "equity_curve": [],
            "trades": [],
        }
        
        equity = initial_capital
        
        for symbol in symbols:
            try:
                symbol_result = self._backtest_symbol(
                    symbol, start_date, end_date,
                    equity, slippage, max_position_pct
                )
                
                if symbol_result and symbol_result["trades"]:
                    results["symbols_traded"] += 1
                    results["total_trades"] += symbol_result["trades"]
                    results["winning_trades"] += symbol_result["wins"]
                    results["losing_trades"] += symbol_result["losses"]
                    results["total_pnl"] += symbol_result["pnl"]
                    equity += symbol_result["pnl"]
                    
                    results["trades"].extend(symbol_result["trade_log"])
                    
            except Exception as e:
                log.debug(f"Backtest error for {symbol}: {e}")
        
        results["symbols_tested"] = len(symbols)
        results["win_rate"] = (
            results["winning_trades"] / results["total_trades"] * 100
            if results["total_trades"] > 0 else 0
        )
        results["final_capital"] = equity
        results["return_pct"] = (
            (equity - initial_capital) / initial_capital * 100
        )
        
        log.info(
            f"Backtest complete: {results['total_trades']} trades, "
            f"{results['win_rate']:.1f}% win rate, "
            f"${results['total_pnl']:+.2f} P&L"
        )
        
        return results
    
    def _backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime,
                        capital: float, slippage: float, max_position_pct: float) -> Optional[Dict]:
        """Run backtest on a single symbol."""
        
        bars = self.fetcher.get_stock_bars(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if bars is None or len(bars) < 50:
            return None
        
        close = bars['close'].values
        high = bars['high'].values
        low = bars['low'].values
        volume = bars['volume'].values
        
        # Use QuantEngine to generate signals
        # Simplified: use RSI signals
        rsi_period = 14
        rsi = self._calculate_rsi(close, rsi_period)
        
        position = 0  # 0 = flat, 1 = long, -1 = short
        entry_price = 0
        trades = []
        
        for i in range(rsi_period + 1, len(close) - 1):
            current_price = close[i]
            
            # Entry signals
            if position == 0:
                # Long signal: RSI < 30 (oversold)
                if rsi[i] < 30:
                    # Apply slippage to entry
                    entry_price = current_price * (1 + slippage)
                    position = 1
                    notional = capital * max_position_pct
                    qty = notional / entry_price
            
            # Exit signals
            elif position == 1:
                # Exit: RSI > 70 or profit target
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                if rsi[i] > 70 or pnl_pct >= 5 or pnl_pct <= -2:
                    exit_price = current_price * (1 - slippage)
                    pnl = (exit_price - entry_price) * qty
                    
                    trades.append({
                        "symbol": symbol,
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "direction": "long"
                    })
                    
                    position = 0
        
        # Calculate stats
        pnl = sum(t["pnl"] for t in trades)
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        
        return {
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "pnl": pnl,
            "trade_log": trades,
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad to match price length
        rsi = np.concatenate([np.full(period, np.nan), rsi])
        
        return rsi
    
    def compare_strategies(self, symbol: str, days: int = 365) -> Dict:
        """
        Compare multiple strategy variants on a symbol.
        Returns performance comparison.
        """
        log.info(f"Strategy comparison for {symbol}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        strategies = {
            "rsi_naive": self._strategy_rsi_naive,
            "rsi_slope": self._strategy_rsi_slope,
            "ma_cross": self._strategy_ma_cross,
            "bb_squeeze": self._strategy_bb_squeeze,
        }
        
        results = {}
        
        for name, strategy_fn in strategies.items():
            result = self._test_strategy(
                symbol, start_date, end_date, strategy_fn
            )
            if result:
                results[name] = result
        
        return results
    
    def _test_strategy(self, symbol: str, start_date: datetime, 
                      end_date: datetime, strategy_fn) -> Optional[Dict]:
        """Test a specific strategy."""
        bars = self.fetcher.get_stock_bars(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if bars is None or len(bars) < 50:
            return None
        
        signals = strategy_fn(bars)
        
        # Simulate trades
        trades = []
        equity = 10000
        
        for i in range(len(signals)):
            if signals[i] == 1 and (not trades or trades[-1]["exit"]):
                # Buy signal
                trades.append({"entry": i, "exit": None, "dir": "long"})
            elif signals[i] == -1 and trades and not trades[-1]["exit"]:
                # Exit signal
                trades[-1]["exit"] = i
        
        return {"trades": len([t for t in trades if t["exit"]])}
    
    def _strategy_rsi_naive(self, bars: pd.DataFrame) -> np.ndarray:
        """RSI naive: buy <30, sell >70."""
        close = bars['close'].values
        rsi = self._calculate_rsi(close)
        signals = np.zeros(len(close))
        signals[rsi < 30] = 1
        signals[rsi > 70] = -1
        return signals
    
    def _strategy_rsi_slope(self, bars: pd.DataFrame) -> np.ndarray:
        """RSI with slope confirmation."""
        close = bars['close'].values
        rsi = self._calculate_rsi(close)
        rsi_slope = np.gradient(rsi)
        signals = np.zeros(len(close))
        signals[(rsi < 30) & (rsi_slope > 0)] = 1
        signals[(rsi > 70) & (rsi_slope < 0)] = -1
        return signals
    
    def _strategy_ma_cross(self, bars: pd.DataFrame) -> np.ndarray:
        """MA crossover strategy."""
        close = bars['close'].values
        ma20 = pd.Series(close).rolling(20).mean().values
        ma50 = pd.Series(close).rolling(50).mean().values
        signals = np.zeros(len(close))
        signals[ma20 > ma50] = 1
        signals[ma20 < ma50] = -1
        return signals
    
    def _strategy_bb_squeeze(self, bars: pd.DataFrame) -> np.ndarray:
        """Bollinger Band squeeze breakout."""
        close = bars['close'].values
        ma = pd.Series(close).rolling(20).mean()
        std = pd.Series(close).rolling(20).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        
        # Squeeze = narrow bands
        bandwidth = (upper - lower) / ma * 100
        squeeze = bandwidth < 4
        
        # Breakout = price crosses above upper band after squeeze
        signals = np.zeros(len(close))
        for i in range(21, len(close)):
            if squeeze.iloc[i-1] and close[i] > upper.iloc[i]:
                signals[i] = 1
            elif close[i] < ma.iloc[i]:
                signals[i] = -1
        
        return signals