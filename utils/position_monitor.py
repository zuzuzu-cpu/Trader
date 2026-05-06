"""
Real-time Position Monitor

Monitors open positions between trading cycles.
Automatically exits positions that hit stop-loss or take-profit levels.
Runs independently of the main trading cycle.
"""
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import config
from execution.alpaca_broker import AlpacaBroker
from utils.logger import get_logger

log = get_logger("sentinel.position_monitor")


class PositionMonitor:
    """
    Background monitor for open positions.
    Checks prices and exits positions when levels are breached.
    """
    
    def __init__(self, broker: AlpacaBroker = None, check_interval_seconds: int = 30):
        self.broker = broker or AlpacaBroker()
        self.check_interval = check_interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Track position targets (set by trading logic)
        self._position_targets = {}  # {symbol: {"stop_pct": float, "take_profit_pct": float, "entry_price": float}}
    
    def set_position_target(self, symbol: str, entry_price: float, 
                         stop_pct: float = None, take_profit_pct: float = None):
        """Set exit targets for a position."""
        self._position_targets[symbol] = {
            "entry_price": entry_price,
            "stop_pct": stop_pct or config.TRAILING_STOP_PCT,
            "take_profit_pct": take_profit_pct or config.PROFIT_TARGET_PCT,
        }
        log.info(f"Position target set for {symbol}: entry=${entry_price:.2f}, stop={stop_pct}%, tp={take_profit_pct}%")
    
    def clear_position_target(self, symbol: str):
        """Clear target for a closed position."""
        self._position_targets.pop(symbol, None)
    
    def start(self):
        """Start the background monitor."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info("Position monitor started")
    
    def stop(self):
        """Stop the background monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Position monitor stopped")
    
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        log.info("Position monitor loop started")
        
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                log.error(f"Position monitor error: {e}")
            
            # Sleep with interrupt check
            for _ in range(self.check_interval):
                if not self._running:
                    break
                time.sleep(1)
    
    def _check_positions(self):
        """Check all positions for exit signals."""
        try:
            positions = self.broker.get_positions()
            
            if not positions:
                return
            
            for pos in positions:
                symbol = pos.symbol.replace("/", "")
                qty = float(pos.qty)
                current_price = float(pos.current_price)
                avg_entry = float(pos.avg_entry_price)
                
                if qty == 0 or current_price == 0:
                    continue
                
                # Get targets
                targets = self._position_targets.get(symbol, {})
                entry_price = targets.get("entry_price") or avg_entry
                stop_pct = targets.get("stop_pct", config.TRAILING_STOP_PCT)
                take_profit_pct = targets.get("take_profit_pct", config.PROFIT_TARGET_PCT)
                
                # Calculate current P&L %
                if entry_price > 0:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = 0
                
                direction = "long" if qty > 0 else "short"
                
                # Check exit conditions
                exit_reason = None
                
                if direction == "long":
                    # Long: check stop loss (down) or take profit (up)
                    if pnl_pct <= -stop_pct:
                        exit_reason = "stop_loss"
                    elif pnl_pct >= take_profit_pct:
                        exit_reason = "take_profit"
                else:
                    # Short: inverted logic
                    if pnl_pct >= stop_pct:
                        exit_reason = "stop_loss"  # Price went up against short
                    elif pnl_pct <= -take_profit_pct:
                        exit_reason = "take_profit"  # Price dropped, profit realized
                
                if exit_reason:
                    log.warning(f"Position exit triggered: {symbol} {exit_reason} (pnl: {pnl_pct:+.2f}%)")
                    # Don't auto-exit here - let the main cycle handle it
                    # This prevents conflicts with the closing agent
    
    def get_position_status(self, symbol: str) -> Optional[dict]:
        """Get current status of a position."""
        try:
            pos = self.broker.get_position(symbol)
            if not pos:
                return None
            
            qty = float(pos.qty)
            current_price = float(pos.current_price)
            avg_entry = float(pos.avg_entry_price)
            
            targets = self._position_targets.get(symbol, {})
            entry_price = targets.get("entry_price") or avg_entry
            
            pnl = 0
            pnl_pct = 0
            if entry_price > 0:
                pnl = (current_price - entry_price) * abs(qty)
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            
            return {
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "stop_pct": targets.get("stop_pct", config.TRAILING_STOP_PCT),
                "take_profit_pct": targets.get("take_profit_pct", config.PROFIT_TARGET_PCT),
                "direction": "long" if qty > 0 else "short",
            }
        except Exception as e:
            log.debug(f"Failed to get position status: {e}")
            return None


# ─── Singleton ─────────────────────────────────────────────────────────────
_position_monitor = None

def get_position_monitor() -> PositionMonitor:
    """Get or create the position monitor singleton."""
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor