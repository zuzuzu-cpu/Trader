import json
import time
from pathlib import Path
import config
from utils.logger import get_logger

log = get_logger("sentinel.live_state")

class LiveState:
    """
    Singleton for tracking the exact current step of the bot and writing it 
    to a high-frequency JSON file that the web dashboard can poll.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiveState, cls).__new__(cls)
            cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.state_file = config.DATA_DIR / "live_status.json"
        self.state = {
            "step": "Initializing",
            "details": "Booting up Sentinel Autotrader...",
            "progress": 0,
            "timestamp": time.time(),
            "active_symbol": None
        }
        self._flush()

    def update(self, step: str = None, details: str = None, progress: int = None, active_symbol: str = None):
        """Updates the live state and flushes to disk."""
        if step is not None:
            self.state["step"] = step
        if details is not None:
            self.state["details"] = details
        if progress is not None:
            self.state["progress"] = min(100, max(0, progress))
        if active_symbol is not None:
            self.state["active_symbol"] = active_symbol
            
        self.state["timestamp"] = time.time()
        self._flush()

    def _flush(self):
        """Writes state to JSON atomically."""
        try:
            # Write to a temp file first to prevent the dashboard reading a half-written JSON
            temp_path = self.state_file.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(self.state, f)
            temp_path.replace(self.state_file)
        except Exception as e:
            log.debug(f"Failed to flush live state: {e}")

# Global instance
live_state = LiveState()
