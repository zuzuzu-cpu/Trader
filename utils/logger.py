"""
Structured logging and trade journal for Sentinel Autotrader.
Uses Python's built-in logging with JSON formatting for machine-readable logs
and a SQLite database for persistent trade history and performance tracking.
"""
import logging
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import config


# ─── JSON Log Formatter ─────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Outputs log records as single-line JSON objects for easy parsing."""

    def format(self, record):
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Attach any extra structured data
        for key in ("symbol", "score", "action", "confidence", "notional", "order_id", "cycle_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def get_logger(name: str = "sentinel") -> logging.Logger:
    """Returns a configured logger with console + file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(module)-18s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # File handler (JSON, machine-readable)
    log_file = config.LOG_DIR / f"sentinel_{datetime.now().strftime('%Y%m%d')}.jsonl"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    return logger


# ─── SQLite Trade Journal ────────────────────────────────────────────────────

class TradeJournal:
    """
    Persistent trade journal using SQLite.
    Records every trade decision, execution, and portfolio snapshot.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle_id TEXT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT,
                    side TEXT NOT NULL,
                    notional REAL,
                    qty REAL,
                    fill_price REAL,
                    order_id TEXT,
                    quant_score REAL,
                    sentiment_score REAL,
                    risk_grade TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    status TEXT DEFAULT 'submitted'
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle_id TEXT,
                    symbol TEXT NOT NULL,
                    asset_type TEXT,
                    quant_score REAL,
                    quant_reason TEXT,
                    sentiment_score REAL,
                    risk_grade TEXT,
                    confidence REAL,
                    decision TEXT NOT NULL,
                    reasoning TEXT
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle_id TEXT,
                    equity REAL,
                    buying_power REAL,
                    cash REAL,
                    positions_count INTEGER,
                    total_pl REAL,
                    total_pl_pct REAL,
                    max_drawdown_pct REAL
                );

                CREATE TABLE IF NOT EXISTS cycle_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id TEXT UNIQUE,
                    start_time TEXT,
                    end_time TEXT,
                    universe_size INTEGER,
                    candidates_found INTEGER,
                    trades_executed INTEGER,
                    trades_skipped INTEGER,
                    errors INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );
            """)

    def log_decision(self, cycle_id: str, symbol: str, asset_type: str,
                     quant_score: float, quant_reason: str,
                     sentiment_score: float, risk_grade: str,
                     confidence: float, decision: str, reasoning: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO decisions
                (timestamp, cycle_id, symbol, asset_type, quant_score, quant_reason,
                 sentiment_score, risk_grade, confidence, decision, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), cycle_id, symbol, asset_type,
                quant_score, quant_reason, sentiment_score, risk_grade,
                confidence, decision, reasoning
            ))

    def log_trade(self, cycle_id: str, symbol: str, asset_type: str,
                  side: str, notional: float, order_id: str,
                  quant_score: float, sentiment_score: float,
                  risk_grade: str, confidence: float, reasoning: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades
                (timestamp, cycle_id, symbol, asset_type, side, notional, order_id,
                 quant_score, sentiment_score, risk_grade, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), cycle_id, symbol, asset_type,
                side, notional, order_id, quant_score, sentiment_score,
                risk_grade, confidence, reasoning
            ))

    def log_portfolio_snapshot(self, cycle_id: str, equity: float,
                               buying_power: float, cash: float,
                               positions_count: int, total_pl: float,
                               total_pl_pct: float, max_drawdown_pct: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO portfolio_snapshots
                (timestamp, cycle_id, equity, buying_power, cash,
                 positions_count, total_pl, total_pl_pct, max_drawdown_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), cycle_id,
                equity, buying_power, cash, positions_count,
                total_pl, total_pl_pct, max_drawdown_pct
            ))

    def start_cycle(self, cycle_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cycle_log (cycle_id, start_time, status)
                VALUES (?, ?, 'running')
            """, (cycle_id, datetime.now(timezone.utc).isoformat()))

    def end_cycle(self, cycle_id: str, universe_size: int,
                  candidates_found: int, trades_executed: int,
                  trades_skipped: int, errors: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE cycle_log SET
                    end_time = ?, universe_size = ?, candidates_found = ?,
                    trades_executed = ?, trades_skipped = ?, errors = ?,
                    status = 'completed'
                WHERE cycle_id = ?
            """, (
                datetime.now(timezone.utc).isoformat(), universe_size,
                candidates_found, trades_executed, trades_skipped,
                errors, cycle_id
            ))

    def get_peak_equity(self) -> float:
        """Returns the highest equity ever recorded for drawdown calculation."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT MAX(equity) FROM portfolio_snapshots"
            ).fetchone()
            return result[0] if result and result[0] else 0.0
