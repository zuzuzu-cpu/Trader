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

                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT UNIQUE,
                    pnl REAL,
                    pnl_pct REAL,
                    hold_minutes INTEGER,
                    exit_reason TEXT,
                    closed_at TEXT
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

    # ─── ML Feedback Loop: Trade History ─────────────────────────────

    def get_trade_history(self, limit: int = 20) -> list[dict]:
        """
        Returns the last N trades with outcome data for ML feedback.
        Used to inject trade memory into the DeepSeek Reasoner prompt.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT t.symbol, t.side, t.notional, t.confidence,
                       t.quant_score, t.sentiment_score, t.risk_grade,
                       t.reasoning, t.timestamp,
                       o.pnl, o.pnl_pct, o.hold_minutes, o.exit_reason
                FROM trades t
                LEFT JOIN trade_outcomes o ON t.order_id = o.order_id
                WHERE t.status != 'failed'
                ORDER BY t.id DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_trade_stats(self) -> dict:
        """
        Computes aggregate trade statistics for Kelly Criterion.
        Returns win_rate, avg_win, avg_loss, total_trades, profit_factor.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all closed trades with P&L
            rows = conn.execute("""
                SELECT o.pnl, o.pnl_pct
                FROM trade_outcomes o
                WHERE o.pnl IS NOT NULL
            """).fetchall()

            if not rows:
                return {
                    "total_trades": 0, "win_rate": 0.0,
                    "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                    "profit_factor": 0.0, "kelly_pct": 0.0,
                }

            wins = [r["pnl_pct"] for r in rows if r["pnl_pct"] and r["pnl_pct"] > 0]
            losses = [abs(r["pnl_pct"]) for r in rows if r["pnl_pct"] and r["pnl_pct"] < 0]

            total = len(rows)
            win_rate = len(wins) / total if total > 0 else 0
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.001  # Avoid div/0

            # Profit factor = gross wins / gross losses
            gross_wins = sum(wins) if wins else 0
            gross_losses = sum(losses) if losses else 0.001
            profit_factor = gross_wins / gross_losses

            # Kelly Criterion: f* = (bp - q) / b
            # b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss if avg_loss > 0 else 1
            p = win_rate
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, kelly)  # Never go negative

            return {
                "total_trades": total,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(win_rate, 4),
                "avg_win_pct": round(avg_win, 4),
                "avg_loss_pct": round(avg_loss, 4),
                "profit_factor": round(profit_factor, 4),
                "kelly_pct": round(kelly, 4),
            }

    def log_trade_outcome(self, order_id: str, pnl: float, pnl_pct: float,
                          hold_minutes: int, exit_reason: str):
        """Records the outcome of a closed trade for ML feedback."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_outcomes
                (order_id, pnl, pnl_pct, hold_minutes, exit_reason, closed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (order_id, pnl, pnl_pct, hold_minutes, exit_reason,
                  datetime.now(timezone.utc).isoformat()))

