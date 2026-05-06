import os
import sqlite3
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, jsonify

# Point to the SQLite database
# When running via Docker, this is mounted as a volume
DB_PATH = os.environ.get("DB_PATH", "../data/sentinel.db")

app = Flask(__name__)

def get_db_connection():
    # Adding a timeout of 15 seconds to prevent "database is locked" errors
    # if the main bot thread is actively writing to the database.
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/live_status")
def api_live_status():
    """Returns the live status of the bot from the JSON state file."""
    import json
    status_file = os.path.join(os.path.dirname(DB_PATH), "live_status.json")
    
    try:
        if os.path.exists(status_file):
            # Check if file is stale (> 10 minutes)
            if os.path.getmtime(status_file) < (datetime.now(timezone.utc).timestamp() - 600):
                return jsonify({
                    "step": "Sleeping",
                    "details": "Bot is currently idle or sleeping between cycles.",
                    "progress": 100,
                    "active_symbol": None,
                    "stale": True
                })
                
            with open(status_file, "r") as f:
                return jsonify(json.load(f))
    except Exception as e:
        pass

    return jsonify({
        "step": "Unknown",
        "details": "Waiting for bot to start or post status...",
        "progress": 0,
        "active_symbol": None
    })

@app.route("/api/summary")
def api_summary():
    """Returns top-level portfolio metrics."""
    try:
        conn = get_db_connection()
        snap = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if not snap:
            return jsonify({"status": "no_data"})

        cycle = conn.execute(
            "SELECT * FROM cycle_log ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # Count today's trades
        today = datetime.now(timezone.utc).date().isoformat()
        trades_today = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?",
            (f"{today}%",)
        ).fetchone()[0]

        return jsonify({
            "status": "ok",
            "equity": snap["equity"],
            "buying_power": snap["buying_power"],
            "cash": snap["cash"],
            "positions_count": snap["positions_count"],
            "total_pl": snap["total_pl"],
            "total_pl_pct": snap["total_pl_pct"],
            "max_drawdown_pct": snap["max_drawdown_pct"] * 100,
            "last_update": snap["timestamp"],
            "trades_today": trades_today,
            "status_text": cycle["status"] if cycle else "Unknown",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/api/trades")
def api_trades():
    """Returns recent trades."""
    try:
        conn = get_db_connection()
        trades = conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT 50"
        ).fetchall()
        return jsonify([dict(t) for t in trades])
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/decisions")
def api_decisions():
    """Returns recent AI decisions."""
    try:
        conn = get_db_connection()
        decisions = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT 100"
        ).fetchall()
        return jsonify([dict(d) for d in decisions])
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/equity_chart")
def api_equity_chart():
    """Returns equity curve data for the chart."""
    try:
        conn = get_db_connection()
        # Get snapshots from the last 7 days, max 100 points
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        snaps = conn.execute(
            "SELECT timestamp, equity FROM portfolio_snapshots WHERE timestamp > ? ORDER BY id ASC",
            (cutoff,)
        ).fetchall()

        # Downsample if too many points for the chart
        step = max(1, len(snaps) // 100)
        data = []
        for i in range(0, len(snaps), step):
            snap = snaps[i]
            # Convert ISO string to JS timestamp
            dt = datetime.fromisoformat(snap["timestamp"])
            data.append({
                "x": dt.timestamp() * 1000,
                "y": snap["equity"]
            })

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
