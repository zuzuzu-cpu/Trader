"""
Sentinel Autotrader V3.0 - Main Orchestrator

The central pipeline that coordinates all components:
1. Universe Scanner → discover tradable assets
2. Quant Engine → mathematical screening (15+ indicators)
3. News Hound → AI sentiment analysis
4. Skeptic → AI risk assessment
5. Portfolio Manager → final decision + position sizing
6. Alpaca Broker → execute paper trades

Features:
- Scheduled execution (configurable interval)
- Market hours awareness
- Dynamic universe scanning (200+ stocks, 30 ETFs, 10 crypto)
- Full audit trail via SQLite trade journal
- Max drawdown circuit breaker
- Graceful shutdown handling
"""
import os
import sys
import signal
import uuid
import time
from datetime import datetime, timedelta, timezone

import schedule

import config
from core.data_fetcher import DataFetcher
from core.universe_scanner import UniverseScanner
from core.quant_engine import QuantEngine
from agents.news_hound import NewsHound
from agents.skeptic import Skeptic
from agents.portfolio_mgr import PortfolioManager
from execution.alpaca_broker import AlpacaBroker
from utils.logger import get_logger, TradeJournal

log = get_logger("sentinel")

# ─── Graceful Shutdown ───────────────────────────────────────────────────────
_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    log.warning(f"Shutdown signal received ({signum}). Finishing current cycle...")
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── The Main Pipeline ──────────────────────────────────────────────────────

def run_cycle():
    """
    Executes one full trading cycle:
    Scan → Filter → Analyze → Decide → Execute
    """
    global _shutdown
    cycle_id = str(uuid.uuid4())[:8]
    log.info(f"{'='*60}")
    log.info(f"CYCLE {cycle_id} STARTED at {datetime.now()}")
    log.info(f"{'='*60}")

    # Initialize all components (shared fetcher for caching efficiency)
    fetcher = DataFetcher()
    scanner = UniverseScanner()
    quant = QuantEngine(fetcher=fetcher)
    broker = AlpacaBroker()
    news_hound = NewsHound(fetcher=fetcher)
    skeptic = Skeptic(fetcher=fetcher, broker=broker)
    portfolio_mgr = PortfolioManager()
    journal = TradeJournal()

    journal.start_cycle(cycle_id)
    stats = {"universe": 0, "candidates": 0, "trades": 0, "skipped": 0, "errors": 0}

    try:
        # ─── Step 0: Account Snapshot ────────────────────────────────
        account = broker.get_account()
        equity = account["equity"]
        peak_equity = max(journal.get_peak_equity(), equity)

        log.info(
            f"Account: equity=${equity:,.2f} | "
            f"buying_power=${account['buying_power']:,.2f} | "
            f"cash=${account['cash']:,.2f}"
        )

        # Check drawdown circuit breaker
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown > config.MAX_DRAWDOWN_PCT:
                log.warning(
                    f"CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds "
                    f"{config.MAX_DRAWDOWN_PCT:.0%} limit. Skipping cycle."
                )
                journal.end_cycle(cycle_id, 0, 0, 0, 0, 0)
                return

        # ─── Step 1: Universe Scanning ───────────────────────────────
        log.info("Step 1: Scanning trading universe...")
        universe = scanner.get_full_universe(max_stocks=200)
        stats["universe"] = sum(len(v) for v in universe.values())

        # ─── Step 2: Quantitative Filtering ──────────────────────────
        log.info("Step 2: Running quantitative screening...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        candidates = []

        # Screen stocks
        for i, symbol in enumerate(universe["stocks"]):
            if _shutdown:
                break
            try:
                result = quant.evaluate_stock(symbol, start_str, end_str)
                if result["score"] >= config.QUANT_PASS_SCORE:
                    candidates.append(result)
                    log.debug(f"  PASS: {symbol} score={result['score']:.0f} ({result['reason']})")
            except Exception as e:
                log.debug(f"  Error screening {symbol}: {e}")
                stats["errors"] += 1

            # Progress log every 50 stocks
            if (i + 1) % 50 == 0:
                log.info(f"  Screened {i+1}/{len(universe['stocks'])} stocks, {len(candidates)} candidates so far")

        # Screen ETFs
        for symbol in universe["etfs"]:
            if _shutdown:
                break
            try:
                result = quant.evaluate_etf(symbol, start_str, end_str)
                if result["score"] >= config.QUANT_PASS_SCORE:
                    candidates.append(result)
                    log.debug(f"  PASS: {symbol} score={result['score']:.0f}")
            except Exception as e:
                stats["errors"] += 1

        # Screen crypto
        for symbol in universe["crypto"]:
            if _shutdown:
                break
            try:
                result = quant.evaluate_crypto(symbol, start_str, end_str)
                if result["score"] >= config.QUANT_PASS_SCORE:
                    candidates.append(result)
                    log.debug(f"  PASS: {symbol} score={result['score']:.0f}")
            except Exception as e:
                stats["errors"] += 1

        # Sort by score (best first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        stats["candidates"] = len(candidates)

        log.info(
            f"Quant screening complete: {stats['candidates']} candidates "
            f"from {stats['universe']} assets (threshold: {config.QUANT_PASS_SCORE})"
        )

        if not candidates:
            log.info("No candidates passed quantitative filter. Cycle complete.")
            journal.end_cycle(cycle_id, stats["universe"], 0, 0, 0, stats["errors"])
            return

        # Log top 10 candidates
        log.info("Top candidates:")
        for c in candidates[:10]:
            log.info(f"  {c['symbol']:8s} | Score: {c['score']:5.1f} | {c['reason']}")

        # ─── Steps 3-5: AI Swarm Processing ─────────────────────────
        # Only process the top N candidates to conserve API calls
        max_ai_candidates = min(len(candidates), 15)
        log.info(f"\nStep 3-5: AI Swarm analyzing top {max_ai_candidates} candidates...")

        for q_result in candidates[:max_ai_candidates]:
            if _shutdown:
                break

            symbol = q_result["symbol"]
            asset_type = q_result["asset_type"]

            log.info(f"\n{'─'*40}")
            log.info(f"Analyzing: {symbol} (Score: {q_result['score']:.0f})")

            try:
                # ─── Agent 1: News Hound ─────────────────────────────
                log.info(f"  Agent 1 (News Hound): Analyzing sentiment...")
                sentiment = news_hound.analyze_sentiment(symbol, asset_type)
                log.info(
                    f"  Sentiment: score={sentiment['score']}, "
                    f"confidence={sentiment['confidence']:.2f}, "
                    f"events={sentiment['events']}"
                )

                # ─── Agent 2: The Skeptic ────────────────────────────
                log.info(f"  Agent 2 (Skeptic): Evaluating risk...")
                risk = skeptic.evaluate_risk(symbol, asset_type, q_result, sentiment)
                log.info(
                    f"  Risk: grade={risk['grade']}, "
                    f"score={risk['score']}, "
                    f"flags={risk['flags']}"
                )

                # ─── Agent 3: Portfolio Manager ──────────────────────
                log.info(f"  Agent 3 (Portfolio Mgr): Making final decision...")
                # Refresh account state
                account = broker.get_account()
                equity = account["equity"]

                verdict = portfolio_mgr.decide(
                    symbol, asset_type, q_result, sentiment, risk,
                    equity, peak_equity
                )

                # ─── Log the decision ────────────────────────────────
                journal.log_decision(
                    cycle_id, symbol, asset_type,
                    q_result["score"], q_result["reason"],
                    sentiment["score"], risk["grade"],
                    verdict["confidence"],
                    "TRADE" if verdict["should_trade"] else "SKIP",
                    verdict["reasoning"]
                )

                # ─── Step 5: Execute ─────────────────────────────────
                if verdict["should_trade"]:
                    log.info(
                        f"  ★ EXECUTING: {symbol} | "
                        f"Confidence: {verdict['confidence']:.1f}% | "
                        f"Notional: ${verdict['notional']:.2f} | "
                        f"Stop: {verdict['trailing_stop_pct']:.1f}%"
                    )

                    exec_result = broker.execute_trade(
                        symbol,
                        verdict["notional"],
                        verdict["trailing_stop_pct"]
                    )

                    if exec_result["status"] == "complete":
                        stats["trades"] += 1
                        journal.log_trade(
                            cycle_id, symbol, asset_type, "BUY",
                            verdict["notional"], exec_result["buy_order_id"],
                            q_result["score"], sentiment["score"],
                            risk["grade"], verdict["confidence"],
                            verdict["reasoning"]
                        )
                        log.info(f"  ✓ Trade complete: {symbol} filled @ ${exec_result['fill_price']:.2f}")
                    else:
                        stats["errors"] += 1
                        log.warning(f"  ✗ Trade failed: {symbol} → {exec_result['status']}")
                else:
                    stats["skipped"] += 1
                    log.info(f"  ○ Skipped: {verdict['reasoning'][:100]}")

            except Exception as e:
                stats["errors"] += 1
                log.error(f"  Error processing {symbol}: {e}", exc_info=True)

        # ─── End of Cycle: Portfolio Snapshot ─────────────────────────
        account = broker.get_account()
        positions = broker.get_positions()

        total_pl = account["equity"] - float(os.getenv("INITIAL_EQUITY", account["equity"]))
        total_pl_pct = (total_pl / float(os.getenv("INITIAL_EQUITY", account["equity"]))) * 100

        journal.log_portfolio_snapshot(
            cycle_id, account["equity"], account["buying_power"],
            account["cash"], len(positions), total_pl, total_pl_pct,
            (peak_equity - account["equity"]) / peak_equity if peak_equity > 0 else 0
        )

        journal.end_cycle(
            cycle_id, stats["universe"], stats["candidates"],
            stats["trades"], stats["skipped"], stats["errors"]
        )

    except Exception as e:
        log.error(f"Critical error in cycle {cycle_id}: {e}", exc_info=True)
        stats["errors"] += 1
        try:
            journal.end_cycle(cycle_id, stats["universe"], stats["candidates"],
                              stats["trades"], stats["skipped"], stats["errors"])
        except Exception:
            pass

    # ─── Summary ─────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"CYCLE {cycle_id} COMPLETE")
    log.info(
        f"Universe: {stats['universe']} | Candidates: {stats['candidates']} | "
        f"Trades: {stats['trades']} | Skipped: {stats['skipped']} | "
        f"Errors: {stats['errors']}"
    )
    log.info(f"{'='*60}\n")


# ─── Scheduler ───────────────────────────────────────────────────────────────

def main():
    """
    Entry point. Runs the trading cycle on a schedule.
    """
    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║       SENTINEL AUTOTRADER V3.0 — STARTING           ║")
    log.info("║       Multi-Agent AI Paper Trading Swarm             ║")
    log.info("╚═══════════════════════════════════════════════════════╝")
    log.info(f"Alpaca Base URL: {config.ALPACA_BASE_URL}")
    log.info(f"DeepSeek Model: {config.DEEPSEEK_MODEL}")
    log.info(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}%")
    log.info(f"Max Position: {config.MAX_POSITION_PCT:.1%}")
    log.info(f"Scan Interval: {config.SCAN_INTERVAL_MINUTES} minutes")
    log.info(f"Max Drawdown Limit: {config.MAX_DRAWDOWN_PCT:.0%}")

    # Validate API keys
    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        log.error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")
        sys.exit(1)
    if not config.DEEPSEEK_API_KEY:
        log.error("Missing DEEPSEEK_API_KEY in .env")
        sys.exit(1)

    # Run immediately on startup
    log.info("Running initial cycle...")
    run_cycle()

    # Schedule recurring cycles
    schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(run_cycle)
    log.info(f"Scheduled: every {config.SCAN_INTERVAL_MINUTES} minutes")

    # Main loop
    while not _shutdown:
        schedule.run_pending()
        time.sleep(10)

    log.info("Sentinel Autotrader shut down gracefully.")


if __name__ == "__main__":
    main()
