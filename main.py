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
from core.live_stream import LiveStreamManager
from core.premarket_scanner import PreMarketScanner
from core.market_regime import get_market_regime
from agents.news_hound import NewsHound
from agents.skeptic import Skeptic
from agents.portfolio_mgr import PortfolioManager
from agents.insider_tracker import InsiderTracker
from agents.options_flow import OptionsFlow
from agents.closing_agent import ClosingAgent
from agents.macro_agent import macro_agent
from execution.alpaca_broker import AlpacaBroker
from utils.logger import get_logger, TradeJournal
from utils.async_executor import AsyncExecutor
from utils.telegram import TelegramAlert
from utils.live_state import live_state
from utils.correlation_guard import correlation_guard

log = get_logger("sentinel")
telegram = TelegramAlert()

# ─── V5: Global live stream manager (singleton, runs across cycles) ──────────
live_stream = LiveStreamManager()

# ─── V5: Global Position Monitor (runs between cycles) ─────────────────────
from utils.position_monitor import PositionMonitor
position_monitor = PositionMonitor(broker=None, check_interval_seconds=30)

# ─── Graceful Shutdown ───────────────────────────────────────────────────────
_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    if not _shutdown:
        log.warning(f"Shutdown signal received ({signum}). Finishing current cycle...")
        telegram.shutdown()
        position_monitor.stop()
        _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Position Outcome Tracker (ML Feedback) ─────────────────────────────────

def _track_position_outcomes(broker, journal):
    """
    Checks for closed positions and records their P&L to the trade_outcomes
    table. This feeds the ML feedback loop and Kelly Criterion.
    
    Uses Alpaca's closed positions data for real P&L calculation.
    """
    import sqlite3
    from datetime import datetime, timezone

    try:
        # Get currently held symbols
        positions = _get_cached_positions(broker)
        held_symbols = set()
        for p in positions:
            held_symbols.add(p.symbol.replace("/", ""))

        # Get recent trades that don't have outcomes yet
        with sqlite3.connect(str(journal.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            open_trades = conn.execute("""
                SELECT t.order_id, t.symbol, t.side, t.fill_price, t.notional,
                       t.timestamp, t.qty
                FROM trades t
                LEFT JOIN trade_outcomes o ON t.order_id = o.order_id
                WHERE o.order_id IS NULL
                  AND t.status != 'failed'
                  AND t.fill_price IS NOT NULL
                  AND t.fill_price > 0
            """).fetchall()

        # Get closed positions P&L from Alpaca
        closed_positions = broker.get_closed_positions_pnl()
        closed_pnl_map = {cp["symbol"]: cp for cp in closed_positions}

        for trade in open_trades:
            clean_symbol = trade["symbol"].replace("/", "")
            
            if clean_symbol not in held_symbols:
                # Position has been closed — calculate real P&L
                entry_price = trade["fill_price"]
                entry_qty = float(trade["qty"]) if trade["qty"] else 0
                notional = trade["notional"] or 0
                
                # Try to get real P&L from Alpacaclosed positions
                pnl = 0.0
                pnl_pct = 0.0
                
                # Match by symbol
                for cp in closed_positions:
                    if cp["symbol"].replace("/", "") == clean_symbol:
                        pnl = cp.get("pnl", 0)
                        pnl_pct = cp.get("pnl_pct", 0)
                        break
                
                # If no exact match, estimate from entry price
                if pnl == 0 and entry_qty > 0 and entry_price > 0:
                    # Estimate: we don't know exact exit, so mark for ML feedback
                    # Real P&L will be reflected in equity changes over time
                    pnl = 0.0
                    pnl_pct = 0.0

                entry_time = datetime.fromisoformat(trade["timestamp"].replace("Z", "+00:00")) \
                    if trade["timestamp"] else datetime.now(timezone.utc)
                hold_minutes = int((datetime.now(timezone.utc) - entry_time).total_seconds() / 60)

                # Determine exit reason
                exit_reason = "closed"
                if pnl > 0:
                    exit_reason = "take_profit"
                elif pnl < 0:
                    exit_reason = "stop_loss"
                
                journal.log_trade_outcome(
                    order_id=trade["order_id"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_minutes=hold_minutes,
                    exit_reason=exit_reason
                )
                log.info(
                    f"Position outcome: {trade['symbol']} "
                    f"(P&L: ${pnl:+.2f} / {pnl_pct:+.1f}%, held {hold_minutes}min)"
                )

    except Exception as e:
        log.debug(f"Position tracking error: {e}")


# ─── Position caching (avoids redundant API calls per cycle) ─────────────────
_position_cache = None
_position_cache_ts = 0

def _get_cached_positions(broker, max_age_sec: int = 30):
    """Returns cached positions if fresh, otherwise fetches from broker."""
    global _position_cache, _position_cache_ts
    now = time.time()
    if _position_cache is not None and (now - _position_cache_ts) < max_age_sec:
        return _position_cache
    _position_cache = broker.get_positions()
    _position_cache_ts = now
    return _position_cache


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
    journal = TradeJournal()
    portfolio_mgr = PortfolioManager(journal=journal)
    async_exec = AsyncExecutor()
    # V5 agents
    insider_tracker = InsiderTracker()
    options_flow = OptionsFlow()
    closing_agent = ClosingAgent()
    premarket_scanner = PreMarketScanner(fetcher=fetcher, broker=broker)

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

        # ─── Step 0.1: Market Regime Detection ────────────────────────
        market_regime = get_market_regime()
        regime = market_regime.get_regime(cycle_id)
        log.info(f"Market regime: {regime['regime']} | {regime['guidance'][:80]}...")

        # ─── Step 0.2: Macro Context (once daily) ─────────────────────
        macro_context = macro_agent.get_macro_context()

        # Check drawdown circuit breaker
        if peak_equity > 0:
            drawdown = (peak_equity - equity) / peak_equity
            if drawdown > config.MAX_DRAWDOWN_PCT:
                log.warning(
                    f"CIRCUIT BREAKER: Drawdown {drawdown:.1%} exceeds "
                    f"{config.MAX_DRAWDOWN_PCT:.0%} limit. Skipping cycle."
                )
                telegram.circuit_breaker(drawdown, equity)
                journal.end_cycle(cycle_id, 0, 0, 0, 0, 0)
                return

        # ─── Step 0.5: Proactive Exit Management (4-Tier Closing Agent) ─
        if config.CLOSING_AGENT_ENABLED and not _shutdown:
            log.info("Step 0.5: Tiered Closing Agent evaluating open positions...")
            live_state.update(step="Step 0.5: Exit Management", details="Running tiered exit hierarchy...", progress=2)

            positions = _get_cached_positions(broker)
            verdicts = closing_agent.run(
                positions=positions,
                regime=regime,
                macro_context=macro_context,
                fetcher=fetcher,
                quant=quant,
                news_hound=news_hound,
                broker=broker,
            )

            for v in verdicts:
                if _shutdown: break
                symbol = v.get("symbol", "?")
                action = v["action"]
                reason = v.get("reason", "")
                tier = v.get("tier", "?")

                if action == "SELL_ALL":
                    log.info(f"Tier {tier} EXIT: Closing {symbol} — {reason}")
                    if broker.close_position(symbol):
                        telegram.closing_agent_alert(symbol, action, v.get("confidence", 0), 0, reason, "Sold 100%")
                        closing_agent.add_cooldown(symbol, reason)
                elif action == "SELL_PARTIAL":
                    pct = v.get("sell_pct", 0.5)
                    log.info(f"Tier {tier} EXIT: Partially closing {symbol} by {pct:.0%} — {reason}")
                    if broker.close_position_partial(symbol, pct):
                        telegram.closing_agent_alert(symbol, action, v.get("confidence", 0), 0, reason, f"Sold {pct:.0%}")

        # ─── Step 1: Universe Scanning ───────────────────────────────
        log.info("Step 1: Scanning trading universe...")
        live_state.update(step="Step 1: Universe Scanning", details="Fetching list of tradable assets...", progress=5)
        universe = scanner.get_full_universe(max_stocks=config.MAX_UNIVERSE_SIZE)
        
        # If the stock market is closed, do not bother scanning 7,000 stocks/ETFs.
        # Only trade crypto. Alpaca's API tells us the exact NY market status.
        if config.SKIP_CLOSED_MARKET and not broker.is_market_open():
            log.info("Market is currently CLOSED. Skipping Stock and ETF scans. Only scanning Crypto.")
            universe["stocks"] = []
            universe["etfs"] = []
            
        stats["universe"] = sum(len(v) for v in universe.values())

        # ─── Step 1.5: Batch Prefetching (Rate Limit Focus) ──────────
        # Warm up the cache for thousands of assets in optimized batches
        # This prevents 8,000 individual API calls during screening
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.LOOKBACK_DAYS)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        if not _shutdown:
            log.info("Warming up data cache for full universe...")
            live_state.update(details=f"Prefetching cache for {stats['universe']} assets...", progress=10)
            fetcher.prefetch_stock_bars(universe["stocks"], start_str, end_str)
            fetcher.prefetch_crypto_bars(universe["crypto"], start_str, end_str)

        # ─── V5: Start WebSocket Live Stream (first cycle only) ──────
        if not live_stream.is_running and not _shutdown:
            log.info("V5: Starting WebSocket live stream...")
            # Get currently held positions to ensure we track their live prices instantly
            open_positions = _get_cached_positions(broker)
            held_symbols = [p.symbol for p in open_positions] if open_positions else []
            
            # Merge held stocks with the top universe candidates (remove duplicates)
            stock_stream_list = list(set(held_symbols + universe["stocks"][:500]))
            
            live_stream.start(
                stock_symbols=stock_stream_list,
                crypto_symbols=universe["crypto"],
            )

        # ─── V5: Pre-Market Priority Queue ───────────────────────────
        priority_symbols = []
        if PreMarketScanner.is_premarket() and not _shutdown:
            log.info("V5: Pre-market window — scanning for gap movers...")
            pm_movers = premarket_scanner.scan_premarket_movers(universe["stocks"][:200])
            premarket_scanner.enqueue(pm_movers)
        priority_symbols = premarket_scanner.get_priority_queue()
        if priority_symbols:
            log.info(f"V5: {len(priority_symbols)} pre-market priority candidates")

        # ─── Step 2: Quantitative Filtering (ASYNC) ──────────────────
        log.info(f"Step 2: Running parallel quantitative screening ({config.MAX_WORKERS} workers)...")
        live_state.update(step="Step 2: Quant Screening", details=f"Calculating 15+ indicators across {stats['universe']} assets...", progress=20)

        candidates = []

        # Screen stocks in parallel
        if universe["stocks"] and not _shutdown:
            stock_results = async_exec.map_batched(
                quant.evaluate_stock, universe["stocks"],
                config.BATCH_SIZE, start_str, end_str
            )
            candidates.extend([r for r in stock_results if r.get("score", 0) >= config.QUANT_PASS_SCORE])
            stats["errors"] += len(universe["stocks"]) - len(stock_results)

        # Screen ETFs in parallel
        if universe["etfs"] and not _shutdown:
            etf_results = async_exec.map_batched(
                quant.evaluate_etf, universe["etfs"],
                config.BATCH_SIZE, start_str, end_str
            )
            candidates.extend([r for r in etf_results if r.get("score", 0) >= config.QUANT_PASS_SCORE])
            stats["errors"] += len(universe["etfs"]) - len(etf_results)

        # Screen crypto in parallel
        if universe["crypto"] and not _shutdown:
            crypto_results = async_exec.map_batched(
                quant.evaluate_crypto, universe["crypto"],
                config.BATCH_SIZE, start_str, end_str
            )
            candidates.extend([r for r in crypto_results if r.get("score", 0) >= config.QUANT_PASS_SCORE])
            stats["errors"] += len(universe["crypto"]) - len(crypto_results)

        # Sort by score (best first)
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
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

        for idx, q_result in enumerate(candidates[:max_ai_candidates]):
            if _shutdown:
                break

            symbol = q_result["symbol"]
            asset_type = q_result["asset_type"]

            log.info(f"\n{'─'*40}")
            log.info(f"Analyzing: {symbol} (Score: {q_result['score']:.0f})")
            
            # Calculate dynamic progress (from 30% to 90%)
            prog = 30 + int(60 * (idx / max_ai_candidates))
            live_state.update(step="Step 3: AI Swarm", details=f"Agents analyzing {symbol}...", progress=prog, active_symbol=symbol)

            try:
                # ─── Agent 1: News Hound ─────────────────────────────
                log.info(f"  Agent 1 (News Hound): Analyzing sentiment...")
                sentiment = news_hound.analyze_sentiment(symbol, asset_type)

                # V5: Blend social sentiment into news score (disabled — replaced by SEC/FRED data)
                social = {}

                log.info(
                    f"  Sentiment: score={sentiment['score']}, "
                    f"confidence={sentiment['confidence']:.2f}, "
                    f"events={sentiment['events']}"
                )

                # ─── Agent 2: The Skeptic ────────────────────────────
                log.info(f"  Agent 2 (Skeptic): Evaluating risk...")
                live_state.update(details=f"The Skeptic is evaluating risk for {symbol}...")
                risk = skeptic.evaluate_risk(symbol, asset_type, q_result, sentiment)
                log.info(
                    f"  Risk: grade={risk['grade']}, "
                    f"score={risk['score']}, "
                    f"flags={risk['flags']}"
                )

                # ─── V5: Insider Tracker (SEC Form 4) ────────────────
                insider = {}
                if asset_type != "crypto" and not _shutdown:
                    log.info(f"  V5 (Insider Tracker): Checking SEC Form 4...")
                    insider = insider_tracker.get_insider_score(symbol)
                    if insider.get("score", 0) != 0:
                        log.info(f"  Insider: {insider.get('summary', '')} (score={insider.get('score', 0):+d})")

                # ─── V5: Options Flow ────────────────────────────────
                options = {}
                if asset_type != "crypto" and not _shutdown:
                    log.info(f"  V5 (Options Flow): Analyzing put/call ratio...")
                    options = options_flow.get_options_score(symbol)
                    if options.get("score", 0) != 0:
                        log.info(f"  Options: {options.get('summary', '')} (score={options.get('score', 0):+d})")

                # ─── Agent 3: Portfolio Manager ──────────────────────
                log.info(f"  Agent 3 (Portfolio Mgr): Making final decision...")
                # Refresh account state
                account = broker.get_account()
                equity = account["equity"]

                # Correlation check before committing
                held_symbols = [p.symbol.replace("/", "") for p in _get_cached_positions(broker)]
                correlation_result = correlation_guard.check(symbol, held_symbols, direction=q_result.get("direction", "long"))
                if not correlation_result.get("can_trade", True):
                    log.info(f"Correlation guard: skipping {symbol} — {correlation_result.get('reason', '')}")

                verdict = portfolio_mgr.decide(
                    symbol, asset_type, q_result, sentiment, risk,
                    equity, peak_equity,
                    insider_result=insider,
                    options_result=options,
                    regime=regime,
                    macro_context=macro_context,
                    correlation_result=correlation_result,
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
                    direction = verdict.get("direction", "long")
                    side_label = "BUY" if direction == "long" else "SHORT"
                    
                    live_state.update(step="Step 5: Execution", details=f"Placing {side_label} order for {symbol}...", progress=95)

                    log.info(
                        f"  ★ EXECUTING {side_label}: {symbol} | "
                        f"Confidence: {verdict['confidence']:.1f}% | "
                        f"Notional: ${verdict['notional']:.2f} | "
                        f"Stop: {verdict['trailing_stop_pct']:.1f}%"
                    )

                    exec_result = broker.execute_trade(
                        symbol,
                        verdict["notional"],
                        verdict["trailing_stop_pct"],
                        direction=direction,
                        asset_type=asset_type
                    )

                    if exec_result["status"] == "complete":
                        stats["trades"] += 1
                        journal.log_trade(
                            cycle_id, symbol, asset_type, side_label,
                            verdict["notional"], exec_result.get("order_id", ""),
                            q_result["score"], sentiment["score"],
                            risk["grade"], verdict["confidence"],
                            verdict["reasoning"]
                        )
                        # V5: Live Stream dynamic subscription
                        if asset_type in ["stock", "etf"]:
                            live_stream.subscribe([symbol])
                        
                        # V5: Rich trade explanation card
                        telegram.trade_card(
                            symbol, side_label, verdict["notional"],
                            verdict["confidence"], exec_result.get("fill_price", 0),
                            verdict["trailing_stop_pct"], verdict,
                            q_result=q_result, sentiment=sentiment,
                            insider=insider, options=options,
                        )
                        log.info(f"  ✓ Trade complete: {symbol} filled @ ${exec_result['fill_price']:.2f}")
                    elif exec_result["status"] == "market_closed":
                        stats["skipped"] += 1
                        log.info(f"  ○ Market closed — skipped {symbol} (not crypto)")
                        telegram.trade_skipped(symbol, "Market closed — only crypto trades on weekends/after hours")
                    else:
                        stats["errors"] += 1
                        log.warning(f"  ✗ Trade failed: {symbol} → {exec_result['status']}")
                        telegram.error(f"Trade failed for {symbol}: {exec_result['status']}")
                else:
                    stats["skipped"] += 1
                    log.info(f"  ○ Skipped: {verdict['reasoning'][:100]}")
                    # V5: Near-miss skip card
                    telegram.skip_card(symbol, verdict["confidence"], verdict, q_result)
                    telegram.trade_skipped(symbol, verdict["reasoning"])

            except Exception as e:
                stats["errors"] += 1
                log.error(f"  Error processing {symbol}: {e}", exc_info=True)

        # ─── Track Position Outcomes (ML Feedback) ─────────────────────
        # Record P&L for any positions that have closed since last cycle
        try:
            _track_position_outcomes(broker, journal)
        except Exception as e:
            log.debug(f"Position outcome tracking error: {e}")

        # ─── End of Cycle: Portfolio Snapshot ─────────────────────────
        account = broker.get_account()
        positions = _get_cached_positions(broker)

        total_pl = account["equity"] - config.INITIAL_EQUITY
        total_pl_pct = (total_pl / config.INITIAL_EQUITY) * 100

        journal.log_portfolio_snapshot(
            cycle_id, account["equity"], account["buying_power"],
            account["cash"], len(positions), total_pl, total_pl_pct,
            (peak_equity - account["equity"]) / peak_equity if peak_equity > 0 else 0
        )

        journal.end_cycle(
            cycle_id, stats["universe"], stats["candidates"],
            stats["trades"], stats["skipped"], stats["errors"]
        )

        telegram.cycle_summary(
            cycle_id, stats["universe"], stats["candidates"],
            stats["trades"], stats["skipped"], stats["errors"],
            account["equity"]
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
    live_state.update(step="Sleeping", details="Cycle complete. Waiting for next schedule...", progress=100, active_symbol=None)


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

    telegram.startup()

    # Start position monitor (runs in background between cycles)
    if config.LIVE_STREAM_ENABLED:
        position_monitor.start()

    # Main loop
    while not _shutdown:
        schedule.run_pending()
        time.sleep(10)

    log.info("Sentinel Autotrader shut down gracefully.")


if __name__ == "__main__":
    main()
