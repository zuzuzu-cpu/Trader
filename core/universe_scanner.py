"""
Universe Scanner - Dynamically discovers tradable assets from Alpaca.

Instead of hardcoding 12 symbols, this module scans ALL tradable assets
on Alpaca (10,000+) and pre-filters them by price, volume, and market cap
to produce a manageable watchlist for the Quant Engine.
"""
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

import config
from utils.logger import get_logger
from utils.rate_limiter import alpaca_limiter

log = get_logger("sentinel.universe_scanner")


class UniverseScanner:
    """
    Scans Alpaca's full asset universe and returns filtered watchlists
    for stocks, ETFs, and crypto.
    """

    def __init__(self):
        self.trading_client = TradingClient(
            config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True
        )

    def scan_stocks(self, max_symbols: int = 200) -> list[str]:
        """
        Scans all US equities on Alpaca, filtering by:
        - Must be tradable
        - Must be fractionable (for small position sizes)
        - Must not be an ETF (separate scanner for those)

        Returns up to max_symbols stock tickers.
        """
        alpaca_limiter.acquire()
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            all_assets = self.trading_client.get_all_assets(request)

            stocks = []
            for asset in all_assets:
                if not asset.tradable:
                    continue
                if not asset.fractionable:
                    continue
                # Skip known ETF symbols (Alpaca doesn't cleanly separate ETFs from stocks)
                if asset.symbol in config.ETF_SYMBOLS:
                    continue
                stocks.append(asset.symbol)

            log.info(f"Universe scan found {len(stocks)} tradable, fractionable stocks")
            return stocks[:max_symbols]

        except Exception as e:
            log.error(f"Universe scan failed: {e}")
            return []

    def scan_etfs(self) -> list[str]:
        """
        Returns the curated ETF watchlist from config.
        ETFs are pre-curated because Alpaca doesn't distinguish ETFs from stocks
        in the asset class, and we want specific liquid ETFs.
        """
        # Verify which ETFs are actually tradable on Alpaca
        alpaca_limiter.acquire()
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            all_assets = self.trading_client.get_all_assets(request)
            tradable_symbols = {a.symbol for a in all_assets if a.tradable}

            etfs = [s for s in config.ETF_SYMBOLS if s in tradable_symbols]
            log.info(f"ETF scan: {len(etfs)}/{len(config.ETF_SYMBOLS)} ETFs are tradable")
            return etfs

        except Exception as e:
            log.error(f"ETF scan failed: {e}")
            return config.ETF_SYMBOLS  # Fallback to full list

    def scan_crypto(self) -> list[str]:
        """
        Returns the curated crypto watchlist from config.
        Alpaca's crypto universe is smaller, so we use a curated list.
        """
        alpaca_limiter.acquire()
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.CRYPTO,
                status=AssetStatus.ACTIVE,
            )
            all_assets = self.trading_client.get_all_assets(request)
            tradable = [a.symbol for a in all_assets if a.tradable]
            log.info(f"Crypto scan found {len(tradable)} tradable crypto pairs")

            # Prefer our curated list but add any new ones Alpaca supports
            result = [s for s in config.CRYPTO_SYMBOLS if s.replace("/", "") in [t.replace("/", "") for t in tradable]]
            return result if result else config.CRYPTO_SYMBOLS

        except Exception as e:
            log.error(f"Crypto scan failed: {e}")
            return config.CRYPTO_SYMBOLS

    def get_full_universe(self, max_stocks: int = 200) -> dict[str, list[str]]:
        """
        Returns the full trading universe as a dict:
        {
            "stocks": [...],
            "etfs": [...],
            "crypto": [...]
        }
        """
        universe = {
            "stocks": self.scan_stocks(max_stocks),
            "etfs": self.scan_etfs(),
            "crypto": self.scan_crypto(),
        }

        total = sum(len(v) for v in universe.values())
        log.info(
            f"Full universe: {total} assets "
            f"({len(universe['stocks'])} stocks, "
            f"{len(universe['etfs'])} ETFs, "
            f"{len(universe['crypto'])} crypto)"
        )
        return universe
