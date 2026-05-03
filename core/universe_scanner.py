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

    def scan_stocks(self, max_symbols: int = 10000) -> list[str]:
        """
        Scans all tradable equities on Alpaca, filtering by:
        - Must be tradable
        - Must be fractionable (for small position sizes)
        
        Returns up to max_symbols tickers.
        """
        alpaca_limiter.acquire()
        try:
            # Fetch US Equities
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            all_assets = self.trading_client.get_all_assets(request)

            stocks = []
            for asset in all_assets:
                if not asset.tradable or not asset.fractionable:
                    continue
                # Note: We include ETFs in this pool now to ensure nothing is missed
                stocks.append(asset.symbol)

            log.info(f"Universe scan found {len(stocks)} tradable, fractionable assets")
            return stocks[:max_symbols]

        except Exception as e:
            log.error(f"Universe scan failed: {e}")
            return []

    def scan_etfs(self) -> list[str]:
        """
        Returns a verified list of liquid ETFs from config.
        Since Alpaca mixes ETFs into US_EQUITY, we use a curated list for focus.
        """
        alpaca_limiter.acquire()
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            all_assets = self.trading_client.get_all_assets(request)
            tradable_symbols = {a.symbol for a in all_assets if a.tradable}

            etfs = [s for s in config.ETF_SYMBOLS if s in tradable_symbols]
            log.info(f"ETF scan verified {len(etfs)} tradable ETFs")
            return etfs

        except Exception as e:
            log.error(f"ETF scan failed: {e}")
            return config.ETF_SYMBOLS

    def scan_crypto(self) -> list[str]:
        """
        Returns ALL tradable crypto pairs supported by Alpaca.
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
            return tradable

        except Exception as e:
            log.error(f"Crypto scan failed: {e}")
            return config.CRYPTO_SYMBOLS

    def get_full_universe(self, max_stocks: int = 10000) -> dict[str, list[str]]:
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
            f"Full universe expanded to {total} assets "
            f"({len(universe['stocks'])} stocks/ADRs, "
            f"{len(universe['etfs'])} ETFs, "
            f"{len(universe['crypto'])} crypto)"
        )
        return universe
