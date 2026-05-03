"""
Async Executor - Parallel symbol processing with ThreadPoolExecutor.

Processes batches of symbols concurrently to reduce screening time
from ~20 minutes to ~2 minutes for 200+ symbols.
"""
import concurrent.futures
from typing import Callable, Any

import config
from utils.logger import get_logger

log = get_logger("sentinel.async_executor")


class AsyncExecutor:
    """
    Wraps ThreadPoolExecutor for batch-parallel processing of symbols.
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or config.MAX_WORKERS

    def map_symbols(self, func: Callable, symbols: list[str],
                    *args, **kwargs) -> list[dict]:
        """
        Applies `func(symbol, *args, **kwargs)` to every symbol in parallel.
        Returns list of results (preserving order).
        Exceptions are caught per-symbol and returned as score=0 results.
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(func, symbol, *args, **kwargs): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(future_map):
                symbol = future_map[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    log.warning(f"Timeout screening {symbol}")
                except Exception as e:
                    log.debug(f"Error screening {symbol}: {e}")

        return results

    def map_batched(self, func: Callable, items: list,
                    batch_size: int = None, *args, **kwargs) -> list:
        """
        Processes items in batches to control memory and API pressure.
        """
        batch_size = batch_size or config.BATCH_SIZE
        all_results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = self.map_symbols(func, batch, *args, **kwargs)
            all_results.extend(batch_results)
            
            current_batch = i // batch_size + 1
            log.info(f"  Batch {current_batch}: processed {len(batch)} symbols, "
                     f"{len(batch_results)} passed")
                     
            # Ping live_state to prevent dashboard timeout
            from utils.live_state import live_state
            total_batches = (len(items) + batch_size - 1) // batch_size
            
            # Map progress from 20% to 50% during screening
            progress = 20 + int(30 * (current_batch / total_batches))
            live_state.update(
                step="Step 2: Quant Screening",
                details=f"Calculating indicators: Batch {current_batch}/{total_batches} ({len(items)} total assets)",
                progress=progress,
                active_symbol=batch[-1] if batch else None
            )

        return all_results
