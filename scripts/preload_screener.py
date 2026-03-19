"""Preload Screener.in data for Nifty 500 stocks.

Run this script offline (before demo or as a weekly cron) to pre-cache
Screener.in financial data for all Nifty 500 constituents.

Usage:
    python -m scripts.preload_screener              # all Nifty 500
    python -m scripts.preload_screener --symbols RELIANCE ZOMATO TCS
    python -m scripts.preload_screener --max 50     # first 50 only
    python -m scripts.preload_screener --refresh     # ignore existing cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# Ensure project root is on path
sys.path.insert(0, ".")

from app.tools.market_breadth import get_nifty_constituents
from app.rag.screener_scraper import scrape_company_page, _load_from_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUEST_DELAY = 2.0  # seconds between requests to respect rate limits


def get_all_symbols() -> list[str]:
    """Extract unique symbols from Nifty 500 CSV (without .NS suffix)."""
    sector_map = get_nifty_constituents()
    seen: set[str] = set()
    symbols: list[str] = []
    for tickers in sector_map.values():
        for t in tickers:
            clean = t.upper().replace(".NS", "").replace(".BO", "")
            if clean not in seen:
                seen.add(clean)
                symbols.append(clean)
    return sorted(symbols)


def preload(
    symbols: list[str] | None = None,
    max_stocks: int | None = None,
    refresh: bool = False,
    delay: float = REQUEST_DELAY,
) -> None:
    """Pre-cache Screener.in data for given symbols."""
    if symbols is None:
        symbols = get_all_symbols()
        logger.info("Loaded %d symbols from Nifty 500 CSV", len(symbols))

    if max_stocks:
        symbols = symbols[:max_stocks]

    total = len(symbols)
    success = 0
    skipped = 0
    failed = 0
    start = time.time()

    logger.info("Starting preload for %d symbols (refresh=%s)", total, refresh)

    for i, symbol in enumerate(symbols, 1):
        # Skip if already cached and not refreshing
        if not refresh and _load_from_cache(symbol, max_age_hours=24) is not None:
            skipped += 1
            if i % 50 == 0:
                logger.info("[%d/%d] %s — cached (skipped)", i, total, symbol)
            continue

        try:
            data = scrape_company_page(symbol, use_cache=not refresh)
            if "error" in data:
                failed += 1
                logger.warning("[%d/%d] %s — FAILED: %s", i, total, symbol, data.get("error"))
            else:
                success += 1
                sections = sum(1 for k in ["quarterly_results", "annual_pl", "balance_sheet",
                                           "cash_flows", "ratios", "shareholding"]
                               if k in data)
                logger.info("[%d/%d] %s — OK (%d sections)", i, total, symbol, sections)
        except Exception as e:
            failed += 1
            logger.error("[%d/%d] %s — ERROR: %s", i, total, symbol, e)

        # Rate limiting (skip delay after last symbol)
        if i < total:
            time.sleep(delay)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Preload complete in %.1f minutes", elapsed / 60)
    logger.info("  Success: %d  |  Skipped (cached): %d  |  Failed: %d", success, skipped, failed)
    logger.info("  Total: %d / %d", success + skipped, total)


def main():
    parser = argparse.ArgumentParser(description="Preload Screener.in data for Nifty 500")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to preload")
    parser.add_argument("--max", type=int, help="Max number of symbols to process")
    parser.add_argument("--refresh", action="store_true", help="Ignore existing cache, re-scrape all")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests (seconds)")
    args = parser.parse_args()

    preload(
        symbols=args.symbols,
        max_stocks=args.max,
        refresh=args.refresh,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
