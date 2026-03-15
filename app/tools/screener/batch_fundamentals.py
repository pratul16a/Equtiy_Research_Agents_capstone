"""Bulk fundamental data fetcher using ThreadPoolExecutor with rate limiting."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import yfinance as yf

from app.utils.cache import cache_get, cache_set, retry_on_error, yfinance_rate_limiter

logger = logging.getLogger(__name__)

_INFO_TTL = 3600  # 1 hour cache for screening data
_FINANCIALS_TTL = 3600


@retry_on_error(max_retries=1, base_delay=1.0)
def _fetch_single_info(ticker: str) -> dict | None:
    """Fetch .info for a single ticker with rate limiting."""
    cache_key = f"screener_info:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    yfinance_rate_limiter.wait()
    try:
        info = yf.Ticker(ticker).info
        if info and info.get("regularMarketPrice") is not None:
            cache_set(cache_key, info, ttl=_INFO_TTL)
            return info
    except Exception as e:
        logger.debug("Failed to fetch info for %s: %s", ticker, e)
    return None


def fetch_bulk_info(
    tickers: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
    max_workers: int = 4,
) -> dict[str, dict]:
    """Fetch .info for multiple tickers using ThreadPoolExecutor.

    Returns {ticker: info_dict} for tickers that succeeded.
    """
    results: dict[str, dict] = {}
    total = len(tickers)

    # Check cache first to minimize API calls
    uncached: list[str] = []
    for t in tickers:
        cached = cache_get(f"screener_info:{t}")
        if cached is not None:
            results[t] = cached
        else:
            uncached.append(t)

    if progress_cb and results:
        progress_cb(len(results) / total, f"Found {len(results)} cached, fetching {len(uncached)} remaining...")

    if not uncached:
        return results

    completed = len(results)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_fetch_single_info, t): t for t in uncached}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1
            try:
                info = future.result()
                if info:
                    results[ticker] = info
            except Exception as e:
                logger.debug("Info fetch failed for %s: %s", ticker, e)

            if progress_cb and completed % 20 == 0:
                progress_cb(completed / total, f"Fetched info: {completed}/{total} tickers...")

    logger.info("Bulk info: got %d / %d tickers", len(results), total)
    return results


@retry_on_error(max_retries=1, base_delay=1.0)
def fetch_ticker_financials(ticker: str) -> dict[str, Any]:
    """Fetch balance sheet, income statement, and insider transactions for a ticker.

    Returns dict with keys: balance_sheet, financials, insider_transactions.
    All cached for 1 hour.
    """
    cache_key = f"screener_financials:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    yfinance_rate_limiter.wait()
    stock = yf.Ticker(ticker)

    result: dict[str, Any] = {
        "balance_sheet": None,
        "financials": None,
        "cashflow": None,
        "insider_transactions": None,
        "shares_outstanding": None,
        "dividends": None,
    }

    try:
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            result["balance_sheet"] = bs
    except Exception as e:
        logger.debug("Balance sheet failed for %s: %s", ticker, e)

    try:
        fin = stock.financials
        if fin is not None and not fin.empty:
            result["financials"] = fin
    except Exception as e:
        logger.debug("Financials failed for %s: %s", ticker, e)

    try:
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            result["cashflow"] = cf
    except Exception as e:
        logger.debug("Cashflow failed for %s: %s", ticker, e)

    try:
        insider = stock.insider_transactions
        if insider is not None and not insider.empty:
            result["insider_transactions"] = insider
    except Exception as e:
        logger.debug("Insider transactions failed for %s: %s", ticker, e)

    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            result["dividends"] = divs
    except Exception as e:
        logger.debug("Dividends failed for %s: %s", ticker, e)

    try:
        info = stock.info
        result["shares_outstanding"] = info.get("sharesOutstanding")
    except Exception:
        pass

    cache_set(cache_key, result, ttl=_FINANCIALS_TTL)
    return result
