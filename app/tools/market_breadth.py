"""Market breadth analysis — 52-week highs/lows, sector relative strength, and outperformers.

Fetches price data for ~500 Nifty constituents using yf.download() batch API,
computes breadth metrics, sector RS vs Nifty 500, and top outperformers per sector.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any, Callable

import pandas as pd
import yfinance as yf

from app.utils.cache import cache_get, cache_set, retry_on_error, yfinance_rate_limiter

logger = logging.getLogger(__name__)

# Path to the Nifty 500 CSV file
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "nifty500_constituents.csv")

# Batch size for yf.download() calls — keeps each request manageable
_BATCH_SIZE = 100


# ── Constituent loading ──────────────────────────────────────


def get_nifty_constituents() -> dict[str, list[str]]:
    """Load sector-to-tickers mapping from the Nifty 500 CSV file."""
    csv_path = os.path.normpath(_CSV_PATH)
    if not os.path.exists(csv_path):
        logger.warning("CSV not found at %s — using empty map", csv_path)
        return {}

    sector_map: dict[str, list[str]] = {}
    seen: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"].strip()
            sector = row["sector"].strip()
            if ticker and sector and ticker not in seen:
                seen.add(ticker)
                sector_map.setdefault(sector, []).append(ticker)

    total = sum(len(v) for v in sector_map.values())
    logger.info("Loaded %d tickers across %d sectors from CSV", total, len(sector_map))
    return sector_map


def _all_tickers(sector_map: dict[str, list[str]]) -> list[str]:
    """Flatten sector map to a deduplicated list of tickers."""
    seen: set[str] = set()
    tickers: list[str] = []
    for stocks in sector_map.values():
        for t in stocks:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


def _ticker_to_sector(sector_map: dict[str, list[str]]) -> dict[str, str]:
    """Build ticker -> sector lookup."""
    mapping: dict[str, str] = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            mapping[t] = sector
    return mapping


# ── Data fetching ────────────────────────────────────────────


@retry_on_error(max_retries=2, base_delay=2.0)
def _download_batch(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Download a single batch of tickers."""
    yfinance_rate_limiter.wait()
    raw = yf.download(tickers, period=period, group_by="ticker", threads=True, progress=False)

    close_data: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                series = raw["Close"]
            else:
                series = raw[(ticker, "Close")]
            if series is not None and not series.dropna().empty:
                close_data[ticker] = series
        except (KeyError, TypeError):
            pass

    return pd.DataFrame(close_data)


def fetch_bulk_price_data(
    tickers: list[str],
    period: str = "1y",
    progress_cb: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Batch-download daily close prices, splitting into batches of ~100.

    Args:
        tickers: List of ticker symbols.
        period: yfinance period string.
        progress_cb: Optional callback(fraction, message) for UI progress updates.

    Returns a DataFrame with dates as index and tickers as columns (Close prices).
    Cached for 10 minutes.
    """
    cache_key = f"bulk_prices:{len(tickers)}:{period}"
    cached_result = cache_get(cache_key)
    if cached_result is not None:
        if progress_cb:
            progress_cb(1.0, "Using cached price data")
        return cached_result

    # Split into batches
    batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
    total_batches = len(batches)
    logger.info("Downloading price data: %d tickers in %d batches", len(tickers), total_batches)

    all_frames: list[pd.DataFrame] = []
    for idx, batch in enumerate(batches):
        if progress_cb:
            pct = idx / total_batches
            progress_cb(pct, f"Downloading batch {idx + 1}/{total_batches} ({len(batch)} tickers)...")

        df = _download_batch(batch, period)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, axis=1)
    # Remove duplicate columns (if any ticker appeared in multiple batches)
    result = result.loc[:, ~result.columns.duplicated()]

    logger.info("Got price data for %d / %d tickers", len(result.columns), len(tickers))
    if progress_cb:
        progress_cb(1.0, f"Downloaded data for {len(result.columns)} stocks")

    cache_set(cache_key, result, ttl=600)
    return result


def _fetch_benchmark(benchmark: str, period: str = "1y") -> pd.Series | None:
    """Fetch benchmark index price series."""
    for ticker in [benchmark, "^NSEI"]:
        try:
            yfinance_rate_limiter.wait()
            data = yf.download(ticker, period=period, progress=False)
            if data is not None and not data.empty:
                close = data["Close"]
                if hasattr(close, "squeeze"):
                    close = close.squeeze()
                if not close.dropna().empty:
                    logger.info("Using benchmark: %s", ticker)
                    return close
        except Exception as e:
            logger.warning("Benchmark %s failed: %s", ticker, e)
    return None


# ── Computation functions ────────────────────────────────────


def compute_52week_breadth(price_df: pd.DataFrame, sector_map: dict[str, list[str]]) -> dict[str, Any]:
    """Compute 52-week high/low breadth from price DataFrame."""
    t2s = _ticker_to_sector(sector_map)

    stocks_at_high: list[dict] = []
    stocks_at_low: list[dict] = []
    sector_highs: dict[str, int] = {}
    sector_lows: dict[str, int] = {}
    valid_count = 0

    for ticker in price_df.columns:
        series = price_df[ticker].dropna()
        if len(series) < 20:
            continue

        valid_count += 1
        current = float(series.iloc[-1])
        high_52w = float(series.max())
        low_52w = float(series.min())
        sector = t2s.get(ticker, "Other")

        if current >= 0.98 * high_52w:
            stocks_at_high.append({
                "ticker": ticker, "sector": sector,
                "price": round(current, 2), "high_52w": round(high_52w, 2),
                "pct_from_high": round((current / high_52w - 1) * 100, 2),
            })
            sector_highs[sector] = sector_highs.get(sector, 0) + 1

        if current <= 1.02 * low_52w:
            stocks_at_low.append({
                "ticker": ticker, "sector": sector,
                "price": round(current, 2), "low_52w": round(low_52w, 2),
                "pct_from_low": round((current / low_52w - 1) * 100, 2),
            })
            sector_lows[sector] = sector_lows.get(sector, 0) + 1

    at_high = len(stocks_at_high)
    at_low = len(stocks_at_low)
    high_low_ratio = round(at_high / max(at_low, 1), 2)

    if at_high > 3 * max(at_low, 1) and (at_high / max(valid_count, 1)) > 0.05:
        signal = "Bullish"
    elif at_low > 3 * max(at_high, 1) and (at_low / max(valid_count, 1)) > 0.05:
        signal = "Bearish"
    else:
        signal = "Neutral"

    return {
        "total_stocks": valid_count,
        "at_52w_high": at_high,
        "at_52w_low": at_low,
        "high_pct": round(at_high / max(valid_count, 1) * 100, 1),
        "low_pct": round(at_low / max(valid_count, 1) * 100, 1),
        "high_low_ratio": high_low_ratio,
        "breadth_signal": signal,
        "stocks_at_high": sorted(stocks_at_high, key=lambda x: x["pct_from_high"], reverse=True),
        "stocks_at_low": sorted(stocks_at_low, key=lambda x: x["pct_from_low"]),
        "sector_highs": sector_highs,
        "sector_lows": sector_lows,
    }


def _compute_return(series: pd.Series, days: int) -> float | None:
    """Compute return over the last N trading days."""
    s = series.dropna()
    if len(s) < days + 1:
        return None
    current = float(s.iloc[-1])
    past = float(s.iloc[-days - 1]) if days < len(s) else float(s.iloc[0])
    if past == 0:
        return None
    return round((current / past - 1) * 100, 2)


def compute_sector_relative_strength(
    price_df: pd.DataFrame,
    sector_map: dict[str, list[str]],
    benchmark_ticker: str = "^CRSLDX",
) -> list[dict[str, Any]]:
    """Compute sector relative strength vs benchmark across timeframes."""
    benchmark_series = _fetch_benchmark(benchmark_ticker, period="1y")

    bm_returns: dict[str, float] = {}
    timeframes = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}
    if benchmark_series is not None:
        for label, days in timeframes.items():
            ret = _compute_return(benchmark_series, days)
            bm_returns[label] = ret if ret is not None else 0.0
    else:
        logger.warning("No benchmark data — relative strength will use absolute returns")
        bm_returns = {label: 0.0 for label in timeframes}

    results: list[dict] = []
    for sector, tickers in sector_map.items():
        sector_returns: dict[str, list[float]] = {label: [] for label in timeframes}

        for ticker in tickers:
            if ticker not in price_df.columns:
                continue
            series = price_df[ticker]
            for label, days in timeframes.items():
                ret = _compute_return(series, days)
                if ret is not None:
                    sector_returns[label].append(ret)

        if not any(sector_returns[label] for label in timeframes):
            continue

        row: dict[str, Any] = {"sector": sector}
        for label in timeframes:
            vals = sector_returns[label]
            avg_ret = round(sum(vals) / len(vals), 2) if vals else 0.0
            row[f"abs_{label}"] = avg_ret
            row[f"rs_{label}"] = round(avg_ret - bm_returns.get(label, 0.0), 2)

        row["is_outperforming"] = row.get("rs_3m", 0) > 0 and row.get("rs_1m", 0) > 0
        results.append(row)

    results.sort(key=lambda x: x.get("rs_3m", 0), reverse=True)
    for i, row in enumerate(results):
        row["rank"] = i + 1

    return results


def get_sector_outperformers(
    price_df: pd.DataFrame,
    sector_map: dict[str, list[str]],
    strong_sectors: list[str],
    top_n: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Get top N outperforming stocks in each strong sector by 3M return."""
    result: dict[str, list[dict]] = {}

    for sector in strong_sectors:
        tickers = sector_map.get(sector, [])
        stock_returns: list[dict] = []

        for ticker in tickers:
            if ticker not in price_df.columns:
                continue
            series = price_df[ticker]
            ret_3m = _compute_return(series, 63)
            ret_1m = _compute_return(series, 21)
            current = float(series.dropna().iloc[-1]) if not series.dropna().empty else 0.0

            if ret_3m is not None:
                stock_returns.append({
                    "ticker": ticker,
                    "name": ticker.replace(".NS", "").replace(".BO", ""),
                    "return_3m": ret_3m,
                    "return_1m": ret_1m if ret_1m is not None else 0.0,
                    "current_price": round(current, 2),
                })

        stock_returns.sort(key=lambda x: x["return_3m"], reverse=True)
        result[sector] = stock_returns[:top_n]

    return result


# ── Orchestrator ─────────────────────────────────────────────


def run_market_breadth_analysis(
    progress_cb: Callable[[float, str], None] | None = None,
    max_stocks: int | None = None,
) -> dict[str, Any]:
    """Run the full market breadth analysis pipeline.

    Args:
        progress_cb: Optional callback(fraction, message) for UI progress updates.
        max_stocks: If set, limit to this many tickers (for testing).

    Returns a dict with keys: breadth, sector_rs, strong_sectors, outperformers.
    """
    sector_map = get_nifty_constituents()
    all_t = _all_tickers(sector_map)

    if max_stocks and max_stocks < len(all_t):
        all_t = all_t[:max_stocks]
        # Trim sector_map to only include tickers in the subset
        subset = set(all_t)
        sector_map = {s: [t for t in tl if t in subset] for s, tl in sector_map.items()}
        sector_map = {s: tl for s, tl in sector_map.items() if tl}

    logger.info("Market breadth: %d unique tickers across %d sectors", len(all_t), len(sector_map))

    # Step 1: Fetch price data (this is the slow part)
    price_df = fetch_bulk_price_data(all_t, period="1y", progress_cb=progress_cb)
    if price_df.empty:
        raise RuntimeError("Failed to fetch price data for Nifty constituents")

    # Step 2: 52-week breadth
    breadth = compute_52week_breadth(price_df, sector_map)

    # Step 3: Sector relative strength
    sector_rs = compute_sector_relative_strength(price_df, sector_map)

    # Step 4: Identify strong sectors and their outperformers
    strong_sectors = [s["sector"] for s in sector_rs if s.get("is_outperforming")]
    outperformers = get_sector_outperformers(price_df, sector_map, strong_sectors, top_n=3)

    return {
        "breadth": breadth,
        "sector_rs": sector_rs,
        "strong_sectors": strong_sectors,
        "outperformers": outperformers,
    }
