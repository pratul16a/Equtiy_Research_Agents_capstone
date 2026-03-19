"""3-Category Stock Screener — Strengthening Industries, Momentum, Value Bottoms.

Each category runs its own screening criteria and returns scored results.
Price-first filtering: batch-download prices → technical screens → eliminate ~60%
→ then fetch .info for survivors only (optimization).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from app.tools.market_breadth import (
    get_nifty_constituents,
    fetch_bulk_price_data,
    compute_52week_breadth,
    compute_sector_relative_strength,
    compute_sector_profit_growth,
    _all_tickers,
    _ticker_to_sector,
)
from app.tools.screener.batch_fundamentals import fetch_bulk_info
from app.tools.screener.technical_screens import (
    screen_above_200dma,
    screen_rsi_range,
    screen_rsi_oversold,
    screen_volume_breakout,
    screen_weekly_ma_alignment,
)
from app.tools.screener.quality_screens import (
    screen_opm_improvement,
    screen_cf_turnaround,
    screen_qoq_profit_growth,
)
from app.tools.screener.quantitative import (
    screen_roe,
    screen_pb_vs_historical,
    screen_ps_vs_historical,
)
from app.tools.screener.qualitative import (
    screen_capacity_utilization,
    screen_announcements,
)
from app.tools.screener.geopolitical import screen_geopolitical_risk
from app.tools.screener.regulatory import screen_regulatory_tailwind
from app.tools.screener.smart_money import screen_smart_money_lag
from app.tools.screener.mgmt_credibility import screen_management_credibility
from app.tools.screener.promoter_anomaly import screen_promoter_anomalies

logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────


def _merge_and_score(
    screen_results: dict[str, list[dict]],
    sector_map: dict[str, str],
    category: str,
) -> list[dict[str, Any]]:
    """Merge multiple screen results into a per-stock scored list.

    Each stock gets: ticker, sector, category, criteria_passed (list), score (count).
    """
    stock_data: dict[str, dict[str, Any]] = {}

    for criterion, stocks in screen_results.items():
        for stock in stocks:
            ticker = stock["ticker"]
            if ticker not in stock_data:
                stock_data[ticker] = {
                    "ticker": ticker,
                    "sector": stock.get("sector", sector_map.get(ticker, "Other")),
                    "category": category,
                    "criteria_passed": [],
                    "criteria_details": {},
                    "score": 0,
                }
            stock_data[ticker]["criteria_passed"].append(criterion)
            stock_data[ticker]["criteria_details"][criterion] = stock
            stock_data[ticker]["score"] += 1

    result = sorted(stock_data.values(), key=lambda x: x["score"], reverse=True)
    return result


def _get_tickers_from_results(screen_results: dict[str, list[dict]]) -> set[str]:
    """Extract unique tickers from screen results."""
    tickers = set()
    for stocks in screen_results.values():
        for s in stocks:
            tickers.add(s["ticker"])
    return tickers


# ── Category A: Strengthening Industries ──────────────────────


def screen_category_a(
    sector_map: dict[str, list[str]],
    price_df: pd.DataFrame,
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Category A: Stocks from Strengthening Industries (top-down sector play).

    Phase 1: Identify strong sectors (52W breadth, RS > Nifty 500, profit growth ≥10%)
    Phase 2: Within strong sectors, filter by policy/geo tailwinds.
    """
    if progress_cb:
        progress_cb(0.0, "Cat A: Identifying strong sectors...")

    # Phase 1: Sector-level screening
    breadth = compute_52week_breadth(price_df, sector_map)
    sector_rs = compute_sector_relative_strength(price_df, sector_map)
    profit_growth = compute_sector_profit_growth(sector_map, bulk_info)

    # Strong sectors: outperforming + profit growth ≥ 10%
    outperforming_sectors = {s["sector"] for s in sector_rs if s.get("is_outperforming")}
    growing_sectors = {s for s, data in profit_growth.items() if data["above_threshold"]}

    strong_sectors = list(outperforming_sectors & growing_sectors)
    if not strong_sectors:
        # Fallback: use outperforming sectors even without profit growth data
        strong_sectors = list(outperforming_sectors)

    logger.info("Cat A: %d strong sectors identified: %s", len(strong_sectors), strong_sectors)

    if progress_cb:
        progress_cb(0.4, f"Cat A: {len(strong_sectors)} strong sectors, filtering stocks...")

    # Phase 2: Stocks within strong sectors + policy/geo
    sector_tickers = []
    for sector in strong_sectors:
        sector_tickers.extend(sector_map.get(sector, []))
    sector_tickers = list(set(sector_tickers))

    # Filter bulk_info to only sector tickers
    sector_bulk = {t: bulk_info[t] for t in sector_tickers if t in bulk_info}
    sector_t2s = {t: t2s[t] for t in sector_tickers if t in t2s}

    results: dict[str, list[dict]] = {}

    # Run geo and regulatory screens on sector stocks
    results["geopolitical"] = screen_geopolitical_risk(sector_bulk, sector_t2s)
    results["regulatory"] = screen_regulatory_tailwind(sector_bulk, sector_t2s)

    # Also include all sector stocks (they passed sector-level criteria)
    results["strong_sector"] = [
        {"ticker": t, "sector": t2s.get(t, "Other"), "passed": True}
        for t in sector_tickers if t in bulk_info
    ]

    if progress_cb:
        progress_cb(1.0, f"Cat A: {len(_get_tickers_from_results(results))} stocks passed")

    return _merge_and_score(results, t2s, "A_Strengthening")


# ── Category B: Momentum Stocks ──────────────────────────────


def screen_category_b(
    price_df: pd.DataFrame,
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    tickers: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Category B: Momentum Stocks (price + fundamental inflection).

    Criteria: QoQ profit growth, RSI 40-70, weekly MA alignment, OPM improving,
    CF turnaround, volume breakout.
    """
    if progress_cb:
        progress_cb(0.0, "Cat B: Running momentum screens...")

    results: dict[str, list[dict]] = {}

    # Run independent screens in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(screen_rsi_range, price_df, t2s, 40, 70): "rsi_range",
            executor.submit(screen_weekly_ma_alignment, tickers, t2s): "weekly_ma",
            executor.submit(screen_volume_breakout, tickers, t2s): "volume_breakout",
            executor.submit(screen_opm_improvement, bulk_info, t2s): "opm_improvement",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning("Cat B screen %s failed: %s", key, e)
                results[key] = []

    if progress_cb:
        progress_cb(0.6, "Cat B: Running QoQ profit and CF screens...")

    # Sequential screens (depend on ticker financials — heavier API load)
    results["qoq_profit"] = screen_qoq_profit_growth(bulk_info, t2s)
    results["cf_turnaround"] = screen_cf_turnaround(bulk_info, t2s)

    if progress_cb:
        progress_cb(1.0, f"Cat B: {len(_get_tickers_from_results(results))} stocks passed")

    return _merge_and_score(results, t2s, "B_Momentum")


# ── Category C: Value Bottoms ─────────────────────────────────


def screen_category_c(
    price_df: pd.DataFrame,
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    tickers: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Category C: Value Bottoms (contrarian turnaround plays).

    Criteria: Low ROE, P/B < 5yr avg, P/S < 5yr avg, RSI < 40, capacity utilization,
    capex announcements, management credibility, geo risk, regulatory tailwind.
    """
    if progress_cb:
        progress_cb(0.0, "Cat C: Running value screens...")

    results: dict[str, list[dict]] = {}

    # Price-based screens (parallel, fast)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(screen_rsi_oversold, price_df, t2s, 40): "rsi_oversold",
            executor.submit(screen_pb_vs_historical, bulk_info, price_df, t2s): "pb_below_5yr",
            executor.submit(screen_ps_vs_historical, bulk_info, price_df, t2s): "ps_below_5yr",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logger.warning("Cat C screen %s failed: %s", key, e)
                results[key] = []

    if progress_cb:
        progress_cb(0.4, "Cat C: Running fundamental + governance screens...")

    # Fundamental screens
    results["low_roe"] = screen_roe(bulk_info, t2s)
    results["capacity_util"] = screen_capacity_utilization(tickers, t2s)

    # Announcement screens
    try:
        announcements = screen_announcements(tickers, t2s)
        results["capex"] = announcements.get("capex", [])
        results["new_business"] = announcements.get("new_business", [])
    except Exception as e:
        logger.warning("Cat C announcement screens failed: %s", e)

    if progress_cb:
        progress_cb(0.7, "Cat C: Running governance screens...")

    # Governance / USP screens on value stocks
    results["mgmt_credibility"] = screen_management_credibility(bulk_info, t2s)
    results["geopolitical"] = screen_geopolitical_risk(bulk_info, t2s)
    results["regulatory"] = screen_regulatory_tailwind(bulk_info, t2s)

    if progress_cb:
        progress_cb(1.0, f"Cat C: {len(_get_tickers_from_results(results))} stocks passed")

    return _merge_and_score(results, t2s, "C_ValueBottoms")


# ── USP Layer ─────────────────────────────────────────────────


def apply_usp_layer(
    recommended_tickers: list[str],
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run USP indicators on recommended stocks only (~30-50, not 500).

    Returns: {ticker: {geopolitical, smart_money, regulatory, mgmt_credibility, promoter}}
    """
    if progress_cb:
        progress_cb(0.0, f"USP: Analyzing {len(recommended_tickers)} stocks...")

    # Filter data to recommended stocks only
    rec_bulk = {t: bulk_info[t] for t in recommended_tickers if t in bulk_info}
    rec_t2s = {t: t2s[t] for t in recommended_tickers if t in t2s}

    # Run all 5 USP modules
    usp_raw: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(screen_geopolitical_risk, rec_bulk, rec_t2s): "geopolitical",
            executor.submit(screen_smart_money_lag, rec_bulk, rec_t2s): "smart_money",
            executor.submit(screen_regulatory_tailwind, rec_bulk, rec_t2s): "regulatory",
            executor.submit(screen_management_credibility, rec_bulk, rec_t2s): "mgmt_credibility",
            executor.submit(screen_promoter_anomalies, recommended_tickers, rec_t2s, rec_bulk): "promoter",
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                usp_raw[key] = future.result()
            except Exception as e:
                logger.warning("USP module %s failed: %s", key, e)
                usp_raw[key] = []

    if progress_cb:
        progress_cb(0.8, "USP: Assembling per-stock scores...")

    # Reshape: from {module: [stocks]} to {ticker: {module: data}}
    per_stock: dict[str, dict[str, Any]] = {}
    for module_name, stocks in usp_raw.items():
        for stock in stocks:
            ticker = stock.get("ticker", "")
            if ticker not in per_stock:
                per_stock[ticker] = {}
            per_stock[ticker][module_name] = stock

    if progress_cb:
        progress_cb(1.0, f"USP: Scored {len(per_stock)} stocks across 5 dimensions")

    return per_stock


# ── Main Orchestrator ─────────────────────────────────────────


def run_category_screener(
    categories: list[str] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    max_stocks: int | None = None,
) -> dict[str, Any]:
    """Run the full 3-category screening pipeline.

    Args:
        categories: List of categories to run ("A", "B", "C"). None = all 3.
        progress_cb: Optional callback(fraction, message).
        max_stocks: Limit universe size (for testing).

    Returns dict with keys: category_results, usp_scores, universe_info, regime_data.
    """
    if categories is None:
        categories = ["A", "B", "C"]

    # ── Load universe ────────────────────────────────────────
    sector_map = get_nifty_constituents()
    all_tickers = _all_tickers(sector_map)
    t2s = _ticker_to_sector(sector_map)

    if max_stocks and max_stocks < len(all_tickers):
        all_tickers = all_tickers[:max_stocks]

    if progress_cb:
        progress_cb(0.02, f"Loading {len(all_tickers)} tickers...")

    # ── Phase 1: Batch price download (fast, ~10s for 500) ───
    if progress_cb:
        progress_cb(0.05, "Phase 1: Downloading price data...")

    price_df = fetch_bulk_price_data(
        all_tickers, period="1y",
        progress_cb=lambda f, m: progress_cb(0.05 + f * 0.10, m) if progress_cb else None,
    )

    # ── Phase 2: Price-first filtering ───────────────────────
    # Run quick technical screens to eliminate ~60% before slow .info calls
    if progress_cb:
        progress_cb(0.15, "Phase 2: Technical pre-filter...")

    # Stocks that pass ANY technical screen survive for .info fetch
    tech_survivors: set[str] = set()

    above_200 = screen_above_200dma(price_df, t2s)
    tech_survivors.update(s["ticker"] for s in above_200)

    rsi_mid = screen_rsi_range(price_df, t2s, lo=20, hi=80)  # broad range for pre-filter
    tech_survivors.update(s["ticker"] for s in rsi_mid)

    # Also keep all tickers if tech screens eliminate too aggressively
    if len(tech_survivors) < len(all_tickers) * 0.3:
        tech_survivors = set(all_tickers)  # fallback: don't over-filter
        logger.info("Tech pre-filter too aggressive — keeping all tickers")

    filtered_tickers = list(tech_survivors)
    logger.info("Price-first filter: %d → %d tickers for .info fetch",
                len(all_tickers), len(filtered_tickers))

    # ── Phase 3: Fetch fundamentals (only for survivors) ─────
    if progress_cb:
        progress_cb(0.20, f"Phase 3: Fetching fundamentals for {len(filtered_tickers)} stocks...")

    bulk_info = fetch_bulk_info(
        filtered_tickers,
        progress_cb=lambda f, m: progress_cb(0.20 + f * 0.30, m) if progress_cb else None,
    )

    if progress_cb:
        progress_cb(0.50, f"Data ready: {len(bulk_info)} stocks with fundamentals")

    # ── Phase 4: Run category screens (PARALLEL) ─────────────
    if progress_cb:
        progress_cb(0.52, "Running all category screens in parallel...")

    category_results: dict[str, list[dict]] = {}

    # Build tasks for parallel execution
    _cat_tasks: dict[str, tuple] = {}
    if "A" in categories:
        _cat_tasks["A"] = (screen_category_a, (sector_map, price_df, bulk_info, t2s))
    if "B" in categories:
        _cat_tasks["B"] = (screen_category_b, (price_df, bulk_info, t2s, filtered_tickers))
    if "C" in categories:
        _cat_tasks["C"] = (screen_category_c, (price_df, bulk_info, t2s, filtered_tickers))

    with ThreadPoolExecutor(max_workers=len(_cat_tasks)) as executor:
        future_to_cat = {
            executor.submit(fn, *args): cat
            for cat, (fn, args) in _cat_tasks.items()
        }
        for future in as_completed(future_to_cat):
            cat = future_to_cat[future]
            try:
                category_results[cat] = future.result()
                if progress_cb:
                    progress_cb(0.52 + 0.36 * len(category_results) / len(_cat_tasks),
                                f"Category {cat} done: {len(category_results[cat])} stocks")
            except Exception as e:
                logger.error("Category %s screening failed: %s", cat, e)
                category_results[cat] = []

    # ── Phase 5: USP layer on all survivors (parallel with categories) ──
    all_survivors = set()
    for cat_stocks in category_results.values():
        for stock in cat_stocks:
            all_survivors.add(stock["ticker"])

    recommended_tickers = list(all_survivors)
    logger.info("Total survivors across categories: %d", len(recommended_tickers))

    if progress_cb:
        progress_cb(0.88, f"Running USP analysis on {len(recommended_tickers)} stocks...")

    usp_scores = apply_usp_layer(
        recommended_tickers, bulk_info, t2s,
    )

    # ── Build summary ────────────────────────────────────────
    summary = {
        "universe_size": len(all_tickers),
        "after_tech_filter": len(filtered_tickers),
        "with_fundamentals": len(bulk_info),
        "category_counts": {cat: len(stocks) for cat, stocks in category_results.items()},
        "total_survivors": len(recommended_tickers),
        "usp_scored": len(usp_scores),
    }

    if progress_cb:
        progress_cb(1.0, f"Screening complete: {len(recommended_tickers)} stocks across {len(categories)} categories")

    return {
        "category_results": category_results,
        "usp_scores": usp_scores,
        "summary": summary,
        "universe_info": {"type": "nifty500", "ticker_count": len(all_tickers), "technical_survivors": len(filtered_tickers)},
        "bulk_info": bulk_info,
        "price_df": price_df,
        "sector_map": sector_map,
        "t2s": t2s,
    }
