"""Stock screener orchestrator — 4-phase pipeline for fundamental screening."""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.tools.market_breadth import (
    get_nifty_constituents,
    fetch_bulk_price_data,
    _all_tickers,
    _ticker_to_sector,
)
from app.tools.screener.batch_fundamentals import fetch_bulk_info
from app.tools.screener.quantitative import (
    screen_roe,
    screen_pb_vs_historical,
    screen_ps_vs_historical,
)
from app.tools.screener.qualitative import (
    screen_insider_transactions,
    screen_announcements,
    screen_capacity_utilization,
)
from app.tools.screener.quality_screens import (
    screen_debt_reduction,
    screen_opm_improvement,
    screen_cf_turnaround,
    screen_dividend_initiation,
)
from app.tools.screener.technical_screens import (
    screen_above_200dma,
    screen_volume_breakout,
)
from app.tools.screener.governance_screens import (
    screen_low_pledge,
    screen_increasing_institutional,
)
from app.tools.screener.geopolitical import screen_geopolitical_risk
from app.tools.screener.smart_money import screen_smart_money_lag
from app.tools.screener.regulatory import screen_regulatory_tailwind
from app.tools.screener.promoter_anomaly import screen_promoter_anomalies
from app.tools.screener.composite_scorer import score_all_stocks

logger = logging.getLogger(__name__)

# All available screening criteria
ALL_CRITERIA = [
    # Quantitative (original)
    "roe",
    "pb_vs_historical",
    "ps_vs_historical",
    # Qualitative (original)
    "capacity_utilization",
    "insider_buying",
    "capex",
    "new_business",
    "order_booking",
    # Financial quality (Phase 1A)
    "debt_reduction",
    "opm_improvement",
    "cf_turnaround",
    "dividend_initiation",
    # Technical (Phase 1B)
    "above_200dma",
    "volume_breakout",
    # Governance (Phase 1C)
    "low_pledge",
    "increasing_institutional",
    # USP screens (Phases 2-6)
    "geopolitical",
    "smart_money_lag",
    "regulatory_tailwind",
    "promoter_anomaly",
]


def _merge_results(
    all_screen_results: dict[str, list[dict]],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Merge per-criterion results into a unified per-stock view.

    Returns list of dicts, one per stock that passed at least one criterion.
    """
    stock_data: dict[str, dict[str, Any]] = {}

    for criterion, stocks in all_screen_results.items():
        for stock in stocks:
            ticker = stock["ticker"]
            if ticker not in stock_data:
                stock_data[ticker] = {
                    "ticker": ticker,
                    "sector": stock.get("sector", sector_map.get(ticker, "Other")),
                    "criteria_passed": [],
                    "criteria_details": {},
                    "score": 0,
                }
            stock_data[ticker]["criteria_passed"].append(criterion)
            stock_data[ticker]["criteria_details"][criterion] = stock
            stock_data[ticker]["score"] += 1

    result = sorted(stock_data.values(), key=lambda x: x["score"], reverse=True)
    return result


def run_stock_screener(
    universe: str = "nifty500",
    criteria: list[str] | None = None,
    min_score: int = 1,
    breadth_data: dict | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    max_stocks: int | None = None,
) -> dict[str, Any]:
    """Run the full stock screening pipeline.

    Args:
        universe: "nifty500" for all Nifty 500 stocks, or "strong_sectors" to
                  screen only strong sectors from breadth analysis.
        criteria: List of criteria to apply (from ALL_CRITERIA). None = all.
        min_score: Minimum number of criteria a stock must pass.
        breadth_data: Output from run_market_breadth_analysis() if universe="strong_sectors".
        progress_cb: Optional callback(fraction, message) for UI progress.
        max_stocks: If set, limit to this many tickers (for testing).

    Returns dict with keys: stocks, summary, criteria_used, universe_info.
    """
    if criteria is None:
        criteria = list(ALL_CRITERIA)

    # ── Determine universe ────────────────────────────────────
    sector_constituents = get_nifty_constituents()
    all_t = _all_tickers(sector_constituents)
    t2s = _ticker_to_sector(sector_constituents)

    if universe == "strong_sectors" and breadth_data:
        strong = breadth_data.get("strong_sectors", [])
        if strong:
            tickers = []
            for sector in strong:
                tickers.extend(sector_constituents.get(sector, []))
            # Deduplicate
            seen = set()
            tickers = [t for t in tickers if t not in seen and not seen.add(t)]
            universe_info = {
                "type": "strong_sectors",
                "sectors": strong,
                "ticker_count": len(tickers),
            }
        else:
            logger.warning("No strong sectors found — falling back to full Nifty 500")
            tickers = all_t
            universe_info = {"type": "nifty500", "ticker_count": len(tickers)}
    else:
        tickers = all_t
        universe_info = {"type": "nifty500", "ticker_count": len(tickers)}

    if max_stocks and max_stocks < len(tickers):
        tickers = tickers[:max_stocks]
        universe_info["ticker_count"] = len(tickers)
        universe_info["limited"] = True

    logger.info("Screener universe: %d tickers (%s)", len(tickers), universe_info["type"])

    # ── Phase 1: Fetch bulk data ──────────────────────────────
    if progress_cb:
        progress_cb(0.05, f"Phase 1: Fetching data for {len(tickers)} stocks...")

    bulk_info = fetch_bulk_info(tickers, progress_cb=lambda f, m: progress_cb(0.05 + f * 0.20, m) if progress_cb else None)
    price_df = fetch_bulk_price_data(tickers, period="1y", progress_cb=lambda f, m: progress_cb(0.25 + f * 0.15, m) if progress_cb else None)

    if progress_cb:
        progress_cb(0.40, f"Phase 1 complete: {len(bulk_info)} stocks with info data")

    # ── Phase 2a: Quantitative screens ────────────────────────
    all_results: dict[str, list[dict]] = {}

    if "roe" in criteria:
        if progress_cb:
            progress_cb(0.42, "Phase 2a: Screening ROE...")
        all_results["roe"] = screen_roe(bulk_info, t2s)

    if "pb_vs_historical" in criteria:
        if progress_cb:
            progress_cb(0.46, "Phase 2a: Screening P/B vs historical...")
        all_results["pb_vs_historical"] = screen_pb_vs_historical(bulk_info, price_df, t2s)

    if "ps_vs_historical" in criteria:
        if progress_cb:
            progress_cb(0.50, "Phase 2a: Screening P/S vs historical...")
        all_results["ps_vs_historical"] = screen_ps_vs_historical(bulk_info, price_df, t2s)

    # ── Phase 2b: Financial quality screens ──────────────────
    if "debt_reduction" in criteria:
        if progress_cb:
            progress_cb(0.53, "Phase 2b: Screening debt reduction...")
        all_results["debt_reduction"] = screen_debt_reduction(bulk_info, t2s)

    if "opm_improvement" in criteria:
        if progress_cb:
            progress_cb(0.55, "Phase 2b: Screening OPM improvement...")
        all_results["opm_improvement"] = screen_opm_improvement(bulk_info, t2s)

    if "cf_turnaround" in criteria:
        if progress_cb:
            progress_cb(0.57, "Phase 2b: Screening cash flow turnaround...")
        all_results["cf_turnaround"] = screen_cf_turnaround(bulk_info, t2s)

    if "dividend_initiation" in criteria:
        if progress_cb:
            progress_cb(0.59, "Phase 2b: Screening dividend initiation...")
        all_results["dividend_initiation"] = screen_dividend_initiation(bulk_info, t2s)

    # ── Phase 2c: Technical screens ──────────────────────────
    if "above_200dma" in criteria:
        if progress_cb:
            progress_cb(0.61, "Phase 2c: Screening price vs 200 DMA...")
        all_results["above_200dma"] = screen_above_200dma(price_df, t2s)

    if "volume_breakout" in criteria:
        if progress_cb:
            progress_cb(0.63, "Phase 2c: Screening volume breakout...")
        all_results["volume_breakout"] = screen_volume_breakout(tickers, t2s)

    # ── Phase 2d: Governance screens ─────────────────────────
    if "low_pledge" in criteria:
        if progress_cb:
            progress_cb(0.65, "Phase 2d: Screening promoter pledge...")
        all_results["low_pledge"] = screen_low_pledge(bulk_info, t2s)

    if "increasing_institutional" in criteria:
        if progress_cb:
            progress_cb(0.67, "Phase 2d: Screening institutional holding...")
        all_results["increasing_institutional"] = screen_increasing_institutional(tickers, t2s)

    if progress_cb:
        quant_count = sum(len(v) for v in all_results.values())
        progress_cb(0.70, f"Phase 2 complete: {quant_count} quantitative + quality matches")

    # ── Phase 3: Semi-quantitative screens ────────────────────
    # Get tickers that passed at least one criterion for deeper analysis
    quant_tickers = set()
    for stocks in all_results.values():
        for s in stocks:
            quant_tickers.add(s["ticker"])
    # Also include all tickers for insider/announcement screens (broader search)
    phase3_tickers = list(set(tickers) & set(bulk_info.keys()))

    if "insider_buying" in criteria:
        if progress_cb:
            progress_cb(0.72, "Phase 3: Screening insider transactions...")
        all_results["insider_buying"] = screen_insider_transactions(phase3_tickers, t2s)

    if any(c in criteria for c in ["capex", "new_business", "order_booking"]):
        if progress_cb:
            progress_cb(0.78, "Phase 3: Screening announcements & news...")
        announcement_results = screen_announcements(phase3_tickers, t2s)
        for key in ["capex", "new_business", "order_booking"]:
            if key in criteria and key in announcement_results:
                all_results[key] = announcement_results[key]

    if progress_cb:
        progress_cb(0.88, "Phase 3 complete")

    # ── Phase 4: Capacity utilization ─────────────────────────
    if "capacity_utilization" in criteria:
        if progress_cb:
            progress_cb(0.85, "Phase 4: Screening capacity utilization...")
        all_results["capacity_utilization"] = screen_capacity_utilization(phase3_tickers, t2s)

    # ── Phase 5: USP screens ────────────────────────────────
    if "geopolitical" in criteria:
        if progress_cb:
            progress_cb(0.87, "Phase 5: Geopolitical risk scoring...")
        all_results["geopolitical"] = screen_geopolitical_risk(bulk_info, t2s)

    if "smart_money_lag" in criteria:
        if progress_cb:
            progress_cb(0.89, "Phase 5: Smart money lag detection...")
        all_results["smart_money_lag"] = screen_smart_money_lag(bulk_info, t2s)

    if "regulatory_tailwind" in criteria:
        if progress_cb:
            progress_cb(0.91, "Phase 5: Regulatory tailwind scanning...")
        all_results["regulatory_tailwind"] = screen_regulatory_tailwind(bulk_info, t2s)

    if "promoter_anomaly" in criteria:
        if progress_cb:
            progress_cb(0.93, "Phase 5: Promoter anomaly detection...")
        all_results["promoter_anomaly"] = screen_promoter_anomalies(phase3_tickers, t2s, bulk_info)

    if progress_cb:
        progress_cb(0.95, "Merging and scoring results...")

    # ── Merge and filter ──────────────────────────────────────
    merged = _merge_results(all_results, t2s)
    filtered = [s for s in merged if s["score"] >= min_score]

    # ── Apply composite scoring with regime adjustment ────────
    breadth_signal = "Neutral"
    if breadth_data:
        breadth_signal = breadth_data.get("breadth", {}).get("breadth_signal", "Neutral")
    filtered = score_all_stocks(filtered, breadth_signal)

    # Build summary
    sector_counts: dict[str, int] = {}
    for stock in filtered:
        sec = stock["sector"]
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    top_sector = max(sector_counts, key=sector_counts.get) if sector_counts else "N/A"

    summary = {
        "total_screened": len(bulk_info),
        "total_passing": len(filtered),
        "min_score": min_score,
        "top_sector": top_sector,
        "sector_distribution": dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)),
        "criteria_hits": {k: len(v) for k, v in all_results.items()},
    }

    if progress_cb:
        progress_cb(1.0, f"Screening complete: {len(filtered)} stocks passed (min score: {min_score})")

    return {
        "stocks": filtered,
        "summary": summary,
        "criteria_used": criteria,
        "universe_info": universe_info,
    }
