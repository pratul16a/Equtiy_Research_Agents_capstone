"""2-Category Stock Screener — Momentum + Value Bottom.

Momentum (17 criteria): Buy what's already working.
    Strong stock + strong sector + volume proof. RSI 55-75.

Value Bottom (28+ criteria): Buy what nobody wants yet.
    Cheap stock + turnaround catalyst + safety nets. RSI <40.

Price-first filtering: batch-download prices → technical screens → eliminate ~60%
→ then fetch .info for survivors only (optimization).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any, Callable

import pandas as pd

# Timeout for individual screen tasks (seconds)
_SCREEN_TIMEOUT = 120

from app.tools.market_breadth import (
    get_nifty_constituents,
    fetch_bulk_price_data,
    compute_sector_relative_strength,
    screen_sector_rotation_signal,
    screen_sector_highs_gt_lows,
    _all_tickers,
    _ticker_to_sector,
)
from app.utils.cache import yfinance_rate_limiter
from app.tools.screener.batch_fundamentals import fetch_bulk_info
from app.tools.screener.technical_screens import (
    screen_above_200dma,
    screen_rsi_range,
    screen_rsi_oversold,
    screen_volume_breakout,
    screen_golden_alignment,
    screen_near_52w_high,
    screen_macd_bullish,
    screen_obv_rising,
)
from app.tools.screener.quality_screens import (
    screen_opm_improvement,
    screen_cf_turnaround,
    screen_debt_reduction,
    screen_dividend_initiation,
    screen_working_capital_improving,
    screen_piotroski_fscore,
    screen_altman_zscore,
)
from app.tools.screener.quantitative import (
    screen_roe,
    screen_pb_vs_historical,
    screen_ps_vs_historical,
    screen_roe_above,
    screen_ev_ebitda_below_median,
    screen_fcf_yield_above_sector,
)
from app.tools.screener.qualitative import (
    screen_capacity_utilization,
    screen_announcements,
    screen_insider_transactions,
    screen_no_sebi_red_flags,
    screen_capex_going_live,
    screen_delivery_pct,
)
from app.tools.screener.ownership_screens import (
    screen_promoter_holding_strong,
    screen_institutional_increasing,
)
from app.tools.screener.geopolitical import (
    screen_geopolitical_risk,
    screen_supply_chain_advantage,
    screen_export_pli_beneficiary,
)
from app.tools.screener.gate_defs import (
    GateDef,
    MOMENTUM_GATES,
    MOMENTUM_OVERALL,
    VALUE_GATES,
    VALUE_OVERALL,
    TIER_CONFIG,
    TIER_ORDER,
    MOMENTUM_TIER_THRESHOLDS,
    VALUE_TIER_THRESHOLDS,
    evaluate_gates,
    compute_overall_score,
)
from app.tools.screener.regulatory import screen_regulatory_tailwind
from app.tools.screener.smart_money import screen_smart_money_lag
from app.tools.screener.mgmt_credibility import screen_management_credibility
from app.tools.screener.promoter_anomaly import screen_promoter_anomalies

logger = logging.getLogger(__name__)


# ── Criteria registries (used by UI tree view) ────────────────

MOMENTUM_CRITERIA: dict[str, dict] = {
    # Gate 1: Sector Tide (HARD, 2/3)
    "sector_rs_positive":    {"label": "Sector RS positive (1M & 3M)", "gate": "Sector Tide", "default": True, "cost": "fast"},
    "sector_highs_gt_lows":  {"label": "Sector 52W highs > lows", "gate": "Sector Tide", "default": True, "cost": "fast"},
    "sector_top4":           {"label": "Sector in top 4 by 3M RS", "gate": "Sector Tide", "default": True, "cost": "fast"},
    # Gate 2: Trend Structure (HARD, 3/5, above_200dma mandatory)
    "above_200dma":          {"label": "Price > 200 DMA", "gate": "Trend Structure", "default": True, "cost": "fast", "mandatory": True},
    "golden_alignment":      {"label": "Golden alignment (P>50>200 DMA)", "gate": "Trend Structure", "default": True, "cost": "fast"},
    "rsi_55_75":             {"label": "RSI 55-75 (momentum zone)", "gate": "Trend Structure", "default": True, "cost": "fast"},
    "near_52w_high":         {"label": "Within 10% of 52W high", "gate": "Trend Structure", "default": True, "cost": "fast"},
    "macd_bullish":          {"label": "MACD above signal line", "gate": "Trend Structure", "default": True, "cost": "fast"},
    # Gate 3: Volume Conviction (SOFT, 1/2 active)
    "volume_breakout":       {"label": "Volume breakout (2x avg)", "gate": "Volume Conviction", "default": True, "cost": "medium"},
    "delivery_pct":          {"label": "Delivery % > 60% (NSE)", "gate": "Volume Conviction", "default": False, "cost": "slow", "deferred": True},
    "obv_rising":            {"label": "OBV rising 20 days", "gate": "Volume Conviction", "default": True, "cost": "medium"},
    # Gate 4: Smart Money Flow (SOFT, 1/2 active)
    "promoter_holding":      {"label": "Promoter holding > 50%", "gate": "Smart Money Flow", "default": True, "cost": "fast"},
    "institutional_interest": {"label": "Institutional interest increasing", "gate": "Smart Money Flow", "default": True, "cost": "fast"},
    "mf_scheme_count":       {"label": "MF scheme count increasing", "gate": "Smart Money Flow", "default": False, "cost": "slow", "deferred": True},
    # Gate 5: Fundamental Backbone (SOFT, 1/3)
    "opm_improvement":       {"label": "Improving operating margins", "gate": "Fundamental Backbone", "default": True, "cost": "fast"},
    "order_booking":         {"label": "Order booking announcements", "gate": "Fundamental Backbone", "default": False, "cost": "slow"},
    "roe_above_12":          {"label": "ROE > 12% or Revenue Growth > 20%", "gate": "Fundamental Backbone", "default": True, "cost": "fast"},
}

VALUE_CRITERIA: dict[str, dict] = {
    # Gate 1: Valuation Floor (HARD, 3/5)
    "low_roe":               {"label": "ROE < 10% (cheap valuation)", "gate": "Valuation Floor", "default": True, "cost": "fast"},
    "pb_below_avg":          {"label": "P/B below 4yr avg", "gate": "Valuation Floor", "default": True, "cost": "fast"},
    "ps_below_avg":          {"label": "P/S below 4yr avg", "gate": "Valuation Floor", "default": True, "cost": "fast"},
    "ev_ebitda_below_median": {"label": "EV/EBITDA below sector median", "gate": "Valuation Floor", "default": True, "cost": "fast"},
    "fcf_yield_above_sector": {"label": "FCF Yield > sector avg", "gate": "Valuation Floor", "default": True, "cost": "fast"},
    # Gate 2: Turnaround Signals (HARD, 2/5)
    "debt_reduction":        {"label": "Debt reduction trend", "gate": "Turnaround Signals", "default": True, "cost": "fast"},
    "opm_improvement":       {"label": "Improving operating margins", "gate": "Turnaround Signals", "default": True, "cost": "fast"},
    "cf_turnaround":         {"label": "Cash flow turnaround", "gate": "Turnaround Signals", "default": True, "cost": "fast"},
    "dividend_initiation":   {"label": "Dividend initiation/resumption", "gate": "Turnaround Signals", "default": True, "cost": "fast"},
    "working_capital":       {"label": "Working capital improving", "gate": "Turnaround Signals", "default": True, "cost": "fast"},
    # Gate 3: Trap Avoidance (HARD, 2/3)
    "piotroski":             {"label": "Piotroski F-Score >= 5", "gate": "Trap Avoidance", "default": True, "cost": "fast"},
    "altman_z":              {"label": "Altman Z-Score > 1.8", "gate": "Trap Avoidance", "default": True, "cost": "fast"},
    "no_sebi_red_flags":     {"label": "No SEBI/audit red flags", "gate": "Trap Avoidance", "default": False, "cost": "slow"},
    # Gate 4: Technical Timing (SOFT, 1/3 active)
    "rsi_oversold":          {"label": "RSI < 40 (oversold)", "gate": "Technical Timing", "default": True, "cost": "fast"},
    "above_200dma":          {"label": "Price > 200 DMA", "gate": "Technical Timing", "default": True, "cost": "fast"},
    "delivery_pct":          {"label": "Delivery % > 60% (NSE)", "gate": "Technical Timing", "default": False, "cost": "slow", "deferred": True},
    "sector_rotation":       {"label": "Sector rotation signal", "gate": "Technical Timing", "default": True, "cost": "fast"},
    # Gate 5: Re-rating Catalysts (SOFT, 2/5)
    "capacity_util":         {"label": "Low capacity utilization", "gate": "Re-rating Catalysts", "default": False, "cost": "medium"},
    "capex":                 {"label": "Capex announcements", "gate": "Re-rating Catalysts", "default": False, "cost": "slow"},
    "new_business":          {"label": "New business/diversification", "gate": "Re-rating Catalysts", "default": False, "cost": "slow"},
    "order_booking":         {"label": "Order booking announcements", "gate": "Re-rating Catalysts", "default": False, "cost": "slow"},
    "capex_going_live":      {"label": "Capex going live", "gate": "Re-rating Catalysts", "default": False, "cost": "slow"},
    # Gate 6: Insider Conviction (SOFT, 2/4 active)
    "insider_buying":        {"label": "Insider buying activity", "gate": "Insider Conviction", "default": True, "cost": "medium"},
    "promoter_holding":      {"label": "Promoter holding > 50%", "gate": "Insider Conviction", "default": True, "cost": "fast"},
    "pledge_reducing":       {"label": "Pledge % reducing", "gate": "Insider Conviction", "default": False, "cost": "slow", "deferred": True},
    "mgmt_credibility":      {"label": "Management credibility", "gate": "Insider Conviction", "default": True, "cost": "fast"},
    "promoter_behavior":     {"label": "Promoter behavior signals", "gate": "Insider Conviction", "default": False, "cost": "slow"},
    # Gate 7: Macro Tailwinds (BONUS, 0 min)
    "geopolitical":          {"label": "Geopolitical resilience", "gate": "Macro Tailwinds", "default": True, "cost": "fast"},
    "smart_money":           {"label": "Smart money lag", "gate": "Macro Tailwinds", "default": True, "cost": "fast"},
    "regulatory":            {"label": "Regulatory tailwind", "gate": "Macro Tailwinds", "default": True, "cost": "fast"},
    "supply_chain_advantage": {"label": "Supply chain advantage (China+1)", "gate": "Macro Tailwinds", "default": True, "cost": "fast"},
    "export_pli_beneficiary": {"label": "Export / PLI beneficiary", "gate": "Macro Tailwinds", "default": True, "cost": "fast"},
}


def get_default_criteria(category: str) -> set[str]:
    """Return the set of criteria keys enabled by default for a category."""
    registry = MOMENTUM_CRITERIA if category == "Momentum" else VALUE_CRITERIA
    return {k for k, v in registry.items() if v["default"]}


# ── Shared helpers ────────────────────────────────────────────


def _merge_and_score(
    screen_results: dict[str, list[dict]],
    sector_map: dict[str, str],
    category: str,
    min_criteria: int = 1,
) -> list[dict[str, Any]]:
    """DEPRECATED — use _merge_and_evaluate_gates() instead.

    Kept for backward compatibility. Flat count-based scoring.
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

    result = [s for s in stock_data.values() if s["score"] >= min_criteria]
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def _merge_and_evaluate_gates(
    screen_results: dict[str, list[dict]],
    sector_map: dict[str, str],
    category: str,
    gates: list[GateDef],
    overall_threshold: int,
    disabled_criteria: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Gate-based merge: score each stock, evaluate gates, apply overall threshold.

    A stock passes if: all non-bonus gates pass AND overall_score >= threshold.
    """
    if disabled_criteria is None:
        disabled_criteria = set()

    # Step 1: build per-stock criteria sets
    # Screens may return all stocks with passed=True/False, or only passing stocks.
    # Only count a criterion as "passed" if stock["passed"] is True (default for legacy screens).
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
                }
            # Always store details (raw data) for export/debugging
            stock_data[ticker]["criteria_details"][criterion] = stock
            # Only count as passed if the screen marked it as passed
            if stock.get("passed", True):
                stock_data[ticker]["criteria_passed"].append(criterion)

    # Step 2: evaluate gates and overall score per stock
    result: list[dict[str, Any]] = []
    for ticker, data in stock_data.items():
        passed_set = set(data["criteria_passed"])

        all_gates_pass, gate_details = evaluate_gates(passed_set, gates, disabled_criteria)
        score, active_total, meets_threshold = compute_overall_score(
            passed_set, gates, overall_threshold, disabled_criteria,
        )

        data["score"] = score
        data["active_total"] = active_total
        data["overall_threshold"] = overall_threshold
        data["meets_threshold"] = meets_threshold
        data["all_gates_pass"] = all_gates_pass
        data["gate_results"] = gate_details
        data["gates_passed"] = sum(1 for g in gate_details if g["passed"])
        data["gates_total"] = len(gate_details)
        data["passed"] = all_gates_pass and meets_threshold

        # Tier classification
        data["score_ratio"] = score / active_total if active_total > 0 else 0
        data["hard_gates_pass"] = all(
            g["passed"] for g in gate_details if g["gate_type"] == "hard"
        )
        data["soft_gates_failed"] = sum(
            1 for g in gate_details if g["gate_type"] == "soft" and not g["passed"]
        )

        # Pick tier thresholds based on category
        tier_thresholds = MOMENTUM_TIER_THRESHOLDS if category == "Momentum" else VALUE_TIER_THRESHOLDS
        data["tier"] = _classify_tier(data, tier_thresholds)
        # Update "passed" to mean stock is in a passing tier (buy_zone/watchlist/monitor)
        data["passed"] = data["tier"] in ("buy_zone", "watchlist", "monitor")

        tier_info = TIER_CONFIG.get(data["tier"], TIER_CONFIG["failed"])
        data["tier_label"] = tier_info["label"]
        data["tier_color"] = tier_info["color"]
        data["tier_icon"] = tier_info["icon"]

        # Near miss reason
        if data["tier"] == "near_miss":
            failed_soft = [
                g["name"] for g in gate_details
                if g["gate_type"] == "soft" and not g["passed"]
            ]
            data["near_miss_reason"] = (
                f"All hard gates passed. Failed {', '.join(failed_soft)} "
                f"({data['soft_gates_failed']} soft gate). "
                f"Score {score}/{active_total}."
            )
        else:
            data["near_miss_reason"] = ""

        # Include all stocks (even failed) so UI can show why they failed
        result.append(data)

    # Sort: by tier order, then by score descending within tier
    _tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    result.sort(key=lambda x: (_tier_rank.get(x["tier"], 9), -x["score"]))
    return result


def _classify_tier(stock: dict, tier_thresholds: dict) -> str:
    """Classify a stock into a conviction tier based on absolute score thresholds.

    Buy Zone:  all gates pass + score >= buy_zone threshold
    Watchlist: all gates pass + score >= watchlist threshold
    Monitor:   all gates pass + score >= monitor threshold
    Near Miss: 0 hard gate failures + exactly 1 soft gate failure + score >= near_miss_score
    Failed:    everything else
    """
    score = stock["score"]

    # Passing tiers: ALL gates (hard + soft) must pass
    if stock["all_gates_pass"]:
        if score >= tier_thresholds["buy_zone"]:
            return "buy_zone"
        elif score >= tier_thresholds["watchlist"]:
            return "watchlist"
        elif score >= tier_thresholds["monitor"]:
            return "monitor"

    # Near miss: all hard gates pass, exactly 1 soft gate failed, score above near_miss_score
    if (
        stock["hard_gates_pass"]
        and stock["soft_gates_failed"] == 1
        and score >= tier_thresholds["near_miss_score"]
    ):
        return "near_miss"

    return "failed"


def _get_tickers_from_results(screen_results: dict[str, list[dict]]) -> set[str]:
    """Extract unique tickers from screen results."""
    tickers = set()
    for stocks in screen_results.values():
        for s in stocks:
            tickers.add(s["ticker"])
    return tickers


# ── Momentum Category (17 criteria) ──────────────────────────


def screen_momentum(
    sector_map: dict[str, list[str]],
    price_df: pd.DataFrame,
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    tickers: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
    enabled_criteria: set[str] | None = None,
    min_criteria: int | None = None,
) -> list[dict[str, Any]]:
    """Momentum: Buy what's already working.

    Sector filter first (eliminates ~60%), then 14 remaining screens.
    RSI 55-75 zone. Min 5 of 17 criteria to pass.

    Criteria:
      1. Sector RS positive (1M, 3M)
      2. Sector 52W highs > lows
      3. Sector in top 4 by 3M RS
      4. Price > 200 DMA
      5. Golden alignment (Price > 50DMA > 200DMA)
      6. RSI 55-75
      7. Within 10% of 52W high
      8. MACD above signal line
      9. Volume breakout (2x 20-day avg)
     10. Delivery % > 60% (DEFERRED)
     11. OBV rising 20 days
     12. Promoter holding > 50%
     13. Institutional interest increasing
     14. MF scheme count increasing (DEFERRED)
     15. Improving operating margins
     16. Order booking announcements
     17. ROE > 12%
    """
    # Use default criteria if none specified
    if enabled_criteria is None:
        enabled_criteria = get_default_criteria("Momentum")

    # Default min_criteria: ~50% of enabled screens (target 20-25 survivors)
    if min_criteria is None:
        min_criteria = max(5, len(enabled_criteria) // 2)

    def _on(key: str) -> bool:
        return key in enabled_criteria

    if progress_cb:
        progress_cb(0.0, f"Momentum: {len(enabled_criteria)} criteria, min {min_criteria} to pass...")

    results: dict[str, list[dict]] = {}

    # ── Sector criteria (scored, NOT used as pre-filter) ──────
    # All stocks get scored — sector criteria add points but don't eliminate
    sector_rs = compute_sector_relative_strength(price_df, sector_map)

    if _on("sector_rs_positive"):
        strong_sectors = set()
        for row in sector_rs:
            if row.get("rs_1m", 0) > 0 and row.get("rs_3m", 0) > 0:
                strong_sectors.add(row["sector"])
        results["sector_rs_positive"] = [
            {"ticker": t, "sector": t2s.get(t, "Other"), "passed": True}
            for t in tickers if t2s.get(t) in strong_sectors
        ]

    if _on("sector_top4"):
        top4_sectors = {row["sector"] for row in sector_rs[:4]}
        results["sector_top4"] = [
            {"ticker": t, "sector": t2s.get(t, "Other"), "passed": True}
            for t in tickers if t2s.get(t) in top4_sectors
        ]

    if _on("sector_highs_gt_lows"):
        results["sector_highs_gt_lows"] = screen_sector_highs_gt_lows(price_df, t2s)

    logger.info("Momentum: scoring ALL %d tickers (no sector pre-filter)", len(tickers))

    if progress_cb:
        progress_cb(0.2, f"Momentum: Technical screens on {len(tickers)} stocks...")

    # ── All screens run on FULL universe ──────────────────────
    avail_cols = [t for t in tickers if t in price_df.columns]
    all_price_df = price_df[avail_cols] if avail_cols else pd.DataFrame()
    all_bulk = {t: bulk_info[t] for t in tickers if t in bulk_info}
    all_t2s = {t: t2s[t] for t in tickers if t in t2s}

    # Pre-download volume data only if volume/OBV screens are enabled
    need_volume = _on("volume_breakout") or _on("obv_rising")
    volume_df = pd.DataFrame()
    if need_volume:
        from app.utils.cache import cache_get, cache_set
        vol_cache_key = f"volume_data:{len(tickers)}"
        volume_df = cache_get(vol_cache_key)
        if volume_df is None:
            from app.tools.screener.technical_screens import _download_volume_batch, _BATCH_SIZE
            vol_batches = [tickers[i:i + _BATCH_SIZE] for i in range(0, len(tickers), _BATCH_SIZE)]
            vol_frames = []
            for batch in vol_batches:
                df = _download_volume_batch(batch)
                if not df.empty:
                    vol_frames.append(df)
            volume_df = pd.concat(vol_frames, axis=1) if vol_frames else pd.DataFrame()
            if not volume_df.empty:
                volume_df = volume_df.loc[:, ~volume_df.columns.duplicated()]
                cache_set(vol_cache_key, volume_df, ttl=600)

    # Build parallel screen tasks based on enabled criteria
    parallel_screens: dict = {}
    if _on("above_200dma"):
        parallel_screens["above_200dma"] = lambda: screen_above_200dma(all_price_df, all_t2s)
    if _on("golden_alignment"):
        parallel_screens["golden_alignment"] = lambda: screen_golden_alignment(all_price_df, all_t2s)
    if _on("rsi_55_75"):
        parallel_screens["rsi_55_75"] = lambda: screen_rsi_range(all_price_df, all_t2s, 55, 75)
    if _on("near_52w_high"):
        parallel_screens["near_52w_high"] = lambda: screen_near_52w_high(all_price_df, all_t2s, 0.10)
    if _on("macd_bullish"):
        parallel_screens["macd_bullish"] = lambda: screen_macd_bullish(all_price_df, all_t2s)
    if _on("volume_breakout"):
        parallel_screens["volume_breakout"] = lambda: screen_volume_breakout(tickers, all_t2s, volume_df)

    if parallel_screens:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(fn): key for key, fn in parallel_screens.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result(timeout=_SCREEN_TIMEOUT)
                except TimeoutError:
                    logger.warning("Momentum screen %s timed out after %ds", key, _SCREEN_TIMEOUT)
                    results[key] = []
                except Exception as e:
                    logger.warning("Momentum screen %s failed: %s", key, e)
                    results[key] = []

    if progress_cb:
        progress_cb(0.6, "Momentum: OBV + fundamental screens...")

    if _on("obv_rising"):
        try:
            results["obv_rising"] = screen_obv_rising(all_price_df, all_t2s, volume_df=volume_df)
        except Exception as e:
            logger.warning("OBV screen failed: %s", e)
            results["obv_rising"] = []

    if _on("promoter_holding"):
        results["promoter_holding"] = screen_promoter_holding_strong(all_bulk, all_t2s)
    if _on("institutional_interest"):
        results["institutional_interest"] = screen_institutional_increasing(all_bulk, all_t2s)

    if _on("delivery_pct"):
        try:
            results["delivery_pct"] = screen_delivery_pct(tickers[:50], all_t2s, threshold=60.0)
        except Exception as e:
            logger.warning("Delivery %% screen failed: %s", e)
            results["delivery_pct"] = []

    if _on("opm_improvement"):
        results["opm_improvement"] = screen_opm_improvement(all_bulk, all_t2s)
    if _on("roe_above_12"):
        results["roe_above_12"] = screen_roe_above(all_bulk, all_t2s, threshold=0.12)

    if _on("order_booking"):
        if progress_cb:
            progress_cb(0.8, "Momentum: Announcement screens...")
        try:
            announcements = screen_announcements(tickers[:100], all_t2s)
            results["order_booking"] = announcements.get("order_booking", [])
        except Exception as e:
            logger.warning("Momentum announcement screen failed: %s", e)
            results["order_booking"] = []

    if progress_cb:
        total = len(_get_tickers_from_results(results))
        progress_cb(1.0, f"Momentum: {total} unique stocks across {len(results)} screens")

    # Compute disabled criteria (all criteria NOT in enabled set)
    all_mom_keys = {k for g in MOMENTUM_GATES for k in g["criteria_keys"]}
    disabled = all_mom_keys - enabled_criteria

    return _merge_and_evaluate_gates(
        results, t2s, "Momentum", MOMENTUM_GATES,
        overall_threshold=MOMENTUM_OVERALL["threshold"],
        disabled_criteria=disabled,
    )


# ── Value Bottom Category (33 criteria, 7 gates) ─────────────


def screen_value_bottom(
    sector_map: dict[str, list[str]],
    price_df: pd.DataFrame,
    bulk_info: dict[str, dict],
    t2s: dict[str, str],
    tickers: list[str],
    progress_cb: Callable[[float, str], None] | None = None,
    enabled_criteria: set[str] | None = None,
    min_criteria: int | None = None,
) -> list[dict[str, Any]]:
    """Value Bottom: Buy what nobody wants yet.

    Two-phase: cheap valuation pre-filter, then expensive screens on candidates.
    RSI <40 zone. Min 6 of 28 criteria to pass.

    Criteria:
      1. ROE < 10% (2yr)          2. P/B below 4yr avg
      3. P/S below 4yr avg        4. EV/EBITDA below sector median
      5. FCF Yield > sector avg   6. Debt reduction trend
      7. Improving OPM             8. Cash flow turnaround
      9. Dividend initiation      10. Working capital improving
     11. Piotroski F-Score >= 5   12. Altman Z-Score > 1.8
     13. No SEBI/audit red flags  14. RSI < 40
     15. Price > 200 DMA          16. Delivery % > 60% (DEFERRED)
     17. Low capacity utilization  18. Insider buying activity
     19. Capex announcements      20. New business announcements
     21. Order booking            22. Capex going live
     23. Promoter holding > 50%   24. Pledge reducing (DEFERRED)
     25. Management credibility   26. Geopolitical resilience
     27. Smart money lag          28. Regulatory tailwind
     29. Promoter behavior        30. Sector rotation signal
    """
    if enabled_criteria is None:
        enabled_criteria = get_default_criteria("ValueBottom")

    # Default min_criteria: ~40% of enabled screens (target 20-25 survivors)
    if min_criteria is None:
        min_criteria = max(5, len(enabled_criteria) * 2 // 5)

    def _on(key: str) -> bool:
        return key in enabled_criteria

    if progress_cb:
        progress_cb(0.0, f"Value Bottom: {len(enabled_criteria)} criteria, min {min_criteria} to pass...")

    results: dict[str, list[dict]] = {}
    all_bulk = {t: bulk_info[t] for t in tickers if t in bulk_info}
    all_t2s = {t: t2s[t] for t in tickers if t in t2s}

    logger.info("Value Bottom: scoring ALL %d tickers (no pre-filter)", len(tickers))

    # ── Valuation screens (from .info — fast) ─────────────────
    if _on("low_roe"):
        results["low_roe"] = screen_roe(all_bulk, all_t2s)
    if _on("ev_ebitda_below_median"):
        results["ev_ebitda_below_median"] = screen_ev_ebitda_below_median(all_bulk, all_t2s)
    if _on("fcf_yield_above_sector"):
        results["fcf_yield_above_sector"] = screen_fcf_yield_above_sector(all_bulk, all_t2s)
    if _on("promoter_holding"):
        results["promoter_holding"] = screen_promoter_holding_strong(all_bulk, all_t2s)

    # Price-based screens (parallel, fast)
    avail_cols = [t for t in tickers if t in price_df.columns]
    ticker_price_df = price_df[avail_cols] if avail_cols else pd.DataFrame()

    price_screens: dict = {}
    if _on("rsi_oversold"):
        price_screens["rsi_oversold"] = lambda: screen_rsi_oversold(ticker_price_df, all_t2s, 40)
    if _on("above_200dma"):
        price_screens["above_200dma"] = lambda: screen_above_200dma(ticker_price_df, all_t2s)
    if _on("pb_below_avg"):
        price_screens["pb_below_avg"] = lambda: screen_pb_vs_historical(all_bulk, ticker_price_df, all_t2s)
    if _on("ps_below_avg"):
        price_screens["ps_below_avg"] = lambda: screen_ps_vs_historical(all_bulk, ticker_price_df, all_t2s)

    if price_screens:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fn): key for key, fn in price_screens.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result(timeout=_SCREEN_TIMEOUT)
                except TimeoutError:
                    logger.warning("Value Bottom screen %s timed out after %ds", key, _SCREEN_TIMEOUT)
                    results[key] = []
                except Exception as e:
                    logger.warning("Value Bottom screen %s failed: %s", key, e)
                    results[key] = []

    if progress_cb:
        progress_cb(0.25, f"Value Bottom: Quality screens on {len(tickers)} stocks...")

    # ── Quality screens (all on full universe) ─────────────────
    if _on("debt_reduction"):
        results["debt_reduction"] = screen_debt_reduction(all_bulk, all_t2s)
    if _on("opm_improvement"):
        results["opm_improvement"] = screen_opm_improvement(all_bulk, all_t2s)
    if _on("cf_turnaround"):
        results["cf_turnaround"] = screen_cf_turnaround(all_bulk, all_t2s)
    if _on("dividend_initiation"):
        results["dividend_initiation"] = screen_dividend_initiation(all_bulk, all_t2s)
    if _on("working_capital"):
        results["working_capital"] = screen_working_capital_improving(all_bulk, all_t2s)

    if progress_cb:
        progress_cb(0.45, "Value Bottom: Piotroski + Altman scores...")

    if _on("piotroski"):
        results["piotroski"] = screen_piotroski_fscore(all_bulk, all_t2s, min_score=5)
    if _on("altman_z"):
        results["altman_z"] = screen_altman_zscore(all_bulk, all_t2s, min_z=1.8)

    if progress_cb:
        progress_cb(0.60, "Value Bottom: Qualitative screens...")

    # ── Qualitative / announcement screens (capped for speed) ──
    ann_tickers = tickers[:100]
    need_announcements = _on("capex") or _on("new_business") or _on("order_booking")
    if need_announcements:
        try:
            announcements = screen_announcements(ann_tickers, all_t2s)
            if _on("capex"):
                results["capex"] = announcements.get("capex", [])
            if _on("new_business"):
                results["new_business"] = announcements.get("new_business", [])
            if _on("order_booking"):
                results["order_booking"] = announcements.get("order_booking", [])
        except Exception as e:
            logger.warning("Value Bottom announcement screens failed: %s", e)

    if _on("capex_going_live"):
        try:
            results["capex_going_live"] = screen_capex_going_live(ann_tickers, all_t2s)
        except Exception as e:
            logger.warning("Capex going live screen failed: %s", e)
            results["capex_going_live"] = []

    if _on("no_sebi_red_flags"):
        try:
            results["no_sebi_red_flags"] = screen_no_sebi_red_flags(ann_tickers, all_t2s)
        except Exception as e:
            logger.warning("SEBI red flags screen failed: %s", e)
            results["no_sebi_red_flags"] = []

    if _on("insider_buying"):
        results["insider_buying"] = screen_insider_transactions(ann_tickers, all_t2s)
    if _on("capacity_util"):
        results["capacity_util"] = screen_capacity_utilization(ann_tickers, all_t2s)

    if _on("delivery_pct"):
        try:
            results["delivery_pct"] = screen_delivery_pct(tickers[:50], all_t2s, threshold=60.0)
        except Exception as e:
            logger.warning("Delivery %% screen failed: %s", e)
            results["delivery_pct"] = []

    if progress_cb:
        progress_cb(0.80, "Value Bottom: USP screens...")

    # ── USP-type screens ──────────────────────────────────────
    if _on("mgmt_credibility"):
        results["mgmt_credibility"] = screen_management_credibility(all_bulk, all_t2s)
    if _on("geopolitical"):
        results["geopolitical"] = screen_geopolitical_risk(all_bulk, all_t2s)
    if _on("smart_money"):
        results["smart_money"] = screen_smart_money_lag(all_bulk, all_t2s)
    if _on("regulatory"):
        results["regulatory"] = screen_regulatory_tailwind(all_bulk, all_t2s)
    if _on("promoter_behavior"):
        results["promoter_behavior"] = screen_promoter_anomalies(tickers, all_t2s, all_bulk)

    if _on("sector_rotation"):
        try:
            rotation_sectors = screen_sector_rotation_signal(price_df, sector_map)
            rotating_sector_names = {r["sector"] for r in rotation_sectors}
            results["sector_rotation"] = [
                {"ticker": t, "sector": t2s.get(t, "Other"), "passed": True}
                for t in tickers if t2s.get(t) in rotating_sector_names
            ]
        except Exception as e:
            logger.warning("Sector rotation screen failed: %s", e)
            results["sector_rotation"] = []

    # Gate 7 new screens
    if _on("supply_chain_advantage"):
        results["supply_chain_advantage"] = screen_supply_chain_advantage(all_bulk, all_t2s)
    if _on("export_pli_beneficiary"):
        results["export_pli_beneficiary"] = screen_export_pli_beneficiary(all_bulk, all_t2s)

    if progress_cb:
        total = len(_get_tickers_from_results(results))
        progress_cb(1.0, f"Value Bottom: {total} unique stocks across {len(results)} screens")

    # Compute disabled criteria
    all_val_keys = {k for g in VALUE_GATES for k in g["criteria_keys"]}
    disabled = all_val_keys - enabled_criteria

    return _merge_and_evaluate_gates(
        results, t2s, "ValueBottom", VALUE_GATES,
        overall_threshold=VALUE_OVERALL["threshold"],
        disabled_criteria=disabled,
    )


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
                usp_raw[key] = future.result(timeout=_SCREEN_TIMEOUT)
            except TimeoutError:
                logger.warning("USP module %s timed out after %ds", key, _SCREEN_TIMEOUT)
                usp_raw[key] = []
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
        progress_cb(0.85, f"USP: Scored {len(per_stock)} stocks across 5 dimensions")

    # ── LLM Commentary Layer ──────────────────────────────────
    try:
        from app.tools.screener.usp_commentary import (
            generate_usp_commentary,
            generate_portfolio_insights,
        )

        if progress_cb:
            progress_cb(0.86, f"USP: Generating LLM commentary for {len(per_stock)} stocks...")

        commentary = generate_usp_commentary(
            per_stock, bulk_info,
            progress_cb=lambda f, m: progress_cb(0.86 + f * 0.10, m) if progress_cb else None,
        )
        for ticker, comm in commentary.items():
            if ticker in per_stock:
                per_stock[ticker]["_commentary"] = comm

        if progress_cb:
            progress_cb(0.97, "USP: Generating portfolio insights...")

        portfolio_insights = generate_portfolio_insights(per_stock)
        if portfolio_insights:
            per_stock["_portfolio_insights"] = {"text": portfolio_insights}

    except Exception as e:
        logger.warning("USP commentary generation failed (non-fatal): %s", e)

    if progress_cb:
        progress_cb(1.0, f"USP: Complete — {len(per_stock)} stocks scored with commentary")

    return per_stock


# ── Main Orchestrator ─────────────────────────────────────────


def run_category_screener(
    categories: list[str] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    max_stocks: int | None = None,
    enabled_criteria: dict[str, set[str]] | None = None,
    min_criteria: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run the 2-category screening pipeline.

    Args:
        categories: List of categories to run ("Momentum", "ValueBottom"). None = both.
        progress_cb: Optional callback(fraction, message).
        max_stocks: Limit universe size (for testing).
        enabled_criteria: {category: set_of_criterion_keys}. None = use defaults.
        min_criteria: {category: min_count}. None = use defaults (~50% of enabled).

    Returns dict with keys: category_results, usp_scores, universe_info, regime_data.
    """
    # Reset rate limiter state from prior runs to avoid stale timestamps
    yfinance_rate_limiter.reset()

    if categories is None:
        categories = ["Momentum", "ValueBottom"]

    # ── Load universe ────────────────────────────────────────
    sector_map = get_nifty_constituents()
    all_tickers = _all_tickers(sector_map)
    t2s = _ticker_to_sector(sector_map)

    if max_stocks and max_stocks < len(all_tickers):
        all_tickers = all_tickers[:max_stocks]

    if progress_cb:
        progress_cb(0.02, f"Loading {len(all_tickers)} tickers...")

    # ── Phase 1: Batch price download ────────────────────────
    if progress_cb:
        progress_cb(0.05, "Phase 1: Downloading price data...")

    price_df = fetch_bulk_price_data(
        all_tickers, period="1y",
        progress_cb=lambda f, m: progress_cb(0.05 + f * 0.10, m) if progress_cb else None,
    )

    # ── Phase 2: Price-first filtering ───────────────────────
    if progress_cb:
        progress_cb(0.15, "Phase 2: Technical pre-filter...")

    tech_survivors: set[str] = set()

    above_200 = screen_above_200dma(price_df, t2s)
    tech_survivors.update(s["ticker"] for s in above_200)

    rsi_mid = screen_rsi_range(price_df, t2s, lo=20, hi=80)
    tech_survivors.update(s["ticker"] for s in rsi_mid)

    if len(tech_survivors) < len(all_tickers) * 0.3:
        tech_survivors = set(all_tickers)
        logger.info("Tech pre-filter too aggressive — keeping all tickers")

    filtered_tickers = list(tech_survivors)
    logger.info("Price-first filter: %d → %d tickers for .info fetch",
                len(all_tickers), len(filtered_tickers))

    # ── Phase 3: Fetch fundamentals ──────────────────────────
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
        progress_cb(0.52, "Running category screens in parallel...")

    category_results: dict[str, list[dict]] = {}

    if enabled_criteria is None:
        enabled_criteria = {}
    if min_criteria is None:
        min_criteria = {}

    _cat_tasks: dict[str, tuple] = {}
    if "Momentum" in categories:
        mom_criteria = enabled_criteria.get("Momentum")
        mom_min = min_criteria.get("Momentum")
        _cat_tasks["Momentum"] = (screen_momentum, (sector_map, price_df, bulk_info, t2s, filtered_tickers, None, mom_criteria, mom_min))
    if "ValueBottom" in categories:
        val_criteria = enabled_criteria.get("ValueBottom")
        val_min = min_criteria.get("ValueBottom")
        _cat_tasks["ValueBottom"] = (screen_value_bottom, (sector_map, price_df, bulk_info, t2s, filtered_tickers, None, val_criteria, val_min))

    with ThreadPoolExecutor(max_workers=max(1, len(_cat_tasks))) as executor:
        future_to_cat = {
            executor.submit(fn, *args): cat
            for cat, (fn, args) in _cat_tasks.items()
        }
        for future in as_completed(future_to_cat):
            cat = future_to_cat[future]
            try:
                category_results[cat] = future.result(timeout=300)  # 5min max per category
                if progress_cb:
                    progress_cb(0.52 + 0.36 * len(category_results) / len(_cat_tasks),
                                f"{cat} done: {len(category_results[cat])} stocks")
            except TimeoutError:
                logger.error("Category %s timed out after 5 minutes", cat)
                category_results[cat] = []
            except Exception as e:
                logger.error("Category %s screening failed: %s", cat, e)
                category_results[cat] = []

    # ── Phase 5: USP layer on passed stocks only ────────────
    all_survivors = set()
    for cat_stocks in category_results.values():
        for stock in cat_stocks:
            if stock.get("passed", True):  # backward compat: if no "passed" key, include
                all_survivors.add(stock["ticker"])

    recommended_tickers = list(all_survivors)
    logger.info("Total survivors across categories: %d", len(recommended_tickers))

    if progress_cb:
        progress_cb(0.88, f"Running USP analysis on {len(recommended_tickers)} stocks...")

    usp_scores = apply_usp_layer(recommended_tickers, bulk_info, t2s)

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
        "gate_configs": {
            "Momentum": MOMENTUM_GATES,
            "ValueBottom": VALUE_GATES,
        },
    }


# ── Single-Stock Diagnostic ──────────────────────────────────


def run_single_stock_diagnostic(
    ticker: str,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run ALL screening criteria on a single stock and report pass/fail for each.

    Returns a dict with per-criterion results for both Momentum and Value gates.
    """
    import yfinance as yf
    from app.tools.screener.technical_screens import _download_volume_batch

    yfinance_rate_limiter.reset()

    if progress_cb:
        progress_cb(0.02, f"Loading data for {ticker}...")

    # ── Determine sector ──
    sector_map = get_nifty_constituents()
    t2s = _ticker_to_sector(sector_map)
    sector = t2s.get(ticker, "Other")
    t2s_single = {ticker: sector}

    # ── Fetch price data ──
    if progress_cb:
        progress_cb(0.05, "Downloading price data...")
    price_df = fetch_bulk_price_data([ticker], period="1y")

    # ── Fetch fundamentals ──
    if progress_cb:
        progress_cb(0.15, "Fetching fundamentals...")
    bulk_info = fetch_bulk_info([ticker])
    info = bulk_info.get(ticker, {})

    # ── Fetch volume data ──
    if progress_cb:
        progress_cb(0.25, "Fetching volume data...")
    try:
        volume_df = _download_volume_batch([ticker])
    except Exception:
        volume_df = pd.DataFrame()

    # ── Sector-level data (needed for sector gates) ──
    if progress_cb:
        progress_cb(0.30, "Computing sector data...")
    all_tickers = _all_tickers(sector_map)
    # Use cached price data for sector computation if available
    from app.utils.cache import cache_get
    sector_price_df = cache_get(f"bulk_prices:{len(all_tickers)}:1y")
    if sector_price_df is None:
        # Minimal: just use single-stock price (sector screens will return empty)
        sector_price_df = price_df

    sector_rs = compute_sector_relative_strength(sector_price_df, sector_map)

    # ── Run ALL screens ──
    if progress_cb:
        progress_cb(0.35, "Running all screens...")

    criteria_results: dict[str, dict[str, Any]] = {}

    def _run_screen(name: str, fn, label: str = ""):
        """Run a screen and record pass/fail + details for single ticker."""
        try:
            result = fn()
            # result is list[dict] — check if our ticker is in it
            if isinstance(result, dict):
                # screen_announcements returns {criterion: [stocks]}
                for sub_key, stocks in result.items():
                    passed = any(s.get("ticker") == ticker for s in stocks)
                    detail = next((s for s in stocks if s.get("ticker") == ticker), {})
                    criteria_results[sub_key] = {
                        "passed": passed,
                        "detail": detail,
                        "label": sub_key,
                    }
                return
            passed = any(s.get("ticker") == ticker for s in result)
            detail = next((s for s in result if s.get("ticker") == ticker), {})
            criteria_results[name] = {
                "passed": passed,
                "detail": detail,
                "label": label or name,
            }
        except Exception as e:
            criteria_results[name] = {
                "passed": False,
                "detail": {"error": str(e)},
                "label": label or name,
            }

    avail_cols = [ticker] if ticker in price_df.columns else []
    single_price = price_df[avail_cols] if avail_cols else pd.DataFrame()
    single_bulk = {ticker: info} if info else {}

    # ── Sector criteria ──
    strong_sectors = set()
    for row in sector_rs:
        if row.get("rs_1m", 0) > 0 and row.get("rs_3m", 0) > 0:
            strong_sectors.add(row["sector"])
    criteria_results["sector_rs_positive"] = {
        "passed": sector in strong_sectors,
        "detail": {"sector": sector, "strong_sectors": list(strong_sectors)[:10]},
        "label": "sector_rs_positive",
    }

    top4 = {row["sector"] for row in sector_rs[:4]}
    criteria_results["sector_top4"] = {
        "passed": sector in top4,
        "detail": {"sector": sector, "top4": list(top4)},
        "label": "sector_top4",
    }

    _run_screen("sector_highs_gt_lows",
                lambda: screen_sector_highs_gt_lows(sector_price_df, t2s))

    if progress_cb:
        progress_cb(0.45, "Running technical screens...")

    # ── Technical screens ──
    _run_screen("above_200dma", lambda: screen_above_200dma(single_price, t2s_single))
    _run_screen("golden_alignment", lambda: screen_golden_alignment(single_price, t2s_single))
    _run_screen("rsi_55_75", lambda: screen_rsi_range(single_price, t2s_single, 55, 75))
    _run_screen("rsi_oversold", lambda: screen_rsi_oversold(single_price, t2s_single, 40))
    _run_screen("near_52w_high", lambda: screen_near_52w_high(single_price, t2s_single, 0.10))
    _run_screen("macd_bullish", lambda: screen_macd_bullish(single_price, t2s_single))
    _run_screen("volume_breakout", lambda: screen_volume_breakout([ticker], t2s_single, volume_df))
    _run_screen("obv_rising", lambda: screen_obv_rising(single_price, t2s_single, volume_df=volume_df))

    if progress_cb:
        progress_cb(0.55, "Running fundamental screens...")

    # ── Fundamental / quality screens ──
    _run_screen("promoter_holding", lambda: screen_promoter_holding_strong(single_bulk, t2s_single))
    _run_screen("institutional_interest", lambda: screen_institutional_increasing(single_bulk, t2s_single))
    _run_screen("opm_improvement", lambda: screen_opm_improvement(single_bulk, t2s_single))
    _run_screen("roe_above_12", lambda: screen_roe_above(single_bulk, t2s_single, 0.12))
    _run_screen("low_roe", lambda: screen_roe(single_bulk, t2s_single))
    _run_screen("debt_reduction", lambda: screen_debt_reduction(single_bulk, t2s_single))
    _run_screen("cf_turnaround", lambda: screen_cf_turnaround(single_bulk, t2s_single))
    _run_screen("dividend_initiation", lambda: screen_dividend_initiation(single_bulk, t2s_single))
    _run_screen("working_capital", lambda: screen_working_capital_improving(single_bulk, t2s_single))
    _run_screen("ev_ebitda_below_median", lambda: screen_ev_ebitda_below_median(single_bulk, t2s_single))
    _run_screen("fcf_yield_above_sector", lambda: screen_fcf_yield_above_sector(single_bulk, t2s_single))
    _run_screen("pb_below_avg", lambda: screen_pb_vs_historical(single_bulk, single_price, t2s_single))
    _run_screen("ps_below_avg", lambda: screen_ps_vs_historical(single_bulk, single_price, t2s_single))

    if progress_cb:
        progress_cb(0.65, "Running quality scores...")

    _run_screen("piotroski", lambda: screen_piotroski_fscore(single_bulk, t2s_single, min_score=5))
    _run_screen("altman_z", lambda: screen_altman_zscore(single_bulk, t2s_single, min_z=1.8))

    if progress_cb:
        progress_cb(0.75, "Running announcement screens...")

    # ── Announcement / qualitative screens ──
    _run_screen("announcements", lambda: screen_announcements([ticker], t2s_single))
    _run_screen("capex_going_live", lambda: screen_capex_going_live([ticker], t2s_single))
    _run_screen("no_sebi_red_flags", lambda: screen_no_sebi_red_flags([ticker], t2s_single))
    _run_screen("insider_buying", lambda: screen_insider_transactions([ticker], t2s_single))
    _run_screen("capacity_util", lambda: screen_capacity_utilization([ticker], t2s_single))

    try:
        _run_screen("delivery_pct", lambda: screen_delivery_pct([ticker], t2s_single, threshold=60.0))
    except Exception:
        criteria_results["delivery_pct"] = {"passed": False, "detail": {"error": "unavailable"}, "label": "delivery_pct"}

    if progress_cb:
        progress_cb(0.85, "Running USP screens...")

    # ── USP / geopolitical screens ──
    _run_screen("geopolitical", lambda: screen_geopolitical_risk(single_bulk, t2s_single))
    _run_screen("smart_money", lambda: screen_smart_money_lag(single_bulk, t2s_single))
    _run_screen("regulatory", lambda: screen_regulatory_tailwind(single_bulk, t2s_single))
    _run_screen("mgmt_credibility", lambda: screen_management_credibility(single_bulk, t2s_single))
    _run_screen("promoter_behavior", lambda: screen_promoter_anomalies([ticker], t2s_single, single_bulk))
    _run_screen("supply_chain_advantage", lambda: screen_supply_chain_advantage(single_bulk, t2s_single))
    _run_screen("export_pli_beneficiary", lambda: screen_export_pli_beneficiary(single_bulk, t2s_single))

    # Sector rotation
    try:
        rotation_sectors = screen_sector_rotation_signal(sector_price_df, sector_map)
        rotating = {r["sector"] for r in rotation_sectors}
        criteria_results["sector_rotation"] = {
            "passed": sector in rotating,
            "detail": {"sector": sector, "rotating_sectors": list(rotating)[:10]},
            "label": "sector_rotation",
        }
    except Exception as e:
        criteria_results["sector_rotation"] = {"passed": False, "detail": {"error": str(e)}, "label": "sector_rotation"}

    if progress_cb:
        progress_cb(0.92, "Evaluating gates...")

    # ── Evaluate gates for both categories ──
    passed_set = {k for k, v in criteria_results.items() if v.get("passed")}

    mom_gates_pass, mom_gate_details = evaluate_gates(passed_set, MOMENTUM_GATES)
    mom_score, mom_active, mom_meets = compute_overall_score(
        passed_set, MOMENTUM_GATES, MOMENTUM_OVERALL["threshold"],
    )

    val_gates_pass, val_gate_details = evaluate_gates(passed_set, VALUE_GATES)
    val_score, val_active, val_meets = compute_overall_score(
        passed_set, VALUE_GATES, VALUE_OVERALL["threshold"],
    )

    if progress_cb:
        progress_cb(1.0, f"Diagnostic complete: {len(passed_set)} criteria passed")

    return {
        "ticker": ticker,
        "sector": sector,
        "info": info,
        "criteria_results": criteria_results,
        "passed_set": passed_set,
        "momentum": {
            "gates_pass": mom_gates_pass,
            "gate_details": mom_gate_details,
            "score": mom_score,
            "active_total": mom_active,
            "meets_threshold": mom_meets,
            "final_pass": mom_gates_pass and mom_meets,
        },
        "value": {
            "gates_pass": val_gates_pass,
            "gate_details": val_gate_details,
            "score": val_score,
            "active_total": val_active,
            "meets_threshold": val_meets,
            "final_pass": val_gates_pass and val_meets,
        },
    }
