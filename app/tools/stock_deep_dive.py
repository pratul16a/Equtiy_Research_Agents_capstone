"""Stock Deep Dive — Profile builder for individual stock analysis.

Aggregates all data sources (LangGraph pipeline + USP modules) into a single
comprehensive stock profile for conversational Q&A.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.config import get_llm
from app.graph import run_research
from app.tools.screener.batch_fundamentals import _fetch_single_info, fetch_ticker_financials
from app.tools.screener.geopolitical import compute_geopolitical_score
from app.tools.screener.smart_money import _compute_fundamental_improvement, _compute_institutional_flow
from app.tools.screener.mgmt_credibility import compute_credibility_score
from app.tools.screener.regulatory import scan_policy_signals, compute_regulatory_score
from app.tools.screener.promoter_anomaly import detect_promoter_buying_pattern, detect_related_party_anomaly

logger = logging.getLogger(__name__)


def _resolve_sector(info: dict) -> str:
    """Extract sector from yfinance info dict."""
    return info.get("sector", info.get("industry", "Other"))


def build_stock_profile(
    ticker: str,
    exchange: str = "NSE",
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive stock profile by running all analysis modules.

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS").
        exchange: "NSE" or "BSE".
        progress_cb: Optional callback(fraction, message) for UI progress.

    Returns a dict with keys: ticker, info, financials_data, research_state,
    geopolitical, fundamental_improvement, institutional_flow, smart_money_lag,
    credibility, regulatory, promoter_buying, related_party.
    """
    profile: dict[str, Any] = {"ticker": ticker, "exchange": exchange}

    # ── Step 1: Basic info ────────────────────────────────────
    if progress_cb:
        progress_cb(0.05, "Fetching basic stock info...")

    info = _fetch_single_info(ticker)
    if not info:
        raise ValueError(f"Could not fetch data for {ticker}. Verify the ticker is correct.")
    profile["info"] = info
    sector = _resolve_sector(info)
    profile["sector"] = sector

    # ── Step 2: Detailed financials ───────────────────────────
    if progress_cb:
        progress_cb(0.10, "Fetching detailed financials...")

    financials_data = fetch_ticker_financials(ticker)
    profile["financials_data"] = financials_data

    # ── Step 3: LangGraph pipeline ────────────────────────────
    if progress_cb:
        progress_cb(0.15, "Running full research pipeline (this takes 1-2 minutes)...")

    try:
        research_state = run_research(ticker, exchange)
        profile["research_state"] = research_state
    except Exception as e:
        logger.error("LangGraph pipeline failed for %s: %s", ticker, e)
        profile["research_state"] = {"errors": [str(e)]}

    if progress_cb:
        progress_cb(0.70, "Research pipeline complete. Running USP analyses...")

    # ── Step 4: Geopolitical risk ─────────────────────────────
    if progress_cb:
        progress_cb(0.72, "Computing geopolitical risk score...")

    try:
        geo = compute_geopolitical_score(ticker, info, sector)
        profile["geopolitical"] = geo
    except Exception as e:
        logger.debug("Geopolitical scoring failed: %s", e)
        profile["geopolitical"] = {"overall_score": 50, "risk_level": "Unknown"}

    # ── Step 5: Smart money lag ───────────────────────────────
    if progress_cb:
        progress_cb(0.76, "Analyzing smart money lag...")

    try:
        imp_score, imp_details = _compute_fundamental_improvement(ticker, info)
        flow_score, flow_details = _compute_institutional_flow(ticker, info)
        profile["fundamental_improvement"] = {"score": imp_score, "details": imp_details}
        profile["institutional_flow"] = {"score": flow_score, "details": flow_details}
        profile["smart_money_lag"] = round(imp_score - flow_score, 1)
    except Exception as e:
        logger.debug("Smart money analysis failed: %s", e)
        profile["fundamental_improvement"] = {"score": 0, "details": {}}
        profile["institutional_flow"] = {"score": 0, "details": {}}
        profile["smart_money_lag"] = 0

    # ── Step 6: Management credibility ────────────────────────
    if progress_cb:
        progress_cb(0.80, "Assessing management credibility...")

    try:
        llm = get_llm()
        cred = compute_credibility_score(ticker, info, sector, llm)
        profile["credibility"] = cred
    except Exception as e:
        logger.debug("Credibility scoring failed: %s", e)
        profile["credibility"] = {"score": 50, "reasoning": "Assessment unavailable", "method": "error"}

    # ── Step 7: Regulatory tailwind ───────────────────────────
    if progress_cb:
        progress_cb(0.85, "Scanning regulatory environment...")

    try:
        active_policies = scan_policy_signals()
        reg = compute_regulatory_score(ticker, sector, active_policies)
        profile["regulatory"] = reg
    except Exception as e:
        logger.debug("Regulatory scoring failed: %s", e)
        profile["regulatory"] = {"score": 50, "net_signal": "Unknown"}

    # ── Step 8: Promoter behavior ─────────────────────────────
    if progress_cb:
        progress_cb(0.90, "Analyzing promoter behavior...")

    try:
        buying = detect_promoter_buying_pattern(ticker, info)
        rpt = detect_related_party_anomaly(ticker)
        profile["promoter_buying"] = buying
        profile["related_party"] = rpt
    except Exception as e:
        logger.debug("Promoter analysis failed: %s", e)
        profile["promoter_buying"] = {"signal": "unknown", "has_data": False}
        profile["related_party"] = {"rpt_count": 0, "has_anomaly": False}

    if progress_cb:
        progress_cb(1.0, "Stock profile complete!")

    return profile
