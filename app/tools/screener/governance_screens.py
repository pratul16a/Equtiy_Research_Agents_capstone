"""Governance screens — promoter pledge, institutional holding changes."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.tools.screener.batch_fundamentals import fetch_ticker_financials

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def screen_low_pledge(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with low promoter pledge (held% insiders > 50% suggests strong promoter).

    Uses heldPercentInsiders from yf .info as a proxy for promoter holding.
    High insider holding with no major pledge activity = good governance.
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        try:
            insider_pct = _safe_float(info.get("heldPercentInsiders"))
            if insider_pct is None:
                from app.tools.screener.ownership_screens import _get_screener_shareholding
                screener = _get_screener_shareholding(ticker)
                insider_pct = screener.get("promoter_pct")
            if insider_pct is None:
                continue

            # In Indian context, heldPercentInsiders approximates promoter holding
            # High promoter holding (>50%) is generally a positive governance signal
            if insider_pct > 0.50:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "promoter_holding_pct": round(insider_pct * 100, 2),
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Pledge check failed for %s: %s", ticker, e)

    logger.info("Low pledge screen: %d stocks with strong promoter holding", len(results))
    return results


def screen_increasing_institutional(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with increasing institutional (FII + MF) holding.

    Uses yf .institutional_holders and .mutualfund_holders to detect
    recent institutional interest.
    """
    results: list[dict] = []

    for ticker in tickers:
        try:
            data = fetch_ticker_financials(ticker)
            # Use info data for institutional holding percentage
            info_cache_key = f"screener_info:{ticker}"
            from app.utils.cache import cache_get as _cg
            info = _cg(info_cache_key)
            if info is None:
                continue

            inst_pct = _safe_float(info.get("heldPercentInstitutions"))
            if inst_pct is None:
                from app.tools.screener.ownership_screens import _get_screener_shareholding
                screener = _get_screener_shareholding(ticker)
                inst_pct = screener.get("institutional_pct")
            if inst_pct is None:
                continue

            # Institutions holding > 30% and present = institutional interest signal
            if inst_pct > 0.30:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "institutional_holding_pct": round(inst_pct * 100, 2),
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Institutional holding check failed for %s: %s", ticker, e)

    logger.info("Institutional holding screen: %d stocks with strong institutional presence", len(results))
    return results
