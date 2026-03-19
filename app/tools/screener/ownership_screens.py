"""Ownership screens — promoter holding, institutional interest, delivery %, pledge, MF schemes."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def screen_promoter_holding_strong(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Screen for stocks with promoter/insider holding > threshold (default 50%).

    Aligned incentives — promoter isn't dumping into rally or recovery.
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        held = info.get("heldPercentInsiders")
        if held is None:
            continue
        try:
            held_pct = float(held)
        except (ValueError, TypeError):
            continue

        if held_pct > threshold:
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "promoter_holding_pct": round(held_pct * 100, 2),
                "threshold": round(threshold * 100, 1),
                "passed": True,
            })

    logger.info("Promoter holding screen (>%s%%): %d stocks passed", threshold * 100, len(results))
    return results


def screen_institutional_increasing(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Screen for stocks with institutional holding above threshold (proxy for increasing interest).

    FII/DII adding positions = smart money confirmation.
    Uses current institutional holding % as proxy (historical trend not available in yfinance).
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        held = info.get("heldPercentInstitutions")
        if held is None:
            continue
        try:
            held_pct = float(held)
        except (ValueError, TypeError):
            continue

        if held_pct > threshold:
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "institutional_holding_pct": round(held_pct * 100, 2),
                "passed": True,
            })

    logger.info("Institutional holding screen (>%s%%): %d stocks passed", threshold * 100, len(results))
    return results


# ── Deferred screens (no data source in yfinance) ───────────


def screen_delivery_pct(
    tickers: list[str],
    sector_map: dict[str, str],
    threshold: float = 60.0,
) -> list[dict[str, Any]]:
    """Delivery % > threshold — delegates to qualitative.screen_delivery_pct (uses nselib)."""
    from app.tools.screener.qualitative import screen_delivery_pct as _real_screen
    return _real_screen(tickers, sector_map, threshold)


def screen_mf_scheme_increasing(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """DEFERRED: MF scheme count increasing. Needs AMFI API data."""
    logger.info("MF scheme count screen: DEFERRED — AMFI API not yet integrated")
    return []


def screen_pledge_reducing(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """DEFERRED: Pledge % reducing. Needs NSE shareholding pattern data."""
    logger.info("Pledge reducing screen: DEFERRED — NSE shareholding data not yet integrated")
    return []
