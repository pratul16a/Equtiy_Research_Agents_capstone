"""Ownership screens — promoter holding, institutional interest, delivery %, pledge, MF schemes."""

from __future__ import annotations

import logging
from typing import Any

from app.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


def _get_screener_shareholding(ticker: str) -> dict[str, float | None]:
    """Fallback: scrape shareholding from Screener.in when yfinance returns None.

    Returns dict with keys: promoter_pct, institutional_pct, fii_pct, dii_pct, pledge_pct.
    All values as fractions (0-1), matching yfinance convention.
    """
    cache_key = f"screener_shareholding:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    result: dict[str, float | None] = {
        "promoter_pct": None,
        "institutional_pct": None,
        "fii_pct": None,
        "dii_pct": None,
        "pledge_pct": None,
    }

    try:
        from app.rag.screener_scraper import scrape_company_page

        data = scrape_company_page(ticker.replace(".NS", "").replace(".BO", ""))
        shareholding = data.get("shareholding", {})
        rows = shareholding.get("rows", [])

        if not rows or len(rows) < 2:
            cache_set(cache_key, result, ttl=3600)
            return result

        # Parse header to find latest quarter column
        header = rows[0] if rows else []
        # Data rows: typically ["Promoters", "64.23%", "63.89%", ...]
        for row in rows[1:]:
            if not row:
                continue
            label = row[0].lower().strip()
            # Get the most recent value (last column with data)
            latest_val = None
            for cell in reversed(row[1:]):
                cell = cell.strip().replace("%", "").replace(",", "")
                try:
                    latest_val = float(cell) / 100.0  # Convert to fraction
                    break
                except (ValueError, TypeError):
                    continue

            if latest_val is None:
                continue

            # Strip trailing + from Screener.in labels (e.g., "Promoters+", "FIIs+")
            label = label.rstrip("+").rstrip("s").strip()

            if "promoter" in label and "pledge" not in label:
                result["promoter_pct"] = latest_val
            elif label in ("fii", "foreign institution"):
                result["fii_pct"] = latest_val
            elif label in ("dii", "domestic institution"):
                result["dii_pct"] = latest_val
            elif "pledge" in label:
                result["pledge_pct"] = latest_val
            elif "institution" in label and "foreign" not in label and "domestic" not in label:
                result["institutional_pct"] = latest_val

        # Compute total institutional if not found directly
        if result["institutional_pct"] is None and result["fii_pct"] is not None:
            dii = result["dii_pct"] or 0.0
            result["institutional_pct"] = result["fii_pct"] + dii

        logger.info("Screener.in shareholding for %s: promoter=%.1f%%, inst=%.1f%%",
                     ticker,
                     (result["promoter_pct"] or 0) * 100,
                     (result["institutional_pct"] or 0) * 100)

    except Exception as e:
        logger.debug("Screener.in shareholding fallback failed for %s: %s", ticker, e)

    cache_set(cache_key, result, ttl=3600)
    return result


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
        # Fallback to Screener.in if yfinance returns None
        if held is None:
            screener = _get_screener_shareholding(ticker)
            held = screener.get("promoter_pct")
        if held is None:
            continue
        try:
            held_pct = float(held)
        except (ValueError, TypeError):
            continue

        results.append({
            "ticker": ticker,
            "sector": sector_map.get(ticker, "Other"),
            "promoter_holding_pct": round(held_pct * 100, 2),
            "threshold": round(threshold * 100, 1),
            "passed": held_pct > threshold,
        })

    logger.info("Promoter holding screen (>%s%%): %d stocks passed", threshold * 100, sum(1 for r in results if r["passed"]))
    return results


def screen_institutional_increasing(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    threshold: float = 0.30,
) -> list[dict[str, Any]]:
    """Screen for stocks with institutional holding above threshold (proxy for increasing interest).

    FII/DII adding positions = smart money confirmation.
    Uses current institutional holding % as proxy (historical trend not available in yfinance).
    Falls back to Screener.in data when yfinance returns None.
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        held = info.get("heldPercentInstitutions")
        # Fallback to Screener.in if yfinance returns None
        if held is None:
            screener = _get_screener_shareholding(ticker)
            held = screener.get("institutional_pct")
        if held is None:
            continue
        try:
            held_pct = float(held)
        except (ValueError, TypeError):
            continue

        results.append({
            "ticker": ticker,
            "sector": sector_map.get(ticker, "Other"),
            "institutional_holding_pct": round(held_pct * 100, 2),
            "passed": held_pct > threshold,
        })

    logger.info("Institutional holding screen (>%s%%): %d stocks passed", threshold * 100, sum(1 for r in results if r["passed"]))
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
