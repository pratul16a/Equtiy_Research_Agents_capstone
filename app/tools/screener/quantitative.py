"""Quantitative screening — ROE, P/B, P/S, EV/EBITDA, FCF yield filters."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.tools.screener.batch_fundamentals import fetch_ticker_financials

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def screen_roe(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with ROE < 10% for the last 2 years.

    Phase 1: Quick filter using current ROE from .info (< 15% threshold).
    Phase 2: Deep check using historical financials for last 2 years.
    """
    # Phase 1: Quick filter
    candidates: list[str] = []
    for ticker, info in bulk_info.items():
        roe = _safe_float(info.get("returnOnEquity"))
        if roe is not None and roe < 0.15:
            candidates.append(ticker)
        elif roe is None:
            candidates.append(ticker)  # check if we can compute

    logger.info("ROE quick filter: %d / %d candidates", len(candidates), len(bulk_info))

    # Phase 2: Deep check
    results: list[dict] = []
    for ticker in candidates:
        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            bs = data.get("balance_sheet")
            if fin is None or bs is None:
                continue

            # Get Net Income and Shareholders Equity for last 2 years
            roe_years: list[dict] = []
            for i in range(min(2, len(fin.columns), len(bs.columns))):
                net_income = _safe_float(fin.iloc[:, i].get("Net Income"))
                # Try multiple field names for equity
                equity = None
                for field in ["Stockholders Equity", "Total Equity Gross Minority Interest",
                              "Common Stock Equity", "Total Stockholder Equity"]:
                    equity = _safe_float(bs.iloc[:, i].get(field))
                    if equity and equity != 0:
                        break

                if net_income is not None and equity and equity != 0:
                    roe_val = net_income / equity
                    year_label = str(fin.columns[i])[:10]
                    roe_years.append({"year": year_label, "roe": round(roe_val * 100, 2)})

            if len(roe_years) >= 2 and all(y["roe"] < 10.0 for y in roe_years[:2]):
                current_roe = _safe_float(bulk_info.get(ticker, {}).get("returnOnEquity"))
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "current_roe": round(current_roe * 100, 2) if current_roe else None,
                    "roe_history": roe_years,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("ROE deep check failed for %s: %s", ticker, e)

    logger.info("ROE screen: %d stocks passed", len(results))
    return results


def screen_pb_vs_historical(
    bulk_info: dict[str, dict],
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with current P/B below 5-year average P/B."""
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        current_pb = _safe_float(info.get("priceToBook"))
        if current_pb is None or current_pb <= 0:
            continue

        try:
            data = fetch_ticker_financials(ticker)
            bs = data.get("balance_sheet")
            shares = _safe_float(data.get("shares_outstanding"))
            if bs is None or not shares or shares == 0:
                continue

            # Compute historical P/B for each year
            historical_pbs: list[float] = []
            for i in range(min(5, len(bs.columns))):
                equity = None
                for field in ["Stockholders Equity", "Total Equity Gross Minority Interest",
                              "Common Stock Equity"]:
                    equity = _safe_float(bs.iloc[:, i].get(field))
                    if equity and equity > 0:
                        break
                if not equity or equity <= 0:
                    continue

                book_per_share = equity / shares
                if book_per_share <= 0:
                    continue

                # Get price at that year's date
                year_date = bs.columns[i]
                if ticker in price_df.columns:
                    price_series = price_df[ticker].dropna()
                    # Find closest price to the balance sheet date
                    try:
                        idx = price_series.index.get_indexer([year_date], method="nearest")[0]
                        if 0 <= idx < len(price_series):
                            hist_price = float(price_series.iloc[idx])
                            hist_pb = hist_price / book_per_share
                            historical_pbs.append(hist_pb)
                    except Exception:
                        pass

            if len(historical_pbs) >= 2:
                avg_pb = sum(historical_pbs) / len(historical_pbs)
                if current_pb < avg_pb:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "current_pb": round(current_pb, 2),
                        "avg_pb_5yr": round(avg_pb, 2),
                        "discount_pct": round((1 - current_pb / avg_pb) * 100, 1),
                        "passed": True,
                    })
        except Exception as e:
            logger.debug("P/B historical check failed for %s: %s", ticker, e)

    logger.info("P/B screen: %d stocks passed", len(results))
    return results


def screen_ps_vs_historical(
    bulk_info: dict[str, dict],
    price_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with current P/S below 5-year average P/S."""
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        current_ps = _safe_float(info.get("priceToSalesTrailing12Months"))
        if current_ps is None or current_ps <= 0:
            continue

        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            shares = _safe_float(data.get("shares_outstanding"))
            if fin is None or not shares or shares == 0:
                continue

            historical_pss: list[float] = []
            for i in range(min(5, len(fin.columns))):
                revenue = _safe_float(fin.iloc[:, i].get("Total Revenue"))
                if not revenue or revenue <= 0:
                    continue

                year_date = fin.columns[i]
                if ticker in price_df.columns:
                    price_series = price_df[ticker].dropna()
                    try:
                        idx = price_series.index.get_indexer([year_date], method="nearest")[0]
                        if 0 <= idx < len(price_series):
                            hist_price = float(price_series.iloc[idx])
                            hist_ps = (hist_price * shares) / revenue
                            historical_pss.append(hist_ps)
                    except Exception:
                        pass

            if len(historical_pss) >= 2:
                avg_ps = sum(historical_pss) / len(historical_pss)
                if current_ps < avg_ps:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "current_ps": round(current_ps, 2),
                        "avg_ps_5yr": round(avg_ps, 2),
                        "discount_pct": round((1 - current_ps / avg_ps) * 100, 1),
                        "passed": True,
                    })
        except Exception as e:
            logger.debug("P/S historical check failed for %s: %s", ticker, e)

    logger.info("P/S screen: %d stocks passed", len(results))
    return results


# ── ROE Above Threshold ─────────────────────────────────────


def _compute_roe_from_financials(ticker: str) -> float | None:
    """Fallback: compute ROE from Net Income / Equity via financial statements."""
    try:
        data = fetch_ticker_financials(ticker)
        fin = data.get("financials")
        bs = data.get("balance_sheet")
        if fin is None or bs is None or fin.empty or bs.empty:
            return None
        ni = _safe_float(fin.iloc[:, 0].get("Net Income"))
        eq = None
        for field in ["Stockholders Equity", "Total Equity Gross Minority Interest",
                      "Common Stock Equity", "Total Stockholder Equity"]:
            eq = _safe_float(bs.iloc[:, 0].get(field))
            if eq and eq != 0:
                break
        if ni is not None and eq and eq != 0:
            return ni / eq
    except Exception:
        pass
    return None


def _compute_revenue_growth_from_financials(ticker: str) -> float | None:
    """Fallback: compute YoY revenue growth from financial statements."""
    try:
        data = fetch_ticker_financials(ticker)
        fin = data.get("financials")
        if fin is None or fin.empty or len(fin.columns) < 2:
            return None
        rev_curr = _safe_float(fin.iloc[:, 0].get("Total Revenue"))
        rev_prev = _safe_float(fin.iloc[:, 1].get("Total Revenue"))
        if rev_curr and rev_prev and rev_prev > 0:
            return (rev_curr - rev_prev) / rev_prev
    except Exception:
        pass
    return None


def _compute_fcf_yield(ticker: str, market_cap: float | None) -> float | None:
    """Fallback: compute FCF yield from OCF - Capex / Market Cap."""
    if not market_cap or market_cap <= 0:
        return None
    try:
        data = fetch_ticker_financials(ticker)
        cf = data.get("cashflow")
        if cf is None or cf.empty:
            return None
        ocf = _safe_float(cf.iloc[:, 0].get("Operating Cash Flow"))
        capex = _safe_float(cf.iloc[:, 0].get("Capital Expenditure"))
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is negative
            return fcf / market_cap
    except Exception:
        pass
    return None


def screen_roe_above(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    threshold: float = 0.12,
    revenue_growth_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    """Screen for stocks with ROE > threshold OR YoY revenue growth > 20%.

    Quality filter — strong returns or high-growth companies.
    Falls back to financial statement computation when .info fields are missing.
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        roe = _safe_float(info.get("returnOnEquity"))
        rev_growth = _safe_float(info.get("revenueGrowth"))

        # Fallback: compute from financial statements if .info is missing
        if roe is None:
            roe = _compute_roe_from_financials(ticker)
        if rev_growth is None:
            rev_growth = _compute_revenue_growth_from_financials(ticker)

        roe_pass = roe is not None and roe > threshold
        growth_pass = rev_growth is not None and rev_growth > revenue_growth_threshold

        if roe_pass or growth_pass:
            reason = []
            if roe_pass:
                reason.append(f"ROE {roe * 100:.1f}%")
            if growth_pass:
                reason.append(f"RevGrowth {rev_growth * 100:.1f}%")
            results.append({
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "current_roe": round(roe * 100, 2) if roe is not None else None,
                "revenue_growth_yoy": round(rev_growth * 100, 2) if rev_growth is not None else None,
                "reason": " + ".join(reason),
                "passed": True,
            })

    logger.info("ROE/Growth screen: %d stocks passed", len(results))
    return results


# ── EV/EBITDA Below Sector Median ───────────────────────────


def screen_ev_ebitda_below_median(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with EV/EBITDA below their sector median.

    Catches asset-light cheap stocks that P/B misses (IT, services).
    """
    # Collect EV/EBITDA per sector
    sector_values: dict[str, list[tuple[str, float]]] = {}
    for ticker, info in bulk_info.items():
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
        if ev_ebitda is None or ev_ebitda <= 0:
            continue
        sector = sector_map.get(ticker, "Other")
        sector_values.setdefault(sector, []).append((ticker, ev_ebitda))

    # Compute sector medians
    sector_medians: dict[str, float] = {}
    for sector, values in sector_values.items():
        sorted_vals = sorted(v for _, v in values)
        mid = len(sorted_vals) // 2
        sector_medians[sector] = sorted_vals[mid] if len(sorted_vals) % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2

    results: list[dict] = []
    for sector, values in sector_values.items():
        median = sector_medians[sector]
        for ticker, ev_ebitda in values:
            if ev_ebitda < median:
                results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "ev_ebitda": round(ev_ebitda, 2),
                    "sector_median": round(median, 2),
                    "discount_pct": round((1 - ev_ebitda / median) * 100, 1),
                    "passed": True,
                })

    logger.info("EV/EBITDA below median screen: %d stocks passed", len(results))
    return results


# ── FCF Yield Above Sector Average ──────────────────────────


def screen_fcf_yield_above_sector(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with FCF yield > sector average.

    Generating real cash relative to price — not an accounting illusion.
    """
    # Collect FCF yield per sector (with fallback to financial statements)
    sector_yields: dict[str, list[tuple[str, float]]] = {}
    for ticker, info in bulk_info.items():
        fcf = _safe_float(info.get("freeCashflow"))
        mcap = _safe_float(info.get("marketCap"))
        if mcap is None or mcap <= 0:
            continue
        # Fallback: compute FCF from cashflow statement
        if fcf is None:
            computed_yield = _compute_fcf_yield(ticker, mcap)
            if computed_yield is not None:
                fcf = computed_yield * mcap  # reconstruct for sector avg calc
        if fcf is None:
            continue
        fcf_yield = fcf / mcap
        sector = sector_map.get(ticker, "Other")
        sector_yields.setdefault(sector, []).append((ticker, fcf_yield))

    # Compute sector averages
    sector_avgs: dict[str, float] = {}
    for sector, values in sector_yields.items():
        sector_avgs[sector] = sum(v for _, v in values) / len(values)

    results: list[dict] = []
    for sector, values in sector_yields.items():
        avg = sector_avgs[sector]
        for ticker, fcf_yield in values:
            if fcf_yield > avg and fcf_yield > 0:
                results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "fcf_yield": round(fcf_yield * 100, 2),
                    "sector_avg": round(avg * 100, 2),
                    "passed": True,
                })

    logger.info("FCF yield above sector avg screen: %d stocks passed", len(results))
    return results
