"""Financial quality screens — debt reduction, OPM improvement, CF turnaround, dividend initiation."""

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


def screen_debt_reduction(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with Total Debt decreasing for 2+ consecutive years."""
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            data = fetch_ticker_financials(ticker)
            bs = data.get("balance_sheet")
            if bs is None or len(bs.columns) < 3:
                continue

            debts: list[dict] = []
            for i in range(min(4, len(bs.columns))):
                debt = None
                for field in ["Total Debt", "Long Term Debt", "Total Non Current Liabilities Net Minority Interest"]:
                    debt = _safe_float(bs.iloc[:, i].get(field))
                    if debt is not None:
                        break
                if debt is not None:
                    year_label = str(bs.columns[i])[:10]
                    debts.append({"year": year_label, "debt": debt})

            if len(debts) < 3:
                continue

            # Check if debt is decreasing for at least 2 consecutive years
            # debts[0] is most recent, debts[1] is previous year, etc.
            consecutive_decreases = 0
            for i in range(len(debts) - 1):
                if debts[i]["debt"] < debts[i + 1]["debt"]:
                    consecutive_decreases += 1
                else:
                    break

            if consecutive_decreases >= 2:
                pct_reduction = round(
                    (1 - debts[0]["debt"] / debts[consecutive_decreases]["debt"]) * 100, 1
                ) if debts[consecutive_decreases]["debt"] > 0 else 0

                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "consecutive_years_reducing": consecutive_decreases,
                    "debt_reduction_pct": pct_reduction,
                    "debt_history": debts[:consecutive_decreases + 1],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Debt reduction check failed for %s: %s", ticker, e)

    logger.info("Debt reduction screen: %d stocks passed", len(results))
    return results


def screen_opm_improvement(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with Operating Profit Margin expanding 2 consecutive years."""
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            if fin is None or len(fin.columns) < 3:
                continue

            margins: list[dict] = []
            for i in range(min(4, len(fin.columns))):
                op_income = _safe_float(fin.iloc[:, i].get("Operating Income"))
                revenue = None
                for field in ["Total Revenue", "Revenue"]:
                    revenue = _safe_float(fin.iloc[:, i].get(field))
                    if revenue and revenue > 0:
                        break

                if op_income is not None and revenue and revenue > 0:
                    opm = op_income / revenue
                    year_label = str(fin.columns[i])[:10]
                    margins.append({"year": year_label, "opm": round(opm * 100, 2)})

            if len(margins) < 3:
                continue

            # Check if OPM is improving for 2 consecutive years (most recent vs previous)
            consecutive_improvements = 0
            for i in range(len(margins) - 1):
                if margins[i]["opm"] > margins[i + 1]["opm"]:
                    consecutive_improvements += 1
                else:
                    break

            if consecutive_improvements >= 2:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "current_opm": margins[0]["opm"],
                    "opm_expansion": round(margins[0]["opm"] - margins[consecutive_improvements]["opm"], 2),
                    "consecutive_years_improving": consecutive_improvements,
                    "opm_history": margins[:consecutive_improvements + 1],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("OPM improvement check failed for %s: %s", ticker, e)

    logger.info("OPM improvement screen: %d stocks passed", len(results))
    return results


def screen_cf_turnaround(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks where Operating Cash Flow turned positive after being negative."""
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            data = fetch_ticker_financials(ticker)
            cf = data.get("cashflow")
            if cf is None or len(cf.columns) < 2:
                continue

            ocf_values: list[dict] = []
            for i in range(min(4, len(cf.columns))):
                ocf = None
                for field in ["Operating Cash Flow", "Total Cash From Operating Activities",
                              "Cash Flow From Continuing Operating Activities"]:
                    ocf = _safe_float(cf.iloc[:, i].get(field))
                    if ocf is not None:
                        break
                if ocf is not None:
                    year_label = str(cf.columns[i])[:10]
                    ocf_values.append({"year": year_label, "ocf": ocf})

            if len(ocf_values) < 2:
                continue

            # Current year positive, at least one of the previous years negative
            if ocf_values[0]["ocf"] > 0 and any(v["ocf"] < 0 for v in ocf_values[1:]):
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "current_ocf_cr": round(ocf_values[0]["ocf"] / 1e7, 2),
                    "ocf_history": [
                        {"year": v["year"], "ocf_cr": round(v["ocf"] / 1e7, 2)}
                        for v in ocf_values
                    ],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("CF turnaround check failed for %s: %s", ticker, e)

    logger.info("CF turnaround screen: %d stocks passed", len(results))
    return results


def screen_dividend_initiation(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks that started/resumed dividends after a gap."""
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            data = fetch_ticker_financials(ticker)
            divs = data.get("dividends")
            if divs is None or divs.empty:
                continue

            # Group dividends by year
            div_years = sorted(divs.index.year.unique())
            if len(div_years) < 1:
                continue

            most_recent_year = div_years[-1]

            # Check for gap: dividend in recent year but a year with no dividend before that
            if len(div_years) >= 2:
                # Look for a gap in the sequence
                has_gap = False
                for i in range(len(div_years) - 1):
                    if div_years[i + 1] - div_years[i] > 1:
                        has_gap = True
                        break

                if has_gap:
                    recent_div = float(divs[divs.index.year == most_recent_year].sum())
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "most_recent_div_year": int(most_recent_year),
                        "recent_dividend_total": round(recent_div, 2),
                        "dividend_years": [int(y) for y in div_years],
                        "passed": True,
                    })
            elif len(div_years) == 1:
                # First-ever dividend
                recent_div = float(divs[divs.index.year == most_recent_year].sum())
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "most_recent_div_year": int(most_recent_year),
                    "recent_dividend_total": round(recent_div, 2),
                    "dividend_years": [int(most_recent_year)],
                    "first_dividend": True,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Dividend initiation check failed for %s: %s", ticker, e)

    logger.info("Dividend initiation screen: %d stocks passed", len(results))
    return results


def screen_qoq_profit_growth(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    min_growth: float = 10.0,
    min_quarters: int = 2,
) -> list[dict[str, Any]]:
    """Screen for stocks with QoQ profit growth > min_growth% for last min_quarters quarters.

    Used by Category B: sustained earnings momentum.

    Args:
        bulk_info: {ticker: info_dict} from yfinance.
        sector_map: {ticker: sector} lookup.
        min_growth: Minimum QoQ profit growth % (default 10%).
        min_quarters: Number of consecutive quarters required (default 2).
    """
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            if fin is None:
                continue

            # yfinance .quarterly_financials — try to get quarterly data
            import yfinance as yf
            from app.utils.cache import cache_get as _cg, cache_set as _cs, yfinance_rate_limiter as _rl

            qcache_key = f"quarterly_fin:{ticker}"
            qfin = _cg(qcache_key)
            if qfin is None:
                _rl.wait()
                stock = yf.Ticker(ticker)
                qfin = stock.quarterly_financials
                if qfin is not None and not qfin.empty:
                    _cs(qcache_key, qfin, ttl=3600)
                else:
                    continue

            if qfin is None or len(qfin.columns) < min_quarters + 1:
                continue

            # Extract net income per quarter (columns are most recent first)
            profits: list[dict] = []
            for i in range(min(min_quarters + 2, len(qfin.columns))):
                net_income = None
                for field in ["Net Income", "Net Income Common Stockholders"]:
                    net_income = _safe_float(qfin.iloc[:, i].get(field))
                    if net_income is not None:
                        break
                if net_income is not None:
                    quarter_label = str(qfin.columns[i])[:10]
                    profits.append({"quarter": quarter_label, "net_income": net_income})

            if len(profits) < min_quarters + 1:
                continue

            # Check QoQ growth for last min_quarters consecutive quarters
            consecutive_growth = 0
            growth_rates: list[dict] = []
            for i in range(len(profits) - 1):
                prev = profits[i + 1]["net_income"]
                curr = profits[i]["net_income"]
                if prev and prev > 0:
                    growth = ((curr - prev) / abs(prev)) * 100
                    growth_rates.append({
                        "quarter": profits[i]["quarter"],
                        "growth_pct": round(growth, 2),
                    })
                    if growth >= min_growth:
                        consecutive_growth += 1
                    else:
                        break
                else:
                    break

            if consecutive_growth >= min_quarters:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "consecutive_quarters": consecutive_growth,
                    "growth_rates": growth_rates[:consecutive_growth],
                    "latest_growth_pct": growth_rates[0]["growth_pct"] if growth_rates else None,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("QoQ profit growth check failed for %s: %s", ticker, e)

    logger.info("QoQ profit growth screen (>%s%% for %d quarters): %d stocks passed",
                min_growth, min_quarters, len(results))
    return results
