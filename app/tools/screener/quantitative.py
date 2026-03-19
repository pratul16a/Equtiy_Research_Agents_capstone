"""Quantitative screening — ROE, P/B, P/S filters."""

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
