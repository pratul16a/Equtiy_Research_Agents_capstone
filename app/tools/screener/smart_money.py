"""Smart Money Lag Detector — USP #2.

Finds stocks where fundamentals improved but institutions haven't bought yet.
High positive lag = undiscovered turnaround = potential early alpha.

Smart Money Lag = Fundamental Improvement Score - Institutional Flow Score
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.tools.screener.batch_fundamentals import fetch_ticker_financials
from app.utils.cache import cache_get

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def _compute_fundamental_improvement(ticker: str, info: dict) -> tuple[float, dict]:
    """Compute fundamental improvement score (0-100).

    Checks: ROE improving, debt reducing, margins expanding, revenue growth.
    Returns (score, details_dict).
    """
    data = fetch_ticker_financials(ticker)
    fin = data.get("financials")
    bs = data.get("balance_sheet")
    signals: dict[str, Any] = {}
    sub_scores: list[float] = []

    # 1. ROE improvement (current vs 2yr ago)
    if fin is not None and bs is not None and len(fin.columns) >= 3 and len(bs.columns) >= 3:
        try:
            ni_curr = _safe_float(fin.iloc[:, 0].get("Net Income"))
            eq_curr = None
            for f_name in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                eq_curr = _safe_float(bs.iloc[:, 0].get(f_name))
                if eq_curr and eq_curr > 0:
                    break

            ni_old = _safe_float(fin.iloc[:, 2].get("Net Income"))
            eq_old = None
            for f_name in ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]:
                eq_old = _safe_float(bs.iloc[:, 2].get(f_name))
                if eq_old and eq_old > 0:
                    break

            if ni_curr and eq_curr and eq_curr > 0 and ni_old and eq_old and eq_old > 0:
                roe_curr = ni_curr / eq_curr
                roe_old = ni_old / eq_old
                roe_change = roe_curr - roe_old
                signals["roe_improvement"] = round(roe_change * 100, 2)
                # Score: +5pp improvement = 100, 0 = 50, -5pp = 0
                sub_scores.append(max(0, min(100, 50 + roe_change * 100 * 10)))
        except Exception:
            pass

    # 2. Debt reduction
    if bs is not None and len(bs.columns) >= 3:
        try:
            debt_curr = None
            debt_old = None
            for f_name in ["Total Debt", "Long Term Debt"]:
                d = _safe_float(bs.iloc[:, 0].get(f_name))
                if d is not None:
                    debt_curr = d
                    break
            for f_name in ["Total Debt", "Long Term Debt"]:
                d = _safe_float(bs.iloc[:, 2].get(f_name))
                if d is not None:
                    debt_old = d
                    break

            if debt_curr is not None and debt_old is not None and debt_old > 0:
                debt_change = (debt_curr - debt_old) / debt_old
                signals["debt_change_pct"] = round(debt_change * 100, 1)
                # Reduction is positive signal: -30% reduction = 100, 0 = 50, +30% increase = 0
                sub_scores.append(max(0, min(100, 50 - debt_change * 100 * 1.67)))
        except Exception:
            pass

    # 3. Margin expansion
    if fin is not None and len(fin.columns) >= 3:
        try:
            rev_curr = _safe_float(fin.iloc[:, 0].get("Total Revenue"))
            op_curr = _safe_float(fin.iloc[:, 0].get("Operating Income"))
            rev_old = _safe_float(fin.iloc[:, 2].get("Total Revenue"))
            op_old = _safe_float(fin.iloc[:, 2].get("Operating Income"))

            if all(v and v > 0 for v in [rev_curr, rev_old]) and op_curr is not None and op_old is not None:
                margin_curr = op_curr / rev_curr
                margin_old = op_old / rev_old
                margin_change = margin_curr - margin_old
                signals["margin_expansion"] = round(margin_change * 100, 2)
                sub_scores.append(max(0, min(100, 50 + margin_change * 100 * 10)))
        except Exception:
            pass

    # 4. Revenue growth
    if fin is not None and len(fin.columns) >= 3:
        try:
            rev_curr = _safe_float(fin.iloc[:, 0].get("Total Revenue"))
            rev_old = _safe_float(fin.iloc[:, 2].get("Total Revenue"))
            if rev_curr and rev_old and rev_old > 0:
                rev_growth = (rev_curr - rev_old) / rev_old
                signals["revenue_growth_2yr"] = round(rev_growth * 100, 1)
                # 30% growth over 2yr = 100, 0 = 30, -20% = 0
                sub_scores.append(max(0, min(100, 30 + rev_growth * 100 * 2.33)))
        except Exception:
            pass

    if not sub_scores:
        return 0.0, signals

    score = sum(sub_scores) / len(sub_scores)
    return round(score, 1), signals


def _compute_institutional_flow(ticker: str, info: dict) -> tuple[float, dict]:
    """Compute institutional flow score (0-100).

    High = institutions are already buying (no lag opportunity).
    Low = institutions haven't noticed yet (potential lag alpha).
    """
    signals: dict[str, Any] = {}
    sub_scores: list[float] = []

    # 1. Current institutional holding level
    inst_pct = _safe_float(info.get("heldPercentInstitutions"))
    if inst_pct is not None:
        signals["institutional_pct"] = round(inst_pct * 100, 2)
        # Higher institutional holding = higher flow score (less opportunity for lag)
        sub_scores.append(min(100, inst_pct * 100 * 1.5))

    # 2. Short interest as a contrarian signal
    short_pct = _safe_float(info.get("shortPercentOfFloat"))
    if short_pct is not None:
        signals["short_pct"] = round(short_pct * 100, 2)
        # Low short interest = institutions are neutral/positive = higher flow score
        sub_scores.append(min(100, (1 - short_pct) * 100))

    # 3. Analyst recommendations as institutional sentiment proxy
    rec = _safe_float(info.get("recommendationMean"))
    if rec is not None:
        signals["analyst_recommendation"] = rec
        # 1=Strong Buy, 5=Strong Sell. Lower = more institutional attention
        sub_scores.append(max(0, min(100, (5 - rec) * 25)))

    if not sub_scores:
        return 50.0, signals  # Neutral if no data

    score = sum(sub_scores) / len(sub_scores)
    return round(score, 1), signals


def screen_smart_money_lag(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    min_lag: float = 20.0,
) -> list[dict[str, Any]]:
    """Screen for stocks with high smart money lag (fundamentals improved, institutions haven't noticed).

    Lag = Fundamental Improvement Score - Institutional Flow Score
    Stocks with lag > min_lag are flagged as potential alpha opportunities.
    """
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        try:
            improvement_score, imp_details = _compute_fundamental_improvement(ticker, info)
            flow_score, flow_details = _compute_institutional_flow(ticker, info)

            lag = improvement_score - flow_score

            if lag >= min_lag:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "smart_money_lag": round(lag, 1),
                    "fundamental_improvement": improvement_score,
                    "institutional_flow": flow_score,
                    "improvement_details": imp_details,
                    "flow_details": flow_details,
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Smart money lag check failed for %s: %s", ticker, e)

    results.sort(key=lambda x: x["smart_money_lag"], reverse=True)
    logger.info("Smart money lag screen: %d stocks with lag >= %.0f", len(results), min_lag)
    return results
