"""India-specific financial metrics — ROCE, promoter holding, FII/DII activity.

Includes TTL caching, rate limiting, and retry logic for resilient API calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yfinance as yf
from langchain_core.tools import tool

from app.utils.cache import get_yf_ticker, yfinance_rate_limiter

logger = logging.getLogger(__name__)


@tool
def get_shareholding_pattern(ticker: str) -> str:
    """Get the shareholding pattern for an Indian stock — promoter, FII, DII, and public holdings.

    Uses yfinance major_holders and institutional_holders data.
    Ticker should include .NS or .BO suffix.
    """
    try:
        stock, info = get_yf_ticker(ticker)
        result: dict[str, Any] = {"ticker": ticker}

        # Major holders (percentages)
        try:
            yfinance_rate_limiter.wait()
            major = stock.major_holders
            if major is not None and not major.empty:
                holder_data = {}
                for _, row in major.iterrows():
                    holder_data[str(row.iloc[1]).strip()] = str(row.iloc[0]).strip()
                result["major_holders"] = holder_data
            else:
                result["major_holders"] = {}
        except Exception as e:
            logger.warning("major_holders unavailable for %s: %s", ticker, e)
            result["major_holders"] = {}

        # Institutional holders
        try:
            yfinance_rate_limiter.wait()
            inst = stock.institutional_holders
            if inst is not None and not inst.empty:
                result["top_institutional_holders"] = [
                    {
                        "holder": str(row.get("Holder", "N/A")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "pct_out": float(row.get("% Out", 0)) if row.get("% Out") else None,
                    }
                    for _, row in inst.head(10).iterrows()
                ]
            else:
                result["top_institutional_holders"] = []
        except Exception as e:
            logger.warning("institutional_holders unavailable for %s: %s", ticker, e)
            result["top_institutional_holders"] = []

        result["note"] = (
            "For detailed promoter/FII/DII breakdown, check BSE corporate filings. "
            "yfinance provides institutional holder data which partially covers FII/DII."
        )

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_shareholding_pattern failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to get shareholding for {ticker}: {e}"})


@tool
def get_promoter_pledge_info(ticker: str) -> str:
    """Analyze promoter pledge status for an Indian stock.

    High promoter pledging (>20%) is a red flag in Indian markets.
    Uses available yfinance data and provides analysis context.
    """
    try:
        stock, info = get_yf_ticker(ticker)

        result = {
            "ticker": ticker,
            "company": info.get("longName", "N/A"),
        }

        try:
            yfinance_rate_limiter.wait()
            major = stock.major_holders
            if major is not None and not major.empty:
                result["holder_summary"] = {
                    str(row.iloc[1]).strip(): str(row.iloc[0]).strip()
                    for _, row in major.iterrows()
                }
            else:
                result["holder_summary"] = {}
        except Exception as e:
            logger.warning("major_holders unavailable for pledge info %s: %s", ticker, e)
            result["holder_summary"] = {}

        result["pledge_analysis_note"] = (
            "Promoter pledge data requires BSE/NSE quarterly filings. "
            "Key thresholds: <10% pledge is healthy, 10-20% needs monitoring, >20% is a red flag. "
            "Check the BSE corporate announcements tool for the latest shareholding pattern filing."
        )

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_promoter_pledge_info failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to get pledge info for {ticker}: {e}"})


@tool
def get_fii_dii_activity(ticker: str) -> str:
    """Get Foreign Institutional Investor (FII) and Domestic Institutional Investor (DII) activity.

    Provides institutional holder breakdown and context for Indian stocks.
    """
    try:
        stock, info = get_yf_ticker(ticker)

        result: dict[str, Any] = {
            "ticker": ticker,
            "company": info.get("longName", "N/A"),
        }

        # Get institutional holders
        try:
            yfinance_rate_limiter.wait()
            inst = stock.institutional_holders
            if inst is not None and not inst.empty:
                holders = []
                for _, row in inst.head(15).iterrows():
                    holders.append({
                        "holder": str(row.get("Holder", "N/A")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                        "date_reported": str(row.get("Date Reported", "N/A")),
                    })
                result["institutional_holders"] = holders
                result["total_institutions"] = len(inst)
            else:
                result["institutional_holders"] = []
                result["total_institutions"] = 0
        except Exception as e:
            logger.warning("institutional_holders unavailable for %s: %s", ticker, e)
            result["institutional_holders"] = []
            result["total_institutions"] = 0

        # Get mutual fund holders (maps to DII in Indian context)
        try:
            yfinance_rate_limiter.wait()
            mf = stock.mutualfund_holders
            if mf is not None and not mf.empty:
                mf_holders = []
                for _, row in mf.head(10).iterrows():
                    mf_holders.append({
                        "holder": str(row.get("Holder", "N/A")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                    })
                result["mutual_fund_holders"] = mf_holders
            else:
                result["mutual_fund_holders"] = []
        except Exception as e:
            logger.warning("mutualfund_holders unavailable for %s: %s", ticker, e)
            result["mutual_fund_holders"] = []

        result["analysis_note"] = (
            "FII buying signals global confidence; DII buying signals domestic conviction. "
            "Increasing FII + DII holding is bullish. FII selling with DII buying is neutral. "
            "Both selling is bearish."
        )

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_fii_dii_activity failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to get FII/DII data for {ticker}: {e}"})


@tool
def compute_roce(ticker: str) -> str:
    """Compute Return on Capital Employed (ROCE) for an Indian stock.

    ROCE = EBIT / Capital Employed
    Capital Employed = Total Assets - Current Liabilities

    ROCE is one of the most important metrics in Indian equity analysis.
    Benchmarks: >20% excellent, 15-20% good, 10-15% average, <10% poor.
    """
    try:
        stock, info = get_yf_ticker(ticker)

        yfinance_rate_limiter.wait()
        financials = stock.financials
        yfinance_rate_limiter.wait()
        balance = stock.balance_sheet

        if financials is None or financials.empty or balance is None or balance.empty:
            return json.dumps({"error": f"Insufficient data for ROCE calculation for {ticker}"})

        # Get EBIT
        ebit = None
        if "EBIT" in financials.index:
            ebit = float(financials.loc["EBIT"].iloc[0])
        elif "Operating Income" in financials.index:
            ebit = float(financials.loc["Operating Income"].iloc[0])

        if ebit is None:
            return json.dumps({"error": "EBIT data not found in financials"})

        # Get Capital Employed
        total_assets = float(balance.loc["Total Assets"].iloc[0]) if "Total Assets" in balance.index else None
        current_liab = float(balance.loc["Current Liabilities"].iloc[0]) if "Current Liabilities" in balance.index else None

        if total_assets is None or current_liab is None:
            return json.dumps({"error": "Balance sheet data insufficient for capital employed calculation"})

        capital_employed = total_assets - current_liab

        if capital_employed <= 0:
            return json.dumps({"error": "Negative or zero capital employed — company may have unusual balance sheet"})

        roce = ebit / capital_employed

        # Historical ROCE (if multiple years available)
        historical_roce = []
        years = min(4, financials.shape[1], balance.shape[1])
        for i in range(years):
            try:
                yr_ebit = None
                if "EBIT" in financials.index:
                    yr_ebit = float(financials.iloc[financials.index.get_loc("EBIT"), i])
                elif "Operating Income" in financials.index:
                    yr_ebit = float(financials.iloc[financials.index.get_loc("Operating Income"), i])

                yr_ta = float(balance.iloc[balance.index.get_loc("Total Assets"), i])
                yr_cl = float(balance.iloc[balance.index.get_loc("Current Liabilities"), i])
                yr_ce = yr_ta - yr_cl
                if yr_ce > 0 and yr_ebit is not None:
                    historical_roce.append({
                        "year": str(financials.columns[i].date()) if hasattr(financials.columns[i], "date") else str(financials.columns[i]),
                        "roce": round(yr_ebit / yr_ce, 4),
                        "ebit_crores": round(yr_ebit / 1e7, 2),
                        "capital_employed_crores": round(yr_ce / 1e7, 2),
                    })
            except (KeyError, IndexError, TypeError):
                continue

        quality = "Excellent" if roce > 0.20 else "Good" if roce > 0.15 else "Average" if roce > 0.10 else "Poor"

        result = {
            "ticker": ticker,
            "current_roce": round(roce, 4),
            "roce_pct": f"{roce * 100:.2f}%",
            "quality_rating": quality,
            "ebit_crores": round(ebit / 1e7, 2),
            "capital_employed_crores": round(capital_employed / 1e7, 2),
            "historical_roce": historical_roce,
            "benchmarks": {
                "excellent": ">20%",
                "good": "15-20%",
                "average": "10-15%",
                "poor": "<10%",
            },
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("compute_roce failed for %s: %s", ticker, e)
        return json.dumps({"error": f"ROCE calculation failed for {ticker}: {e}"})


# ── Tool list for agents ─────────────────────────────────────

INDIAN_METRICS_TOOLS = [
    get_shareholding_pattern,
    get_promoter_pledge_info,
    get_fii_dii_activity,
    compute_roce,
]
