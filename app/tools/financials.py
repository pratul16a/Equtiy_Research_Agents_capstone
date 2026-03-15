"""Financial data tools — yfinance wrappers adapted for Indian stocks (NSE/BSE).

Includes TTL caching, rate limiting, and retry logic for resilient API calls.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yfinance as yf
from langchain_core.tools import tool

from app.config import INDIAN_RISK_FREE_RATE, INDIAN_MARKET_PREMIUM, INDIAN_TERMINAL_GROWTH
from app.utils.cache import cached, get_yf_ticker, retry_on_error, yfinance_rate_limiter

logger = logging.getLogger(__name__)


# ── Cached yfinance helpers ──────────────────────────────────

@cached(ttl=300, key_prefix="yf_financials")
@retry_on_error(max_retries=2, base_delay=1.0)
def _fetch_financials(ticker: str) -> dict:
    """Fetch and cache financial statements for a ticker."""
    yfinance_rate_limiter.wait()
    stock = yf.Ticker(ticker)
    result = {}

    for attr, label in [("financials", "income"), ("balance_sheet", "balance"), ("cashflow", "cashflow")]:
        df = getattr(stock, attr, None)
        if df is not None and not df.empty:
            data = df.T.head(4).to_dict()
            result[label] = {
                str(col): {str(k): float(v) if v == v else None for k, v in values.items()}
                for col, values in data.items()
            }
    return result


# ── Data-Fetching Tools ──────────────────────────────────────


@tool
def get_company_info(ticker: str) -> str:
    """Get basic company information for an Indian stock including sector, market cap, and description.

    Ticker should include .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    """
    try:
        stock, info = get_yf_ticker(ticker)
        result = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "market_cap_crores": round(info.get("marketCap", 0) / 1e7, 2),
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "pe_trailing": info.get("trailingPE", None),
            "pe_forward": info.get("forwardPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "book_value": info.get("bookValue", None),
            "price_to_book": info.get("priceToBook", None),
            "description": info.get("longBusinessSummary", "N/A")[:500],
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_company_info failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch company info for {ticker}: {e}"})


@tool
def get_income_statement(ticker: str) -> str:
    """Get the annual income statement for an Indian stock (last 4 years). Values in INR."""
    try:
        data = _fetch_financials(ticker)
        income = data.get("income")
        if not income:
            return json.dumps({"error": f"No income statement data for {ticker}"})
        return json.dumps(income, indent=2, default=str)
    except Exception as e:
        logger.error("get_income_statement failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch income statement for {ticker}: {e}"})


@tool
def get_balance_sheet(ticker: str) -> str:
    """Get the annual balance sheet for an Indian stock (last 4 years). Values in INR."""
    try:
        data = _fetch_financials(ticker)
        balance = data.get("balance")
        if not balance:
            return json.dumps({"error": f"No balance sheet data for {ticker}"})
        return json.dumps(balance, indent=2, default=str)
    except Exception as e:
        logger.error("get_balance_sheet failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch balance sheet for {ticker}: {e}"})


@tool
def get_cash_flow(ticker: str) -> str:
    """Get the annual cash flow statement for an Indian stock (last 4 years). Values in INR."""
    try:
        data = _fetch_financials(ticker)
        cashflow = data.get("cashflow")
        if not cashflow:
            return json.dumps({"error": f"No cash flow data for {ticker}"})
        return json.dumps(cashflow, indent=2, default=str)
    except Exception as e:
        logger.error("get_cash_flow failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch cash flow for {ticker}: {e}"})


@tool
def get_stock_price_history(ticker: str, period: str = "1y") -> str:
    """Get historical stock price data for an Indian stock.

    Period can be: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max.
    Prices in INR.
    """
    try:
        yfinance_rate_limiter.wait()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return json.dumps({"error": f"No price history for {ticker}"})

        summary = {
            "ticker": ticker,
            "currency": "INR",
            "period": period,
            "data_points": len(df),
            "latest_close": round(float(df["Close"].iloc[-1]), 2),
            "period_high": round(float(df["High"].max()), 2),
            "period_low": round(float(df["Low"].min()), 2),
            "period_return_pct": round(
                float((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100), 2
            ),
            "avg_volume": int(df["Volume"].mean()),
            "recent_prices": [
                {
                    "date": str(idx.date()),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                }
                for idx, row in df.tail(30).iterrows()
            ],
        }
        return json.dumps(summary, indent=2)
    except Exception as e:
        logger.error("get_stock_price_history failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch price history for {ticker}: {e}"})


# ── Analysis / Computation Tools ─────────────────────────────


@tool
def compute_financial_ratios(ticker: str) -> str:
    """Compute key financial ratios for an Indian stock including ROCE.

    Returns: P/E, P/B, EV/EBITDA, ROE, ROA, ROCE, D/E, margins, growth metrics.
    """
    try:
        stock, info = get_yf_ticker(ticker)

        yfinance_rate_limiter.wait()
        balance = stock.balance_sheet
        yfinance_rate_limiter.wait()
        financials = stock.financials

        ratios: dict[str, Any] = {}

        # Valuation ratios
        ratios["pe_trailing"] = info.get("trailingPE")
        ratios["pe_forward"] = info.get("forwardPE")
        ratios["pb_ratio"] = info.get("priceToBook")
        ratios["ps_ratio"] = info.get("priceToSalesTrailing12Months")
        ratios["ev_to_ebitda"] = info.get("enterpriseToEbitda")
        ratios["ev_to_revenue"] = info.get("enterpriseToRevenue")
        ratios["peg_ratio"] = info.get("pegRatio")

        # Profitability ratios
        ratios["profit_margin"] = info.get("profitMargins")
        ratios["operating_margin"] = info.get("operatingMargins")
        ratios["gross_margin"] = info.get("grossMargins")
        ratios["roe"] = info.get("returnOnEquity")
        ratios["roa"] = info.get("returnOnAssets")

        # ROCE — key Indian market metric
        if balance is not None and not balance.empty and financials is not None and not financials.empty:
            try:
                ebit = None
                if "EBIT" in financials.index:
                    ebit = float(financials.loc["EBIT"].iloc[0])
                elif "Operating Income" in financials.index:
                    ebit = float(financials.loc["Operating Income"].iloc[0])

                total_assets = None
                if "Total Assets" in balance.index:
                    total_assets = float(balance.loc["Total Assets"].iloc[0])

                current_liabilities = None
                if "Current Liabilities" in balance.index:
                    current_liabilities = float(balance.loc["Current Liabilities"].iloc[0])

                if ebit and total_assets and current_liabilities:
                    capital_employed = total_assets - current_liabilities
                    if capital_employed > 0:
                        ratios["roce"] = round(ebit / capital_employed, 4)
            except (KeyError, IndexError, TypeError):
                pass

        # Leverage ratios
        ratios["debt_to_equity"] = info.get("debtToEquity")
        ratios["current_ratio"] = info.get("currentRatio")
        ratios["quick_ratio"] = info.get("quickRatio")

        # Growth
        ratios["earnings_growth"] = info.get("earningsGrowth")
        ratios["revenue_growth"] = info.get("revenueGrowth")

        # Dividend
        ratios["dividend_yield"] = info.get("dividendYield")
        ratios["payout_ratio"] = info.get("payoutRatio")

        # Filter out None values
        ratios = {k: round(v, 4) if isinstance(v, float) else v for k, v in ratios.items() if v is not None}

        return json.dumps(ratios, indent=2, default=str)
    except Exception as e:
        logger.error("compute_financial_ratios failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Failed to compute ratios for {ticker}: {e}"})


@tool
def compute_dcf_valuation(
    ticker: str,
    growth_rate: float = 0.10,
    discount_rate: float = 0.135,
    terminal_growth: float = 0.05,
    projection_years: int = 5,
) -> str:
    """Compute a DCF (Discounted Cash Flow) intrinsic value estimate for an Indian stock.

    Default assumptions use Indian market parameters:
    - Growth rate: 10% (Indian growth companies)
    - Discount rate: 13.5% (Indian WACC: Rf=7.2% + beta*7%)
    - Terminal growth: 5% (India GDP growth)

    Args:
        ticker: Indian stock ticker with .NS or .BO suffix
        growth_rate: Expected FCF growth rate (default 10%)
        discount_rate: WACC / discount rate (default 13.5%)
        terminal_growth: Terminal growth rate (default 5%)
        projection_years: Number of years to project (default 5)
    """
    try:
        stock, info = get_yf_ticker(ticker)

        yfinance_rate_limiter.wait()
        cashflow = stock.cashflow

        if cashflow is None or cashflow.empty:
            return json.dumps({"error": "No cash flow data available for DCF"})

        # Get the most recent Free Cash Flow
        if "Free Cash Flow" in cashflow.index:
            fcf = float(cashflow.loc["Free Cash Flow"].iloc[0])
        elif "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
            ocf = float(cashflow.loc["Operating Cash Flow"].iloc[0])
            capex = float(cashflow.loc["Capital Expenditure"].iloc[0])
            fcf = ocf + capex  # capex is typically negative
        else:
            return json.dumps({"error": "Cannot determine Free Cash Flow from available data"})

        shares_outstanding = info.get("sharesOutstanding", 0)
        current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))

        if not shares_outstanding or not current_price:
            return json.dumps({"error": "Missing shares outstanding or current price"})

        # Project future FCFs
        projected_fcfs = []
        for year in range(1, projection_years + 1):
            projected_fcf = fcf * (1 + growth_rate) ** year
            discounted_fcf = projected_fcf / (1 + discount_rate) ** year
            projected_fcfs.append({
                "year": year,
                "projected_fcf_crores": round(projected_fcf / 1e7, 2),
                "discounted_fcf_crores": round(discounted_fcf / 1e7, 2),
            })

        # Terminal value
        terminal_fcf = fcf * (1 + growth_rate) ** projection_years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value / (1 + discount_rate) ** projection_years

        # Total enterprise value
        total_pv_fcfs = sum(pf["discounted_fcf_crores"] for pf in projected_fcfs) * 1e7
        enterprise_value = total_pv_fcfs + discounted_terminal

        # Intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding

        result = {
            "currency": "INR",
            "current_fcf_crores": round(fcf / 1e7, 2),
            "assumptions": {
                "growth_rate": growth_rate,
                "discount_rate_wacc": discount_rate,
                "terminal_growth": terminal_growth,
                "projection_years": projection_years,
                "risk_free_rate": INDIAN_RISK_FREE_RATE,
                "equity_risk_premium": INDIAN_MARKET_PREMIUM,
            },
            "projected_fcfs": projected_fcfs,
            "terminal_value_crores": round(terminal_value / 1e7, 2),
            "discounted_terminal_crores": round(discounted_terminal / 1e7, 2),
            "enterprise_value_crores": round(enterprise_value / 1e7, 2),
            "shares_outstanding_crores": round(shares_outstanding / 1e7, 2),
            "intrinsic_value_per_share": round(intrinsic_value, 2),
            "current_price": round(current_price, 2),
            "upside_pct": round((intrinsic_value / current_price - 1) * 100, 2),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error("compute_dcf_valuation failed for %s: %s", ticker, e)
        return json.dumps({"error": f"DCF calculation failed for {ticker}: {e}"})


# ── Tool lists for agents ────────────────────────────────────

DATA_TOOLS = [
    get_company_info,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_stock_price_history,
]

ANALYSIS_TOOLS = [
    compute_financial_ratios,
    compute_dcf_valuation,
]
