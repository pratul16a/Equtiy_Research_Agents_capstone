"""MCP Server — Indian Financial Data tools (yfinance + Indian metrics).

Run standalone: python mcp_servers/financial_data_server.py
Or via MCP: mcp dev mcp_servers/financial_data_server.py
"""

from __future__ import annotations

import json
import sys
import logging

import yfinance as yf
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("indian-financial-data")


# ── Data-Fetching Tools ──────────────────────────────────────


@mcp.tool()
async def get_company_info(ticker: str) -> str:
    """Get basic company information for an Indian stock.

    Ticker should include .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    Returns: name, sector, industry, market cap, P/E, beta, description.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
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
        return json.dumps({"error": f"Failed to fetch company info for {ticker}: {e}"})


@mcp.tool()
async def get_income_statement(ticker: str) -> str:
    """Get the annual income statement for an Indian stock (last 4 years). Values in INR."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.financials
        if df is None or df.empty:
            return json.dumps({"error": f"No income statement data for {ticker}"})

        data = df.T.head(4).to_dict()
        result = {}
        for col, values in data.items():
            result[str(col)] = {str(k): float(v) if v == v else None for k, v in values.items()}
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch income statement for {ticker}: {e}"})


@mcp.tool()
async def get_balance_sheet(ticker: str) -> str:
    """Get the annual balance sheet for an Indian stock (last 4 years). Values in INR."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.balance_sheet
        if df is None or df.empty:
            return json.dumps({"error": f"No balance sheet data for {ticker}"})

        data = df.T.head(4).to_dict()
        result = {}
        for col, values in data.items():
            result[str(col)] = {str(k): float(v) if v == v else None for k, v in values.items()}
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch balance sheet for {ticker}: {e}"})


@mcp.tool()
async def get_cash_flow(ticker: str) -> str:
    """Get the annual cash flow statement for an Indian stock (last 4 years). Values in INR."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.cashflow
        if df is None or df.empty:
            return json.dumps({"error": f"No cash flow data for {ticker}"})

        data = df.T.head(4).to_dict()
        result = {}
        for col, values in data.items():
            result[str(col)] = {str(k): float(v) if v == v else None for k, v in values.items()}
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch cash flow for {ticker}: {e}"})


@mcp.tool()
async def get_stock_price_history(ticker: str, period: str = "1y") -> str:
    """Get historical stock price data for an Indian stock. Prices in INR.

    Args:
        ticker: Indian stock ticker with .NS or .BO suffix
        period: Time period — 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
    """
    try:
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
        return json.dumps({"error": f"Failed to fetch price history for {ticker}: {e}"})


# ── Indian-Specific Metrics ──────────────────────────────────


@mcp.tool()
async def get_shareholding_pattern(ticker: str) -> str:
    """Get the shareholding pattern for an Indian stock — promoter, FII, DII, and public holdings.

    Uses yfinance major_holders and institutional_holders data.
    """
    try:
        stock = yf.Ticker(ticker)
        result = {"ticker": ticker}

        try:
            major = stock.major_holders
            if major is not None and not major.empty:
                holder_data = {}
                for _, row in major.iterrows():
                    holder_data[str(row.iloc[1]).strip()] = str(row.iloc[0]).strip()
                result["major_holders"] = holder_data
        except Exception:
            result["major_holders"] = "Data unavailable"

        try:
            inst = stock.institutional_holders
            if inst is not None and not inst.empty:
                result["top_institutional_holders"] = [
                    {
                        "holder": str(row.get("Holder", "N/A")),
                        "shares": int(row.get("Shares", 0)),
                        "value": float(row.get("Value", 0)),
                    }
                    for _, row in inst.head(10).iterrows()
                ]
        except Exception:
            result["top_institutional_holders"] = "Data unavailable"

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to get shareholding for {ticker}: {e}"})


@mcp.tool()
async def compute_roce(ticker: str) -> str:
    """Compute Return on Capital Employed (ROCE) for an Indian stock.

    ROCE = EBIT / Capital Employed (Total Assets - Current Liabilities).
    Benchmarks: >20% excellent, 15-20% good, 10-15% average, <10% poor.
    """
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        balance = stock.balance_sheet

        if financials is None or financials.empty or balance is None or balance.empty:
            return json.dumps({"error": f"Insufficient data for ROCE calculation for {ticker}"})

        ebit = None
        if "EBIT" in financials.index:
            ebit = float(financials.loc["EBIT"].iloc[0])
        elif "Operating Income" in financials.index:
            ebit = float(financials.loc["Operating Income"].iloc[0])

        if ebit is None:
            return json.dumps({"error": "EBIT data not found"})

        total_assets = float(balance.loc["Total Assets"].iloc[0]) if "Total Assets" in balance.index else None
        current_liab = float(balance.loc["Current Liabilities"].iloc[0]) if "Current Liabilities" in balance.index else None

        if total_assets is None or current_liab is None:
            return json.dumps({"error": "Balance sheet data insufficient"})

        capital_employed = total_assets - current_liab
        if capital_employed <= 0:
            return json.dumps({"error": "Negative or zero capital employed"})

        roce = ebit / capital_employed
        quality = "Excellent" if roce > 0.20 else "Good" if roce > 0.15 else "Average" if roce > 0.10 else "Poor"

        result = {
            "ticker": ticker,
            "current_roce": round(roce, 4),
            "roce_pct": f"{roce * 100:.2f}%",
            "quality_rating": quality,
            "ebit_crores": round(ebit / 1e7, 2),
            "capital_employed_crores": round(capital_employed / 1e7, 2),
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"ROCE calculation failed for {ticker}: {e}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
