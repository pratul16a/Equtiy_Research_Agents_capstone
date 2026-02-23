"""Financial data tools — wrappers around yfinance for structured data retrieval."""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf
from langchain_core.tools import tool


# ── Data-Fetching Tools ──────────────────────────────────────


@tool
def get_company_info(ticker: str) -> str:
    """Get basic company information including sector, industry, market cap, and description."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
            "52_week_high": info.get("fiftyTwoWeekHigh", 0),
            "52_week_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "pe_trailing": info.get("trailingPE", None),
            "pe_forward": info.get("forwardPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "beta": info.get("beta", None),
            "description": info.get("longBusinessSummary", "N/A")[:500],
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch company info for {ticker}: {e}"})


@tool
def get_income_statement(ticker: str) -> str:
    """Get the annual income statement for a stock (last 4 years)."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.financials  # annual income statement
        if df is None or df.empty:
            return json.dumps({"error": f"No income statement data for {ticker}"})

        # Transpose so each row is a year, convert to dict
        data = df.T.head(4).to_dict()
        # Convert keys to strings for JSON
        result = {}
        for col, values in data.items():
            result[str(col)] = {str(k): float(v) if v == v else None for k, v in values.items()}
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch income statement for {ticker}: {e}"})


@tool
def get_balance_sheet(ticker: str) -> str:
    """Get the annual balance sheet for a stock (last 4 years)."""
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


@tool
def get_cash_flow(ticker: str) -> str:
    """Get the annual cash flow statement for a stock (last 4 years)."""
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


@tool
def get_stock_price_history(ticker: str, period: str = "1y") -> str:
    """Get historical stock price data. Period can be: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or df.empty:
            return json.dumps({"error": f"No price history for {ticker}"})

        # Return summary stats + last 30 data points
        summary = {
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


# ── Analysis / Computation Tools ─────────────────────────────


@tool
def compute_financial_ratios(ticker: str) -> str:
    """Compute key financial ratios: P/E, P/B, EV/EBITDA, ROE, D/E, margins, etc."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance = stock.balance_sheet

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

        # Filter out None values for cleaner output
        ratios = {k: round(v, 4) if isinstance(v, float) else v for k, v in ratios.items() if v is not None}

        return json.dumps(ratios, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to compute ratios for {ticker}: {e}"})


@tool
def compute_dcf_valuation(ticker: str, growth_rate: float = 0.08, discount_rate: float = 0.10, terminal_growth: float = 0.03, projection_years: int = 5) -> str:
    """Compute a simple DCF (Discounted Cash Flow) intrinsic value estimate.

    Args:
        ticker: Stock ticker symbol
        growth_rate: Expected FCF growth rate (default 8%)
        discount_rate: WACC / discount rate (default 10%)
        terminal_growth: Terminal growth rate (default 3%)
        projection_years: Number of years to project (default 5)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
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
                "projected_fcf": round(projected_fcf / 1e9, 2),
                "discounted_fcf": round(discounted_fcf / 1e9, 2),
            })

        # Terminal value
        terminal_fcf = fcf * (1 + growth_rate) ** projection_years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value / (1 + discount_rate) ** projection_years

        # Total enterprise value
        total_pv_fcfs = sum(pf["discounted_fcf"] for pf in projected_fcfs) * 1e9
        enterprise_value = total_pv_fcfs + discounted_terminal

        # Intrinsic value per share
        intrinsic_value = enterprise_value / shares_outstanding

        result = {
            "current_fcf_billions": round(fcf / 1e9, 2),
            "assumptions": {
                "growth_rate": growth_rate,
                "discount_rate": discount_rate,
                "terminal_growth": terminal_growth,
                "projection_years": projection_years,
            },
            "projected_fcfs": projected_fcfs,
            "terminal_value_billions": round(terminal_value / 1e9, 2),
            "discounted_terminal_billions": round(discounted_terminal / 1e9, 2),
            "enterprise_value_billions": round(enterprise_value / 1e9, 2),
            "shares_outstanding_millions": round(shares_outstanding / 1e6, 2),
            "intrinsic_value_per_share": round(intrinsic_value, 2),
            "current_price": round(current_price, 2),
            "upside_pct": round((intrinsic_value / current_price - 1) * 100, 2),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"DCF calculation failed for {ticker}: {e}"})


@tool
def get_peer_comparison(ticker: str) -> str:
    """Get peer company comparison — fetches key metrics for companies in the same sector."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        # Use yfinance screener or a fallback list of known peers
        # For simplicity, we'll use recommendations/peers if available
        peers_data = []

        # Try to get peer tickers from yfinance
        peer_tickers = []
        try:
            # yfinance doesn't have a direct peers API, so use a sector-based approach
            import yfinance as yf_inner

            # Fallback: use known sector peers
            sector_peers = {
                "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN"],
                "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
                "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY"],
                "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX"],
                "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
                "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T"],
            }
            peer_tickers = sector_peers.get(sector, [])[:5]
            # Remove self from peers
            peer_tickers = [p for p in peer_tickers if p.upper() != ticker.upper()][:4]
        except Exception:
            pass

        # Add self as first entry for comparison
        all_tickers = [ticker.upper()] + peer_tickers

        for t in all_tickers:
            try:
                s = yf.Ticker(t)
                i = s.info
                peers_data.append({
                    "ticker": t,
                    "name": i.get("longName", t),
                    "market_cap_b": round(i.get("marketCap", 0) / 1e9, 2),
                    "pe_trailing": i.get("trailingPE"),
                    "pe_forward": i.get("forwardPE"),
                    "ev_ebitda": i.get("enterpriseToEbitda"),
                    "profit_margin": i.get("profitMargins"),
                    "roe": i.get("returnOnEquity"),
                    "revenue_growth": i.get("revenueGrowth"),
                    "debt_to_equity": i.get("debtToEquity"),
                })
            except Exception:
                continue

        result = {
            "target_ticker": ticker.upper(),
            "sector": sector,
            "industry": industry,
            "peers": peers_data,
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Peer comparison failed for {ticker}: {e}"})


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
    get_peer_comparison,
]
