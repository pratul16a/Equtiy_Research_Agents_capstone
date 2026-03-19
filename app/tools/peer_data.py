"""Indian sector peer mapping and comparison tool.

Includes TTL caching and rate limiting for resilient API calls.
"""

from __future__ import annotations

import json
import logging

import yfinance as yf
from langchain_core.tools import tool

from app.utils.cache import cached, yfinance_rate_limiter

logger = logging.getLogger(__name__)


# ── Indian sector → Nifty peer mapping ───────────────────────

INDIAN_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "GAIL.NS", "NTPC.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
    "Consumer Cyclical": ["TITAN.NS", "TRENT.NS", "DMART.NS", "ASIANPAINT.NS", "PIDILITIND.NS"],
    "Automobiles": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS", "APOLLOHOSP.NS"],
    "Basic Materials": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS", "UPL.NS"],
    "Real Estate": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS"],
    "Communication Services": ["BHARTIARTL.NS", "TATACOMM.NS"],
    "Industrials": ["LT.NS", "ADANIENT.NS", "ADANIPORTS.NS", "SIEMENS.NS", "ABB.NS", "HAL.NS"],
    "Utilities": ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "NHPC.NS"],
    "Consumer Internet": ["ZOMATO.NS", "NYKAA.NS", "POLICYBZR.NS", "PAYTM.NS", "CARTRADE.NS", "EASEMYTRIP.NS"],
}

# Broader sector aliases to map yfinance sector names to our mapping
SECTOR_ALIASES: dict[str, str] = {
    "Technology": "Technology",
    "Information Technology": "Technology",
    "Financial Services": "Financial Services",
    "Financials": "Financial Services",
    "Energy": "Energy",
    "Oil & Gas": "Energy",
    "Consumer Defensive": "Consumer Defensive",
    "Consumer Staples": "Consumer Defensive",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Discretionary": "Consumer Cyclical",
    "Automobiles": "Automobiles",
    "Auto": "Automobiles",
    "Healthcare": "Healthcare",
    "Health Care": "Healthcare",
    "Pharma": "Healthcare",
    "Basic Materials": "Basic Materials",
    "Materials": "Basic Materials",
    "Metals & Mining": "Basic Materials",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
    "Telecom": "Communication Services",
    "Industrials": "Industrials",
    "Capital Goods": "Industrials",
    "Infrastructure": "Industrials",
    "Utilities": "Utilities",
    "Consumer Internet": "Consumer Internet",
    "Internet & Direct Marketing Retail": "Consumer Internet",
    "Internet Software & Services": "Consumer Internet",
    "Internet Content & Information": "Consumer Internet",
}


@cached(ttl=300, key_prefix="peer_info")
def _fetch_peer_info(peer_ticker: str) -> dict | None:
    """Fetch info for a single peer ticker with caching."""
    try:
        yfinance_rate_limiter.wait()
        s = yf.Ticker(peer_ticker)
        i = s.info
        return {
            "ticker": peer_ticker,
            "name": i.get("longName", peer_ticker),
            "market_cap_crores": round(i.get("marketCap", 0) / 1e7, 2),
            "pe_trailing": i.get("trailingPE"),
            "pe_forward": i.get("forwardPE"),
            "pb_ratio": i.get("priceToBook"),
            "ev_ebitda": i.get("enterpriseToEbitda"),
            "profit_margin": i.get("profitMargins"),
            "roe": i.get("returnOnEquity"),
            "revenue_growth": i.get("revenueGrowth"),
            "debt_to_equity": i.get("debtToEquity"),
            "dividend_yield": i.get("dividendYield"),
            "current_price": i.get("currentPrice", i.get("regularMarketPrice", 0)),
        }
    except Exception as e:
        logger.warning("Failed to fetch peer info for %s: %s", peer_ticker, e)
        return None


@tool
def get_peer_comparison(ticker: str) -> str:
    """Get peer company comparison for an Indian stock — key metrics for companies in the same sector.

    Compares the target stock with Nifty sector peers on valuation, profitability, and growth.
    """
    try:
        yfinance_rate_limiter.wait()
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get("sector", "")
        industry = info.get("industry", "")

        # Resolve sector to our peer mapping
        mapped_sector = SECTOR_ALIASES.get(sector, sector)
        peer_tickers = INDIAN_SECTOR_PEERS.get(mapped_sector, [])

        # Remove self from peers and limit to 5
        peer_tickers = [p for p in peer_tickers if p.upper() != ticker.upper()][:5]

        # Add self as first entry
        all_tickers = [ticker.upper()] + peer_tickers
        peers_data = []

        for t in all_tickers:
            peer_info = _fetch_peer_info(t)
            if peer_info is not None:
                peers_data.append(peer_info)

        result = {
            "target_ticker": ticker.upper(),
            "sector": sector,
            "mapped_sector": mapped_sector,
            "industry": industry,
            "currency": "INR",
            "peers_count": len(peers_data),
            "peers": peers_data,
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_peer_comparison failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Peer comparison failed for {ticker}: {e}"})


# ── Tool list for agents ─────────────────────────────────────

PEER_TOOLS = [get_peer_comparison]
