"""MCP Server — Indian Corporate Filings tools (BSE/NSE announcements, earnings).

Run standalone: python mcp_servers/corporate_filings_server.py
Or via MCP: mcp dev mcp_servers/corporate_filings_server.py
"""

from __future__ import annotations

import json
import sys
import logging

import requests
import yfinance as yf
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP("indian-corporate-filings")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


@mcp.tool()
async def get_bse_announcements(ticker: str, count: int = 10) -> str:
    """Fetch recent corporate announcements from BSE for an Indian company.

    Includes board meetings, results, dividends, corporate actions, and regulatory filings.

    Args:
        ticker: Indian stock ticker (e.g., 'RELIANCE.NS')
        count: Number of announcements to fetch (default 10)
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")

    try:
        url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
        params = {
            "strCat": "-1",
            "strPrevDate": "",
            "strScrip": base_name,
            "strSearch": "P",
            "strToDate": "",
            "strType": "C",
        }

        response = requests.get(url, params=params, headers=HEADERS, timeout=15)

        if response.status_code == 200:
            try:
                data = response.json()
                announcements = []

                if isinstance(data, dict) and "Table" in data:
                    for item in data["Table"][:count]:
                        announcements.append({
                            "date": item.get("DT_TM", ""),
                            "headline": item.get("NEWSSUB", ""),
                            "category": item.get("CATEGORYNAME", ""),
                            "attachment_url": item.get("ATTACHMENTNAME", ""),
                        })
                elif isinstance(data, list):
                    for item in data[:count]:
                        announcements.append({
                            "date": item.get("DT_TM", item.get("NEWS_DT", "")),
                            "headline": item.get("NEWSSUB", item.get("HEAD_LINE", "")),
                            "category": item.get("CATEGORYNAME", ""),
                        })

                return json.dumps({
                    "ticker": ticker,
                    "source": "BSE India",
                    "announcements_found": len(announcements),
                    "announcements": announcements,
                }, indent=2, default=str)
            except (ValueError, KeyError):
                pass

        return json.dumps({
            "ticker": ticker,
            "source": "BSE India",
            "announcements": [],
            "note": f"Could not fetch BSE announcements for {base_name}. "
                    f"Visit https://www.bseindia.com for direct access.",
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"BSE announcements fetch failed for {ticker}: {e}",
            "fallback": f"Check https://www.bseindia.com/stock-share-price/{base_name}/ for announcements.",
        })


@mcp.tool()
async def get_earnings_transcript_summary(ticker: str) -> str:
    """Fetch earnings call transcript context for an Indian company.

    Provides links and guidance for accessing the latest earnings call transcript.
    Sources: Trendlyne, AlphaStreet India, company investor relations.

    Args:
        ticker: Indian stock ticker (e.g., 'RELIANCE.NS')
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get("longName", base_name)
        website = info.get("website", "N/A")

        calendar = {}
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    calendar = {k: str(v) for k, v in cal.items()}
        except Exception:
            pass

        result = {
            "ticker": ticker,
            "company": company_name,
            "upcoming_events": calendar if calendar else "No upcoming events data",
            "transcript_sources": [
                {
                    "source": "Trendlyne",
                    "url": f"https://trendlyne.com/equity/{base_name}/",
                    "note": "Free earnings call transcripts for NSE/BSE companies",
                },
                {
                    "source": "AlphaStreet India",
                    "url": f"https://alphastreet.com/india/?s={company_name}",
                    "note": "Live audio and transcript streaming for earnings calls",
                },
                {
                    "source": "Company IR Page",
                    "url": website,
                    "note": "Many Indian companies publish transcripts on their investor relations page",
                },
            ],
            "management_analysis_note": (
                "Key things to look for in Indian earnings calls: "
                "1) Management guidance on revenue growth and margins "
                "2) Capex plans and capacity expansion "
                "3) Commentary on demand environment "
                "4) Working capital and cash flow trends "
                "5) Views on regulatory changes (GST, SEBI, RBI policy)"
            ),
        }

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Earnings transcript fetch failed for {ticker}: {e}"})


@mcp.tool()
async def get_corporate_actions(ticker: str) -> str:
    """Get recent dividends, stock splits, and bonus issues for an Indian stock.

    Args:
        ticker: Indian stock ticker (e.g., 'RELIANCE.NS')
    """
    try:
        stock = yf.Ticker(ticker)

        actions = {"ticker": ticker, "currency": "INR"}

        # Dividends
        try:
            dividends = stock.dividends
            if dividends is not None and not dividends.empty:
                recent_divs = dividends.tail(10)
                actions["dividends"] = [
                    {"date": str(idx.date()), "amount_inr": round(float(val), 2)}
                    for idx, val in recent_divs.items()
                ]
            else:
                actions["dividends"] = []
        except Exception:
            actions["dividends"] = []

        # Splits
        try:
            splits = stock.splits
            if splits is not None and not splits.empty:
                recent_splits = splits[splits != 0].tail(5)
                actions["splits"] = [
                    {"date": str(idx.date()), "ratio": f"{float(val)}:1"}
                    for idx, val in recent_splits.items()
                ]
            else:
                actions["splits"] = []
        except Exception:
            actions["splits"] = []

        return json.dumps(actions, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Corporate actions fetch failed for {ticker}: {e}"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
