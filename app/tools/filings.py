"""Indian corporate filings tools — BSE/NSE announcements and earnings transcripts.

Includes retry logic and caching for resilient API calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import requests
from langchain_core.tools import tool

from app.utils.cache import retry_on_error, get_yf_ticker

logger = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


@retry_on_error(max_retries=2, base_delay=2.0, exceptions=(requests.RequestException, requests.Timeout))
def _fetch_bse_api(base_name: str, count: int) -> list[dict]:
    """Fetch BSE announcements with retry logic."""
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
    response.raise_for_status()

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

    return announcements


@tool
def get_bse_announcements(ticker: str, count: int = 10) -> str:
    """Fetch recent corporate announcements from BSE for an Indian company.

    Includes board meetings, results, dividends, corporate actions, and regulatory filings.
    Ticker should include .NS or .BO suffix (base symbol is extracted).

    Args:
        ticker: Indian stock ticker (e.g., 'RELIANCE.NS')
        count: Number of announcements to fetch (default 10)
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")

    try:
        announcements = _fetch_bse_api(base_name, count)

        return json.dumps({
            "ticker": ticker,
            "source": "BSE India",
            "announcements_found": len(announcements),
            "announcements": announcements,
        }, indent=2, default=str)
    except Exception as e:
        logger.warning("BSE announcements fetch failed for %s: %s", ticker, e)
        # Graceful fallback instead of error
        return json.dumps({
            "ticker": ticker,
            "source": "BSE India",
            "announcements": [],
            "note": (
                f"Could not fetch BSE announcements for {base_name} after retries. "
                "Visit https://www.bseindia.com for direct access to corporate filings."
            ),
        }, indent=2)


@tool
def get_earnings_transcript_summary(ticker: str) -> str:
    """Fetch earnings call transcript context for an Indian company.

    Provides links and guidance for accessing the latest earnings call transcript.
    Sources: Trendlyne, AlphaStreet India, company investor relations.

    Args:
        ticker: Indian stock ticker (e.g., 'RELIANCE.NS')
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")

    try:
        stock, info = get_yf_ticker(ticker)
        company_name = info.get("longName", base_name)
        website = info.get("website", "N/A")

        # Get recent calendar events (earnings dates)
        calendar = {}
        try:
            cal = stock.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    calendar = {k: str(v) for k, v in cal.items()}
        except Exception as e:
            logger.warning("Calendar data unavailable for %s: %s", ticker, e)

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
                "3) Commentary on demand environment and competitive dynamics "
                "4) Working capital and cash flow trends "
                "5) Views on regulatory changes (GST, SEBI, RBI policy)"
            ),
        }

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error("get_earnings_transcript_summary failed for %s: %s", ticker, e)
        return json.dumps({"error": f"Earnings transcript fetch failed for {ticker}: {e}"})


# ── Tool list for agents ─────────────────────────────────────

FILING_TOOLS = [get_bse_announcements, get_earnings_transcript_summary]
