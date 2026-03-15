"""Qualitative screening — insider activity, BSE announcements, news keyword search, capacity utilization."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests

from app.utils.cache import cache_get, cache_set, retry_on_error, yfinance_rate_limiter
from app.tools.screener.batch_fundamentals import fetch_ticker_financials

logger = logging.getLogger(__name__)

# ── Keyword patterns for announcement/news screening ────────

CRITERIA_KEYWORDS = {
    "capex": re.compile(
        r"capex|capacity expansion|new plant|commissioning|commercial operation"
        r"|COD|goes live|ramp.?up|capacity addition|greenfield|brownfield",
        re.IGNORECASE,
    ),
    "order_booking": re.compile(
        r"order win|order book|order inflow|received order|awarded contract"
        r"|LOI|letter of intent|new order|order value|order worth",
        re.IGNORECASE,
    ),
    "new_business": re.compile(
        r"new product|new business|diversification|new vertical|new segment"
        r"|launch|foray|joint venture|JV|strategic partnership|collaboration"
        r"|commerciali[sz]ation",
        re.IGNORECASE,
    ),
}

_BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


# ── Insider transactions ────────────────────────────────────


def screen_insider_transactions(
    tickers: list[str],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with net insider buying in the last 6 months.

    Uses yfinance .insider_transactions data.
    """
    results: list[dict] = []

    for ticker in tickers:
        try:
            data = fetch_ticker_financials(ticker)
            insider_df = data.get("insider_transactions")
            if insider_df is None or insider_df.empty:
                continue

            # Count buys vs sells
            buys = 0
            sells = 0
            buy_value = 0.0
            sell_value = 0.0

            for _, row in insider_df.iterrows():
                text = str(row.get("Text", row.get("Transaction", ""))).lower()
                shares = abs(float(row.get("Shares", row.get("shares", 0)) or 0))
                value = abs(float(row.get("Value", row.get("value", 0)) or 0))

                if any(w in text for w in ["purchase", "buy", "acquisition"]):
                    buys += 1
                    buy_value += value if value else shares
                elif any(w in text for w in ["sale", "sell", "disposition"]):
                    sells += 1
                    sell_value += value if value else shares

            if buys > sells:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "insider_buys": buys,
                    "insider_sells": sells,
                    "net_buying": True,
                    "buy_value": round(buy_value, 2),
                    "sell_value": round(sell_value, 2),
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Insider screen failed for %s: %s", ticker, e)

    logger.info("Insider screen: %d stocks with net buying", len(results))
    return results


# ── BSE announcement keyword search ─────────────────────────


@retry_on_error(max_retries=1, base_delay=2.0, exceptions=(requests.RequestException,))
def _fetch_bse_announcements(base_name: str, count: int = 20) -> list[dict]:
    """Fetch BSE announcements for a ticker."""
    cache_key = f"screener_bse:{base_name}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    params = {
        "strCat": "-1",
        "strPrevDate": "",
        "strScrip": base_name,
        "strSearch": "P",
        "strToDate": "",
        "strType": "C",
    }

    response = requests.get(url, params=params, headers=_BSE_HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()

    announcements = []
    items = []
    if isinstance(data, dict) and "Table" in data:
        items = data["Table"][:count]
    elif isinstance(data, list):
        items = data[:count]

    for item in items:
        announcements.append({
            "date": item.get("DT_TM", item.get("NEWS_DT", "")),
            "headline": item.get("NEWSSUB", item.get("HEAD_LINE", "")),
            "category": item.get("CATEGORYNAME", ""),
        })

    cache_set(cache_key, announcements, ttl=3600)
    return announcements


def _fetch_news_headlines(base_name: str, max_articles: int = 15) -> list[dict]:
    """Fetch recent news headlines from Google News India RSS."""
    cache_key = f"screener_news:{base_name}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"https://news.google.com/rss/search?q={quote_plus(base_name)}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "published": entry.get("published", entry.get("updated", "")),
                "link": entry.get("link", ""),
            })
        cache_set(cache_key, articles, ttl=3600)
        return articles
    except Exception as e:
        logger.debug("News fetch failed for %s: %s", base_name, e)
        return []


def screen_announcements(
    tickers: list[str],
    sector_map: dict[str, str],
    max_workers: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """Screen BSE announcements and news for capex, order booking, and new business keywords.

    Returns {criterion_name: [list of matching stocks with details]}.
    """
    results: dict[str, list[dict]] = {key: [] for key in CRITERIA_KEYWORDS}

    def _check_ticker(ticker: str) -> dict[str, list[dict]]:
        base_name = ticker.upper().replace(".NS", "").replace(".BO", "")
        matches: dict[str, list[dict]] = {key: [] for key in CRITERIA_KEYWORDS}

        # Fetch BSE announcements
        try:
            announcements = _fetch_bse_announcements(base_name)
            for ann in announcements:
                headline = ann.get("headline", "")
                for criterion, pattern in CRITERIA_KEYWORDS.items():
                    if pattern.search(headline):
                        matches[criterion].append({
                            "source": "BSE",
                            "headline": headline,
                            "date": ann.get("date", ""),
                        })
        except Exception as e:
            logger.debug("BSE check failed for %s: %s", ticker, e)

        # Fetch news
        try:
            articles = _fetch_news_headlines(base_name)
            for article in articles:
                title = article.get("title", "")
                for criterion, pattern in CRITERIA_KEYWORDS.items():
                    if pattern.search(title):
                        matches[criterion].append({
                            "source": "News",
                            "headline": title,
                            "date": article.get("published", ""),
                        })
        except Exception as e:
            logger.debug("News check failed for %s: %s", ticker, e)

        return matches

    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_check_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                matches = future.result()
                for criterion, hits in matches.items():
                    if hits:
                        results[criterion].append({
                            "ticker": ticker,
                            "sector": sector_map.get(ticker, "Other"),
                            "matches": hits,
                            "match_count": len(hits),
                            "passed": True,
                        })
            except Exception as e:
                logger.debug("Announcement screen failed for %s: %s", ticker, e)

    for criterion, stocks in results.items():
        logger.info("Announcement screen [%s]: %d stocks matched", criterion, len(stocks))

    return results


# ── Capacity utilization ─────────────────────────────────────


def screen_capacity_utilization(
    tickers: list[str],
    sector_map: dict[str, str],
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """Screen for stocks with low capacity utilization using asset turnover as proxy.

    Asset turnover = Revenue / Total Assets. Low values (<0.5) suggest underutilized capacity.
    If use_llm=True, attempts LLM extraction from earnings context (future enhancement).
    """
    results: list[dict] = []

    for ticker in tickers:
        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            bs = data.get("balance_sheet")
            if fin is None or bs is None:
                continue

            # Asset turnover proxy
            revenue = None
            total_assets = None

            if len(fin.columns) > 0:
                for field in ["Total Revenue", "Revenue"]:
                    val = fin.iloc[:, 0].get(field)
                    if val is not None:
                        try:
                            revenue = float(val)
                            if revenue > 0:
                                break
                        except (ValueError, TypeError):
                            pass
                        revenue = None

            if len(bs.columns) > 0:
                for field in ["Total Assets"]:
                    val = bs.iloc[:, 0].get(field)
                    if val is not None:
                        try:
                            total_assets = float(val)
                            if total_assets > 0:
                                break
                        except (ValueError, TypeError):
                            pass
                        total_assets = None

            if revenue and total_assets and total_assets > 0:
                asset_turnover = revenue / total_assets
                # Low asset turnover suggests low capacity utilization
                if asset_turnover < 0.5:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "asset_turnover": round(asset_turnover, 3),
                        "revenue": round(revenue / 1e7, 2),  # in Cr
                        "total_assets": round(total_assets / 1e7, 2),
                        "method": "asset_turnover_proxy",
                        "passed": True,
                    })
        except Exception as e:
            logger.debug("Capacity util screen failed for %s: %s", ticker, e)

    logger.info("Capacity utilization screen: %d stocks with low asset turnover", len(results))
    return results
