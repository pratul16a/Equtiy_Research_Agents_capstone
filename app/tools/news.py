"""Indian financial news tools — MoneyControl, Economic Times, Google News India RSS feeds.

Includes retry logic and caching for resilient news fetching.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from urllib.parse import quote_plus

import feedparser
from langchain_core.tools import tool

from app.utils.cache import cached, retry_on_error

logger = logging.getLogger(__name__)


# ── Indian financial news RSS feeds ──────────────────────────
INDIAN_RSS_FEEDS = {
    "google_news_india": "https://news.google.com/rss/search?q={query}+stock+india&hl=en-IN&gl=IN&ceid=IN:en",
    "economic_times_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "moneycontrol_news": "https://www.moneycontrol.com/rss/latestnews.xml",
}

HEADERS = {
    "User-Agent": "IndianEquityResearchBot/1.0 (research@example.com)",
}


@retry_on_error(max_retries=2, base_delay=1.0, exceptions=(Exception,))
def _fetch_single_feed(source_name: str, feed_url: str, base_name: str, max_articles: int) -> list[dict]:
    """Fetch articles from a single RSS feed with retry logic."""
    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries[:max_articles]:
        published = ""
        if hasattr(entry, "published"):
            published = entry.published
        elif hasattr(entry, "updated"):
            published = entry.updated

        title = entry.get("title", "")
        summary = entry.get("summary", entry.get("description", ""))[:500]

        # For non-ticker-specific feeds, filter for relevance
        if "{query}" not in INDIAN_RSS_FEEDS.get(source_name, ""):
            if base_name.lower() not in (title + summary).lower():
                continue

        articles.append({
            "source": source_name,
            "title": title,
            "summary": summary,
            "link": entry.get("link", ""),
            "published": published,
        })

    return articles


@tool
def fetch_indian_financial_news(ticker: str, max_articles: int = 10) -> str:
    """Fetch recent financial news for an Indian stock from Google News India, ET, MoneyControl.

    Args:
        ticker: Stock ticker (e.g., 'RELIANCE.NS') — the base name is extracted for search
        max_articles: Maximum number of articles to return (default 10)
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")
    articles = []
    errors = []

    for source_name, feed_url_template in INDIAN_RSS_FEEDS.items():
        try:
            if "{query}" in feed_url_template:
                feed_url = feed_url_template.format(query=quote_plus(base_name))
            else:
                feed_url = feed_url_template

            feed_articles = _fetch_single_feed(source_name, feed_url, base_name, max_articles)
            articles.extend(feed_articles)
        except Exception as e:
            logger.warning("Feed %s failed after retries: %s", source_name, e)
            errors.append(f"{source_name}: {e}")

    # Deduplicate by title
    seen_titles: set[str] = set()
    unique_articles = []
    for article in articles:
        title_lower = article["title"].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)

    unique_articles = unique_articles[:max_articles]

    result = {
        "ticker": ticker,
        "search_term": base_name,
        "articles_found": len(unique_articles),
        "fetch_date": datetime.now().isoformat(),
        "articles": unique_articles,
    }
    if errors:
        result["feed_errors"] = errors

    return json.dumps(result, indent=2, default=str)


@tool
def fetch_indian_market_news(query: str = "nifty sensex indian stock market", max_articles: int = 8) -> str:
    """Fetch general Indian market news — Nifty, Sensex, RBI, SEBI updates.

    Args:
        query: Search terms for market news (default: 'nifty sensex indian stock market')
        max_articles: Maximum articles to return (default 8)
    """
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })

        # Also fetch from ET markets feed
        try:
            et_feed = feedparser.parse("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
            for entry in et_feed.entries[:5]:
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source": "economic_times",
                })
        except Exception as e:
            logger.warning("ET markets feed failed: %s", e)

        # Deduplicate
        seen: set[str] = set()
        unique = []
        for a in articles:
            key = a["title"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return json.dumps({
            "query": query,
            "articles_found": len(unique[:max_articles]),
            "articles": unique[:max_articles],
        }, indent=2, default=str)
    except Exception as e:
        logger.error("fetch_indian_market_news failed: %s", e)
        return json.dumps({"error": f"Indian market news fetch failed: {e}"})


# ── Tool list for agents ─────────────────────────────────────

NEWS_TOOLS = [fetch_indian_financial_news, fetch_indian_market_news]
