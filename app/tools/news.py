"""News & sentiment analysis tools — fetch and score financial news."""

from __future__ import annotations

import json
from datetime import datetime

import feedparser
import requests
from langchain_core.tools import tool


# ── Free RSS feeds for financial news ────────────────────────
RSS_FEEDS = {
    "yahoo_finance": "https://finance.yahoo.com/rss/headline?s={ticker}",
    "google_news": "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
    "seeking_alpha": "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
}

HEADERS = {
    "User-Agent": "EquityResearchBot/1.0 (research@example.com)",
}


@tool
def fetch_financial_news(ticker: str, max_articles: int = 10) -> str:
    """Fetch recent financial news articles for a stock ticker from RSS feeds.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        max_articles: Maximum number of articles to return (default 10)
    """
    articles = []
    errors = []

    for source_name, feed_url_template in RSS_FEEDS.items():
        try:
            feed_url = feed_url_template.format(ticker=ticker.upper())
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_articles]:
                published = ""
                if hasattr(entry, "published"):
                    published = entry.published
                elif hasattr(entry, "updated"):
                    published = entry.updated

                articles.append({
                    "source": source_name,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:500],
                    "link": entry.get("link", ""),
                    "published": published,
                })
        except Exception as e:
            errors.append(f"{source_name}: {e}")

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for article in articles:
        title_lower = article["title"].lower().strip()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_articles.append(article)

    # Limit total
    unique_articles = unique_articles[:max_articles]

    result = {
        "ticker": ticker.upper(),
        "articles_found": len(unique_articles),
        "fetch_date": datetime.now().isoformat(),
        "articles": unique_articles,
    }
    if errors:
        result["feed_errors"] = errors

    return json.dumps(result, indent=2, default=str)


@tool
def fetch_market_news(query: str = "stock market", max_articles: int = 5) -> str:
    """Fetch general market news from Google News RSS feed.

    Args:
        query: Search terms for market news (default 'stock market')
        max_articles: Maximum articles to return (default 5)
    """
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })

        return json.dumps({
            "query": query,
            "articles_found": len(articles),
            "articles": articles,
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Market news fetch failed: {e}"})


NEWS_TOOLS = [fetch_financial_news, fetch_market_news]
