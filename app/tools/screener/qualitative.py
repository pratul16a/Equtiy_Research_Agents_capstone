"""Qualitative screening — insider activity, announcements (yfinance+news), capacity utilization, delivery %.

Data sources (in priority order):
    1. yfinance .news — already fetched per-ticker, zero extra API calls
    2. Google News India RSS — free, fast, broader coverage
    3. nselib — NSE corporate actions, delivery %, block/bulk deals

BSE API removed — unreliable, slow, and the main cause of screener hangs.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any
from urllib.parse import quote_plus

import feedparser
import yfinance as yf

from app.utils.cache import cache_get, cache_set, yfinance_rate_limiter
from app.tools.screener.batch_fundamentals import fetch_ticker_financials

logger = logging.getLogger(__name__)

# ── Keyword patterns for announcement/news screening ────────

CRITERIA_KEYWORDS = {
    "capex": re.compile(
        r"capex|capacity expansion|new plant|commissioning|commercial operation"
        r"|COD|goes live|ramp.?up|capacity addition|greenfield|brownfield"
        r"|invest.*crore|invest.*million|invest.*billion|capital expenditure"
        r"|new facility|manufacturing unit|expansion plan|setting up.*plant",
        re.IGNORECASE,
    ),
    "order_booking": re.compile(
        r"order win|order book|order inflow|received order|awarded contract"
        r"|LOI|letter of intent|new order|order value|order worth"
        r"|bags.*order|secures.*order|wins.*contract|lands.*deal"
        r"|agreement.*with|deal.*with|signs.*contract|inks.*deal",
        re.IGNORECASE,
    ),
    "new_business": re.compile(
        r"new product|new business|diversification|new vertical|new segment"
        r"|launch|foray|joint venture|JV|strategic partnership|collaboration"
        r"|commerciali[sz]ation|acquisition|acquires|acquire.*stake"
        r"|subsidiary|merger|MedTech|biosimilar|enters.*sector"
        r"|share purchase agreement|SPA|postal ballot|ESOP",
        re.IGNORECASE,
    ),
}

_SEBI_RED_FLAGS = re.compile(
    r"SEBI.*order|SEBI.*penalty|SEBI.*ban|audit.*qualification|qualified.*opinion"
    r"|show.?cause|investigation|insider.?trading.*violation|non.?compliance"
    r"|forensic.*audit|fraud|misstatement",
    re.IGNORECASE,
)

_CAPEX_LIVE_PATTERN = re.compile(
    r"commissioning|commercial production|goes live|goes on stream|COD achieved"
    r"|capacity operational|ramp.?up complete|plant inaugurated"
    r"|commerciali[sz]ation.*commence|revenue.*commence|first shipment"
    r"|new.*plant.*operational|trial production"
    r"|commence.*operation|facility.*operational|production.*begin"
    r"|execute.*SPA|complete.*acquisition|becomes.*subsidiary",
    re.IGNORECASE,
)


# ── News fetchers ─────────────────────────────────────────────


def _fetch_yf_news(ticker: str, max_articles: int = 10) -> list[dict]:
    """Fetch news from yfinance .news (Yahoo Finance).

    Returns list of {title, summary, date, source}.
    """
    cache_key = f"screener_yf_news:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
    except Exception as e:
        logger.debug("yfinance news failed for %s: %s", ticker, e)
        return []

    articles = []
    for item in raw_news[:max_articles]:
        content = item.get("content", {})
        articles.append({
            "title": content.get("title", ""),
            "summary": content.get("summary", ""),
            "date": content.get("pubDate", ""),
            "source": content.get("provider", {}).get("displayName", "Yahoo"),
        })

    cache_set(cache_key, articles, ttl=3600)
    return articles


def _fetch_google_news(base_name: str, max_articles: int = 15) -> list[dict]:
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
                "summary": "",
                "date": entry.get("published", entry.get("updated", "")),
                "source": "Google News",
            })
        cache_set(cache_key, articles, ttl=3600)
        return articles
    except Exception as e:
        logger.debug("Google News fetch failed for %s: %s", base_name, e)
        return []


def _get_all_news(ticker: str) -> list[dict]:
    """Get news from both yfinance and Google News (combined, deduplicated)."""
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Primary: yfinance news (free, fast)
    yf_news = _fetch_yf_news(ticker)

    # Secondary: Google News RSS
    g_news = _fetch_google_news(base_name)

    # Combine and deduplicate by title similarity
    all_news = list(yf_news)
    seen_titles = {a["title"].lower()[:50] for a in yf_news}
    for article in g_news:
        key = article["title"].lower()[:50]
        if key not in seen_titles:
            seen_titles.add(key)
            all_news.append(article)

    return all_news


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


# ── Announcement keyword search (yfinance + Google News) ─────


def screen_announcements(
    tickers: list[str],
    sector_map: dict[str, str],
    max_workers: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Screen news for capex, order booking, and new business keywords.

    Uses yfinance .news + Google News RSS (no BSE API).
    Returns {criterion_name: [list of matching stocks with details]}.
    """
    results: dict[str, list[dict]] = {key: [] for key in CRITERIA_KEYWORDS}

    def _check_ticker(ticker: str) -> dict[str, list[dict]]:
        matches: dict[str, list[dict]] = {key: [] for key in CRITERIA_KEYWORDS}
        all_news = _get_all_news(ticker)

        for article in all_news:
            text = f"{article['title']} {article.get('summary', '')}"
            for criterion, pattern in CRITERIA_KEYWORDS.items():
                if pattern.search(text):
                    matches[criterion].append({
                        "source": article.get("source", "News"),
                        "headline": article["title"],
                        "date": article.get("date", ""),
                    })

        return matches

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_check_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker, timeout=120):
            ticker = future_to_ticker[future]
            try:
                matches = future.result(timeout=30)
                for criterion, hits in matches.items():
                    if hits:
                        results[criterion].append({
                            "ticker": ticker,
                            "sector": sector_map.get(ticker, "Other"),
                            "matches": hits,
                            "match_count": len(hits),
                            "passed": True,
                        })
            except (TimeoutError, Exception) as e:
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
    """
    results: list[dict] = []

    for ticker in tickers:
        try:
            data = fetch_ticker_financials(ticker)
            fin = data.get("financials")
            bs = data.get("balance_sheet")
            if fin is None or bs is None:
                continue

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
                if asset_turnover < 0.5:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "asset_turnover": round(asset_turnover, 3),
                        "revenue": round(revenue / 1e7, 2),
                        "total_assets": round(total_assets / 1e7, 2),
                        "method": "asset_turnover_proxy",
                        "passed": True,
                    })
        except Exception as e:
            logger.debug("Capacity util screen failed for %s: %s", ticker, e)

    logger.info("Capacity utilization screen: %d stocks with low asset turnover", len(results))
    return results


# ── SEBI / Audit Red Flags ──────────────────────────────────


def screen_no_sebi_red_flags(
    tickers: list[str],
    sector_map: dict[str, str],
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Screen for stocks with NO SEBI/audit red flags in recent news.

    Uses yfinance .news + Google News (no BSE API).
    Absence of red flags is the positive signal — clean stocks pass.
    """
    results: list[dict] = []

    def _check_clean(ticker: str) -> bool:
        all_news = _get_all_news(ticker)
        for article in all_news:
            text = f"{article['title']} {article.get('summary', '')}"
            if _SEBI_RED_FLAGS.search(text):
                return False
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_check_clean, t): t for t in tickers}
        for future in as_completed(future_to_ticker, timeout=120):
            ticker = future_to_ticker[future]
            try:
                is_clean = future.result(timeout=30)
                if is_clean:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "clean": True,
                        "passed": True,
                    })
            except (TimeoutError, Exception):
                # If check fails, give benefit of doubt
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "clean": True,
                    "passed": True,
                })

    logger.info("SEBI red flags screen: %d clean stocks out of %d", len(results), len(tickers))
    return results


# ── Capex Going Live / Commercializing ──────────────────────


def screen_capex_going_live(
    tickers: list[str],
    sector_map: dict[str, str],
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Screen for stocks with capex/new business going live in recent news.

    Uses yfinance .news + Google News (no BSE API).
    Revenue inflection is imminent, not years away.
    """
    results: list[dict] = []

    def _check_ticker(ticker: str) -> list[dict]:
        matches: list[dict] = []
        all_news = _get_all_news(ticker)
        for article in all_news:
            text = f"{article['title']} {article.get('summary', '')}"
            if _CAPEX_LIVE_PATTERN.search(text):
                matches.append({
                    "source": article.get("source", "News"),
                    "headline": article["title"],
                    "date": article.get("date", ""),
                })
        return matches

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(_check_ticker, t): t for t in tickers}
        for future in as_completed(future_to_ticker, timeout=120):
            ticker = future_to_ticker[future]
            try:
                matches = future.result(timeout=30)
                if matches:
                    results.append({
                        "ticker": ticker,
                        "sector": sector_map.get(ticker, "Other"),
                        "matches": matches,
                        "match_count": len(matches),
                        "passed": True,
                    })
            except (TimeoutError, Exception):
                pass

    logger.info("Capex going live screen: %d stocks with imminent commercialization", len(results))
    return results


# ── Delivery % Screen (NEW — uses nselib) ────────────────────


def screen_delivery_pct(
    tickers: list[str],
    sector_map: dict[str, str],
    threshold: float = 60.0,
) -> list[dict[str, Any]]:
    """Screen for stocks with delivery % > threshold (default 60%).

    High delivery % = genuine buying, not just speculative trading.
    Uses nselib to fetch NSE delivery data.
    """
    results: list[dict] = []

    # Batch fetch: nselib delivers data per-symbol
    for ticker in tickers:
        base_name = ticker.upper().replace(".NS", "").replace(".BO", "")
        cache_key = f"delivery_pct:{base_name}"
        cached = cache_get(cache_key)

        if cached is not None:
            if cached.get("passed"):
                results.append(cached)
            continue

        try:
            from nselib import capital_market
            df = capital_market.price_volume_and_deliverable_position_data(
                base_name, period="1M"
            )

            if df is None or df.empty:
                cache_set(cache_key, {"passed": False}, ttl=3600)
                continue

            # Column name: %DlyQttoTradedQty (delivery % to traded qty)
            del_col = None
            for col in df.columns:
                if "DlyQttoTradedQty" in col:
                    del_col = col
                    break

            if del_col is None:
                cache_set(cache_key, {"passed": False}, ttl=3600)
                continue

            # Average delivery % over last month
            delivery_values = df[del_col].apply(
                lambda x: float(str(x).replace(",", "")) if x and str(x).strip() else 0
            )
            avg_delivery = float(delivery_values.mean())

            entry = {
                "ticker": ticker,
                "sector": sector_map.get(ticker, "Other"),
                "avg_delivery_pct": round(avg_delivery, 2),
                "days_sampled": len(df),
                "passed": avg_delivery >= threshold,
            }
            cache_set(cache_key, entry, ttl=3600)

            if entry["passed"]:
                results.append(entry)

        except Exception as e:
            logger.debug("Delivery % screen failed for %s: %s", base_name, e)
            cache_set(cache_key, {"passed": False}, ttl=3600)

    logger.info("Delivery %% screen (>%.0f%%): %d stocks passed", threshold, len(results))
    return results
