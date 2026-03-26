"""LLM-powered USP commentary generation — per-stock and portfolio-level.

Enhanced with:
- Rich structured output (thesis, catalysts, risks, what-to-watch)
- Google News RSS for real-time context grounding (free, no API key)
- Dimension-level notes for card rendering
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from app.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)


# ── Google News RSS (free, no API key) ────────────────────────


def _fetch_stock_news(ticker: str, max_headlines: int = 8) -> list[str]:
    """Fetch recent news headlines for a stock via Google News RSS.

    Free, no API key needed. Returns list of headline strings.
    Cached for 2 hours to avoid hitting Google too often.
    """
    cache_key = f"gnews_rss:{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # Clean ticker for search: TORNTPOWER.NS -> TORNTPOWER NSE
    clean = ticker.replace(".NS", "").replace(".BO", "")
    query = quote_plus(f"{clean} NSE stock")

    headlines: list[str] = []
    try:
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=5) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        for item in root.iter("item"):
            title = item.findtext("title", "")
            if title:
                # Strip source suffix (e.g., " - Economic Times")
                parts = title.rsplit(" - ", 1)
                headline = parts[0].strip()
                if headline:
                    headlines.append(headline)
            if len(headlines) >= max_headlines:
                break
    except Exception as e:
        logger.debug("Google News RSS fetch failed for %s: %s", ticker, e)

    cache_set(cache_key, headlines, ttl=7200)
    return headlines


def _fetch_news_batch(tickers: list[str], max_workers: int = 5) -> dict[str, list[str]]:
    """Fetch news for multiple tickers in parallel."""
    result: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_stock_news, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result[ticker] = future.result(timeout=10)
            except Exception:
                result[ticker] = []
    return result


# ── LLM Call ──────────────────────────────────────────────────


def _call_llm(system: str, user: str) -> str:
    """Make a single LLM call and return the text response."""
    from app.config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.3)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content


def _parse_commentary_json(text: str) -> dict[str, Any]:
    """Extract JSON dict from LLM response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse USP commentary JSON, returning raw text")
        return {"_raw": cleaned}


# ── Per-Stock Commentary ──────────────────────────────────────


def generate_stock_commentary(
    ticker: str,
    usp_data: dict[str, Any],
    sector: str = "Unknown",
    recent_news: list[str] | None = None,
) -> dict[str, Any]:
    """Generate rich LLM commentary for one stock's USP dimensions.

    Returns: {
        investment_thesis: str,
        key_catalysts: [str, ...],
        key_risks: [str, ...],
        what_to_watch: str,
        dimension_notes: {geopolitical: str, smart_money: str, ...}
    }
    """
    from app.prompts.usp_commentary import build_stock_commentary_prompt

    system, user = build_stock_commentary_prompt(ticker, usp_data, sector, recent_news)
    try:
        raw = _call_llm(system, user)
        parsed = _parse_commentary_json(raw)

        # Normalize: ensure dimension_notes exists for backward compat
        if "dimension_notes" not in parsed and "_raw" not in parsed:
            # Old format — 5 dimension keys at top level
            dim_keys = {"geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"}
            if dim_keys & set(parsed.keys()):
                parsed["dimension_notes"] = {k: parsed.pop(k) for k in dim_keys if k in parsed}

        return parsed
    except Exception as e:
        logger.warning("USP commentary failed for %s: %s", ticker, e)
        return {}


def generate_usp_commentary(
    per_stock_usp: dict[str, dict[str, Any]],
    bulk_info: dict[str, dict] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    max_workers: int = 5,
) -> dict[str, dict[str, Any]]:
    """Generate rich LLM commentary for all stocks in parallel.

    Steps:
    1. Fetch Google News RSS for all tickers (parallel, ~2s)
    2. Generate LLM commentary for each stock (parallel, ~3-5s each)

    Returns: {ticker: {investment_thesis, key_catalysts, key_risks, what_to_watch, dimension_notes}}
    """
    if not per_stock_usp:
        return {}

    bulk_info = bulk_info or {}
    tickers = [t for t in per_stock_usp.keys() if not t.startswith("_")]
    total = len(tickers)
    result: dict[str, dict[str, Any]] = {}

    if progress_cb:
        progress_cb(0.0, f"Fetching news for {total} stocks...")

    # Step 1: Fetch news for all tickers in parallel
    news_map = _fetch_news_batch(tickers, max_workers=max_workers)

    if progress_cb:
        progress_cb(0.1, f"Generating rich commentary for {total} stocks...")

    # Step 2: Generate LLM commentary in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker in tickers:
            sector = per_stock_usp[ticker].get("geopolitical", {}).get("sector", "Unknown")
            if sector == "Unknown" and ticker in bulk_info:
                sector = bulk_info[ticker].get("sector", "Unknown")
            news = news_map.get(ticker, [])
            futures[executor.submit(
                generate_stock_commentary, ticker, per_stock_usp[ticker], sector, news
            )] = ticker

        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                commentary = future.result(timeout=45)
                if commentary:
                    result[ticker] = commentary
            except Exception as e:
                logger.warning("Commentary generation timed out for %s: %s", ticker, e)
            done += 1
            if progress_cb:
                progress_cb(0.1 + 0.9 * done / total, f"USP commentary: {done}/{total} stocks done")

    return result


def generate_portfolio_insights(
    per_stock_usp: dict[str, dict[str, Any]],
    category: str = "Momentum",
) -> str:
    """Generate cross-stock portfolio-level insights.

    Returns: Markdown string with 4 portfolio insights.
    """
    clean = {k: v for k, v in per_stock_usp.items() if not k.startswith("_")}
    if len(clean) < 2:
        return ""

    from app.prompts.usp_commentary import build_portfolio_insights_prompt

    system, user = build_portfolio_insights_prompt(clean, category)
    try:
        return _call_llm(system, user)
    except Exception as e:
        logger.warning("Portfolio insights generation failed: %s", e)
        return ""
