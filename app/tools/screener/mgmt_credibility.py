"""Management Credibility Score — USP #3.

LLM-powered analysis of management's track record of delivering on promises.
Fetches BSE announcements with guidance keywords, compares with actual financials,
and uses LLM to rate credibility.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote_plus

import feedparser

from app.utils.cache import cache_get, cache_set
from app.tools.screener.batch_fundamentals import fetch_ticker_financials

logger = logging.getLogger(__name__)

_GUIDANCE_PATTERN = re.compile(
    r"guidance|outlook|target|expects|forecast|projects|aims to|plans to|vision\s+20\d{2}",
    re.IGNORECASE,
)

_CREDIBILITY_PROMPT = """You are analyzing management credibility for {company_name} ({ticker}).

Management statements / guidance found in recent announcements and news:
{guidance_statements}

Actual recent financial results:
{actual_results}

Score management credibility from 0-100:
- 80-100: Consistently delivers on promises, reliable forward guidance
- 60-79: Mostly delivers, minor misses
- 40-59: Mixed track record
- 20-39: Frequently misses targets
- 0-19: Chronic over-promising

Respond in this exact JSON format:
{{"score": <int>, "reasoning": "<brief reasoning>", "examples": ["<example1>", "<example2>"]}}
"""


def _fetch_guidance_from_news(base_name: str, max_articles: int = 10) -> list[dict]:
    """Fetch recent news mentioning management guidance."""
    cache_key = f"mgmt_guidance_news:{base_name}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"https://news.google.com/rss/search?q={quote_plus(base_name)}+guidance+outlook+india&hl=en-IN&gl=IN&ceid=IN:en"
    results: list[dict] = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "")
            if _GUIDANCE_PATTERN.search(title):
                results.append({
                    "source": "News",
                    "headline": title,
                    "date": entry.get("published", ""),
                })
    except Exception as e:
        logger.debug("Guidance news fetch failed for %s: %s", base_name, e)

    cache_set(cache_key, results, ttl=3600)
    return results


def _get_actual_financials_summary(ticker: str, info: dict) -> str:
    """Build a concise summary of actual financial results."""
    data = fetch_ticker_financials(ticker)
    fin = data.get("financials")
    parts: list[str] = []

    # From info
    for key, label in [
        ("totalRevenue", "Revenue"),
        ("netIncomeToCommon", "Net Income"),
        ("operatingMargins", "Operating Margin"),
        ("returnOnEquity", "ROE"),
        ("revenueGrowth", "Revenue Growth"),
        ("earningsGrowth", "Earnings Growth"),
    ]:
        val = info.get(key)
        if val is not None:
            if "Margin" in label or "ROE" in label or "Growth" in label:
                parts.append(f"{label}: {val*100:.1f}%")
            elif val > 1e9:
                parts.append(f"{label}: {val/1e9:.1f}B")
            elif val > 1e7:
                parts.append(f"{label}: {val/1e7:.1f}Cr")
            else:
                parts.append(f"{label}: {val}")

    return ", ".join(parts) if parts else "Limited financial data available"


def compute_credibility_score(
    ticker: str,
    info: dict,
    sector: str,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Compute management credibility score for a single stock.

    If LLM is provided, uses it for scoring. Otherwise falls back to heuristic.
    """
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")
    company_name = info.get("longName", info.get("shortName", base_name))

    # Fetch guidance statements
    guidance = _fetch_guidance_from_news(base_name)

    # Get actual results
    actual_summary = _get_actual_financials_summary(ticker, info)

    if llm and guidance:
        # LLM-powered scoring
        try:
            guidance_text = "\n".join(
                f"- [{g['date']}] {g['headline']}" for g in guidance[:10]
            )
            prompt = _CREDIBILITY_PROMPT.format(
                company_name=company_name,
                ticker=ticker,
                guidance_statements=guidance_text,
                actual_results=actual_summary,
            )
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            import json
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "score": parsed.get("score", 50),
                    "reasoning": parsed.get("reasoning", ""),
                    "examples": parsed.get("examples", []),
                    "guidance_count": len(guidance),
                    "method": "llm",
                }
        except Exception as e:
            logger.debug("LLM credibility scoring failed for %s: %s", ticker, e)

    # Heuristic fallback: Use financial consistency as proxy
    score = 50  # Neutral default
    reasoning_parts: list[str] = []

    # Revenue growth consistency
    rev_growth = info.get("revenueGrowth")
    earn_growth = info.get("earningsGrowth")
    if rev_growth is not None and rev_growth > 0:
        score += 10
        reasoning_parts.append("Positive revenue growth")
    if earn_growth is not None and earn_growth > 0:
        score += 10
        reasoning_parts.append("Positive earnings growth")

    # Margin stability
    op_margin = info.get("operatingMargins")
    if op_margin is not None and op_margin > 0.15:
        score += 10
        reasoning_parts.append("Healthy operating margins")

    # Guidance news presence
    if len(guidance) >= 3:
        score += 5
        reasoning_parts.append(f"{len(guidance)} guidance statements found")

    return {
        "score": min(100, max(0, score)),
        "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Limited data for assessment",
        "examples": [],
        "guidance_count": len(guidance),
        "method": "heuristic",
    }


def screen_management_credibility(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
    llm: Any | None = None,
    min_score: int = 60,
) -> list[dict[str, Any]]:
    """Screen for stocks with high management credibility score."""
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        try:
            sector = sector_map.get(ticker, "Other")
            cred = compute_credibility_score(ticker, info, sector, llm)

            if cred["score"] >= min_score:
                results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "credibility_score": cred["score"],
                    "reasoning": cred["reasoning"],
                    "method": cred["method"],
                    "guidance_count": cred["guidance_count"],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Credibility check failed for %s: %s", ticker, e)

    logger.info("Management credibility screen: %d stocks with score >= %d", len(results), min_score)
    return results
