"""Regulatory Tailwind/Headwind Scanner — USP #4.

Scans news for active policy signals and maps them to stocks/sectors
using a curated policy mapping config.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote_plus

import feedparser

from app.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "policy_mapping.json")
_policy_config: dict | None = None


def _load_policy_config() -> dict:
    global _policy_config
    if _policy_config is not None:
        return _policy_config
    path = os.path.normpath(_POLICY_PATH)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _policy_config = json.load(f)
    else:
        logger.warning("Policy mapping not found at %s", path)
        _policy_config = {}
    return _policy_config


def _scan_news_for_keywords(keywords: list[str], max_articles: int = 10) -> list[dict]:
    """Scan Google News India for policy-related keywords."""
    query = " OR ".join(f'"{kw}"' for kw in keywords[:3])  # Limit query length
    cache_key = f"policy_news:{hash(query)}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+india&hl=en-IN&gl=IN&ceid=IN:en"
    results: list[dict] = []
    try:
        feed = feedparser.parse(url)
        pattern = re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)
        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "")
            if pattern.search(title):
                results.append({
                    "title": title,
                    "published": entry.get("published", ""),
                })
    except Exception as e:
        logger.debug("Policy news scan failed: %s", e)

    cache_set(cache_key, results, ttl=3600)
    return results


def scan_policy_signals() -> dict[str, dict[str, Any]]:
    """Scan news for active policy signals across all configured policies.

    Returns {policy_name: {active: bool, news_count: int, headlines: [...]}}.
    """
    config = _load_policy_config()
    active_policies: dict[str, dict[str, Any]] = {}

    for policy_name, policy_def in config.items():
        keywords = policy_def.get("keywords", [])
        if not keywords:
            continue

        news = _scan_news_for_keywords(keywords)
        active_policies[policy_name] = {
            "active": len(news) > 0,
            "news_count": len(news),
            "headlines": [n["title"] for n in news[:5]],
            "impact": policy_def.get("impact", "medium"),
        }

    return active_policies


def compute_regulatory_score(
    ticker: str,
    sector: str,
    active_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute regulatory tailwind/headwind score for a stock.

    Score: 100 = strong tailwinds, 50 = neutral, 0 = strong headwinds.
    """
    config = _load_policy_config()
    tailwind_count = 0
    headwind_count = 0
    tailwind_policies: list[str] = []
    headwind_policies: list[str] = []

    impact_weights = {"high": 3, "medium": 2, "low": 1}

    for policy_name, policy_def in config.items():
        # Skip inactive policies
        policy_signal = active_policies.get(policy_name, {})
        is_active = policy_signal.get("active", False)
        weight = impact_weights.get(policy_def.get("impact", "medium"), 2)

        # Check if this stock/sector benefits
        tailwind_sectors = policy_def.get("tailwind_sectors", [])
        tailwind_tickers = policy_def.get("tailwind_tickers", [])
        headwind_sectors = policy_def.get("headwind_sectors", [])
        headwind_tickers = policy_def.get("headwind_tickers", [])

        is_tailwind = ticker in tailwind_tickers or sector in tailwind_sectors
        is_headwind = ticker in headwind_tickers or sector in headwind_sectors

        if is_tailwind:
            # Active policy with recent news = stronger signal
            effective_weight = weight * (2 if is_active else 1)
            tailwind_count += effective_weight
            tailwind_policies.append(policy_name)

        if is_headwind:
            effective_weight = weight * (2 if is_active else 1)
            headwind_count += effective_weight
            headwind_policies.append(policy_name)

    # Score: neutral at 50, each tailwind point adds, each headwind subtracts
    max_possible = max(tailwind_count + headwind_count, 1)
    net_score = 50 + (tailwind_count - headwind_count) * 50 / max(max_possible, 10)
    score = max(0, min(100, round(net_score)))

    return {
        "score": score,
        "tailwind_count": len(tailwind_policies),
        "headwind_count": len(headwind_policies),
        "tailwind_policies": tailwind_policies,
        "headwind_policies": headwind_policies,
        "net_signal": "Tailwind" if score > 60 else "Headwind" if score < 40 else "Neutral",
    }


def screen_regulatory_tailwind(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with net regulatory tailwind."""
    active_policies = scan_policy_signals()
    results: list[dict] = []

    for ticker in bulk_info:
        try:
            sector = sector_map.get(ticker, "Other")
            reg = compute_regulatory_score(ticker, sector, active_policies)

            if reg["score"] > 55:  # At least slightly positive
                results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "regulatory_score": reg["score"],
                    "net_signal": reg["net_signal"],
                    "tailwind_policies": reg["tailwind_policies"],
                    "headwind_policies": reg["headwind_policies"],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Regulatory check failed for %s: %s", ticker, e)

    logger.info("Regulatory tailwind screen: %d stocks with net tailwind", len(results))
    return results
