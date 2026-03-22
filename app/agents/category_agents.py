"""Category Screening Agents — 2 agents that run screening categories in parallel.

Each agent wraps the corresponding category_screener function and adds
self-assessment logic: skip if <3 stocks pass.

Designed for LangGraph fan-out/fan-in:
    regime_node -> [momentum_node, value_bottom_node] -> merge_categories_node
"""

from __future__ import annotations

import logging
from typing import Any

from app.tools.screener.category_screener import (
    screen_momentum,
    screen_value_bottom,
)

logger = logging.getLogger(__name__)

MIN_STOCKS_PER_CATEGORY = 3  # Self-assessment threshold

# V4
def _run_category(
    category_fn,
    category_name: str,
    state: dict,
    regime_weights: dict,
) -> dict[str, Any]:
    """Run a single category screener with self-assessment.

    Returns:
        {category, stocks, count, skipped, reason}
    """
    try:
        stocks = category_fn(
            sector_map=state.get("sector_map", {}),
            price_df=state.get("price_df"),
            bulk_info=state.get("bulk_info", {}),
            t2s=state.get("t2s", {}),
            tickers=state.get("filtered_tickers", []),
        )

        # Apply regime weight to each stock's score
        weight = regime_weights.get(category_name, 1.0)
        for stock in stocks:
            if "score" in stock:
                stock["weighted_score"] = round(stock["score"] * weight, 2)
            else:
                stock["weighted_score"] = stock.get("criteria_met", 0) * weight

        # Self-assessment: skip if too few stocks
        if len(stocks) < MIN_STOCKS_PER_CATEGORY:
            logger.info(
                "Category %s self-skip: only %d stocks (min %d)",
                category_name, len(stocks), MIN_STOCKS_PER_CATEGORY,
            )
            return {
                "category": category_name,
                "stocks": stocks,
                "count": len(stocks),
                "skipped": True,
                "reason": f"Only {len(stocks)} stocks passed (minimum {MIN_STOCKS_PER_CATEGORY})",
            }

        # Sort by weighted score descending
        stocks.sort(key=lambda s: s.get("weighted_score", 0), reverse=True)

        return {
            "category": category_name,
            "stocks": stocks,
            "count": len(stocks),
            "skipped": False,
            "reason": None,
        }

    except Exception as e:
        logger.error("Category %s agent failed: %s", category_name, e)
        return {
            "category": category_name,
            "stocks": [],
            "count": 0,
            "skipped": True,
            "reason": f"Error: {e}",
        }


def momentum_node(state: dict) -> dict:
    """LangGraph node: Momentum — buy what's already working."""
    regime = state.get("regime", {})
    weights = regime.get("weights", {"Momentum": 1.0, "ValueBottom": 1.0})

    result = _run_category(
        category_fn=screen_momentum,
        category_name="Momentum",
        state=state,
        regime_weights=weights,
    )

    return {"category_results": [result]}


def value_bottom_node(state: dict) -> dict:
    """LangGraph node: Value Bottom — buy what nobody wants yet."""
    regime = state.get("regime", {})
    weights = regime.get("weights", {"Momentum": 1.0, "ValueBottom": 1.0})

    result = _run_category(
        category_fn=screen_value_bottom,
        category_name="ValueBottom",
        state=state,
        regime_weights=weights,
    )

    return {"category_results": [result]}


def merge_categories_node(state: dict) -> dict:
    """LangGraph node: merge fan-in results from 2 category agents.

    Collects all survivors, deduplicates, and prepares for USP layer.
    """
    category_results = state.get("category_results", [])

    # Deduplicate tickers across categories
    all_tickers: list[str] = []
    seen: set[str] = set()
    for cat_result in category_results:
        for stock in cat_result.get("stocks", []):
            ticker = stock.get("ticker", "")
            if ticker and ticker not in seen:
                seen.add(ticker)
                all_tickers.append(ticker)

    active_categories = [c for c in category_results if not c.get("skipped")]

    logger.info(
        "Merged %d unique stocks from %d active categories (of %d total)",
        len(all_tickers), len(active_categories), len(category_results),
    )

    return {
        "recommended_tickers": all_tickers,
        "active_categories": len(active_categories),
    }
