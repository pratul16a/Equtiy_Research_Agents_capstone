"""Category Screening Agents — 3 agents that run screening categories in parallel.

Each agent wraps the corresponding category_screener function and adds
self-assessment logic: skip if <3 stocks pass.

Designed for LangGraph fan-out/fan-in:
    regime_node -> [cat_a_node, cat_b_node, cat_c_node] -> merge_categories_node
"""

from __future__ import annotations

import logging
from typing import Any

from app.tools.screener.category_screener import (
    screen_category_a,
    screen_category_b,
    screen_category_c,
)

logger = logging.getLogger(__name__)

MIN_STOCKS_PER_CATEGORY = 3  # Self-assessment threshold


def _run_category(
    category_fn,
    category_name: str,
    price_df,
    bulk_info: dict,
    t2s: dict,
    regime_weights: dict,
    progress_cb=None,
) -> dict[str, Any]:
    """Run a single category screener with self-assessment.

    Returns:
        {category, stocks, count, skipped, reason}
    """
    try:
        stocks = category_fn(
            price_df=price_df,
            bulk_info=bulk_info,
            t2s=t2s,
            progress_cb=progress_cb,
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


def cat_a_node(state: dict) -> dict:
    """LangGraph node: Category A — Strengthening Industries."""
    regime = state.get("regime", {})
    weights = regime.get("weights", {"A": 1.0, "B": 1.0, "C": 1.0})

    result = _run_category(
        category_fn=screen_category_a,
        category_name="A",
        price_df=state.get("price_df"),
        bulk_info=state.get("bulk_info", {}),
        t2s=state.get("t2s", {}),
        regime_weights=weights,
    )

    return {"category_results": [result]}


def cat_b_node(state: dict) -> dict:
    """LangGraph node: Category B — Momentum."""
    regime = state.get("regime", {})
    weights = regime.get("weights", {"A": 1.0, "B": 1.0, "C": 1.0})

    result = _run_category(
        category_fn=screen_category_b,
        category_name="B",
        price_df=state.get("price_df"),
        bulk_info=state.get("bulk_info", {}),
        t2s=state.get("t2s", {}),
        regime_weights=weights,
    )

    return {"category_results": [result]}


def cat_c_node(state: dict) -> dict:
    """LangGraph node: Category C — Value Bottoms."""
    regime = state.get("regime", {})
    weights = regime.get("weights", {"A": 1.0, "B": 1.0, "C": 1.0})

    result = _run_category(
        category_fn=screen_category_c,
        category_name="C",
        price_df=state.get("price_df"),
        bulk_info=state.get("bulk_info", {}),
        t2s=state.get("t2s", {}),
        regime_weights=weights,
    )

    return {"category_results": [result]}


def merge_categories_node(state: dict) -> dict:
    """LangGraph node: merge fan-in results from 3 category agents.

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
