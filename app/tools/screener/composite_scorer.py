"""Composite Scorer with Regime-Adjusted Weights — Phase 7.

Replaces simple count-based scoring with dimensional weighted scoring.
Market breadth signal dynamically adjusts dimension weights.

Dimensions:
- Value (P/B, P/S, ROE)
- Quality (debt reduction, OPM improvement, CF turnaround)
- Catalyst (capex, orders, new business, insider buying, dividends)
- Technical (200 DMA, volume breakout)
- Governance (pledge, institutional, promoter anomaly)
- Geopolitical (trade/policy/commodity/event risk)
- Smart Money Lag (fundamental improvement vs institutional flow)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Criteria → Dimension mapping
CRITERIA_TO_DIMENSION = {
    # Value
    "roe": "value",
    "pb_vs_historical": "value",
    "ps_vs_historical": "value",
    # Quality
    "debt_reduction": "quality",
    "opm_improvement": "quality",
    "cf_turnaround": "quality",
    "capacity_utilization": "quality",
    # Catalyst
    "insider_buying": "catalyst",
    "capex": "catalyst",
    "new_business": "catalyst",
    "order_booking": "catalyst",
    "dividend_initiation": "catalyst",
    # Technical
    "above_200dma": "technical",
    "volume_breakout": "technical",
    # Governance
    "low_pledge": "governance",
    "increasing_institutional": "governance",
    "promoter_anomaly": "governance",
    # USP dimensions
    "geopolitical": "geopolitical",
    "smart_money_lag": "smart_money",
    "regulatory_tailwind": "regulatory",
    "mgmt_credibility": "credibility",
}

# Regime-adjusted weights: {regime: {dimension: weight}}
REGIME_WEIGHTS = {
    "Bearish": {
        "value": 0.30,
        "quality": 0.25,
        "catalyst": 0.12,
        "technical": 0.08,
        "governance": 0.10,
        "geopolitical": 0.05,
        "smart_money": 0.05,
        "regulatory": 0.03,
        "credibility": 0.02,
    },
    "Neutral": {
        "value": 0.22,
        "quality": 0.20,
        "catalyst": 0.18,
        "technical": 0.12,
        "governance": 0.10,
        "geopolitical": 0.06,
        "smart_money": 0.05,
        "regulatory": 0.04,
        "credibility": 0.03,
    },
    "Bullish": {
        "value": 0.15,
        "quality": 0.15,
        "catalyst": 0.25,
        "technical": 0.15,
        "governance": 0.10,
        "geopolitical": 0.06,
        "smart_money": 0.06,
        "regulatory": 0.05,
        "credibility": 0.03,
    },
}

# Category diversity bonus: stocks passing criteria from multiple categories get a multiplier
DIVERSITY_BONUS = {
    3: 1.15,  # 3 categories → 15% bonus
    4: 1.25,  # 4 categories → 25% bonus
    5: 1.35,  # 5+ categories → 35% bonus
}


def _get_dimension_score(
    criteria_passed: list[str],
    criteria_details: dict[str, dict],
    dimension: str,
) -> float:
    """Compute a 0-100 score for a dimension based on criteria that passed."""
    dim_criteria = [c for c in criteria_passed if CRITERIA_TO_DIMENSION.get(c) == dimension]
    if not dim_criteria:
        return 0.0

    # Count of criteria passed in this dimension
    total_possible = sum(1 for c, d in CRITERIA_TO_DIMENSION.items() if d == dimension)
    if total_possible == 0:
        return 0.0

    # Base score from pass ratio
    pass_ratio = len(dim_criteria) / total_possible
    base_score = pass_ratio * 70  # Max 70 from pass ratio

    # Bonus from specific detail values
    detail_bonus = 0.0
    for criterion in dim_criteria:
        detail = criteria_details.get(criterion, {})
        if criterion == "pb_vs_historical":
            discount = detail.get("discount_pct", 0)
            detail_bonus += min(10, discount / 5)
        elif criterion == "ps_vs_historical":
            discount = detail.get("discount_pct", 0)
            detail_bonus += min(10, discount / 5)
        elif criterion == "debt_reduction":
            reduction = abs(detail.get("debt_reduction_pct", 0))
            detail_bonus += min(10, reduction / 5)
        elif criterion == "opm_improvement":
            expansion = detail.get("opm_expansion", 0)
            detail_bonus += min(10, expansion / 2)
        elif criterion == "above_200dma":
            pct_above = detail.get("pct_above_200dma", 0)
            detail_bonus += min(10, pct_above / 5)
        elif criterion == "volume_breakout":
            ratio = detail.get("volume_ratio", 2)
            detail_bonus += min(10, (ratio - 2) * 5)
        else:
            detail_bonus += 5  # Generic bonus for passing

    score = base_score + min(30, detail_bonus)
    return min(100.0, score)


def compute_screener_composite(
    stock: dict[str, Any],
    breadth_signal: str = "Neutral",
) -> dict[str, Any]:
    """Compute composite score for a screener stock result.

    Args:
        stock: Dict with keys: ticker, criteria_passed, criteria_details, score.
        breadth_signal: "Bullish", "Bearish", or "Neutral" from market breadth.

    Returns dict with: composite_score, recommendation, dimension_breakdown, regime_used.
    """
    criteria_passed = stock.get("criteria_passed", [])
    criteria_details = stock.get("criteria_details", {})

    if not criteria_passed:
        return {
            "composite_score": 0,
            "recommendation": "INSUFFICIENT DATA",
            "dimension_breakdown": {},
            "regime_used": breadth_signal,
            "diversity_bonus": 1.0,
        }

    # Get regime weights
    weights = REGIME_WEIGHTS.get(breadth_signal, REGIME_WEIGHTS["Neutral"])

    # Compute per-dimension scores
    dimension_scores: dict[str, float] = {}
    all_dimensions = set(weights.keys())
    for dim in all_dimensions:
        dimension_scores[dim] = _get_dimension_score(criteria_passed, criteria_details, dim)

    # Weighted composite
    raw_score = sum(
        dimension_scores.get(dim, 0) * weight
        for dim, weight in weights.items()
    )

    # Category diversity bonus
    active_dimensions = set()
    for c in criteria_passed:
        dim = CRITERIA_TO_DIMENSION.get(c)
        if dim:
            active_dimensions.add(dim)

    diversity_count = len(active_dimensions)
    bonus = 1.0
    for threshold, multiplier in sorted(DIVERSITY_BONUS.items()):
        if diversity_count >= threshold:
            bonus = multiplier

    composite = min(100, round(raw_score * bonus))

    # Recommendation
    if composite >= 75:
        recommendation = "STRONG BUY"
    elif composite >= 60:
        recommendation = "BUY"
    elif composite >= 40:
        recommendation = "HOLD"
    elif composite >= 25:
        recommendation = "SELL"
    else:
        recommendation = "AVOID"

    return {
        "composite_score": composite,
        "recommendation": recommendation,
        "dimension_breakdown": {dim: round(score, 1) for dim, score in dimension_scores.items() if score > 0},
        "regime_used": breadth_signal,
        "diversity_bonus": bonus,
        "categories_hit": diversity_count,
    }


def score_all_stocks(
    stocks: list[dict[str, Any]],
    breadth_signal: str = "Neutral",
) -> list[dict[str, Any]]:
    """Score and re-rank all stocks from screener results using composite scoring.

    Modifies stocks in-place by adding composite_score fields, then re-sorts.
    """
    for stock in stocks:
        composite = compute_screener_composite(stock, breadth_signal)
        stock["composite_score"] = composite["composite_score"]
        stock["recommendation"] = composite["recommendation"]
        stock["dimension_breakdown"] = composite["dimension_breakdown"]
        stock["regime_used"] = composite["regime_used"]
        stock["diversity_bonus"] = composite["diversity_bonus"]
        stock["categories_hit"] = composite["categories_hit"]

    # Re-sort by composite score (primary) then old count score (secondary)
    stocks.sort(key=lambda x: (x.get("composite_score", 0), x.get("score", 0)), reverse=True)
    return stocks
