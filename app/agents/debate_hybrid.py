"""Hybrid Debate — RAG-driven arguments + single LLM judge call.

Bull/bear arguments are auto-generated from:
    - Screening criteria (passed/failed)
    - USP scores (strengths/weaknesses)
    - RAG financial data (Screener.in)

Only the Judge uses LLM (1 call per stock instead of 5).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_llm
from app.rag.retriever import search_by_symbol

logger = logging.getLogger(__name__)

DEBATE_MODEL = "openai/gpt-4o"
DEBATE_FALLBACK_MODEL = "google/gemini-2.0-flash-001"


def _get_rag_snippets(symbol: str, query: str, k: int = 5) -> list[str]:
    """Fetch RAG snippets for a symbol."""
    results = search_by_symbol(query=query, symbol=symbol, k=k)
    if not results:
        return []
    return [r["content"][:300] for r in results]


def _extract_numbers_from_rag(snippets: list[str]) -> list[str]:
    """Extract lines containing numbers from RAG snippets (data-rich lines)."""
    import re
    data_lines = []
    for snippet in snippets:
        for line in snippet.split("\n"):
            line = line.strip()
            if re.search(r"\d+\.?\d*%|\d{2,}|\bRs\b|\bCr\b|\bINR\b", line) and len(line) > 15:
                data_lines.append(line)
    return data_lines[:15]


def build_bull_case(
    ticker: str,
    criteria_passed: list[str],
    usp_data: dict,
    rag_snippets: list[str],
) -> str:
    """Build structured bull case from data (no LLM)."""
    lines = [f"## Bull Case for {ticker}\n"]

    # Screening strength
    lines.append(f"**Screening Score**: Passed {len(criteria_passed)} criteria")
    criteria_labels = {
        "sector_rs_positive": "Sector showing relative strength (1M & 3M positive)",
        "sector_top4": "Sector ranks in top 4 by 3M relative strength",
        "above_200dma": "Price trading above 200-day moving average — long-term uptrend intact",
        "golden_alignment": "Golden alignment: Price > 50DMA > 200DMA — all timeframes bullish",
        "rsi_55_75": "RSI in sweet spot (55-75) — momentum without being overbought",
        "near_52w_high": "Within 10% of 52-week high — breakout candidate",
        "macd_bullish": "MACD above signal line — momentum accelerating",
        "volume_breakout": "Volume breakout (2x average) — institutional accumulation",
        "obv_rising": "On-Balance Volume rising — volume confirms price trend",
        "promoter_holding": "Promoter holding > 50% — aligned incentives",
        "institutional_interest": "Institutional holding > 30% — smart money validated",
        "delivery_pct": "Delivery % > 60% — genuine buying, not speculative",
        "opm_improvement": "Operating margins improving — business efficiency gaining",
        "roe_above_12": "ROE > 12% — generating strong returns on equity",
        "order_booking": "Recent order wins announced — revenue visibility improving",
    }
    for c in criteria_passed:
        label = criteria_labels.get(c, c.replace("_", " ").title())
        lines.append(f"  - {label}")

    # USP strengths
    strong_dims = []
    for dim in ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]:
        dim_data = usp_data.get(dim, {})
        score = dim_data.get("score", 0)
        if score and score >= 60:
            strong_dims.append(f"{dim.replace('_', ' ').title()}: {score}/100")
    if strong_dims:
        lines.append(f"\n**USP Strengths**: {', '.join(strong_dims)}")

    # RAG evidence
    data_lines = _extract_numbers_from_rag(rag_snippets)
    if data_lines:
        lines.append("\n**Financial Evidence** (from Screener.in):")
        for dl in data_lines[:8]:
            lines.append(f"  - {dl}")

    return "\n".join(lines)


def build_bear_case(
    ticker: str,
    all_criteria: list[str],
    criteria_passed: list[str],
    usp_data: dict,
    rag_risk_snippets: list[str],
) -> str:
    """Build structured bear case from data (no LLM)."""
    lines = [f"## Bear Case for {ticker}\n"]

    # Missing criteria
    possible_criteria = [
        "sector_rs_positive", "sector_top4", "above_200dma", "golden_alignment",
        "rsi_55_75", "near_52w_high", "macd_bullish", "volume_breakout",
        "obv_rising", "promoter_holding", "institutional_interest", "delivery_pct",
        "opm_improvement", "roe_above_12", "order_booking",
    ]
    missing = [c for c in possible_criteria if c not in criteria_passed]
    criteria_concerns = {
        "sector_rs_positive": "Sector lacks relative strength — headwind for stock",
        "sector_top4": "Sector not in top 4 — rotation risk",
        "above_200dma": "Below 200 DMA — long-term trend is down",
        "golden_alignment": "No golden alignment — mixed timeframe signals",
        "rsi_55_75": "RSI outside sweet spot — either overbought or losing momentum",
        "near_52w_high": "Far from 52-week high — significant recovery needed",
        "macd_bullish": "MACD bearish — momentum decelerating",
        "volume_breakout": "No volume breakout — lack of institutional conviction",
        "obv_rising": "OBV declining — volume not confirming price moves",
        "promoter_holding": "Promoter holding < 50% — alignment concern",
        "institutional_interest": "Low institutional interest — smart money absent",
        "delivery_pct": "Low delivery % — speculative trading dominated",
        "opm_improvement": "Operating margins not improving — cost pressures",
        "roe_above_12": "ROE < 12% — subpar capital efficiency",
        "order_booking": "No recent order wins — revenue visibility unclear",
    }

    if missing:
        lines.append(f"**Missing Criteria** ({len(missing)} of {len(possible_criteria)} not met):")
        for c in missing:
            label = criteria_concerns.get(c, c.replace("_", " ").title())
            lines.append(f"  - {label}")

    # USP weaknesses
    weak_dims = []
    for dim in ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]:
        dim_data = usp_data.get(dim, {})
        score = dim_data.get("score", 0)
        if score and score < 40:
            weak_dims.append(f"{dim.replace('_', ' ').title()}: {score}/100")
    if weak_dims:
        lines.append(f"\n**USP Weaknesses**: {', '.join(weak_dims)}")

    # RAG risk evidence
    data_lines = _extract_numbers_from_rag(rag_risk_snippets)
    if data_lines:
        lines.append("\n**Risk Evidence** (from Screener.in):")
        for dl in data_lines[:8]:
            lines.append(f"  - {dl}")

    if not missing and not weak_dims and not data_lines:
        lines.append("Limited bear case — stock passes most criteria with strong USP scores.")

    return "\n".join(lines)


def run_hybrid_debate(
    ticker: str,
    usp_cards: dict[str, dict],
    category_results: list[dict],
) -> dict[str, Any]:
    """Run hybrid debate: RAG-driven arguments + single LLM judge.

    Returns same structure as run_debate() for compatibility.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Get stock data from screening results
    criteria_passed = []
    score = 0
    for cat in category_results:
        for stock in cat.get("stocks", []):
            if stock.get("ticker") == ticker:
                criteria_passed = stock.get("criteria_passed", [])
                score = stock.get("score", 0)
                break

    usp_data = usp_cards.get(ticker, {})

    # Fetch RAG data
    rag_bull = _get_rag_snippets(symbol, f"{symbol} revenue growth margins profit competitive")
    rag_bear = _get_rag_snippets(symbol, f"{symbol} risks debt valuation concerns competition")

    # Build structured arguments (NO LLM)
    bull_text = build_bull_case(ticker, criteria_passed, usp_data, rag_bull)
    bear_text = build_bear_case(ticker, [], criteria_passed, usp_data, rag_bear)

    # Count data points for quality metrics
    import re
    bull_data_points = len(re.findall(r"\d+\.?\d*%|\d{2,}", bull_text))
    bear_data_points = len(re.findall(r"\d+\.?\d*%|\d{2,}", bear_text))

    # Single LLM call: Judge only
    composite = usp_data.get("_composite", "N/A")

    prompt = f"""You are a senior portfolio manager evaluating {ticker}.

## Screening Data
- Criteria passed: {len(criteria_passed)} of 17
- Criteria: {', '.join(criteria_passed)}
- USP Composite: {composite}/100

## Bull Case (data-driven):
{bull_text}

## Bear Case (data-driven):
{bear_text}

## Your Task:
Weigh the evidence and assign a conviction score. Consider:
1. How many criteria passed vs failed
2. USP score strength
3. Quality of financial evidence from Screener.in
4. Whether bull or bear case has stronger data support

Respond with ONLY a JSON object (no markdown fences):
{{
    "conviction_score": <1-10>,
    "recommendation": "BUY|ACCUMULATE|HOLD|SELL",
    "reasoning": "<2-3 sentences explaining your verdict>",
    "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "bull_strength": <1-10>,
    "bear_strength": <1-10>
}}"""

    def _invoke_hybrid_judge(model_name: str) -> dict:
        llm = get_llm(temperature=0.2, model=model_name)
        response = llm.invoke(prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(content)
        required = ["conviction_score", "recommendation", "reasoning"]
        for field in required:
            if field not in result:
                result[field] = "Error: missing field"
        return result

    try:
        verdict = _invoke_hybrid_judge(DEBATE_MODEL)
    except Exception as e:
        logger.warning("Hybrid Judge primary failed for %s: %s — trying fallback", ticker, e)
        try:
            verdict = _invoke_hybrid_judge(DEBATE_FALLBACK_MODEL)
        except Exception as e2:
            logger.error("Hybrid Judge fallback also failed for %s: %s", ticker, e2)
            verdict = {
                "conviction_score": 5,
                "recommendation": "HOLD",
                "reasoning": f"Judge error: {e}",
                "key_factors": [],
                "bull_strength": 5,
                "bear_strength": 5,
            }

    return {
        "ticker": ticker,
        "bull_arguments": [bull_text],
        "bear_arguments": [bear_text],
        "verdict": verdict,
        "rounds": 1,
        "method": "hybrid",
        "quality_metrics": {
            "bull_data_points": bull_data_points,
            "bear_data_points": bear_data_points,
            "criteria_count": len(criteria_passed),
            "usp_composite": composite,
        },
    }
