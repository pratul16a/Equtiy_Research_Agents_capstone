"""USP Validation Agent — runs 5 USP modules on survivors, detects contradictions.

Uses GPT-4o-mini for generating human-readable explanation text per factor.
Deterministic scores + LLM explanations = verifiable + understandable.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_llm
from app.tools.screener.category_screener import apply_usp_layer
from app.ui.usp_cards import transform_usp_data

logger = logging.getLogger(__name__)


def _detect_contradictions(
    category_results: list[dict],
    usp_cards: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """Detect contradictions between category placement and USP scores.

    Returns list of {ticker, category, dimension, score, alert_text}.
    """
    alerts: list[dict[str, str]] = []

    for cat_result in category_results:
        cat_name = cat_result.get("category", "")
        for stock in cat_result.get("stocks", []):
            ticker = stock.get("ticker", "")
            if ticker not in usp_cards:
                continue
            usp = usp_cards[ticker]

            # Momentum stock with bad promoter score
            if cat_name == "B":
                promoter = usp.get("promoter", {})
                if promoter.get("score", 100) < 35:
                    alerts.append({
                        "ticker": ticker,
                        "category": "B (Momentum)",
                        "dimension": "promoter",
                        "score": promoter.get("score", 0),
                        "alert_text": (
                            f"{ticker} passed Cat B (Momentum) but Promoter Score "
                            f"{promoter.get('score', 0)}/100 — governance concern. "
                            f"Debate agents will address this."
                        ),
                    })

            # Value stock with bad management credibility
            if cat_name == "C":
                mgmt = usp.get("mgmt_credibility", {})
                if mgmt.get("score", 100) < 30:
                    alerts.append({
                        "ticker": ticker,
                        "category": "C (Value Bottoms)",
                        "dimension": "mgmt_credibility",
                        "score": mgmt.get("score", 0),
                        "alert_text": (
                            f"{ticker} passed Cat C (Value Bottoms) but Management Credibility "
                            f"{mgmt.get('score', 0)}/100 — turnaround may not materialize."
                        ),
                    })

            # Any stock with very high geopolitical risk
            geo = usp.get("geopolitical", {})
            if geo.get("score", 100) < 25:
                alerts.append({
                    "ticker": ticker,
                    "category": cat_name,
                    "dimension": "geopolitical",
                    "score": geo.get("score", 0),
                    "alert_text": (
                        f"{ticker} has Geopolitical Risk Score {geo.get('score', 0)}/100 — "
                        f"very high external risk exposure."
                    ),
                })

    return alerts


def _generate_usp_explanations(
    usp_cards: dict[str, dict[str, Any]],
    top_n: int = 10,
) -> dict[str, str]:
    """Use GPT-4o-mini to generate human-readable explanation for top stocks.

    Returns {ticker: explanation_text}.
    """
    # Only explain top N stocks by composite score
    sorted_tickers = sorted(
        usp_cards.keys(),
        key=lambda t: usp_cards[t].get("_composite", 0),
        reverse=True,
    )[:top_n]

    if not sorted_tickers:
        return {}

    explanations: dict[str, str] = {}

    try:
        llm = get_llm(temperature=0.2, model="gpt-4o-mini")

        for ticker in sorted_tickers:
            card = usp_cards[ticker]
            composite = card.get("_composite", 0)

            # Build a compact summary of all dimensions
            dim_summaries = []
            for dim in ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]:
                dim_data = card.get(dim, {})
                if dim_data:
                    dim_summaries.append(
                        f"- {dim}: {dim_data.get('score', 'N/A')}/100 ({dim_data.get('level', '')})"
                    )

            prompt = f"""You are an Indian equity research analyst. Write a 2-sentence USP summary for {ticker}.

USP Scores (composite: {composite}/100):
{chr(10).join(dim_summaries)}

Rules:
- Be specific: cite the actual scores
- Highlight the strongest and weakest dimensions
- If any dimension is below 30, flag it as a risk
- Keep it under 80 words

Respond with ONLY the summary text, no headers or formatting."""

            response = llm.invoke(prompt)
            explanations[ticker] = response.content.strip()

    except Exception as e:
        logger.warning("USP explanation generation failed: %s", e)

    return explanations


def validation_node(state: dict) -> dict:
    """LangGraph node: run USP analysis on survivors and detect contradictions."""
    recommended_tickers = state.get("recommended_tickers", [])
    bulk_info = state.get("bulk_info", {})
    t2s = state.get("t2s", {})
    category_results = state.get("category_results", [])

    if not recommended_tickers:
        logger.warning("Validation Agent: No recommended tickers to analyze")
        return {
            "usp_scores": {},
            "usp_cards": {},
            "contradictions": [],
            "usp_explanations": {},
        }

    # Phase 1: Run USP modules (deterministic scoring)
    usp_raw = apply_usp_layer(
        recommended_tickers=recommended_tickers,
        bulk_info=bulk_info,
        t2s=t2s,
    )

    # Phase 2: Transform to card format
    usp_cards = transform_usp_data(usp_raw)

    # Phase 3: Detect contradictions
    contradictions = _detect_contradictions(category_results, usp_cards)
    if contradictions:
        logger.info(
            "Validation Agent: %d contradictions detected", len(contradictions)
        )

    # Phase 4: Generate LLM explanations for top stocks
    usp_explanations = _generate_usp_explanations(usp_cards, top_n=10)

    return {
        "usp_scores": usp_raw,
        "usp_cards": usp_cards,
        "contradictions": contradictions,
        "usp_explanations": usp_explanations,
    }
