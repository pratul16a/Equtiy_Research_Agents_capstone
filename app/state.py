"""AnalystState — shared state schema for the multi-agent equity research graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AnalystState(TypedDict, total=False):
    """Shared state that flows through every node in the research graph.

    Fields are populated progressively by each agent:
        data_agent   → financials
        analysis_agent → ratios, dcf_valuation, peer_comparison
        sentiment_agent → sentiment_scores, news_summaries
        report_agent  → rag_context, final_report
    """

    # ── Input ────────────────────────────────────────────────
    ticker: str

    # ── Data Agent outputs ───────────────────────────────────
    financials: dict[str, Any]
    company_info: dict[str, Any]

    # ── Analysis Agent outputs ───────────────────────────────
    ratios: dict[str, float]
    dcf_valuation: dict[str, Any]
    peer_comparison: list[dict[str, Any]]

    # ── Sentiment Agent outputs ──────────────────────────────
    sentiment_scores: dict[str, Any]
    news_summaries: list[str]

    # ── RAG / Report Agent outputs ───────────────────────────
    rag_context: str
    final_report: str

    # ── Errors / metadata ────────────────────────────────────
    errors: Annotated[list[str], operator.add]

    # ── Message history (for agent reasoning traces) ─────────
    messages: Annotated[list[BaseMessage], add_messages]


def create_initial_state(ticker: str) -> dict:
    """Factory to create a clean initial state for a given ticker."""
    return {
        "ticker": ticker.upper().strip(),
        "financials": {},
        "company_info": {},
        "ratios": {},
        "dcf_valuation": {},
        "peer_comparison": [],
        "sentiment_scores": {},
        "news_summaries": [],
        "rag_context": "",
        "final_report": "",
        "errors": [],
        "messages": [],
    }
