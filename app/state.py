"""IndianAnalystState -- shared state schema for the multi-agent Indian equity research graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class IndianAnalystState(TypedDict, total=False):
    """Shared state that flows through every node in the Indian equity research graph.

    Fields are populated progressively by each agent:
        data_agent      -> financials, company_info, shareholding_pattern, corporate_actions
        analysis_agent  -> ratios, indian_metrics, dcf_valuation, peer_comparison
        sentiment_agent -> sentiment_scores, news_summaries, management_commentary
        report_agent    -> rag_context, final_report
    """

    # Input
    ticker: str       # e.g., "RELIANCE.NS"
    exchange: str     # "NSE" or "BSE"

    # Data Agent outputs
    financials: dict[str, Any]               # income_stmt, balance_sheet, cash_flow
    company_info: dict[str, Any]             # name, sector, market_cap, etc.
    shareholding_pattern: dict[str, Any]     # promoter %, FII %, DII %, public %
    corporate_actions: list[dict[str, Any]]  # dividends, splits, bonuses

    # Analysis Agent outputs
    ratios: dict[str, float]                 # standard + ROCE
    indian_metrics: dict[str, Any]           # promoter pledge %, FII trend, etc.
    dcf_valuation: dict[str, Any]            # DCF with Indian WACC
    peer_comparison: list[dict[str, Any]]    # Indian sector peers

    # Sentiment Agent outputs
    sentiment_scores: dict[str, Any]
    news_summaries: list[str]
    management_commentary: str               # earnings call transcript summary

    # Scoring Engine outputs
    investment_score: dict[str, Any]          # composite score, recommendation, dimension breakdown

    # RAG / Report Agent outputs
    rag_context: str
    final_report: str

    # Errors / metadata
    errors: Annotated[list[str], operator.add]

    # Message history (for agent reasoning traces)
    messages: Annotated[list[BaseMessage], add_messages]


def create_initial_state(ticker: str, exchange: str = "NSE") -> dict:
    """Factory to create a clean initial state for a given Indian stock ticker.

    Automatically appends .NS or .BO suffix if not present.
    """
    ticker = ticker.upper().strip()
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    if not ticker.endswith((".NS", ".BO")):
        ticker = ticker + suffix

    return {
        "ticker": ticker,
        "exchange": exchange.upper(),
        "financials": {},
        "company_info": {},
        "shareholding_pattern": {},
        "corporate_actions": [],
        "ratios": {},
        "indian_metrics": {},
        "dcf_valuation": {},
        "peer_comparison": [],
        "sentiment_scores": {},
        "news_summaries": [],
        "management_commentary": "",
        "investment_score": {},
        "rag_context": "",
        "final_report": "",
        "errors": [],
        "messages": [],
    }
