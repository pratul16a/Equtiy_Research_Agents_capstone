"""IndianAnalystState -- shared state schemas for the multi-agent Indian equity research system.

Two state types:
    ScreenerState   — for the screening pipeline (regime → categories → USP → debate)
    DeepDiveState   — for the deep dive pipeline (data → analysis → sentiment → report)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ── Screener Pipeline State ──────────────────────────────────


class ScreenerState(TypedDict, total=False):
    """State for the 2-category screening + USP + debate pipeline.

    Flow:
        regime_agent → [momentum, value_bottom] (fan-out) → merge → validation → debate → END
    """

    # ── Shared data (populated during universe loading) ──
    price_df: Any                                        # pandas DataFrame of batch prices
    bulk_info: dict[str, dict]                           # {ticker: yfinance .info dict}
    t2s: dict[str, str]                                  # {ticker: sector_name}
    sector_map: dict[str, list[str]]                     # {sector: [tickers]}
    filtered_tickers: list[str]                          # tickers surviving tech pre-filter

    # ── Phase 1: Market Regime ──
    regime: dict[str, Any]                               # {regime, reasoning, weights, vix, nifty_data}

    # ── Phase 2: Category Screening (fan-out/fan-in) ──
    category_results: Annotated[list[dict], operator.add] # [{category, stocks, count, skipped, reason}]
    recommended_tickers: list[str]                        # deduplicated tickers from merge
    active_categories: int                                # how many categories had ≥3 stocks

    # ── Phase 3a: USP Validation ──
    usp_scores: dict[str, dict]                          # raw USP: {ticker: {module: data}}
    usp_cards: dict[str, dict[str, Any]]                 # transformed: {ticker: {module: card_data}}
    contradictions: list[dict[str, str]]                  # [{ticker, category, dimension, score, alert_text}]
    usp_explanations: dict[str, str]                      # {ticker: LLM explanation}

    # ── Phase 3b: RAG (populated in parallel with USP) ──
    rag_ready: bool                                       # True once FAISS index is loaded/built

    # ── Phase 4: Debate ──
    debate_results: list[dict[str, Any]]                  # [{ticker, bull_arguments, bear_arguments, verdict}]

    # ── Metadata ──
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list[BaseMessage], add_messages]


# ── Deep Dive Pipeline State (existing, preserved) ──────────


class DeepDiveState(TypedDict, total=False):
    """State for the per-stock deep dive pipeline.

    Flow:
        data_agent → analysis_agent → sentiment_agent → report_agent → END
    """

    # Input
    ticker: str                                          # e.g., "RELIANCE.NS"
    exchange: str                                        # "NSE" or "BSE"

    # Data Agent outputs
    financials: dict[str, Any]
    company_info: dict[str, Any]
    shareholding_pattern: dict[str, Any]
    corporate_actions: list[dict[str, Any]]

    # Analysis Agent outputs
    ratios: dict[str, float]
    indian_metrics: dict[str, Any]
    dcf_valuation: dict[str, Any]
    peer_comparison: list[dict[str, Any]]

    # Sentiment Agent outputs
    sentiment_scores: dict[str, Any]
    news_summaries: list[str]
    management_commentary: str

    # Scoring Engine outputs
    investment_score: dict[str, Any]

    # RAG / Report Agent outputs
    rag_context: str
    final_report: str

    # Errors / metadata
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list[BaseMessage], add_messages]


# Keep backward-compatible alias
IndianAnalystState = DeepDiveState


def create_initial_state(ticker: str, exchange: str = "NSE") -> dict:
    """Factory to create a clean initial state for a deep dive on a given Indian stock ticker.

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


def create_screener_state(
    categories: list[str] | None = None,
) -> dict:
    """Factory to create a clean initial state for the screening pipeline."""
    return {
        "price_df": None,
        "bulk_info": {},
        "t2s": {},
        "sector_map": {},
        "filtered_tickers": [],
        "regime": {},
        "category_results": [],
        "recommended_tickers": [],
        "active_categories": 0,
        "usp_scores": {},
        "usp_cards": {},
        "contradictions": [],
        "usp_explanations": {},
        "rag_ready": False,
        "debate_results": [],
        "errors": [],
        "messages": [],
    }
