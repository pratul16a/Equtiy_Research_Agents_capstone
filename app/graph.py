"""LangGraph workflows for Indian equity research.

Two graphs:
    1. screener_graph — Regime → Fan-out(2 categories) → Fan-in → USP+RAG → Debate → END
    2. deep_dive_graph — Data → Analysis → Sentiment → Report → END (existing pipeline)
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END

from app.state import ScreenerState, DeepDiveState

logger = logging.getLogger(__name__)


# ── Screener Graph ───────────────────────────────────────────


def _load_universe_node(state: dict) -> dict:
    """Load Nifty 500 universe, batch download prices, fetch bulk .info."""
    import pandas as pd
    import yfinance as yf

    from app.tools.market_breadth import get_nifty_constituents, fetch_bulk_price_data, _all_tickers, _ticker_to_sector
    from app.tools.screener.batch_fundamentals import fetch_bulk_info
    from app.tools.screener.technical_screens import screen_above_200dma, screen_rsi_range

    sector_map = get_nifty_constituents()

    # Build ticker list and ticker-to-sector mapping
    all_tickers = _all_tickers(sector_map)
    t2s = _ticker_to_sector(sector_map)

    logger.info("Universe: %d tickers across %d sectors", len(all_tickers), len(sector_map))

    # Batch download prices
    price_df = fetch_bulk_price_data(all_tickers, period="1y")

    # Technical pre-filter
    tech_survivors: set[str] = set()
    above_200 = screen_above_200dma(price_df, t2s)
    tech_survivors.update(s["ticker"] for s in above_200)
    rsi_mid = screen_rsi_range(price_df, t2s, lo=20, hi=80)
    tech_survivors.update(s["ticker"] for s in rsi_mid)

    if len(tech_survivors) < len(all_tickers) * 0.3:
        tech_survivors = set(all_tickers)

    filtered_tickers = list(tech_survivors)
    logger.info("Tech pre-filter: %d → %d tickers", len(all_tickers), len(filtered_tickers))

    # Fetch bulk fundamentals for survivors
    bulk_info = fetch_bulk_info(filtered_tickers)

    return {
        "price_df": price_df,
        "bulk_info": bulk_info,
        "t2s": t2s,
        "sector_map": sector_map,
        "filtered_tickers": filtered_tickers,
    }


def _rag_ingest_node(state: dict) -> dict:
    """Background RAG ingestion: index Screener.in data for recommended tickers."""
    recommended = state.get("recommended_tickers", [])
    if not recommended:
        return {"rag_ready": False}

    try:
        from app.rag.ingest import ingest_screener_data
        # Only ingest top 15 stocks to keep it fast
        symbols = recommended[:15]
        ingest_screener_data(symbols, use_local_embeddings=True)
        return {"rag_ready": True}
    except Exception as e:
        logger.warning("RAG ingestion failed (non-fatal): %s", e)
        return {"rag_ready": False}


def build_screener_graph():
    """Build the screening pipeline graph.

    Flow:
        load_universe → regime → [momentum, value_bottom] (fan-out)
                      → merge_categories (fan-in)
                      → [validation, rag_ingest] (parallel)
                      → debate → END
    """
    from app.agents.regime_agent import regime_node
    from app.agents.category_agents import (
        momentum_node,
        value_bottom_node,
        merge_categories_node,
    )
    from app.agents.validation_agent import validation_node
    from app.agents.debate_agents import debate_node

    workflow = StateGraph(ScreenerState)

    # Add nodes
    workflow.add_node("load_universe", _load_universe_node)
    workflow.add_node("regime", regime_node)
    workflow.add_node("momentum", momentum_node)
    workflow.add_node("value_bottom", value_bottom_node)
    workflow.add_node("merge_categories", merge_categories_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("rag_ingest", _rag_ingest_node)
    workflow.add_node("debate", debate_node)

    # Entry: load universe first
    workflow.set_entry_point("load_universe")
    workflow.add_edge("load_universe", "regime")

    # Regime → fan-out to 2 categories
    workflow.add_edge("regime", "momentum")
    workflow.add_edge("regime", "value_bottom")

    # Fan-in: both categories merge
    workflow.add_edge("momentum", "merge_categories")
    workflow.add_edge("value_bottom", "merge_categories")

    # After merge: USP validation + RAG ingestion in parallel
    workflow.add_edge("merge_categories", "validation")
    workflow.add_edge("merge_categories", "rag_ingest")

    # Both must complete before debate
    workflow.add_edge("validation", "debate")
    workflow.add_edge("rag_ingest", "debate")

    # Debate → END
    workflow.add_edge("debate", END)

    return workflow.compile()


def run_screener(categories: list[str] | None = None) -> dict:
    """Run the full 2-category screening + USP + debate pipeline."""
    from app.state import create_screener_state

    graph = build_screener_graph()
    initial_state = create_screener_state(categories)
    return graph.invoke(initial_state)


# ── Deep Dive Graph (existing, preserved) ────────────────────


def build_deep_dive_graph():
    """Build the per-stock deep dive pipeline.

    Flow:
        data_agent → analysis_agent → sentiment_agent → report_agent → END
    """
    from app.agents.data_agent import data_node
    from app.agents.analysis_agent import analysis_node
    from app.agents.sentiment_agent import sentiment_node
    from app.agents.report_agent import report_node

    workflow = StateGraph(DeepDiveState)

    workflow.add_node("data_agent", data_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("report_agent", report_node)

    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile()


# Backward-compatible aliases
def build_graph():
    """Legacy alias for build_deep_dive_graph."""
    return build_deep_dive_graph()


research_graph = build_deep_dive_graph()


def run_research(ticker: str, exchange: str = "NSE") -> dict:
    """Run the full Indian equity deep dive pipeline for a ticker."""
    from app.state import create_initial_state

    initial_state = create_initial_state(ticker, exchange)
    final_state = research_graph.invoke(initial_state)
    return final_state
