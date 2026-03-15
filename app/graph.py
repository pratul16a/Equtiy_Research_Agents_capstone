"""LangGraph workflow — wires Data -> Analysis -> Sentiment -> Report agents for Indian equity research."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.state import IndianAnalystState
from app.agents.data_agent import data_node
from app.agents.analysis_agent import analysis_node
from app.agents.sentiment_agent import sentiment_node
from app.agents.report_agent import report_node


def build_graph() -> StateGraph:
    """Construct and compile the multi-agent Indian equity research graph.

    Pipeline:
        data_agent -> analysis_agent -> sentiment_agent -> report_agent -> END
    """
    workflow = StateGraph(IndianAnalystState)

    # Add nodes
    workflow.add_node("data_agent", data_node)
    workflow.add_node("analysis_agent", analysis_node)
    workflow.add_node("sentiment_agent", sentiment_node)
    workflow.add_node("report_agent", report_node)

    # Define edges (sequential pipeline)
    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "analysis_agent")
    workflow.add_edge("analysis_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "report_agent")
    workflow.add_edge("report_agent", END)

    return workflow.compile()


# Pre-built compiled graph
research_graph = build_graph()


def run_research(ticker: str, exchange: str = "NSE") -> dict:
    """Run the full Indian equity research pipeline for a ticker and return the final state."""
    from app.state import create_initial_state

    initial_state = create_initial_state(ticker, exchange)
    final_state = research_graph.invoke(initial_state)
    return final_state
