"""Analysis Agent — computes ratios, DCF (Indian WACC), ROCE, peer comparisons, and promoter analysis.

Includes graceful error recovery: returns partial results on individual tool failures.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import get_llm
from app.prompts.templates import ANALYSIS_AGENT_PROMPT
from app.tools.financials import ANALYSIS_TOOLS
from app.tools.indian_metrics import INDIAN_METRICS_TOOLS
from app.tools.peer_data import PEER_TOOLS

logger = logging.getLogger(__name__)

ALL_ANALYSIS_TOOLS = ANALYSIS_TOOLS + INDIAN_METRICS_TOOLS + PEER_TOOLS


def create_analysis_agent(ticker: str):
    """Create an analysis agent with financial computation and Indian metrics tools."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_ANALYSIS_TOOLS)

    
    return llm_with_tools


def analysis_node(state: dict) -> dict:
    """LangGraph node: compute financial ratios, DCF, ROCE, peer comparison, and promoter analysis.

    Reads state['financials'] and populates state['ratios'], state['indian_metrics'],
    state['dcf_valuation'], and state['peer_comparison'].

    Error recovery: individual tool failures are logged and the agent continues
    with remaining tools, returning whatever data was successfully collected.
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Analysis Agent: No ticker provided"]}

    errors: list[str] = []

    try:
        llm_with_tools = create_analysis_agent(ticker)
        tool_map = {t.name: t for t in ALL_ANALYSIS_TOOLS}

        financials_summary = json.dumps(state.get("financials", {}), indent=2, default=str)[:3000]

        prompt = ANALYSIS_AGENT_PROMPT.format(ticker=ticker)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Analyze Indian stock {ticker} using all available tools. "
                f"Here is the financial data already collected:\n\n{financials_summary}\n\n"
                f"Run compute_financial_ratios, compute_dcf_valuation, get_peer_comparison, "
                f"compute_roce, get_shareholding_pattern, get_promoter_pledge_info, and get_fii_dii_activity."
            )},
        ]

        ratios: dict[str, float] = {}
        indian_metrics: dict[str, Any] = {}
        dcf_valuation: dict[str, Any] = {}
        peer_comparison: list[dict] = []
        shareholding_pattern: dict[str, Any] = {}
        max_iterations = 10

        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error("Analysis Agent LLM call failed on iteration %d: %s", iteration, e)
                errors.append(f"Analysis Agent: LLM call failed on iteration {iteration}: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name not in tool_map:
                    logger.warning("Analysis Agent: Unknown tool %s", tool_name)
                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    logger.warning("Analysis Agent: Tool %s failed: %s", tool_name, e)
                    errors.append(f"Analysis Agent: {tool_name} failed: {e}")
                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": f"Tool failed: {e}"}),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

                try:
                    parsed = json.loads(result)
                    if "error" not in parsed:
                        if tool_name == "compute_financial_ratios":
                            ratios = parsed
                        elif tool_name == "compute_dcf_valuation":
                            dcf_valuation = parsed
                        elif tool_name == "get_peer_comparison":
                            peer_comparison = parsed.get("peers", [])
                        elif tool_name == "compute_roce":
                            indian_metrics["roce"] = parsed
                        elif tool_name == "get_shareholding_pattern":
                            shareholding_pattern = parsed
                            indian_metrics["shareholding"] = parsed
                        elif tool_name == "get_promoter_pledge_info":
                            indian_metrics["promoter_pledge"] = parsed
                        elif tool_name == "get_fii_dii_activity":
                            indian_metrics["fii_dii"] = parsed
                    else:
                        errors.append(f"Analysis Agent: {tool_name} returned error: {parsed['error']}")
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return {
            "ratios": ratios,
            "indian_metrics": indian_metrics,
            "dcf_valuation": dcf_valuation,
            "peer_comparison": peer_comparison,
            "shareholding_pattern": shareholding_pattern,
            "errors": errors,
            "messages": [HumanMessage(content=f"[Analysis Agent] Completed analysis for {ticker}")],
        }

    except Exception as e:
        logger.error("Analysis Agent critical error for %s: %s", ticker, e)
        return {
            "errors": [f"Analysis Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Analysis Agent] Error: {str(e)}")],
        }
