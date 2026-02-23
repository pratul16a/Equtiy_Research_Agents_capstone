"""Analysis Agent — computes ratios, DCF models, peer comparisons."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import get_llm
from app.prompts.templates import ANALYSIS_AGENT_PROMPT
from app.tools.financials import ANALYSIS_TOOLS


def create_analysis_agent(ticker: str):
    """Create an analysis agent with financial computation tools."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ANALYSIS_TOOLS)
    return llm_with_tools


def analysis_node(state: dict) -> dict:
    """LangGraph node: compute financial ratios, DCF valuation, and peer comparison.

    Reads state['financials'] and populates state['ratios'],
    state['dcf_valuation'], and state['peer_comparison'].
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Analysis Agent: No ticker provided"]}

    try:
        llm_with_tools = create_analysis_agent(ticker)
        tool_map = {t.name: t for t in ANALYSIS_TOOLS}

        # Provide context from data agent
        financials_summary = json.dumps(state.get("financials", {}), indent=2, default=str)[:3000]

        prompt = ANALYSIS_AGENT_PROMPT.format(ticker=ticker)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Analyze {ticker} using all available tools. "
                f"Here is the financial data already collected:\n\n{financials_summary}\n\n"
                f"Run compute_financial_ratios, compute_dcf_valuation, and get_peer_comparison."
            )},
        ]

        ratios: dict[str, float] = {}
        dcf_valuation: dict[str, Any] = {}
        peer_comparison: list[dict] = []
        analysis_text = ""
        max_iterations = 8

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # Capture the final analysis text
                analysis_text = response.content
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    try:
                        parsed = json.loads(result)
                        if "error" not in parsed:
                            if tool_name == "compute_financial_ratios":
                                ratios = parsed
                            elif tool_name == "compute_dcf_valuation":
                                dcf_valuation = parsed
                            elif tool_name == "get_peer_comparison":
                                peer_comparison = parsed.get("peers", [])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

        return {
            "ratios": ratios,
            "dcf_valuation": dcf_valuation,
            "peer_comparison": peer_comparison,
            "messages": [HumanMessage(content=f"[Analysis Agent] Completed analysis for {ticker}")],
        }

    except Exception as e:
        return {
            "errors": [f"Analysis Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Analysis Agent] Error: {str(e)}")],
        }
