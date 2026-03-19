"""Data Agent — fetches financials, BSE/NSE filings, and corporate data for Indian stocks.

Includes graceful error recovery: returns partial results on individual tool failures.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import get_llm
from app.prompts.templates import DATA_AGENT_PROMPT
from app.tools.financials import DATA_TOOLS
from app.tools.filings import FILING_TOOLS

logger = logging.getLogger(__name__)


def create_data_agent():
    """Create a data-fetching agent with financial and filing tools."""
    llm = get_llm()
    tools = DATA_TOOLS + FILING_TOOLS
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools


def data_node(state: dict) -> dict:
    """LangGraph node: fetch all financial data for an Indian stock.

    Invokes the LLM with tools in a loop until all data is gathered.
    Populates state['financials'], state['company_info'],
    state['shareholding_pattern'], and state['corporate_actions'].

    Error recovery: individual tool failures are logged as warnings and
    the agent continues collecting data from remaining tools.
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Data Agent: No ticker provided"]}

    errors: list[str] = []

    try:
        llm_with_tools, tools = create_data_agent()
        tool_map = {t.name: t for t in tools}

        messages = [
            {"role": "system", "content": DATA_AGENT_PROMPT.format(ticker=ticker)},
            {"role": "user", "content": f"Gather all available financial data for Indian stock: {ticker}"},
        ]

        financials: dict[str, Any] = {}
        company_info: dict[str, Any] = {}
        corporate_actions: list[dict[str, Any]] = []
        max_iterations = 12

        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error("Data Agent LLM call failed on iteration %d: %s", iteration, e)
                errors.append(f"Data Agent: LLM call failed on iteration {iteration}: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name not in tool_map:
                    logger.warning("Data Agent: Unknown tool %s", tool_name)
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
                    logger.warning("Data Agent: Tool %s failed: %s", tool_name, e)
                    errors.append(f"Data Agent: {tool_name} failed: {e}")
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
                        if tool_name == "get_company_info":
                            company_info = parsed
                        elif tool_name == "get_bse_announcements":
                            corporate_actions = parsed.get("announcements", [])
                        else:
                            financials[tool_name] = parsed
                    else:
                        errors.append(f"Data Agent: {tool_name} returned error: {parsed['error']}")
                except (json.JSONDecodeError, TypeError):
                    financials[tool_name] = result

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        return {
            "financials": financials,
            "company_info": company_info,
            "corporate_actions": corporate_actions,
            "errors": errors,
            "messages": [HumanMessage(content=f"[Data Agent] Completed data collection for {ticker}")],
        }

    except Exception as e:
        logger.error("Data Agent critical error for %s: %s", ticker, e)
        return {
            "errors": [f"Data Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Data Agent] Error: {str(e)}")],
        }
