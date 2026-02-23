"""Data Agent — fetches financials, earnings, SEC filings via tool-calling."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import get_llm
from app.prompts.templates import DATA_AGENT_PROMPT
from app.tools.financials import DATA_TOOLS
from app.tools.sec_filings import SEC_TOOLS


def create_data_agent():
    """Create a data-fetching agent with financial and SEC tools."""
    llm = get_llm()
    tools = DATA_TOOLS + SEC_TOOLS
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools


def data_node(state: dict) -> dict:
    """LangGraph node: fetch all financial data for the given ticker.

    Invokes the LLM with tools in a loop until all data is gathered.
    Populates state['financials'] and state['company_info'].
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Data Agent: No ticker provided"]}

    try:
        llm_with_tools, tools = create_data_agent()
        tool_map = {t.name: t for t in tools}

        messages = [
            {"role": "system", "content": DATA_AGENT_PROMPT},
            {"role": "user", "content": f"Gather all available financial data for ticker: {ticker}"},
        ]

        financials: dict[str, Any] = {}
        company_info: dict[str, Any] = {}
        max_iterations = 12  # safety limit

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, the agent is done
            if not response.tool_calls:
                break

            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    # Store results by tool name
                    try:
                        parsed = json.loads(result)
                        if "error" not in parsed:
                            if tool_name == "get_company_info":
                                company_info = parsed
                            else:
                                financials[tool_name] = parsed
                    except (json.JSONDecodeError, TypeError):
                        financials[tool_name] = result

                    # Add tool result as message
                    from langchain_core.messages import ToolMessage
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

        return {
            "financials": financials,
            "company_info": company_info,
            "messages": [HumanMessage(content=f"[Data Agent] Completed data collection for {ticker}")],
        }

    except Exception as e:
        return {
            "errors": [f"Data Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Data Agent] Error: {str(e)}")],
        }
