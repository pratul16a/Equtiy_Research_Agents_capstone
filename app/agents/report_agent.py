"""Report Agent — generates investment memos with bull/bear cases using RAG context."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.config import get_llm
from app.prompts.templates import REPORT_AGENT_PROMPT, REPORT_TEMPLATE


def _make_rag_tool():
    """Create a RAG retrieval tool if FAISS index exists."""

    @tool
    def retrieve_analyst_context(query: str) -> str:
        """Retrieve relevant context from historical analyst reports and investment frameworks.

        Args:
            query: Search query for relevant analyst insights (e.g., 'tech sector valuation frameworks')
        """
        try:
            from app.rag.retriever import get_retriever

            retriever = get_retriever()
            if retriever is None:
                return json.dumps({
                    "note": "No RAG index available. Proceeding without historical context.",
                    "documents": [],
                })

            docs = retriever.invoke(query)
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content[:500],
                    "source": doc.metadata.get("source", "unknown"),
                })
            return json.dumps({"documents": results}, indent=2)
        except Exception as e:
            return json.dumps({
                "note": f"RAG retrieval unavailable: {e}",
                "documents": [],
            })

    return retrieve_analyst_context


def report_node(state: dict) -> dict:
    """LangGraph node: generate the final investment report.

    Reads all state fields (financials, ratios, dcf, sentiment, etc.)
    and produces a comprehensive investment memo in state['final_report'].
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Report Agent: No ticker provided"]}

    try:
        llm = get_llm()
        rag_tool = _make_rag_tool()
        tools = [rag_tool]
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        # Build comprehensive context from all previous agents
        context_parts = []

        # Company info
        company_info = state.get("company_info", {})
        if company_info:
            context_parts.append(f"**Company Info:**\n{json.dumps(company_info, indent=2, default=str)}")

        # Financial ratios
        ratios = state.get("ratios", {})
        if ratios:
            context_parts.append(f"**Financial Ratios:**\n{json.dumps(ratios, indent=2, default=str)}")

        # DCF Valuation
        dcf = state.get("dcf_valuation", {})
        if dcf:
            context_parts.append(f"**DCF Valuation:**\n{json.dumps(dcf, indent=2, default=str)}")

        # Peer Comparison
        peers = state.get("peer_comparison", [])
        if peers:
            context_parts.append(f"**Peer Comparison:**\n{json.dumps(peers, indent=2, default=str)}")

        # Sentiment
        sentiment = state.get("sentiment_scores", {})
        if sentiment:
            context_parts.append(f"**Sentiment Analysis:**\n{json.dumps(sentiment, indent=2, default=str)}")

        # News summaries
        news = state.get("news_summaries", [])
        if news:
            context_parts.append(f"**Recent News:**\n" + "\n".join(f"- {n}" for n in news[:8]))

        # Financials data
        financials = state.get("financials", {})
        if financials:
            # Truncate to avoid token limits
            fin_str = json.dumps(financials, indent=2, default=str)[:4000]
            context_parts.append(f"**Raw Financial Data (truncated):**\n{fin_str}")

        full_context = "\n\n".join(context_parts)

        prompt = REPORT_AGENT_PROMPT.format(ticker=ticker)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Write a comprehensive investment research report for {ticker}.\n\n"
                f"Here is all the research data:\n\n{full_context}\n\n"
                f"First, try to retrieve_analyst_context for relevant historical analysis "
                f"on '{ticker} valuation investment thesis'. Then write the full report."
            )},
        ]

        rag_context = ""
        report_text = ""
        max_iterations = 6

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                report_text = response.content
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    try:
                        parsed = json.loads(result)
                        docs = parsed.get("documents", [])
                        if docs:
                            rag_context = "\n\n".join(d.get("content", "") for d in docs)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

        # Wrap in the report template
        from jinja2 import Template

        template = Template(REPORT_TEMPLATE)
        final_report = template.render(
            ticker=ticker,
            date=datetime.now().strftime("%B %d, %Y"),
            report_content=report_text,
        )

        return {
            "rag_context": rag_context,
            "final_report": final_report,
            "messages": [HumanMessage(content=f"[Report Agent] Investment memo generated for {ticker}")],
        }

    except Exception as e:
        return {
            "errors": [f"Report Agent error: {str(e)}"],
            "final_report": f"# Error generating report for {ticker}\n\n{str(e)}",
            "messages": [HumanMessage(content=f"[Report Agent] Error: {str(e)}")],
        }
