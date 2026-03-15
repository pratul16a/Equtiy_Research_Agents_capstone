"""Report Agent — generates investment memos with bull/bear cases using RAG context (Indian market).

Now includes a quantitative scoring engine that computes a structured BUY/HOLD/SELL
recommendation based on 5 dimensions (Valuation, Quality, Growth, Sentiment, Governance)
before the LLM generates the final report. The LLM must align with or explicitly justify
deviations from the quantitative recommendation.

Includes graceful error recovery: generates a basic report even if RAG or LLM calls fail.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.config import get_llm
from app.prompts.templates import REPORT_AGENT_PROMPT, REPORT_TEMPLATE
from app.scoring import compute_investment_score, format_score_for_prompt

logger = logging.getLogger(__name__)


def _make_rag_tool():
    """Create a RAG retrieval tool if FAISS index exists."""

    @tool
    def retrieve_analyst_context(query: str) -> str:
        """Retrieve relevant context from historical analyst reports and Indian investment frameworks.

        Args:
            query: Search query for relevant insights (e.g., 'Indian IT sector valuation ROCE analysis')
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
            logger.warning("RAG retrieval failed: %s", e)
            return json.dumps({
                "note": f"RAG retrieval unavailable: {e}",
                "documents": [],
            })

    return retrieve_analyst_context


def _generate_fallback_report(ticker: str, score_result: dict, state: dict) -> str:
    """Generate a basic report from available data when LLM fails."""
    parts = [
        f"# Investment Research Report: {ticker}",
        f"**Generated:** {datetime.now().strftime('%B %d, %Y')}",
        "",
        "## Quantitative Assessment",
        f"- **Composite Score:** {score_result['composite_score']}/100",
        f"- **Recommendation:** {score_result['recommendation']}",
        f"- **Confidence:** {score_result['confidence']}",
        "",
        "## Score Breakdown",
    ]

    for dim_name, dim_data in score_result.get("dimension_scores", {}).items():
        parts.append(f"- **{dim_name.title()}:** {dim_data['score']}/100 (weight: {dim_data['weight']}%)")

    company_info = state.get("company_info", {})
    if company_info:
        parts.extend([
            "",
            "## Company Overview",
            f"- **Name:** {company_info.get('name', 'N/A')}",
            f"- **Sector:** {company_info.get('sector', 'N/A')}",
            f"- **Market Cap:** {company_info.get('market_cap_crores', 'N/A')} Cr",
        ])

    parts.extend([
        "",
        "---",
        "*Note: This is a fallback report generated from quantitative data only.*",
        "*The LLM-powered narrative could not be generated. Please retry.*",
    ])

    return "\n".join(parts)


def report_node(state: dict) -> dict:
    """LangGraph node: generate the final Indian equity investment report.

    Steps:
        1. Run the quantitative scoring engine on all available state data
        2. Format the score into a structured context block
        3. Feed the score + all research data to the LLM
        4. LLM generates a report that must align with the quantitative recommendation

    Error recovery: if LLM fails, generates a fallback report from quantitative data.
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Report Agent: No ticker provided"]}

    errors: list[str] = []

    try:
        # ── Step 1: Compute Quantitative Investment Score ──────
        try:
            score_result = compute_investment_score(state)
        except Exception as e:
            logger.error("Scoring engine failed: %s", e)
            score_result = {
                "composite_score": 50,
                "recommendation": "HOLD",
                "confidence": "LOW",
                "confidence_pct": 0,
                "dimension_scores": {},
                "summary": f"Scoring engine error: {e}",
                "thresholds": {},
            }
            errors.append(f"Report Agent: Scoring engine failed: {e}")

        score_text = format_score_for_prompt(score_result)

        llm = get_llm()
        rag_tool = _make_rag_tool()
        tools = [rag_tool]
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        # ── Step 2: Build comprehensive context from all agents ─
        context_parts = []

        # Scoring framework goes FIRST so the LLM sees it prominently
        context_parts.append(score_text)

        company_info = state.get("company_info", {})
        if company_info:
            context_parts.append(f"**Company Info:**\n{json.dumps(company_info, indent=2, default=str)}")

        ratios = state.get("ratios", {})
        if ratios:
            context_parts.append(f"**Financial Ratios (including ROCE):**\n{json.dumps(ratios, indent=2, default=str)}")

        indian_metrics = state.get("indian_metrics", {})
        if indian_metrics:
            context_parts.append(f"**Indian-Specific Metrics (Promoter/FII/DII/ROCE):**\n{json.dumps(indian_metrics, indent=2, default=str)[:2000]}")

        dcf = state.get("dcf_valuation", {})
        if dcf:
            context_parts.append(f"**DCF Valuation (Indian WACC):**\n{json.dumps(dcf, indent=2, default=str)}")

        peers = state.get("peer_comparison", [])
        if peers:
            context_parts.append(f"**Indian Peer Comparison:**\n{json.dumps(peers, indent=2, default=str)}")

        shareholding = state.get("shareholding_pattern", {})
        if shareholding:
            context_parts.append(f"**Shareholding Pattern:**\n{json.dumps(shareholding, indent=2, default=str)}")

        sentiment = state.get("sentiment_scores", {})
        if sentiment:
            context_parts.append(f"**Sentiment Analysis:**\n{json.dumps(sentiment, indent=2, default=str)}")

        news = state.get("news_summaries", [])
        if news:
            context_parts.append("**Recent Indian News:**\n" + "\n".join(f"- {n}" for n in news[:8]))

        commentary = state.get("management_commentary", "")
        if commentary:
            context_parts.append(f"**Management Commentary / Sentiment Detail:**\n{commentary[:1500]}")

        corporate_actions = state.get("corporate_actions", [])
        if corporate_actions:
            context_parts.append(f"**Corporate Actions:**\n{json.dumps(corporate_actions[:5], indent=2, default=str)}")

        financials = state.get("financials", {})
        if financials:
            fin_str = json.dumps(financials, indent=2, default=str)[:4000]
            context_parts.append(f"**Raw Financial Data (truncated, in INR):**\n{fin_str}")

        full_context = "\n\n".join(context_parts)

        # ── Step 3: LLM generates report with scoring context ──
        prompt = REPORT_AGENT_PROMPT.format(
            ticker=ticker,
            recommendation=score_result["recommendation"],
            composite_score=score_result["composite_score"],
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Write a comprehensive Indian equity investment research report for {ticker}.\n\n"
                f"Here is all the research data INCLUDING the quantitative scoring framework:\n\n"
                f"{full_context}\n\n"
                f"The quantitative scoring engine has computed a {score_result['recommendation']} "
                f"recommendation with a composite score of {score_result['composite_score']}/100. "
                f"Your recommendation MUST align with this score unless you can provide explicit, "
                f"data-backed justification for a different view.\n\n"
                f"First, try to retrieve_analyst_context for relevant historical analysis "
                f"on '{ticker} Indian market valuation investment thesis ROCE'. Then write the full report."
            )},
        ]

        rag_context = ""
        report_text = ""
        max_iterations = 6

        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error("Report Agent LLM call failed on iteration %d: %s", iteration, e)
                errors.append(f"Report Agent: LLM call failed on iteration {iteration}: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                report_text = response.content
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        logger.warning("Report Agent: Tool %s failed: %s", tool_name, e)
                        errors.append(f"Report Agent: {tool_name} failed: {e}")
                        messages.append(
                            ToolMessage(
                                content=json.dumps({"error": f"Tool failed: {e}"}),
                                tool_call_id=tool_call["id"],
                            )
                        )
                        continue

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

        # ── Step 4: Wrap in Jinja2 report template ──────────────
        # If LLM failed to produce a report, generate a fallback
        if not report_text:
            logger.warning("Report Agent: LLM produced no report text, using fallback")
            errors.append("Report Agent: LLM produced no report, using quantitative fallback")
            report_text = _generate_fallback_report(ticker, score_result, state)

        from jinja2 import Template

        template = Template(REPORT_TEMPLATE)
        final_report = template.render(
            ticker=ticker,
            date=datetime.now().strftime("%B %d, %Y"),
            recommendation=score_result["recommendation"],
            composite_score=score_result["composite_score"],
            confidence=score_result["confidence"],
            report_content=report_text,
        )

        return {
            "investment_score": score_result,
            "rag_context": rag_context,
            "final_report": final_report,
            "errors": errors,
            "messages": [HumanMessage(
                content=f"[Report Agent] Investment memo generated for {ticker} "
                        f"| Score: {score_result['composite_score']}/100 "
                        f"| Recommendation: {score_result['recommendation']}"
            )],
        }

    except Exception as e:
        logger.error("Report Agent critical error for %s: %s", ticker, e)
        # Generate a minimal fallback report
        fallback_score = {
            "composite_score": 50,
            "recommendation": "HOLD",
            "confidence": "LOW",
            "dimension_scores": {},
        }
        return {
            "investment_score": fallback_score,
            "errors": [f"Report Agent error: {str(e)}"],
            "final_report": f"# Error generating report for {ticker}\n\n{str(e)}",
            "messages": [HumanMessage(content=f"[Report Agent] Error: {str(e)}")],
        }
