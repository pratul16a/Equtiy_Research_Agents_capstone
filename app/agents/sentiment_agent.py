"""Sentiment Agent — analyzes Indian financial news, market mood, and management commentary.

Includes graceful error recovery: returns partial results on individual tool failures.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import get_llm
from app.prompts.templates import SENTIMENT_AGENT_PROMPT
from app.tools.news import NEWS_TOOLS

logger = logging.getLogger(__name__)


def create_sentiment_agent(ticker: str):
    """Create a sentiment analysis agent with Indian news tools."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(NEWS_TOOLS)
    return llm_with_tools


def sentiment_node(state: dict) -> dict:
    """LangGraph node: analyze sentiment from Indian news sources.

    Populates state['sentiment_scores'], state['news_summaries'],
    and state['management_commentary'].

    Error recovery: if news fetching fails, returns neutral sentiment
    with appropriate warnings rather than crashing the pipeline.
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Sentiment Agent: No ticker provided"]}

    errors: list[str] = []

    try:
        llm_with_tools = create_sentiment_agent(ticker)
        tool_map = {t.name: t for t in NEWS_TOOLS}

        prompt = SENTIMENT_AGENT_PROMPT.format(ticker=ticker)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Analyze the current market sentiment for Indian stock {ticker}. "
                f"Fetch Indian financial news and broader Indian market news, "
                f"then provide your sentiment assessment with Indian market context."
            )},
        ]

        all_articles: list[dict] = []
        sentiment_text = ""
        max_iterations = 6

        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error("Sentiment Agent LLM call failed on iteration %d: %s", iteration, e)
                errors.append(f"Sentiment Agent: LLM call failed on iteration {iteration}: {e}")
                break

            messages.append(response)

            if not response.tool_calls:
                sentiment_text = response.content
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name not in tool_map:
                    logger.warning("Sentiment Agent: Unknown tool %s", tool_name)
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
                    logger.warning("Sentiment Agent: Tool %s failed: %s", tool_name, e)
                    errors.append(f"Sentiment Agent: {tool_name} failed: {e}")
                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": f"Tool failed: {e}"}),
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

                try:
                    parsed = json.loads(result)
                    if "articles" in parsed:
                        all_articles.extend(parsed["articles"])
                except (json.JSONDecodeError, TypeError):
                    pass

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )

        sentiment_scores = _extract_sentiment_scores(sentiment_text)

        # If no articles were fetched, note it but don't fail
        if not all_articles:
            errors.append("Sentiment Agent: No news articles could be fetched")

        news_summaries = []
        for article in all_articles[:10]:
            title = article.get("title", "")
            summary = article.get("summary", "")
            source = article.get("source", "")
            if title:
                news_summaries.append(f"[{source}] {title}: {summary[:200]}")

        return {
            "sentiment_scores": sentiment_scores,
            "news_summaries": news_summaries,
            "management_commentary": sentiment_text[:2000] if sentiment_text else "",
            "errors": errors,
            "messages": [HumanMessage(
                content=f"[Sentiment Agent] Completed sentiment analysis for {ticker}\n\n{sentiment_text}"
            )],
        }

    except Exception as e:
        logger.error("Sentiment Agent critical error for %s: %s", ticker, e)
        # Return neutral defaults so the pipeline can continue
        return {
            "sentiment_scores": {"overall_score": 0.0, "overall_label": "neutral", "analysis": ""},
            "news_summaries": [],
            "management_commentary": "",
            "errors": [f"Sentiment Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Sentiment Agent] Error: {str(e)}")],
        }


def _extract_sentiment_scores(text: str) -> dict[str, Any]:
    """Extract sentiment scores from the LLM's analysis text."""
    scores: dict[str, Any] = {
        "overall_score": 0.0,
        "overall_label": "neutral",
        "analysis": text[:1000] if text else "",
    }

    score_patterns = [
        r"(?:overall\s+)?sentiment\s+score[:\s]*([+-]?\d*\.?\d+)",
        r"score[:\s]*([+-]?\d*\.?\d+)\s*/?\s*1?\.?0?",
        r"([+-]?\d*\.?\d+)\s*(?:out of|/)\s*1",
    ]
    for pattern in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if -1.0 <= score <= 1.0:
                    scores["overall_score"] = round(score, 2)
                    if score > 0.2:
                        scores["overall_label"] = "bullish"
                    elif score < -0.2:
                        scores["overall_label"] = "bearish"
                    else:
                        scores["overall_label"] = "neutral"
                    break
            except ValueError:
                continue

    return scores
