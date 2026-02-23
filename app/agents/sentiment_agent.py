"""Sentiment Agent — analyzes news, social media, and management commentary."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from app.config import get_llm
from app.prompts.templates import SENTIMENT_AGENT_PROMPT
from app.tools.news import NEWS_TOOLS


def create_sentiment_agent(ticker: str):
    """Create a sentiment analysis agent with news tools."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(NEWS_TOOLS)
    return llm_with_tools


def sentiment_node(state: dict) -> dict:
    """LangGraph node: analyze sentiment from news and social sources.

    Populates state['sentiment_scores'] and state['news_summaries'].
    """
    ticker = state.get("ticker", "")
    if not ticker:
        return {"errors": ["Sentiment Agent: No ticker provided"]}

    try:
        llm_with_tools = create_sentiment_agent(ticker)
        tool_map = {t.name: t for t in NEWS_TOOLS}

        prompt = SENTIMENT_AGENT_PROMPT.format(ticker=ticker)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"Analyze the current market sentiment for {ticker}. "
                f"Fetch financial news and broader market news, then provide your sentiment assessment."
            )},
        ]

        all_articles: list[dict] = []
        sentiment_text = ""
        max_iterations = 6

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                sentiment_text = response.content
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                    try:
                        parsed = json.loads(result)
                        if "articles" in parsed:
                            all_articles.extend(parsed["articles"])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )

        # Parse sentiment score from the LLM's response
        sentiment_scores = _extract_sentiment_scores(sentiment_text)

        # Build news summaries
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
            "messages": [HumanMessage(content=f"[Sentiment Agent] Completed sentiment analysis for {ticker}\n\n{sentiment_text}")],
        }

    except Exception as e:
        return {
            "errors": [f"Sentiment Agent error: {str(e)}"],
            "messages": [HumanMessage(content=f"[Sentiment Agent] Error: {str(e)}")],
        }


def _extract_sentiment_scores(text: str) -> dict[str, Any]:
    """Extract sentiment scores from the LLM's analysis text."""
    import re

    scores: dict[str, Any] = {
        "overall_score": 0.0,
        "overall_label": "neutral",
        "analysis": text[:1000] if text else "",
    }

    # Try to find a numeric score in the text
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
