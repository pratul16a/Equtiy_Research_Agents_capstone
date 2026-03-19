"""Bull vs Bear Debate Agents — 3 agents with multi-round argumentation.

Architecture:
    Bull Agent (GPT-4o) → argues upside with RAG evidence
    Bear Agent (GPT-4o) → argues downside with RAG evidence
    Judge Agent (GPT-4o) → assigns conviction score (1-10) with reasoning

Uses LangGraph cycles for 2-3 debate rounds per stock.
Only debates top 5 stocks by composite USP + screening score.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_llm
from app.rag.retriever import search_by_symbol

logger = logging.getLogger(__name__)

MAX_DEBATE_STOCKS = 5
DEBATE_ROUNDS = 2


def _get_rag_context(symbol: str, query: str, k: int = 5) -> str:
    """Fetch RAG context for a symbol, return as formatted string."""
    results = search_by_symbol(query=query, symbol=symbol, k=k)
    if not results:
        return "(No RAG data available for this query)"

    chunks = []
    for r in results:
        section = r.get("metadata", {}).get("section", "")
        chunks.append(f"[{section}] {r['content'][:500]}")
    return "\n\n".join(chunks)


def _build_stock_context(ticker: str, usp_cards: dict, category_results: list[dict]) -> str:
    """Build compact context string for debate agents from screening + USP data."""
    lines = [f"## {ticker} — Context for Debate"]

    # USP scores
    usp = usp_cards.get(ticker, {})
    composite = usp.get("_composite", "N/A")
    lines.append(f"\nUSP Composite Score: {composite}/100")
    for dim in ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]:
        dim_data = usp.get(dim, {})
        if dim_data:
            lines.append(f"  {dim}: {dim_data.get('score', 'N/A')}/100 — {dim_data.get('level', '')}")
            for factor in dim_data.get("factors", [])[:3]:
                lines.append(f"    • {factor.get('name', '')}: {factor.get('value', '')}")

    # Category placement
    for cat in category_results:
        for stock in cat.get("stocks", []):
            if stock.get("ticker") == ticker:
                lines.append(f"\nCategory: {cat.get('category', '?')} | Score: {stock.get('score', 'N/A')}")
                lines.append(f"Criteria met: {stock.get('criteria_met', 'N/A')}")
                break

    return "\n".join(lines)


def run_bull_agent(
    ticker: str,
    stock_context: str,
    bear_arguments: str | None = None,
    round_num: int = 1,
) -> str:
    """Bull Agent: argue the upside case with RAG evidence.

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        stock_context: Pre-built context from screening + USP
        bear_arguments: Previous round's bear arguments to rebut
        round_num: Current debate round

    Returns:
        Bull agent's argument text.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Fetch RAG evidence for bull case
    rag_growth = _get_rag_context(symbol, f"{symbol} revenue growth margins profit")
    rag_positive = _get_rag_context(symbol, f"{symbol} competitive advantage market share expansion")

    rebuttal_section = ""
    if bear_arguments:
        rebuttal_section = f"""
## Bear Agent's Arguments (Round {round_num - 1})
{bear_arguments}

You MUST rebut each bear argument with specific data. Do not ignore any point."""

    prompt = f"""You are a BULL Agent — an aggressive equity analyst arguing the UPSIDE case for {ticker}.

{stock_context}

## RAG Evidence (from Screener.in)
### Growth & Profitability Data:
{rag_growth}

### Competitive Position:
{rag_positive}
{rebuttal_section}

## Instructions (Round {round_num} of {DEBATE_ROUNDS}):
1. Present 3-5 SPECIFIC bullish arguments
2. EVERY argument must cite a specific number (revenue growth %, margin %, ROCE %, etc.)
3. Reference USP scores where they support your case
4. If rebutting bear arguments, address each one directly
5. Keep it concise — max 300 words

Format: numbered list of arguments, each with a specific data point."""

    try:
        llm = get_llm(temperature=0.5, model="gpt-4o")
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error("Bull Agent failed for %s: %s", ticker, e)
        return f"Bull Agent error: {e}"


def run_bear_agent(
    ticker: str,
    stock_context: str,
    bull_arguments: str | None = None,
    round_num: int = 1,
) -> str:
    """Bear Agent: argue the downside case with RAG evidence."""
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Fetch RAG evidence for bear case
    rag_risks = _get_rag_context(symbol, f"{symbol} risks debt valuation concerns competition")
    rag_negative = _get_rag_context(symbol, f"{symbol} cons weakness declining")

    rebuttal_section = ""
    if bull_arguments:
        rebuttal_section = f"""
## Bull Agent's Arguments (Round {round_num - 1})
{bull_arguments}

You MUST challenge each bull argument with specific counter-evidence. Do not concede any point easily."""

    prompt = f"""You are a BEAR Agent — a skeptical equity analyst arguing the DOWNSIDE case for {ticker}.

{stock_context}

## RAG Evidence (from Screener.in)
### Risk & Valuation Data:
{rag_risks}

### Weaknesses & Concerns:
{rag_negative}
{rebuttal_section}

## Instructions (Round {round_num} of {DEBATE_ROUNDS}):
1. Present 3-5 SPECIFIC bearish arguments
2. EVERY argument must cite a specific number (valuation multiple, debt ratio, margin compression, etc.)
3. Highlight any USP dimensions where scores are weak
4. Question the sustainability of any positive trends
5. If rebutting bull arguments, poke holes with specific data
6. Keep it concise — max 300 words

Format: numbered list of arguments, each with a specific data point."""

    try:
        llm = get_llm(temperature=0.5, model="gpt-4o")
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error("Bear Agent failed for %s: %s", ticker, e)
        return f"Bear Agent error: {e}"


def run_judge_agent(
    ticker: str,
    stock_context: str,
    bull_arguments: list[str],
    bear_arguments: list[str],
) -> dict[str, Any]:
    """Judge Agent: weigh both sides and assign a conviction score.

    Returns:
        {conviction_score (1-10), recommendation, reasoning, key_factors}
    """
    bull_text = "\n\n---\n\n".join(
        f"### Bull Round {i+1}\n{arg}" for i, arg in enumerate(bull_arguments)
    )
    bear_text = "\n\n---\n\n".join(
        f"### Bear Round {i+1}\n{arg}" for i, arg in enumerate(bear_arguments)
    )

    prompt = f"""You are a JUDGE Agent — a senior portfolio manager evaluating a debate about {ticker}.

{stock_context}

## Bull Case (all rounds):
{bull_text}

## Bear Case (all rounds):
{bear_text}

## Your Task:
1. Evaluate the strength of evidence on both sides
2. Identify which arguments were backed by specific data vs vague claims
3. Weigh the USP scores and RAG evidence cited
4. Assign a conviction score from 1-10:
   - 1-3: SELL / strong bear case wins
   - 4-5: HOLD / balanced, no clear edge
   - 6-7: ACCUMULATE / moderate bull case
   - 8-10: BUY / strong bull case wins

Respond with ONLY a JSON object (no markdown fences):
{{
    "conviction_score": <1-10>,
    "recommendation": "BUY|ACCUMULATE|HOLD|SELL",
    "reasoning": "<2-3 sentences explaining your verdict>",
    "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "bull_strength": <1-10>,
    "bear_strength": <1-10>
}}"""

    try:
        llm = get_llm(temperature=0.2, model="gpt-4o")
        response = llm.invoke(prompt)
        content = response.content.strip().strip("```json").strip("```").strip()
        result = json.loads(content)

        # Validate required fields
        required = ["conviction_score", "recommendation", "reasoning"]
        for field in required:
            if field not in result:
                result[field] = "Error: missing field"

        return result

    except json.JSONDecodeError as e:
        logger.error("Judge Agent JSON parse error for %s: %s", ticker, e)
        return {
            "conviction_score": 5,
            "recommendation": "HOLD",
            "reasoning": f"Judge Agent could not parse response: {e}",
            "key_factors": [],
            "bull_strength": 5,
            "bear_strength": 5,
        }
    except Exception as e:
        logger.error("Judge Agent failed for %s: %s", ticker, e)
        return {
            "conviction_score": 5,
            "recommendation": "HOLD",
            "reasoning": f"Judge Agent error: {e}",
            "key_factors": [],
            "bull_strength": 5,
            "bear_strength": 5,
        }


def run_debate(
    ticker: str,
    usp_cards: dict[str, dict],
    category_results: list[dict],
    rounds: int = DEBATE_ROUNDS,
) -> dict[str, Any]:
    """Run a full multi-round debate for a single stock.

    Returns:
        {ticker, bull_arguments, bear_arguments, verdict, rounds}
    """
    stock_context = _build_stock_context(ticker, usp_cards, category_results)

    bull_args: list[str] = []
    bear_args: list[str] = []

    for round_num in range(1, rounds + 1):
        logger.info("Debate %s: Round %d/%d", ticker, round_num, rounds)

        # Bull goes first
        bull_response = run_bull_agent(
            ticker=ticker,
            stock_context=stock_context,
            bear_arguments=bear_args[-1] if bear_args else None,
            round_num=round_num,
        )
        bull_args.append(bull_response)

        # Bear rebuts
        bear_response = run_bear_agent(
            ticker=ticker,
            stock_context=stock_context,
            bull_arguments=bull_args[-1],
            round_num=round_num,
        )
        bear_args.append(bear_response)

    # Judge evaluates all rounds
    verdict = run_judge_agent(
        ticker=ticker,
        stock_context=stock_context,
        bull_arguments=bull_args,
        bear_arguments=bear_args,
    )

    return {
        "ticker": ticker,
        "bull_arguments": bull_args,
        "bear_arguments": bear_args,
        "verdict": verdict,
        "rounds": rounds,
    }


def _select_top_stocks(
    usp_cards: dict[str, dict],
    category_results: list[dict],
    max_stocks: int = MAX_DEBATE_STOCKS,
) -> list[str]:
    """Select top stocks for debate by composite USP + category score."""
    ticker_scores: dict[str, float] = {}

    for cat_result in category_results:
        for stock in cat_result.get("stocks", []):
            ticker = stock.get("ticker", "")
            cat_score = stock.get("weighted_score", stock.get("score", 0))
            usp_composite = usp_cards.get(ticker, {}).get("_composite", 0)
            # Combined score: 60% screening + 40% USP
            ticker_scores[ticker] = cat_score * 0.6 + usp_composite * 0.4

    sorted_tickers = sorted(ticker_scores, key=ticker_scores.get, reverse=True)
    return sorted_tickers[:max_stocks]


def debate_node(state: dict) -> dict:
    """LangGraph node: run Bull/Bear/Judge debate for top stocks.

    Uses cycles: each stock gets 2 rounds of Bull vs Bear, then Judge.
    """
    usp_cards = state.get("usp_cards", {})
    category_results = state.get("category_results", [])

    # Select top stocks for debate
    top_tickers = _select_top_stocks(usp_cards, category_results)
    if not top_tickers:
        logger.warning("Debate Agent: No stocks to debate")
        return {"debate_results": []}

    logger.info("Debate Agent: Debating %d stocks: %s", len(top_tickers), top_tickers)

    debate_results: list[dict] = []
    for ticker in top_tickers:
        try:
            result = run_debate(
                ticker=ticker,
                usp_cards=usp_cards,
                category_results=category_results,
            )
            debate_results.append(result)
        except Exception as e:
            logger.error("Debate failed for %s: %s", ticker, e)
            debate_results.append({
                "ticker": ticker,
                "bull_arguments": [],
                "bear_arguments": [],
                "verdict": {
                    "conviction_score": 5,
                    "recommendation": "HOLD",
                    "reasoning": f"Debate failed: {e}",
                    "key_factors": [],
                },
                "rounds": 0,
            })

    return {"debate_results": debate_results}
