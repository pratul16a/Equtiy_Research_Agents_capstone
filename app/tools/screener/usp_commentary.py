"""LLM-powered USP commentary generation — per-stock and portfolio-level."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _call_llm(system: str, user: str) -> str:
    """Make a single LLM call and return the text response."""
    from app.config import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_llm(temperature=0.3)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content


def _parse_commentary_json(text: str) -> dict[str, str]:
    """Extract JSON dict from LLM response, handling markdown fences."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse USP commentary JSON, returning raw text")
        return {"_raw": cleaned}


def generate_stock_commentary(
    ticker: str,
    usp_data: dict[str, Any],
    sector: str = "Unknown",
) -> dict[str, str]:
    """Generate LLM commentary for one stock's USP dimensions.

    Returns: {geopolitical: "...", smart_money: "...", regulatory: "...",
              mgmt_credibility: "...", promoter: "..."}
    """
    from app.prompts.usp_commentary import build_stock_commentary_prompt

    system, user = build_stock_commentary_prompt(ticker, usp_data, sector)
    try:
        raw = _call_llm(system, user)
        return _parse_commentary_json(raw)
    except Exception as e:
        logger.warning("USP commentary failed for %s: %s", ticker, e)
        return {}


def generate_usp_commentary(
    per_stock_usp: dict[str, dict[str, Any]],
    bulk_info: dict[str, dict] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    max_workers: int = 5,
) -> dict[str, dict[str, str]]:
    """Generate LLM commentary for all stocks in parallel.

    Args:
        per_stock_usp: {ticker: {module: raw_data}} from apply_usp_layer()
        bulk_info: Optional {ticker: info_dict} for sector lookup
        progress_cb: Optional progress callback
        max_workers: Concurrent LLM calls (default 5)

    Returns: {ticker: {dimension: commentary_text}}
    """
    if not per_stock_usp:
        return {}

    bulk_info = bulk_info or {}
    tickers = list(per_stock_usp.keys())
    total = len(tickers)
    result: dict[str, dict[str, str]] = {}

    if progress_cb:
        progress_cb(0.0, f"Generating USP commentary for {total} stocks...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for ticker in tickers:
            sector = per_stock_usp[ticker].get("geopolitical", {}).get("sector", "Unknown")
            if sector == "Unknown" and ticker in bulk_info:
                sector = bulk_info[ticker].get("sector", "Unknown")
            futures[executor.submit(
                generate_stock_commentary, ticker, per_stock_usp[ticker], sector
            )] = ticker

        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                commentary = future.result(timeout=30)
                if commentary:
                    result[ticker] = commentary
            except Exception as e:
                logger.warning("Commentary generation timed out for %s: %s", ticker, e)
            done += 1
            if progress_cb:
                progress_cb(done / total, f"USP commentary: {done}/{total} stocks done")

    return result


def generate_portfolio_insights(
    per_stock_usp: dict[str, dict[str, Any]],
    category: str = "Momentum",
) -> str:
    """Generate cross-stock portfolio-level insights.

    Returns: Markdown string with 4 portfolio insights.
    """
    if len(per_stock_usp) < 2:
        return ""

    from app.prompts.usp_commentary import build_portfolio_insights_prompt

    system, user = build_portfolio_insights_prompt(per_stock_usp, category)
    try:
        return _call_llm(system, user)
    except Exception as e:
        logger.warning("Portfolio insights generation failed: %s", e)
        return ""
