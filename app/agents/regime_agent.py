"""Market Regime Agent — classifies bull/bear/rotation/mixed from VIX, breadth, FII flows.

Sets category weight multipliers that adjust scoring downstream:
- Bull:     Cat B (Momentum) boosted 1.3x, Cat C (Value) dampened 0.7x
- Bear:     Cat C boosted 1.3x, Cat B dampened 0.7x
- Rotation: Cat A (Sector) boosted 1.3x
- Mixed:    All categories weighted equally (1.0x)
"""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from app.config import get_llm

logger = logging.getLogger(__name__)

# Weight multipliers per regime
REGIME_WEIGHTS = {
    "bull": {"A": 1.0, "B": 1.3, "C": 0.7},
    "bear": {"A": 1.0, "B": 0.7, "C": 1.3},
    "rotation": {"A": 1.3, "B": 1.0, "C": 1.0},
    "mixed": {"A": 1.0, "B": 1.0, "C": 1.0},
}


def _safe_scalar(val) -> float:
    """Convert a possibly-Series value to a plain float."""
    import pandas as pd
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


def _fetch_vix() -> float | None:
    """Fetch India VIX current value."""
    try:
        vix = yf.download("^INDIAVIX", period="5d", progress=False)
        if vix is not None and not vix.empty:
            return _safe_scalar(vix["Close"].iloc[-1])
    except Exception as e:
        logger.warning("Failed to fetch India VIX: %s", e)
    return None


def _fetch_nifty_change() -> dict[str, float | None]:
    """Fetch Nifty 500 recent performance."""
    try:
        nifty = yf.download("^CRSLDX", period="3mo", progress=False)
        if nifty is not None and not nifty.empty:
            close = nifty["Close"]
            current = _safe_scalar(close.iloc[-1])
            month_ago = _safe_scalar(close.iloc[-22]) if len(close) > 22 else current
            three_month_ago = _safe_scalar(close.iloc[0])
            return {
                "current": current,
                "1m_change_pct": round((current - month_ago) / month_ago * 100, 2),
                "3m_change_pct": round((current - three_month_ago) / three_month_ago * 100, 2),
            }
    except Exception as e:
        logger.warning("Failed to fetch Nifty data: %s", e)
    return {"current": None, "1m_change_pct": None, "3m_change_pct": None}


def _classify_regime_rule_based(vix: float | None, nifty: dict) -> tuple[str, str]:
    """Rule-based regime classification as fallback or primary classifier.

    Returns (regime, reasoning).
    """
    signals = []

    # VIX-based signal
    if vix is not None:
        if vix < 14:
            signals.append(("bull", f"VIX={vix:.1f} (low fear)"))
        elif vix < 20:
            signals.append(("mixed", f"VIX={vix:.1f} (moderate)"))
        elif vix < 28:
            signals.append(("rotation", f"VIX={vix:.1f} (elevated, sector rotation likely)"))
        else:
            signals.append(("bear", f"VIX={vix:.1f} (high fear)"))

    # Nifty trend signal
    change_1m = nifty.get("1m_change_pct")
    change_3m = nifty.get("3m_change_pct")
    if change_1m is not None and change_3m is not None:
        if change_1m > 3 and change_3m > 5:
            signals.append(("bull", f"Nifty +{change_1m:.1f}% (1M), +{change_3m:.1f}% (3M)"))
        elif change_1m < -3 and change_3m < -5:
            signals.append(("bear", f"Nifty {change_1m:.1f}% (1M), {change_3m:.1f}% (3M)"))
        elif abs(change_1m) < 2 and abs(change_3m) < 3:
            signals.append(("rotation", f"Nifty flat: {change_1m:.1f}% (1M) — rotation"))
        else:
            signals.append(("mixed", f"Nifty {change_1m:.1f}% (1M), {change_3m:.1f}% (3M)"))

    if not signals:
        return "mixed", "Insufficient data for regime classification"

    # Majority vote
    regime_counts: dict[str, int] = {}
    for regime, _ in signals:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    top_regime = max(regime_counts, key=regime_counts.get)
    reasoning = "; ".join(reason for _, reason in signals)
    return top_regime, reasoning


def detect_regime(
    use_llm: bool = False,
    breadth_data: dict | None = None,
) -> dict[str, Any]:
    """Detect current market regime.

    Args:
        use_llm: If True, use GPT-4o-mini for nuanced classification.
        breadth_data: Optional pre-computed breadth data from market_breadth module.

    Returns:
        {regime, reasoning, weights, vix, nifty_data}
    """
    vix = _fetch_vix()
    nifty = _fetch_nifty_change()

    # Rule-based classification
    regime, reasoning = _classify_regime_rule_based(vix, nifty)

    # Optional LLM enhancement for nuanced classification
    if use_llm:
        try:
            llm = get_llm(temperature=0.1, model="gpt-4o-mini")
            prompt = f"""Classify the Indian stock market regime as exactly one of: bull, bear, rotation, mixed.

Data:
- India VIX: {vix}
- Nifty 500 1-month change: {nifty.get('1m_change_pct')}%
- Nifty 500 3-month change: {nifty.get('3m_change_pct')}%
{f"- Market breadth: {breadth_data}" if breadth_data else ""}

Respond with ONLY a JSON object: {{"regime": "bull|bear|rotation|mixed", "reasoning": "one sentence"}}"""

            response = llm.invoke(prompt)
            import json
            result = json.loads(response.content.strip().strip("```json").strip("```"))
            if result.get("regime") in REGIME_WEIGHTS:
                regime = result["regime"]
                reasoning = result.get("reasoning", reasoning)
        except Exception as e:
            logger.warning("LLM regime classification failed, using rule-based: %s", e)

    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["mixed"])

    return {
        "regime": regime,
        "reasoning": reasoning,
        "weights": weights,
        "vix": vix,
        "nifty_data": nifty,
    }


def regime_node(state: dict) -> dict:
    """LangGraph node: detect market regime and set category weights."""
    try:
        regime_data = detect_regime(use_llm=True)
        return {
            "regime": regime_data,
            "messages": [],
        }
    except Exception as e:
        logger.error("Regime Agent error: %s", e)
        return {
            "regime": {
                "regime": "mixed",
                "reasoning": f"Regime detection failed: {e}",
                "weights": REGIME_WEIGHTS["mixed"],
                "vix": None,
                "nifty_data": {},
            },
            "errors": [f"Regime Agent: {e}"],
        }
