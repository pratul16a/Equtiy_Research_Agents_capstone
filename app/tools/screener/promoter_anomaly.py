"""Promoter Behavior Anomaly Detection — USP #5.

Detects: pledge creep, related party transaction frequency, promoter buying/selling patterns.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from app.tools.screener.batch_fundamentals import fetch_ticker_financials
from app.tools.screener.qualitative import _fetch_bse_announcements

logger = logging.getLogger(__name__)

_RELATED_PARTY_PATTERN = re.compile(
    r"related party|inter.?corporate|group company|subsidiary loan|promoter entity",
    re.IGNORECASE,
)


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def detect_promoter_buying_pattern(
    ticker: str,
    info: dict,
) -> dict[str, Any]:
    """Detect promoter buying/selling patterns from insider transactions.

    Contrarian signal: promoter buying at lows = bullish.
    Promoter selling at highs = caution.
    """
    data = fetch_ticker_financials(ticker)
    insider_df = data.get("insider_transactions")

    result = {"has_data": False, "signal": "neutral", "buys": 0, "sells": 0}

    if insider_df is None or insider_df.empty:
        return result

    buys = 0
    sells = 0
    buy_value = 0.0
    sell_value = 0.0

    for _, row in insider_df.iterrows():
        text = str(row.get("Text", row.get("Transaction", ""))).lower()
        value = abs(float(row.get("Value", row.get("value", 0)) or 0))

        # Focus on promoter/director transactions
        insider_name = str(row.get("Insider", row.get("insider", ""))).lower()
        is_promoter = any(w in insider_name for w in ["promoter", "director", "chairman", "managing"])

        if not is_promoter and "purchase" not in text and "sale" not in text:
            continue

        if any(w in text for w in ["purchase", "buy", "acquisition"]):
            buys += 1
            buy_value += value
        elif any(w in text for w in ["sale", "sell", "disposition"]):
            sells += 1
            sell_value += value

    result["has_data"] = True
    result["buys"] = buys
    result["sells"] = sells
    result["buy_value"] = round(buy_value, 2)
    result["sell_value"] = round(sell_value, 2)

    if buys > sells * 2:
        result["signal"] = "strong_buying"
    elif buys > sells:
        result["signal"] = "net_buying"
    elif sells > buys * 2:
        result["signal"] = "heavy_selling"
    elif sells > buys:
        result["signal"] = "net_selling"

    return result


def detect_related_party_anomaly(ticker: str) -> dict[str, Any]:
    """Detect high frequency of related party transactions from BSE announcements."""
    base_name = ticker.upper().replace(".NS", "").replace(".BO", "")
    result = {"rpt_count": 0, "has_anomaly": False, "headlines": []}

    try:
        announcements = _fetch_bse_announcements(base_name)
        rpt_count = 0
        rpt_headlines: list[str] = []

        for ann in announcements:
            headline = ann.get("headline", "")
            if _RELATED_PARTY_PATTERN.search(headline):
                rpt_count += 1
                rpt_headlines.append(headline)

        result["rpt_count"] = rpt_count
        result["headlines"] = rpt_headlines[:5]
        # More than 3 RPT announcements in recent filings = anomaly
        result["has_anomaly"] = rpt_count >= 3
    except Exception as e:
        logger.debug("RPT check failed for %s: %s", ticker, e)

    return result


def screen_promoter_anomalies(
    tickers: list[str],
    sector_map: dict[str, str],
    bulk_info: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    """Screen for promoter behavior anomalies (positive and negative signals).

    Positive: Strong promoter buying, low related party transactions
    Negative: Heavy selling, excessive RPT
    """
    results: list[dict] = []

    for ticker in tickers:
        try:
            info = (bulk_info or {}).get(ticker, {})
            buying = detect_promoter_buying_pattern(ticker, info)
            rpt = detect_related_party_anomaly(ticker)

            # Score the anomaly
            anomaly_score = 50  # Neutral default
            flags: list[str] = []

            if buying["signal"] == "strong_buying":
                anomaly_score += 25
                flags.append("Strong promoter buying")
            elif buying["signal"] == "net_buying":
                anomaly_score += 15
                flags.append("Net promoter buying")
            elif buying["signal"] == "heavy_selling":
                anomaly_score -= 25
                flags.append("Heavy promoter selling")
            elif buying["signal"] == "net_selling":
                anomaly_score -= 15
                flags.append("Net promoter selling")

            if rpt["has_anomaly"]:
                anomaly_score -= 15
                flags.append(f"High RPT frequency ({rpt['rpt_count']} filings)")

            # High promoter holding from info
            insider_pct = _safe_float(info.get("heldPercentInsiders"))
            if insider_pct is not None and insider_pct > 0.60:
                anomaly_score += 10
                flags.append(f"High promoter holding ({insider_pct*100:.0f}%)")

            # Only include stocks with notable signals (not neutral)
            if anomaly_score != 50 and flags:
                results.append({
                    "ticker": ticker,
                    "sector": sector_map.get(ticker, "Other"),
                    "promoter_score": max(0, min(100, anomaly_score)),
                    "buying_signal": buying["signal"],
                    "promoter_buys": buying["buys"],
                    "promoter_sells": buying["sells"],
                    "rpt_anomaly": rpt["has_anomaly"],
                    "rpt_count": rpt["rpt_count"],
                    "flags": flags,
                    "passed": anomaly_score >= 60,
                })
        except Exception as e:
            logger.debug("Promoter anomaly check failed for %s: %s", ticker, e)

    # Only return stocks that passed (positive anomaly)
    results = [r for r in results if r["passed"]]
    logger.info("Promoter anomaly screen: %d stocks with positive signals", len(results))
    return results
