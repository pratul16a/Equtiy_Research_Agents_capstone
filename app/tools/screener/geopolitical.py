"""Geopolitical & macro risk scoring — USP #1.

Scores each stock on 4 dimensions: trade risk, policy risk, commodity risk, event risk.
Uses sector-level config defaults + stock-specific adjustments (D/E for rate sensitivity, etc.).
Higher score = lower risk (more resilient).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "geopolitical_config.json")
_config: dict | None = None


def _load_config() -> dict:
    global _config
    if _config is not None:
        return _config
    path = os.path.normpath(_CONFIG_PATH)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _config = json.load(f)
    else:
        logger.warning("Geopolitical config not found at %s", path)
        _config = {"sector_profiles": {}, "default_profile": {}, "dimension_weights": {}}
    return _config


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (ValueError, TypeError):
        return None


def _compute_trade_risk_score(profile: dict, info: dict) -> float:
    """Trade risk: high export sensitivity + high import dependency = higher risk (lower score)."""
    trade = profile.get("trade_risk", {})
    export_sens = trade.get("export_sensitivity", 0.3)
    china_benefit = trade.get("china_plus_one_benefit", 0.3)
    import_dep = trade.get("import_dependency", 0.3)

    # Export sensitivity reduces resilience, China+1 benefit improves it
    risk = (export_sens * 0.4 + import_dep * 0.4 - china_benefit * 0.2)
    return max(0, min(100, round((1 - risk) * 100)))


def _compute_policy_risk_score(profile: dict, info: dict) -> float:
    """Policy risk: regulatory sensitivity + rate sensitivity (adjusted by D/E)."""
    policy = profile.get("policy_risk", {})
    reg_sens = policy.get("regulatory_sensitivity", 0.5)
    govt_dep = policy.get("govt_scheme_dependency", 0.3)
    rate_sens = policy.get("rate_sensitivity", 0.5)

    # Adjust rate sensitivity by actual D/E ratio
    de_ratio = _safe_float(info.get("debtToEquity"))
    if de_ratio is not None:
        de_factor = min(de_ratio / 100, 1.5)  # Normalize: 100% D/E = 1.0
        rate_sens = min(1.0, rate_sens * (0.5 + 0.5 * de_factor))

    # Government scheme dependency is positive (reduces risk)
    risk = (reg_sens * 0.4 + rate_sens * 0.4 - govt_dep * 0.2)
    return max(0, min(100, round((1 - risk) * 100)))


def _compute_commodity_risk_score(profile: dict, info: dict) -> float:
    """Commodity risk: crude oil sensitivity + raw material import dependency."""
    commodity = profile.get("commodity_risk", {})
    crude_sens = commodity.get("crude_sensitivity", 0.4)
    rm_import = commodity.get("raw_material_import", 0.4)

    # Higher margin companies are more resilient to commodity shocks
    margin = _safe_float(info.get("operatingMargins"))
    margin_buffer = 0
    if margin is not None and margin > 0.15:
        margin_buffer = min(0.2, (margin - 0.15) * 1.0)

    risk = (crude_sens * 0.5 + rm_import * 0.5) - margin_buffer
    return max(0, min(100, round((1 - max(0, risk)) * 100)))


def _compute_event_risk_score(profile: dict, info: dict) -> float:
    """Event risk: monsoon sensitivity + election cycle sensitivity."""
    event = profile.get("event_risk", {})
    monsoon = event.get("monsoon_sensitivity", 0.3)
    election = event.get("election_sensitivity", 0.4)

    risk = monsoon * 0.5 + election * 0.5
    return max(0, min(100, round((1 - risk) * 100)))


def compute_geopolitical_score(
    ticker: str,
    info: dict,
    sector: str,
) -> dict[str, Any]:
    """Compute geopolitical risk score for a single stock.

    Returns dict with overall score (0-100, higher = more resilient) and sub-scores.
    """
    config = _load_config()
    profiles = config.get("sector_profiles", {})
    profile = profiles.get(sector, config.get("default_profile", {}))
    weights = config.get("dimension_weights", {
        "trade_risk": 0.25, "policy_risk": 0.25,
        "commodity_risk": 0.25, "event_risk": 0.25,
    })

    trade_score = _compute_trade_risk_score(profile, info)
    policy_score = _compute_policy_risk_score(profile, info)
    commodity_score = _compute_commodity_risk_score(profile, info)
    event_score = _compute_event_risk_score(profile, info)

    overall = round(
        trade_score * weights.get("trade_risk", 0.25)
        + policy_score * weights.get("policy_risk", 0.25)
        + commodity_score * weights.get("commodity_risk", 0.25)
        + event_score * weights.get("event_risk", 0.25)
    )

    # Classify risk level
    if overall >= 70:
        risk_level = "Low Risk"
    elif overall >= 50:
        risk_level = "Moderate Risk"
    elif overall >= 30:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    return {
        "overall_score": overall,
        "risk_level": risk_level,
        "trade_risk_score": trade_score,
        "policy_risk_score": policy_score,
        "commodity_risk_score": commodity_score,
        "event_risk_score": event_score,
    }


def screen_geopolitical_risk(
    bulk_info: dict[str, dict],
    sector_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Screen for stocks with low geopolitical risk (high resilience score >= 60)."""
    results: list[dict] = []

    for ticker, info in bulk_info.items():
        try:
            sector = sector_map.get(ticker, "Other")
            scores = compute_geopolitical_score(ticker, info, sector)

            if scores["overall_score"] >= 60:
                results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "geo_score": scores["overall_score"],
                    "risk_level": scores["risk_level"],
                    "trade_risk": scores["trade_risk_score"],
                    "policy_risk": scores["policy_risk_score"],
                    "commodity_risk": scores["commodity_risk_score"],
                    "event_risk": scores["event_risk_score"],
                    "passed": True,
                })
        except Exception as e:
            logger.debug("Geopolitical scoring failed for %s: %s", ticker, e)

    logger.info("Geopolitical screen: %d stocks with low risk", len(results))
    return results
