"""Gate-based screening architecture.

Criteria are grouped into gates (Hard / Soft / Bonus).
- Hard gate: stock MUST meet the minimum or it's eliminated.
- Soft gate: stock MUST meet the minimum, but thresholds are lower.
- Bonus gate: adds to score, never blocks.

A stock passes if: every non-bonus gate passes AND overall score >= threshold.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class GateDef(TypedDict):
    name: str
    gate_type: Literal["hard", "soft", "bonus"]
    min_pass: int
    criteria_keys: list[str]
    mandatory_keys: list[str]   # subset of criteria_keys that MUST pass
    deferred_keys: list[str]    # not yet live; excluded from active count


# ── Momentum Gates (5 gates, 17 criteria) ────────────────────

MOMENTUM_GATES: list[GateDef] = [
    {
        "name": "Sector Tide",
        "gate_type": "hard",
        "min_pass": 2,
        "criteria_keys": ["sector_rs_positive", "sector_highs_gt_lows", "sector_top4"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
    {
        "name": "Trend Structure",
        "gate_type": "hard",
        "min_pass": 3,
        "criteria_keys": ["above_200dma", "golden_alignment", "rsi_55_75", "near_52w_high", "macd_bullish"],
        "mandatory_keys": ["above_200dma"],
        "deferred_keys": [],
    },
    {
        "name": "Volume Conviction",
        "gate_type": "soft",
        "min_pass": 1,
        "criteria_keys": ["volume_breakout", "delivery_pct", "obv_rising"],
        "mandatory_keys": [],
        "deferred_keys": ["delivery_pct"],
    },
    {
        "name": "Smart Money Flow",
        "gate_type": "soft",
        "min_pass": 1,
        "criteria_keys": ["promoter_holding", "institutional_interest", "mf_scheme_count"],
        "mandatory_keys": [],
        "deferred_keys": ["mf_scheme_count"],
    },
    {
        "name": "Fundamental Backbone",
        "gate_type": "soft",
        "min_pass": 1,
        "criteria_keys": ["opm_improvement", "order_booking", "roe_above_12"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
]

MOMENTUM_OVERALL = {"threshold": 11, "active_total": 15}


# ── Value Bottom Gates (7 gates, 33 criteria) ────────────────

VALUE_GATES: list[GateDef] = [
    {
        "name": "Valuation Floor",
        "gate_type": "hard",
        "min_pass": 2,
        "criteria_keys": ["low_roe", "pb_below_avg", "ps_below_avg", "ev_ebitda_below_median", "fcf_yield_above_sector"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
    {
        "name": "Turnaround Signals",
        "gate_type": "hard",
        "min_pass": 2,
        "criteria_keys": ["debt_reduction", "opm_improvement", "cf_turnaround", "dividend_initiation", "working_capital"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
    {
        "name": "Trap Avoidance",
        "gate_type": "hard",
        "min_pass": 2,
        "criteria_keys": ["piotroski", "altman_z", "no_sebi_red_flags"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
    {
        "name": "Technical Timing",
        "gate_type": "soft",
        "min_pass": 1,
        "criteria_keys": ["rsi_oversold", "above_200dma", "delivery_pct", "sector_rotation"],
        "mandatory_keys": [],
        "deferred_keys": ["delivery_pct"],
    },
    {
        "name": "Re-rating Catalysts",
        "gate_type": "soft",
        "min_pass": 2,
        "criteria_keys": ["capacity_util", "capex", "new_business", "order_booking", "capex_going_live"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
    {
        "name": "Insider Conviction",
        "gate_type": "soft",
        "min_pass": 2,
        "criteria_keys": ["insider_buying", "promoter_holding", "pledge_reducing", "mgmt_credibility", "promoter_behavior"],
        "mandatory_keys": [],
        "deferred_keys": ["pledge_reducing"],
    },
    {
        "name": "Macro Tailwinds",
        "gate_type": "bonus",
        "min_pass": 0,
        "criteria_keys": ["geopolitical", "smart_money", "regulatory", "supply_chain_advantage", "export_pli_beneficiary"],
        "mandatory_keys": [],
        "deferred_keys": [],
    },
]

VALUE_OVERALL = {"threshold": 20, "active_total": 30}


# ── Tier Configuration ────────────────────────────────────────

TIER_CONFIG = {
    "buy_zone":  {"label": "Buy Zone",  "icon": "★", "color": "#00D4AA"},
    "watchlist": {"label": "Watchlist",  "icon": "◉", "color": "#4DA6FF"},
    "monitor":   {"label": "Monitor",   "icon": "◎", "color": "#B388FF"},
    "near_miss": {"label": "Near Miss", "icon": "⊘", "color": "#FFA726"},
    "failed":    {"label": "Failed",    "icon": "✗", "color": "#FF4757"},
}
TIER_ORDER = ["buy_zone", "watchlist", "monitor", "near_miss", "failed"]

# Score-based tier thresholds (absolute scores, not ratios)
# Buy Zone: all gates pass + score >= buy_zone
# Watchlist: all gates pass + score >= watchlist (but < buy_zone)
# Monitor: all gates pass + score >= monitor (but < watchlist)
# Near Miss: 0 hard failures + exactly 1 soft failure + score >= near_miss_score
MOMENTUM_TIER_THRESHOLDS = {"buy_zone": 10, "watchlist": 9, "monitor": 8, "near_miss_score": 9}
VALUE_TIER_THRESHOLDS = {"buy_zone": 20, "watchlist": 17, "monitor": 15, "near_miss_score": 18}


# ── Gate Evaluation Engine ────────────────────────────────────


def evaluate_gates(
    criteria_passed: set[str],
    gates: list[GateDef],
    disabled: set[str] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluate all gates for a single stock.

    Args:
        criteria_passed: set of criterion keys the stock passed.
        gates: ordered list of gate definitions.
        disabled: criteria the user turned off (treated like deferred).

    Returns:
        (all_required_gates_pass, gate_details_list)
    """
    if disabled is None:
        disabled = set()

    gate_details: list[dict[str, Any]] = []
    all_pass = True

    for gate in gates:
        # Active keys = total - deferred - disabled
        skip = set(gate["deferred_keys"]) | disabled
        active_keys = [k for k in gate["criteria_keys"] if k not in skip]
        passed_keys = [k for k in active_keys if k in criteria_passed]

        # Mandatory check (only among non-disabled mandatory keys)
        active_mandatory = [k for k in gate["mandatory_keys"] if k not in disabled]
        mandatory_met = all(k in criteria_passed for k in active_mandatory)

        # Adjust min_pass if user disabled so many that original min is unreachable
        effective_min = min(gate["min_pass"], len(active_keys))

        if gate["gate_type"] == "bonus":
            gate_passed = True  # bonus never blocks
        else:
            gate_passed = mandatory_met and len(passed_keys) >= effective_min

        if not gate_passed and gate["gate_type"] != "bonus":
            all_pass = False

        gate_details.append({
            "name": gate["name"],
            "gate_type": gate["gate_type"],
            "passed": gate_passed,
            "passed_count": len(passed_keys),
            "active_count": len(active_keys),
            "min_pass": effective_min,
            "mandatory_met": mandatory_met,
            "passed_keys": passed_keys,
            "failed_keys": [k for k in active_keys if k not in criteria_passed],
            "deferred_keys": [k for k in gate["criteria_keys"] if k in skip],
        })

    return all_pass, gate_details


def compute_overall_score(
    criteria_passed: set[str],
    gates: list[GateDef],
    threshold: int,
    disabled: set[str] | None = None,
) -> tuple[int, int, bool]:
    """Compute overall score across all gates.

    Returns:
        (score, active_total, meets_threshold)
    """
    if disabled is None:
        disabled = set()

    active_keys: set[str] = set()
    for gate in gates:
        skip = set(gate["deferred_keys"]) | disabled
        for k in gate["criteria_keys"]:
            if k not in skip:
                active_keys.add(k)

    score = len(criteria_passed & active_keys)
    return score, len(active_keys), score >= threshold
