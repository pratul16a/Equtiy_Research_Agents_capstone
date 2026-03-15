"""Quantitative Investment Scoring Engine for Indian Equities.

Scores stocks across 5 dimensions and produces a weighted composite score
with clear BUY/HOLD/SELL thresholds. This replaces pure LLM judgment with
a structured, transparent, and reproducible recommendation framework.

Scoring Dimensions (total = 100):
    1. Valuation       (25 pts) - DCF upside, P/E vs peers, P/B
    2. Quality         (25 pts) - ROCE, margins, D/E, cash flow quality
    3. Growth          (20 pts) - Revenue growth, earnings growth, peer-relative
    4. Sentiment       (15 pts) - News sentiment, FII/DII trend
    5. Governance      (15 pts) - Promoter holding, pledge %, institutional trust

Recommendation Thresholds:
    STRONG BUY : >= 75
    BUY        : 60 - 74
    HOLD       : 40 - 59
    SELL       : 25 - 39
    STRONG SELL: < 25
"""

from __future__ import annotations

from typing import Any


# ── Thresholds ────────────────────────────────────────────────
RECOMMENDATION_THRESHOLDS = [
    (75, "STRONG BUY"),
    (60, "BUY"),
    (40, "HOLD"),
    (25, "SELL"),
    (0,  "STRONG SELL"),
]

DIMENSION_WEIGHTS = {
    "valuation": 25,
    "quality": 25,
    "growth": 20,
    "sentiment": 15,
    "governance": 15,
}


# ── Helper: clamp score to 0-100 range ────────────────────────
def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _safe_float(val, default: float = 0.0) -> float:
    """Extract a float from a value that might be a string like '15.2%'."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("%", "").replace(",", "").replace("₹", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


# ── Dimension Scorers ─────────────────────────────────────────

def score_valuation(state: dict) -> dict:
    """Score valuation (0-100) based on DCF upside, P/E relative, P/B.

    Sub-components:
        - DCF Upside (40%): >30% upside = 100, 0% = 50, <-30% = 0
        - P/E vs Peers (35%): Below peer avg = higher score
        - P/B Ratio (25%): <1 = 100, 1-3 = 70-50, >5 = 20
    """
    details = {}
    sub_scores = []

    # DCF upside
    dcf = state.get("dcf_valuation", {})
    upside = _safe_float(dcf.get("upside_pct", 0))
    # Map upside: -50% -> 0, 0% -> 50, +50% -> 100
    dcf_score = _clamp(50 + upside)
    details["dcf_upside_pct"] = round(upside, 2)
    details["dcf_sub_score"] = round(dcf_score, 1)
    sub_scores.append(("dcf_upside", dcf_score, 0.40))

    # P/E relative to peers
    ratios = state.get("ratios", {})
    peers = state.get("peer_comparison", [])
    pe = _safe_float(ratios.get("pe_trailing", 0))
    peer_pes = [_safe_float(p.get("pe_trailing", 0)) for p in peers if _safe_float(p.get("pe_trailing", 0)) > 0]
    if pe > 0 and peer_pes:
        avg_peer_pe = sum(peer_pes) / len(peer_pes)
        # If PE is 50% below peer avg => 100, at avg => 50, 50% above => 0
        pe_ratio = pe / avg_peer_pe if avg_peer_pe > 0 else 1.0
        pe_score = _clamp(100 - (pe_ratio - 0.5) * 100)
        details["pe_trailing"] = round(pe, 2)
        details["peer_avg_pe"] = round(avg_peer_pe, 2)
        details["pe_discount_pct"] = round((1 - pe_ratio) * 100, 1)
    else:
        pe_score = 50  # neutral if data unavailable
        details["pe_trailing"] = pe
        details["peer_avg_pe"] = "N/A"
    details["pe_sub_score"] = round(pe_score, 1)
    sub_scores.append(("pe_relative", pe_score, 0.35))

    # P/B ratio
    pb = _safe_float(ratios.get("pb_ratio", 0))
    if pb > 0:
        # <1 = 100, 1 = 80, 2 = 60, 3 = 40, 5+ = 10
        pb_score = _clamp(100 - (pb - 0.5) * 20)
        details["pb_ratio"] = round(pb, 2)
    else:
        pb_score = 50
        details["pb_ratio"] = "N/A"
    details["pb_sub_score"] = round(pb_score, 1)
    sub_scores.append(("pb_ratio", pb_score, 0.25))

    weighted = sum(s * w for _, s, w in sub_scores)
    return {
        "score": round(weighted, 1),
        "max_score": 100,
        "details": details,
        "sub_scores": {name: round(s, 1) for name, s, _ in sub_scores},
    }


def score_quality(state: dict) -> dict:
    """Score quality (0-100) based on ROCE, margins, D/E, cash flow.

    Sub-components:
        - ROCE (35%): >25% = 100, 15-25% = 60-90, <10% = 20
        - Profit Margin (25%): >20% = 100, 10-20% = 50-90, <5% = 20
        - Debt-to-Equity (20%): <0.3 = 100, 0.3-1 = 60-80, >2 = 20
        - ROE (20%): >20% = 100, 10-20% = 50-90, <5% = 20
    """
    details = {}
    sub_scores = []
    ratios = state.get("ratios", {})
    indian_metrics = state.get("indian_metrics", {})

    # ROCE
    roce_val = _safe_float(ratios.get("roce", 0))
    # Also check indian_metrics for more detailed ROCE
    roce_data = indian_metrics.get("roce", {})
    if roce_data and isinstance(roce_data, dict):
        roce_pct_str = roce_data.get("roce_pct", "")
        if roce_pct_str:
            parsed = _safe_float(roce_pct_str)
            if parsed > 0:
                roce_val = parsed / 100 if parsed > 1 else parsed  # handle "15.2%" vs 0.152

    roce_pct = roce_val * 100 if roce_val < 1 else roce_val  # normalize to percentage
    # Scoring: 30%+ = 100, 20% = 80, 15% = 60, 10% = 40, 5% = 20, 0 = 10
    roce_score = _clamp(roce_pct * 3.3)
    details["roce_pct"] = round(roce_pct, 2)
    details["roce_sub_score"] = round(roce_score, 1)
    sub_scores.append(("roce", roce_score, 0.35))

    # Profit Margin
    margin = _safe_float(ratios.get("profit_margin", 0))
    margin_pct = margin * 100 if abs(margin) < 1 else margin
    margin_score = _clamp(margin_pct * 4)
    details["profit_margin_pct"] = round(margin_pct, 2)
    details["margin_sub_score"] = round(margin_score, 1)
    sub_scores.append(("profit_margin", margin_score, 0.25))

    # Debt to Equity
    de = _safe_float(ratios.get("debt_to_equity", 0))
    if de >= 0:
        # 0 D/E = 100, 0.5 = 80, 1.0 = 60, 2.0 = 30, 3+ = 10
        de_score = _clamp(100 - de * 35)
    else:
        de_score = 50
    details["debt_to_equity"] = round(de, 2)
    details["de_sub_score"] = round(de_score, 1)
    sub_scores.append(("debt_to_equity", de_score, 0.20))

    # ROE
    roe = _safe_float(ratios.get("roe", 0))
    roe_pct = roe * 100 if abs(roe) < 1 else roe
    roe_score = _clamp(roe_pct * 4)
    details["roe_pct"] = round(roe_pct, 2)
    details["roe_sub_score"] = round(roe_score, 1)
    sub_scores.append(("roe", roe_score, 0.20))

    weighted = sum(s * w for _, s, w in sub_scores)
    return {
        "score": round(weighted, 1),
        "max_score": 100,
        "details": details,
        "sub_scores": {name: round(s, 1) for name, s, _ in sub_scores},
    }


def score_growth(state: dict) -> dict:
    """Score growth (0-100) based on revenue growth, earnings growth.

    Sub-components:
        - Revenue Growth (45%): >20% = 100, 10-20% = 60-90, 0-10% = 30-60, <0% = 10
        - Earnings Growth (45%): Same scale as revenue
        - Peer-relative Growth (10%): Above peer avg = bonus
    """
    details = {}
    sub_scores = []
    ratios = state.get("ratios", {})
    peers = state.get("peer_comparison", [])

    # Revenue growth
    rev_growth = _safe_float(ratios.get("revenue_growth", 0))
    rev_pct = rev_growth * 100 if abs(rev_growth) < 1 else rev_growth
    # 30%+ = 100, 20% = 80, 10% = 55, 0% = 30, -10% = 10
    rev_score = _clamp(30 + rev_pct * 2.3)
    details["revenue_growth_pct"] = round(rev_pct, 2)
    details["rev_sub_score"] = round(rev_score, 1)
    sub_scores.append(("revenue_growth", rev_score, 0.45))

    # Earnings growth
    earn_growth = _safe_float(ratios.get("earnings_growth", 0))
    earn_pct = earn_growth * 100 if abs(earn_growth) < 1 else earn_growth
    earn_score = _clamp(30 + earn_pct * 2.3)
    details["earnings_growth_pct"] = round(earn_pct, 2)
    details["earn_sub_score"] = round(earn_score, 1)
    sub_scores.append(("earnings_growth", earn_score, 0.45))

    # Peer-relative growth bonus
    peer_rev = [_safe_float(p.get("revenue_growth", 0)) for p in peers if p.get("revenue_growth")]
    if peer_rev and rev_growth:
        avg_peer_rev = sum(peer_rev) / len(peer_rev)
        peer_relative = (rev_growth - avg_peer_rev)
        peer_pct = peer_relative * 100 if abs(peer_relative) < 1 else peer_relative
        peer_score = _clamp(50 + peer_pct * 3)
        details["peer_avg_rev_growth_pct"] = round(avg_peer_rev * 100 if abs(avg_peer_rev) < 1 else avg_peer_rev, 2)
    else:
        peer_score = 50
    details["peer_relative_sub_score"] = round(peer_score, 1)
    sub_scores.append(("peer_relative", peer_score, 0.10))

    weighted = sum(s * w for _, s, w in sub_scores)
    return {
        "score": round(weighted, 1),
        "max_score": 100,
        "details": details,
        "sub_scores": {name: round(s, 1) for name, s, _ in sub_scores},
    }


def score_sentiment(state: dict) -> dict:
    """Score sentiment (0-100) based on news sentiment and institutional activity.

    Sub-components:
        - News Sentiment (60%): -1 to +1 mapped to 0-100
        - FII/DII Trend (40%): Increasing institutional interest = positive
    """
    details = {}
    sub_scores = []

    # News sentiment score (-1 to +1)
    sentiment = state.get("sentiment_scores", {})
    raw_score = _safe_float(sentiment.get("overall_score", 0))
    # Map -1..+1 to 0..100
    news_score = _clamp((raw_score + 1) * 50)
    label = sentiment.get("overall_label", "neutral")
    details["sentiment_raw"] = round(raw_score, 2)
    details["sentiment_label"] = label
    details["news_sub_score"] = round(news_score, 1)
    sub_scores.append(("news_sentiment", news_score, 0.60))

    # FII/DII institutional trend
    indian_metrics = state.get("indian_metrics", {})
    fii_data = indian_metrics.get("fii_dii", {})
    inst_holders = fii_data.get("institutional_holders", []) if isinstance(fii_data, dict) else []
    mf_holders = fii_data.get("mutual_fund_holders", []) if isinstance(fii_data, dict) else []

    # Heuristic: more institutional holders + mutual fund holders = higher confidence
    inst_count = len(inst_holders) if isinstance(inst_holders, list) else 0
    mf_count = len(mf_holders) if isinstance(mf_holders, list) else 0
    # Normalize: 10+ institutional holders = 100, 0 = 30
    inst_score = _clamp(30 + inst_count * 7)
    # Boost if mutual funds are present
    if mf_count > 5:
        inst_score = min(100, inst_score + 10)
    details["institutional_holders"] = inst_count
    details["mutual_fund_holders"] = mf_count
    details["institutional_sub_score"] = round(inst_score, 1)
    sub_scores.append(("institutional_trend", inst_score, 0.40))

    weighted = sum(s * w for _, s, w in sub_scores)
    return {
        "score": round(weighted, 1),
        "max_score": 100,
        "details": details,
        "sub_scores": {name: round(s, 1) for name, s, _ in sub_scores},
    }


def score_governance(state: dict) -> dict:
    """Score governance (0-100) based on promoter holding and pledge status.

    Sub-components:
        - Promoter Holding (50%): >60% = 90, 50-60% = 75, 40-50% = 60, <30% = 30
        - Promoter Pledge (30%): 0% = 100, <10% = 70, 10-20% = 40, >20% = 10 (red flag)
        - Institutional Trust (20%): High FII% + DII% = strong governance signal
    """
    details = {}
    sub_scores = []
    shareholding = state.get("shareholding_pattern", {})
    indian_metrics = state.get("indian_metrics", {})

    # Promoter holding
    promoter_pct = 0
    # Try shareholding_pattern first
    if shareholding:
        major = shareholding.get("major_holders", {})
        if isinstance(major, dict):
            for key, val in major.items():
                if "insider" in key.lower() or "promoter" in key.lower():
                    promoter_pct = _safe_float(val)
                    break

    # Also check indian_metrics.shareholding
    sh_metrics = indian_metrics.get("shareholding", {})
    if sh_metrics and isinstance(sh_metrics, dict):
        major = sh_metrics.get("major_holders", {})
        if isinstance(major, dict):
            for key, val in major.items():
                if "insider" in key.lower() or "promoter" in key.lower():
                    parsed = _safe_float(val)
                    if parsed > 0:
                        promoter_pct = parsed
                        break

    # Normalize: might be "50.23%" or 0.5023 or 50.23
    if promoter_pct > 1:
        promoter_pct_norm = promoter_pct  # already in percentage
    elif promoter_pct > 0:
        promoter_pct_norm = promoter_pct * 100
    else:
        promoter_pct_norm = 0

    # Scoring: >60% = 90, 50% = 75, 40% = 60, 30% = 45, <20% = 25
    if promoter_pct_norm > 0:
        promoter_score = _clamp(promoter_pct_norm * 1.5)
    else:
        promoter_score = 50  # neutral if unknown
    details["promoter_holding_pct"] = round(promoter_pct_norm, 2)
    details["promoter_sub_score"] = round(promoter_score, 1)
    sub_scores.append(("promoter_holding", promoter_score, 0.50))

    # Promoter pledge
    pledge_data = indian_metrics.get("pledge", {})
    pledge_pct = 0
    if isinstance(pledge_data, dict) and "error" not in pledge_data:
        pledge_pct = _safe_float(pledge_data.get("pledged_pct", 0))

    # Scoring: 0% pledge = 100, 5% = 80, 10% = 55, 20% = 10, 30%+ = 0
    pledge_score = _clamp(100 - pledge_pct * 5)
    details["promoter_pledge_pct"] = round(pledge_pct, 2)
    details["pledge_sub_score"] = round(pledge_score, 1)
    if pledge_pct > 20:
        details["pledge_flag"] = "RED FLAG: >20% promoter shares pledged"
    elif pledge_pct > 10:
        details["pledge_flag"] = "WARNING: >10% promoter shares pledged"
    sub_scores.append(("promoter_pledge", pledge_score, 0.30))

    # Institutional trust (FII + DII %)
    inst_pct = 0
    if isinstance(sh_metrics, dict):
        major = sh_metrics.get("major_holders", {})
        if isinstance(major, dict):
            for key, val in major.items():
                if "institution" in key.lower() or "fii" in key.lower() or "dii" in key.lower():
                    inst_pct += _safe_float(val)
    if inst_pct > 1:
        inst_pct_norm = inst_pct
    elif inst_pct > 0:
        inst_pct_norm = inst_pct * 100
    else:
        inst_pct_norm = 0

    inst_trust_score = _clamp(inst_pct_norm * 2) if inst_pct_norm > 0 else 50
    details["institutional_pct"] = round(inst_pct_norm, 2)
    details["inst_trust_sub_score"] = round(inst_trust_score, 1)
    sub_scores.append(("institutional_trust", inst_trust_score, 0.20))

    weighted = sum(s * w for _, s, w in sub_scores)
    return {
        "score": round(weighted, 1),
        "max_score": 100,
        "details": details,
        "sub_scores": {name: round(s, 1) for name, s, _ in sub_scores},
    }


# ── Main Scoring Function ────────────────────────────────────

def compute_investment_score(state: dict) -> dict:
    """Compute the composite investment score from all available state data.

    Returns a dict with:
        - composite_score (0-100)
        - recommendation (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
        - dimension_scores (breakdown by each dimension)
        - confidence (how many dimensions had sufficient data)
        - summary (human-readable explanation)
    """
    dimensions = {
        "valuation": score_valuation(state),
        "quality": score_quality(state),
        "growth": score_growth(state),
        "sentiment": score_sentiment(state),
        "governance": score_governance(state),
    }

    # Compute weighted composite
    weighted_sum = 0
    total_weight = 0
    data_completeness = 0

    for dim_name, dim_result in dimensions.items():
        weight = DIMENSION_WEIGHTS[dim_name]
        score = dim_result["score"]
        weighted_sum += score * weight
        total_weight += weight

        # Track data completeness
        details = dim_result.get("details", {})
        non_default = sum(1 for v in details.values() if v not in (0, 0.0, "N/A", "", 50))
        if non_default > 0:
            data_completeness += 1

    composite = round(weighted_sum / total_weight, 1) if total_weight > 0 else 50

    # Determine recommendation
    recommendation = "HOLD"
    for threshold, label in RECOMMENDATION_THRESHOLDS:
        if composite >= threshold:
            recommendation = label
            break

    # Confidence based on data availability
    confidence_pct = round((data_completeness / len(dimensions)) * 100)
    if confidence_pct >= 80:
        confidence = "HIGH"
    elif confidence_pct >= 60:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Build human-readable summary
    summary_parts = []
    top_dim = max(dimensions.items(), key=lambda x: x[1]["score"])
    weak_dim = min(dimensions.items(), key=lambda x: x[1]["score"])
    summary_parts.append(f"Composite Score: {composite}/100 -> {recommendation}")
    summary_parts.append(f"Strongest dimension: {top_dim[0].title()} ({top_dim[1]['score']}/100)")
    summary_parts.append(f"Weakest dimension: {weak_dim[0].title()} ({weak_dim[1]['score']}/100)")
    summary_parts.append(f"Data confidence: {confidence} ({confidence_pct}% dimensions with data)")

    # Key flags
    gov_details = dimensions["governance"]["details"]
    if gov_details.get("pledge_flag"):
        summary_parts.append(f"GOVERNANCE ALERT: {gov_details['pledge_flag']}")

    val_details = dimensions["valuation"]["details"]
    dcf_upside = val_details.get("dcf_upside_pct", 0)
    if dcf_upside > 30:
        summary_parts.append(f"VALUATION: Significant DCF upside ({dcf_upside:+.1f}%)")
    elif dcf_upside < -20:
        summary_parts.append(f"VALUATION: DCF suggests overvaluation ({dcf_upside:+.1f}%)")

    qual_details = dimensions["quality"]["details"]
    roce = qual_details.get("roce_pct", 0)
    if roce > 20:
        summary_parts.append(f"QUALITY: Excellent ROCE ({roce:.1f}%)")
    elif roce < 10 and roce > 0:
        summary_parts.append(f"QUALITY: Weak ROCE ({roce:.1f}%)")

    return {
        "composite_score": composite,
        "recommendation": recommendation,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "dimension_scores": {
            name: {
                "score": dim["score"],
                "weight": DIMENSION_WEIGHTS[name],
                "weighted_contribution": round(dim["score"] * DIMENSION_WEIGHTS[name] / 100, 1),
                "details": dim["details"],
                "sub_scores": dim.get("sub_scores", {}),
            }
            for name, dim in dimensions.items()
        },
        "summary": "\n".join(summary_parts),
        "thresholds": {label: threshold for threshold, label in RECOMMENDATION_THRESHOLDS},
    }


def format_score_for_prompt(score_result: dict) -> str:
    """Format the scoring result into a structured text block for the LLM prompt.

    This gives the LLM a clear, data-driven basis for its recommendation
    rather than relying on pure judgment.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("QUANTITATIVE INVESTMENT SCORING FRAMEWORK")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"COMPOSITE SCORE: {score_result['composite_score']}/100")
    lines.append(f"SYSTEM RECOMMENDATION: {score_result['recommendation']}")
    lines.append(f"DATA CONFIDENCE: {score_result['confidence']} ({score_result['confidence_pct']}%)")
    lines.append("")
    lines.append("Thresholds: STRONG BUY >= 75 | BUY >= 60 | HOLD >= 40 | SELL >= 25 | STRONG SELL < 25")
    lines.append("")
    lines.append("-" * 60)
    lines.append("DIMENSION BREAKDOWN:")
    lines.append("-" * 60)

    for dim_name, dim_data in score_result["dimension_scores"].items():
        weight = dim_data["weight"]
        score = dim_data["score"]
        contrib = dim_data["weighted_contribution"]
        lines.append(f"\n{dim_name.upper()} (Weight: {weight}%, Score: {score}/100, Contribution: {contrib} pts)")

        # Sub-scores
        for sub_name, sub_score in dim_data.get("sub_scores", {}).items():
            lines.append(f"  - {sub_name}: {sub_score}/100")

        # Key details
        details = dim_data.get("details", {})
        for key, val in details.items():
            if key.endswith("_sub_score"):
                continue  # already shown above
            if val in (0, 0.0, "N/A", ""):
                continue
            display_key = key.replace("_", " ").title()
            lines.append(f"  {display_key}: {val}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("SCORING SUMMARY:")
    lines.append("-" * 60)
    lines.append(score_result["summary"])
    lines.append("")
    lines.append("IMPORTANT: Your recommendation MUST align with the quantitative score above.")
    lines.append("If you disagree with the score, explain WHY with specific data points.")
    lines.append("Do not override the quantitative recommendation without clear justification.")
    lines.append("=" * 60)

    return "\n".join(lines)
