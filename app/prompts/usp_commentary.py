"""Prompt templates for LLM-generated USP commentary — per-stock and cross-portfolio."""

from __future__ import annotations

import json
from typing import Any


USP_COMMENTARY_SYSTEM = """You are a senior equity research analyst at a top Indian brokerage (IIFL, Motilal Oswal, Kotak Institutional).
Your job: translate raw USP indicator scores into a rich, actionable investment note.

Rules:
- Every claim MUST reference a specific number from the data provided.
- Use Indian market context: Nifty, SEBI, PLI, FII/DII flows, promoter holding norms, RBI policy, monsoon impact.
- Write like a research note, not a textbook. Be opinionated but evidence-based.
- "Strong fundamentals" is BANNED. Use "ROE improved 5pp to 18%, above sector median of 14%" instead.
- Reference recent news headlines when provided — connect USP scores to real-world events.
- Be specific about sector dynamics (e.g., "PLI tailwind for electronics" not "government support").
- Use concrete comparisons: "promoter holding at 62% vs sector avg 45%" not "high promoter holding"."""


USP_RICH_COMMENTARY_TEMPLATE = """Write a rich investment analyst note for {ticker} ({sector} sector).

## Raw USP Data:

### 1. Geopolitical Risk (Score: {geo_score}/100 — {geo_level})
- Trade Risk: {trade_risk}/100 | Policy Risk: {policy_risk}/100
- Commodity Risk: {commodity_risk}/100 | Event Risk: {event_risk}/100
- Risk Level: {risk_level}

### 2. Smart Money Lag (Score: {sm_score}/100 — {sm_level})
- Fundamental Improvement: {fundamental_improvement}/100
- Institutional Flow: {institutional_flow}/100
- Lag (gap): {smart_money_lag}
{sm_details}

### 3. Regulatory (Score: {reg_score}/100 — {reg_level})
- Net Signal: {reg_signal}
- Tailwind Policies: {tailwind_policies}
- Headwind Policies: {headwind_policies}

### 4. Management Credibility (Score: {mgmt_score}/100 — {mgmt_level})
- Method: {mgmt_method}
- Reasoning: {mgmt_reasoning}

### 5. Promoter Behavior (Score: {promoter_score}/100 — {promoter_level})
{promoter_details}

{recent_news_section}

## Output Format — respond in EXACTLY this JSON:
{{
  "investment_thesis": "A 3-4 sentence synthesis connecting ALL 5 USP dimensions into one cohesive investment narrative. What story do these scores tell together? End with a clear directional view (accumulate/hold/avoid).",
  "key_catalysts": ["Catalyst 1 with specific data point", "Catalyst 2 with specific data point", "Catalyst 3 with specific data point"],
  "key_risks": ["Risk 1 with specific score reference", "Risk 2 with specific score reference"],
  "what_to_watch": "One specific, actionable trigger the investor should monitor — e.g., 'Watch for FII holding crossing 15% in next quarter shareholding data' or 'Monitor if PLI disbursement timeline holds per MeitY notification'",
  "dimension_notes": {{
    "geopolitical": "2-3 sentences: what THIS geo score means for THIS company specifically",
    "smart_money": "2-3 sentences: interpret the lag score — is this an opportunity or a warning?",
    "regulatory": "2-3 sentences: which specific policies matter and their likely timeline/impact",
    "mgmt_credibility": "2-3 sentences: what the credibility assessment implies for execution risk",
    "promoter": "2-3 sentences: what promoter behavior signals about insider conviction"
  }}
}}"""


PORTFOLIO_INSIGHTS_SYSTEM = """You are a senior portfolio strategist at a top Indian brokerage.
You analyze collections of stocks that passed a screener and provide portfolio-level insights.
Be specific — use stock names, exact scores, and concrete comparisons. No generic statements.
Write with conviction — this is a strategy note for the desk, not a compliance document."""


PORTFOLIO_INSIGHTS_TEMPLATE = """Given these USP profiles for {count} {category} stocks, provide portfolio-level analysis.

## Stock USP Profiles:
{stock_profiles}

## Respond in EXACTLY this markdown format:

### Strongest Risk-Adjusted Profile
[Which stock has the best overall USP profile and why — cite at least 3 specific dimension scores. Compare it to the weakest profile in the set.]

### Best Smart Money Opportunity
[Which stock has the biggest gap between improving fundamentals and low institutional positioning? Cite the lag score, key fundamental metrics, and current institutional holding. Explain why institutions haven't caught on yet.]

### Portfolio Construction
[How should an investor weight these stocks? Which 2-3 stocks pair well for diversification based on contrasting USP strengths? Which stocks have correlated risks that shouldn't be over-concentrated?]

### Risk Flag
[Which stock's USP profile most contradicts its screener pass? Cite the specific weak dimension, score, and what could go wrong. Be direct — "TICKER should be on a shorter leash because..."]"""


def build_stock_commentary_prompt(
    ticker: str,
    usp_data: dict[str, Any],
    sector: str = "Unknown",
    recent_news: list[str] | None = None,
) -> tuple[str, str]:
    """Build system + user prompt for per-stock USP commentary.

    Args:
        ticker: Stock ticker (e.g., TORNTPHARM.NS)
        usp_data: {module_name: raw_data} from apply_usp_layer()
        sector: Stock sector
        recent_news: Optional list of recent news headlines for context

    Returns:
        (system_prompt, user_prompt)
    """
    geo = usp_data.get("geopolitical", {})
    sm = usp_data.get("smart_money", {})
    reg = usp_data.get("regulatory", {})
    mgmt = usp_data.get("mgmt_credibility", {})
    prom = usp_data.get("promoter", {})

    # Smart money details
    sm_parts = []
    imp = sm.get("improvement_details", {})
    flow = sm.get("flow_details", {})
    if "roe_improvement" in imp:
        sm_parts.append(f"- ROE Change: {imp['roe_improvement']:+.2f}pp")
    if "margin_expansion" in imp:
        sm_parts.append(f"- Margin Change: {imp['margin_expansion']:+.2f}pp")
    if "revenue_growth_2yr" in imp:
        sm_parts.append(f"- Revenue Growth (2yr): {imp['revenue_growth_2yr']:+.1f}%")
    if "debt_change_pct" in imp:
        sm_parts.append(f"- Debt Change: {imp['debt_change_pct']:+.1f}%")
    if "institutional_pct" in flow:
        sm_parts.append(f"- Institutional Holding: {flow['institutional_pct']:.1f}%")

    # Promoter details
    prom_parts = []
    if "promoter_holding" in prom:
        prom_parts.append(f"- Promoter Holding: {prom['promoter_holding']:.1f}%")
    if "pledge_pct" in prom:
        prom_parts.append(f"- Pledge: {prom['pledge_pct']:.1f}%")
    if "buying_signal" in prom:
        prom_parts.append(f"- Insider Activity: {prom['buying_signal']}")
    if "rpt_count" in prom:
        prom_parts.append(f"- Related Party Txns: {prom['rpt_count']}")

    # Recent news section
    news_section = ""
    if recent_news:
        news_lines = "\n".join(f"- {h}" for h in recent_news[:8])
        news_section = f"""### Recent News Headlines (for context — connect scores to real events):
{news_lines}"""

    from app.ui.usp_cards import _score_level

    user_prompt = USP_RICH_COMMENTARY_TEMPLATE.format(
        ticker=ticker,
        sector=sector,
        geo_score=geo.get("geo_score", "N/A"),
        geo_level=_score_level(geo.get("geo_score", 50), "geopolitical"),
        trade_risk=geo.get("trade_risk", "N/A"),
        policy_risk=geo.get("policy_risk", "N/A"),
        commodity_risk=geo.get("commodity_risk", "N/A"),
        event_risk=geo.get("event_risk", "N/A"),
        risk_level=geo.get("risk_level", "Unknown"),
        sm_score=min(100, max(0, 50 + sm.get("smart_money_lag", 0))),
        sm_level=_score_level(min(100, max(0, 50 + sm.get("smart_money_lag", 0))), "smart_money"),
        fundamental_improvement=round(sm.get("fundamental_improvement", 0)),
        institutional_flow=round(sm.get("institutional_flow", 0)),
        smart_money_lag=round(sm.get("smart_money_lag", 0), 1),
        sm_details="\n".join(sm_parts) if sm_parts else "- No detailed metrics available",
        reg_score=reg.get("regulatory_score", reg.get("reg_score", "N/A")),
        reg_level=_score_level(reg.get("regulatory_score", reg.get("reg_score", 50)), "regulatory"),
        reg_signal=reg.get("net_signal", "Unknown"),
        tailwind_policies=", ".join(reg.get("tailwind_policies", [])) or "None",
        headwind_policies=", ".join(reg.get("headwind_policies", [])) or "None",
        mgmt_score=mgmt.get("credibility_score", "N/A"),
        mgmt_level=_score_level(mgmt.get("credibility_score", 50), "mgmt_credibility"),
        mgmt_method=mgmt.get("method", "Unknown"),
        mgmt_reasoning=mgmt.get("reasoning", "No data"),
        promoter_score=prom.get("promoter_score", "N/A"),
        promoter_level=_score_level(prom.get("promoter_score", 50), "promoter"),
        promoter_details="\n".join(prom_parts) if prom_parts else "- No promoter data available",
        recent_news_section=news_section,
    )

    return USP_COMMENTARY_SYSTEM, user_prompt


def build_portfolio_insights_prompt(
    per_stock_usp: dict[str, dict],
    category: str = "Momentum",
) -> tuple[str, str]:
    """Build system + user prompt for cross-stock portfolio insights.

    Args:
        per_stock_usp: {ticker: {module: raw_data}} from apply_usp_layer()
        category: "Momentum" or "ValueBottom"

    Returns:
        (system_prompt, user_prompt)
    """
    from app.ui.usp_cards import _score_level

    profiles = []
    for ticker, modules in sorted(per_stock_usp.items()):
        if ticker.startswith("_"):
            continue
        geo = modules.get("geopolitical", {})
        sm = modules.get("smart_money", {})
        reg = modules.get("regulatory", {})
        mgmt = modules.get("mgmt_credibility", {})
        prom = modules.get("promoter", {})

        sm_score = min(100, max(0, 50 + sm.get("smart_money_lag", 0)))
        imp = sm.get("improvement_details", {})
        flow = sm.get("flow_details", {})

        profile = (
            f"**{ticker}** ({modules.get('geopolitical', {}).get('sector', 'Unknown')}): "
            f"Geo={geo.get('geo_score', 'N/A')}, "
            f"SmartMoney={sm_score} (lag={sm.get('smart_money_lag', 0):.0f}, "
            f"inst={flow.get('institutional_pct', 0):.1f}%), "
            f"Regulatory={reg.get('regulatory_score', reg.get('reg_score', 'N/A'))}, "
            f"Mgmt={mgmt.get('credibility_score', 'N/A')}, "
            f"Promoter={prom.get('promoter_score', 'N/A')}"
        )
        profiles.append(profile)

    user_prompt = PORTFOLIO_INSIGHTS_TEMPLATE.format(
        count=len(profiles),
        category=category,
        stock_profiles="\n".join(profiles),
    )

    return PORTFOLIO_INSIGHTS_SYSTEM, user_prompt
