"""Prompt templates for LLM-generated USP commentary — per-stock and cross-portfolio."""

from __future__ import annotations

import json
from typing import Any


USP_COMMENTARY_SYSTEM = """You are a senior equity research analyst at a top Indian brokerage.
Your job: translate raw USP indicator scores into actionable investment commentary.

Rules:
- Every sentence MUST reference a specific number from the data provided.
- Sentence 1: What the score means for THIS company specifically.
- Sentence 2: Why an investor should care (investment implication).
- Sentence 3 (optional): Key risk or catalyst to watch.
- No generic language. "Strong fundamentals" is banned. Use "ROE improved 5pp to 18%" instead.
- Keep each dimension to 2-3 sentences max. Be dense, not verbose.
- Use Indian market context (Nifty, SEBI, PLI, FII/DII, promoter holding norms)."""


USP_COMMENTARY_TEMPLATE = """Analyze the USP scores for {ticker} ({sector} sector) and write 2-3 sentence investment commentary for each dimension.

## Raw USP Data:

### 1. Geopolitical Risk (Score: {geo_score}/100 — {geo_level})
- Trade Risk: {trade_risk}/100
- Policy Risk: {policy_risk}/100
- Commodity Risk: {commodity_risk}/100
- Event Risk: {event_risk}/100
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

## Output Format (respond in EXACTLY this JSON format):
{{
  "geopolitical": "2-3 sentence commentary...",
  "smart_money": "2-3 sentence commentary...",
  "regulatory": "2-3 sentence commentary...",
  "mgmt_credibility": "2-3 sentence commentary...",
  "promoter": "2-3 sentence commentary..."
}}"""


PORTFOLIO_INSIGHTS_SYSTEM = """You are a senior portfolio strategist at a top Indian brokerage.
You analyze collections of stocks that passed a screener and provide portfolio-level insights.
Be specific — use stock names, exact scores, and concrete comparisons. No generic statements."""


PORTFOLIO_INSIGHTS_TEMPLATE = """Given these USP profiles for {count} {category} stocks, provide 4 concise insights:

## Stock USP Profiles:
{stock_profiles}

## Respond in EXACTLY this format:
1. **Strongest Risk-Adjusted Profile:** [Which stock and why — cite specific scores]
2. **Best Smart Money Opportunity:** [Which stock has the biggest gap between improving fundamentals and low institutional positioning — cite the lag score and key metrics]
3. **Diversification Pairs:** [Which 2 stocks pair well together based on contrasting USP strengths — e.g., one strong on geopolitical, other strong on promoter conviction]
4. **Risk Flag:** [Which stock's USP profile contradicts its screener pass — cite the specific weak dimension and score]"""


def build_stock_commentary_prompt(ticker: str, usp_data: dict[str, Any], sector: str = "Unknown") -> tuple[str, str]:
    """Build system + user prompt for per-stock USP commentary.

    Args:
        ticker: Stock ticker (e.g., TORNTPHARM.NS)
        usp_data: {module_name: raw_data} from apply_usp_layer()
        sector: Stock sector

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

    from app.ui.usp_cards import _score_level

    user_prompt = USP_COMMENTARY_TEMPLATE.format(
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
        reg_score=reg.get("regulatory_score", "N/A"),
        reg_level=_score_level(reg.get("regulatory_score", 50), "regulatory"),
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
            f"Regulatory={reg.get('regulatory_score', 'N/A')}, "
            f"Mgmt={mgmt.get('credibility_score', 'N/A')}, "
            f"Promoter={prom.get('promoter_score', 'N/A')}"
        )
        profiles.append(profile)

    user_prompt = PORTFOLIO_INSIGHTS_TEMPLATE.format(
        count=len(per_stock_usp),
        category=category,
        stock_profiles="\n".join(profiles),
    )

    return PORTFOLIO_INSIGHTS_SYSTEM, user_prompt
