"""USP Card Renderer — transforms raw USP module output into Streamlit UI components.

Provides rendering functions:
1. render_usp_heatmap() — compact colored table for quick comparison
2. render_usp_card() — expandable per-stock card with factor-by-factor evidence
3. render_usp_radar() — Plotly radar chart with peer overlay
4. render_analyst_panel() — rich right-side panel with thesis, catalysts, risks
5. render_contradiction_alerts() — red banners for category/USP conflicts
6. render_portfolio_insights() — cross-stock strategy note
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ── Color/level helpers ──────────────────────────────────────


def _score_color(score: float) -> str:
    """Map a 0-100 score to a color hex (dark theme palette)."""
    if score >= 70:
        return "#00D4AA"  # accent green
    elif score >= 50:
        return "#FFA726"  # accent amber
    elif score >= 30:
        return "#FF8C42"  # warm orange
    else:
        return "#FF4757"  # accent red


def _score_emoji(score: float) -> str:
    if score >= 70:
        return "🟢"
    elif score >= 50:
        return "🟡"
    elif score >= 30:
        return "🟠"
    else:
        return "🔴"


def _score_level(score: float, dimension: str) -> str:
    """Human-readable level for a score in a given dimension."""
    levels = {
        "geopolitical": {70: "LOW RISK", 50: "MODERATE RISK", 30: "HIGH RISK", 0: "VERY HIGH RISK"},
        "smart_money": {70: "STRONG FLOW", 50: "DIVERGENCE", 30: "WEAK FLOW", 0: "NO INSTITUTIONAL INTEREST"},
        "regulatory": {70: "STRONG TAILWIND", 50: "MIXED", 30: "HEADWIND", 0: "STRONG HEADWIND"},
        "mgmt_credibility": {70: "HIGH CREDIBILITY", 50: "MODERATE", 30: "LOW CREDIBILITY", 0: "POOR TRACK RECORD"},
        "promoter": {70: "STABLE", 50: "WATCH", 30: "CONCERN", 0: "RED FLAG"},
    }
    dim_levels = levels.get(dimension, {70: "GOOD", 50: "MODERATE", 30: "WEAK", 0: "POOR"})
    for threshold in sorted(dim_levels.keys(), reverse=True):
        if score >= threshold:
            return dim_levels[threshold]
    return "UNKNOWN"


# ── Transform USP raw data to card format ────────────────────


def _transform_geopolitical(data: dict) -> dict[str, Any]:
    """Transform geopolitical module output to card format."""
    score = data.get("geo_score", 50)
    return {
        "score": score,
        "level": _score_level(score, "geopolitical"),
        "color": _score_color(score),
        "emoji": _score_emoji(score),
        "factors": [
            {"name": "Trade Risk", "value": f"Score: {data.get('trade_risk', 'N/A')}/100", "status": "positive" if data.get("trade_risk", 0) >= 60 else "negative"},
            {"name": "Policy Risk", "value": f"Score: {data.get('policy_risk', 'N/A')}/100", "status": "positive" if data.get("policy_risk", 0) >= 60 else "negative"},
            {"name": "Commodity Risk", "value": f"Score: {data.get('commodity_risk', 'N/A')}/100", "status": "positive" if data.get("commodity_risk", 0) >= 60 else "negative"},
            {"name": "Event Risk", "value": f"Score: {data.get('event_risk', 'N/A')}/100", "status": "positive" if data.get("event_risk", 0) >= 60 else "negative"},
        ],
        "summary": f"{data.get('risk_level', 'Unknown')} — Overall geopolitical resilience score {score}/100",
    }


def _transform_smart_money(data: dict) -> dict[str, Any]:
    """Transform smart money lag module output to card format."""
    lag = data.get("smart_money_lag", 0)
    imp = data.get("fundamental_improvement", 50)
    flow = data.get("institutional_flow", 50)
    score = min(100, max(0, 50 + lag))

    imp_details = data.get("improvement_details", {})
    flow_details = data.get("flow_details", {})

    factors = []
    if "roe_improvement" in imp_details:
        val = imp_details["roe_improvement"]
        factors.append({"name": "ROE Change", "value": f"{val:+.2f}pp", "status": "positive" if val > 0 else "negative"})
    if "margin_expansion" in imp_details:
        val = imp_details["margin_expansion"]
        factors.append({"name": "Margin Change", "value": f"{val:+.2f}pp", "status": "positive" if val > 0 else "negative"})
    if "revenue_growth_2yr" in imp_details:
        val = imp_details["revenue_growth_2yr"]
        factors.append({"name": "Revenue Growth (2yr)", "value": f"{val:+.1f}%", "status": "positive" if val > 10 else "neutral"})
    if "institutional_pct" in flow_details:
        val = flow_details["institutional_pct"]
        factors.append({"name": "Institutional Holding", "value": f"{val:.1f}%", "status": "neutral"})

    if lag > 20:
        interpretation = "Fundamentals improving but institutions haven't caught up — early opportunity"
    elif lag > 0:
        interpretation = "Slight fundamental lead over institutional flows"
    elif lag > -20:
        interpretation = "Institutions and fundamentals roughly aligned"
    else:
        interpretation = "Institutions have priced in more than fundamentals warrant — caution"

    return {
        "score": round(score),
        "level": _score_level(score, "smart_money"),
        "color": _score_color(score),
        "emoji": _score_emoji(score),
        "factors": factors,
        "fundamental_score": round(imp),
        "flow_score": round(flow),
        "lag": round(lag, 1),
        "summary": interpretation,
    }


def _transform_regulatory(data: dict) -> dict[str, Any]:
    """Transform regulatory module output to card format."""
    score = data.get("reg_score", 50)
    policies = data.get("policies_found", [])

    factors = []
    for policy in policies[:5]:
        signal = policy.get("signal", "neutral")
        status = "tailwind" if signal == "tailwind" else "headwind" if signal == "headwind" else "neutral"
        factors.append({
            "name": policy.get("policy_name", "Policy"),
            "value": policy.get("description", ""),
            "status": status,
        })

    if not factors:
        factors.append({"name": "No policy signals", "value": "No recent policy announcements found", "status": "neutral"})

    return {
        "score": score,
        "level": _score_level(score, "regulatory"),
        "color": _score_color(score),
        "emoji": _score_emoji(score),
        "factors": factors,
        "summary": f"Regulatory environment score: {score}/100",
    }


def _transform_mgmt_credibility(data: dict) -> dict[str, Any]:
    """Transform management credibility module output to card format."""
    score = data.get("credibility_score", 50)
    reasoning = data.get("reasoning", "")
    examples = data.get("examples", [])

    factors = []
    for ex in examples[:4]:
        factors.append({"name": "Track Record", "value": ex, "status": "positive" if score >= 60 else "negative"})

    if not factors:
        factors.append({"name": "Credibility", "value": reasoning or "Insufficient data for assessment", "status": "neutral"})

    return {
        "score": score,
        "level": _score_level(score, "mgmt_credibility"),
        "color": _score_color(score),
        "emoji": _score_emoji(score),
        "factors": factors,
        "summary": reasoning or f"Management credibility score: {score}/100",
    }


def _transform_promoter(data: dict) -> dict[str, Any]:
    """Transform promoter anomaly module output to card format."""
    score = data.get("promoter_score", 50)

    factors = []
    if "promoter_holding" in data:
        val = data["promoter_holding"]
        status = "positive" if val >= 50 else "negative" if val < 30 else "neutral"
        factors.append({"name": "Promoter Holding", "value": f"{val:.1f}%", "status": status})

    if "pledge_pct" in data:
        val = data["pledge_pct"]
        status = "positive" if val < 5 else "negative" if val > 20 else "neutral"
        factors.append({"name": "Pledge Status", "value": f"{val:.1f}% pledged", "status": status})

    if "buying_signal" in data:
        signal = data["buying_signal"]
        factors.append({"name": "Insider Activity", "value": signal, "status": "positive" if "buy" in signal.lower() else "neutral"})

    if "rpt_count" in data:
        val = data["rpt_count"]
        status = "positive" if val < 3 else "negative" if val > 8 else "neutral"
        factors.append({"name": "Related Party Txns", "value": f"{val} in last 6 months", "status": status})

    if not factors:
        factors.append({"name": "Promoter Data", "value": "Insufficient data", "status": "neutral"})

    return {
        "score": score,
        "level": _score_level(score, "promoter"),
        "color": _score_color(score),
        "emoji": _score_emoji(score),
        "factors": factors,
        "summary": f"Promoter behavior: {_score_level(score, 'promoter')}",
    }


# Module transformer registry
_TRANSFORMERS = {
    "geopolitical": _transform_geopolitical,
    "smart_money": _transform_smart_money,
    "regulatory": _transform_regulatory,
    "mgmt_credibility": _transform_mgmt_credibility,
    "promoter": _transform_promoter,
}


def transform_usp_data(usp_raw: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Transform raw USP module output into structured card data."""
    result: dict[str, dict[str, Any]] = {}

    for ticker, modules in usp_raw.items():
        if ticker.startswith("_"):
            result[ticker] = modules
            continue

        result[ticker] = {}
        composite_scores = []

        for module_name, raw_data in modules.items():
            if module_name.startswith("_"):
                result[ticker][module_name] = raw_data
                continue
            transformer = _TRANSFORMERS.get(module_name)
            if transformer:
                card = transformer(raw_data)
                result[ticker][module_name] = card
                composite_scores.append(card["score"])

        if composite_scores:
            result[ticker]["_composite"] = round(sum(composite_scores) / len(composite_scores))
        else:
            result[ticker]["_composite"] = 0

    return result


# ── Streamlit Rendering Functions ────────────────────────────


def render_usp_heatmap(usp_cards: dict[str, dict[str, Any]]) -> None:
    """Render the compact heatmap table at the top of Tab 2."""
    if not usp_cards:
        st.info("No USP data available yet. Run the screener first.")
        return

    dimensions = ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]
    dim_labels = {"geopolitical": "Geo", "smart_money": "Smart$", "regulatory": "Reg", "mgmt_credibility": "Mgmt", "promoter": "Promoter"}

    rows = []
    for ticker, modules in sorted(usp_cards.items(), key=lambda x: x[1].get("_composite", 0) if isinstance(x[1], dict) else 0, reverse=True):
        if ticker.startswith("_"):
            continue
        row = {"Stock": ticker.replace(".NS", "")}
        for dim in dimensions:
            card = modules.get(dim, {})
            score = card.get("score", "-")
            emoji = card.get("emoji", "⚪")
            row[dim_labels[dim]] = f"{emoji} {score}" if isinstance(score, (int, float)) else "⚪ -"
        row["Avg"] = modules.get("_composite", 0)
        rows.append(row)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_usp_card(ticker: str, usp_data: dict[str, Any]) -> None:
    """Render the LEFT side: compact USP scores + factors for a single stock."""
    composite = usp_data.get("_composite", 0)
    emoji = _score_emoji(composite)

    with st.expander(f"{emoji} {ticker.replace('.NS', '')} — USP Score: {composite}/100", expanded=False):
        dimensions = ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]
        dim_titles = {
            "geopolitical": "Geopolitical Risk",
            "smart_money": "Smart Money Lag",
            "regulatory": "Regulatory Tailwind/Headwind",
            "mgmt_credibility": "Management Credibility",
            "promoter": "Promoter Behavior",
        }

        # Get dimension notes from commentary
        commentary = usp_data.get("_commentary", {})
        dim_notes = commentary.get("dimension_notes", {})
        # Backward compat: if no dimension_notes, check top-level keys
        if not dim_notes:
            for dim in dimensions:
                if dim in commentary and isinstance(commentary[dim], str):
                    dim_notes[dim] = commentary[dim]

        for dim in dimensions:
            card = usp_data.get(dim)
            if not card:
                st.markdown(f"**{dim_titles[dim]}:** ⚪ Data unavailable")
                continue

            score = card.get("score", 0)
            level = card.get("level", "")
            color = card.get("color", "#8892A0")
            emoji_dim = card.get("emoji", "⚪")

            # Compact header with colored score badge
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">'
                f'<span style="font-weight:600; font-size:0.95em;">{dim_titles[dim]}</span>'
                f'<span style="background:{color}; color:#0E1117; padding:2px 8px; border-radius:10px; '
                f'font-size:0.8em; font-weight:700;">{score}/100</span>'
                f'<span style="color:{color}; font-size:0.8em;">{level}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Render factors as compact pills
            for factor in card.get("factors", []):
                status = factor.get("status", "neutral")
                icon = "✅" if status == "positive" else "🔴" if status in ("negative", "headwind") else "🟢" if status == "tailwind" else "🟡"
                st.markdown(f"- {icon} **{factor['name']}:** {factor['value']}")

            # Dimension-level LLM note (if available)
            note = dim_notes.get(dim)
            if note:
                st.markdown(
                    f'<div style="background:#1a1f2e; border-left:3px solid {color}; '
                    f'padding:8px 12px; border-radius:4px; margin:6px 0; '
                    f'font-size:0.85em; color:#C0C0C0; line-height:1.5;">'
                    f'{note}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()


def render_analyst_panel(ticker: str, usp_data: dict[str, Any]) -> None:
    """Render the RIGHT side: rich Analyst View panel with thesis, catalysts, risks.

    This uses the full real estate below the radar chart.
    """
    commentary = usp_data.get("_commentary", {})
    if not commentary or "_raw" in commentary:
        return

    # ── Investment Thesis ──
    thesis = commentary.get("investment_thesis")
    if thesis:
        st.markdown(
            f'<div style="background:linear-gradient(135deg, #1a2332 0%, #0E1117 100%); '
            f'border:1px solid #00D4AA; border-radius:8px; padding:14px 16px; margin-bottom:12px;">'
            f'<div style="color:#00D4AA; font-size:0.78em; font-weight:700; '
            f'text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Investment Thesis</div>'
            f'<div style="color:#E8E8E8; font-size:0.9em; line-height:1.6;">{thesis}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Key Catalysts ──
    catalysts = commentary.get("key_catalysts", [])
    if catalysts:
        catalyst_html = "".join(
            f'<div style="display:flex; gap:8px; margin-bottom:6px;">'
            f'<span style="color:#00D4AA; font-size:0.85em;">▲</span>'
            f'<span style="color:#D0D0D0; font-size:0.85em;">{c}</span></div>'
            for c in catalysts
        )
        st.markdown(
            f'<div style="background:#111820; border-radius:6px; padding:10px 14px; margin-bottom:10px;">'
            f'<div style="color:#00D4AA; font-size:0.78em; font-weight:700; '
            f'text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Key Catalysts</div>'
            f'{catalyst_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Key Risks ──
    risks = commentary.get("key_risks", [])
    if risks:
        risk_html = "".join(
            f'<div style="display:flex; gap:8px; margin-bottom:6px;">'
            f'<span style="color:#FF4757; font-size:0.85em;">▼</span>'
            f'<span style="color:#D0D0D0; font-size:0.85em;">{r}</span></div>'
            for r in risks
        )
        st.markdown(
            f'<div style="background:#111820; border-radius:6px; padding:10px 14px; margin-bottom:10px;">'
            f'<div style="color:#FF4757; font-size:0.78em; font-weight:700; '
            f'text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Key Risks</div>'
            f'{risk_html}</div>',
            unsafe_allow_html=True,
        )

    # ── What to Watch ──
    watch = commentary.get("what_to_watch")
    if watch:
        st.markdown(
            f'<div style="background:#1a1a2e; border:1px solid #B388FF; border-radius:6px; '
            f'padding:10px 14px; margin-bottom:10px;">'
            f'<div style="color:#B388FF; font-size:0.78em; font-weight:700; '
            f'text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">What to Watch</div>'
            f'<div style="color:#D0D0D0; font-size:0.85em; line-height:1.5;">👁 {watch}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_usp_radar(ticker: str, usp_data: dict[str, Any], category_avg: dict[str, float] | None = None) -> None:
    """Render a Plotly radar chart for USP dimensions with optional peer overlay."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed — radar chart unavailable")
        return

    dimensions = ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]
    labels = ["Geopolitical", "Smart Money", "Regulatory", "Management", "Promoter"]

    values = [usp_data.get(dim, {}).get("score", 0) for dim in dimensions]
    values.append(values[0])
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(0, 212, 170, 0.15)",
        line=dict(color="#00D4AA", width=2),
        name=ticker.replace(".NS", ""),
    ))

    if category_avg:
        avg_values = [category_avg.get(dim, 50) for dim in dimensions]
        avg_values.append(avg_values[0])
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=labels_closed,
            fill="none",
            line=dict(color="#8892A0", width=1, dash="dot"),
            name="Category Avg",
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0E1117",
        polar=dict(
            bgcolor="#0E1117",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2D3748", color="#8892A0"),
            angularaxis=dict(gridcolor="#2D3748", color="#E0E0E0"),
        ),
        font=dict(color="#E0E0E0", size=10),
        showlegend=True,
        height=280,
        margin=dict(l=30, r=30, t=20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892A0", size=9)),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_contradiction_alerts(
    category_results: dict[str, list[dict]],
    usp_cards: dict[str, dict[str, Any]],
) -> None:
    """Render red banners when USP scores contradict category logic."""
    alerts: list[str] = []

    for cat_name, stocks in category_results.items():
        for stock in stocks:
            ticker = stock["ticker"]
            if ticker not in usp_cards:
                continue

            usp = usp_cards[ticker]

            if cat_name == "B" and usp.get("promoter", {}).get("score", 100) < 35:
                alerts.append(
                    f"**{ticker}** passed Cat B (Momentum) but Promoter Score "
                    f"{usp['promoter']['score']}/100 — governance concern. Debate agents will address this."
                )

            if cat_name == "C" and usp.get("mgmt_credibility", {}).get("score", 100) < 30:
                alerts.append(
                    f"**{ticker}** passed Cat C (Value Bottoms) but Management Credibility "
                    f"{usp['mgmt_credibility']['score']}/100 — turnaround may not materialize."
                )

            if usp.get("geopolitical", {}).get("score", 100) < 25:
                alerts.append(
                    f"**{ticker}** has Geopolitical Risk Score {usp['geopolitical']['score']}/100 — "
                    f"very high external risk exposure."
                )

    if alerts:
        for alert in alerts:
            st.warning(f"CONTRADICTION: {alert}")


def render_portfolio_insights(insights_text: str) -> None:
    """Render cross-stock portfolio-level USP insights."""
    if not insights_text:
        return

    st.markdown("### Portfolio Strategy Note")
    st.markdown(
        f'<div style="background:linear-gradient(135deg, #1a1f2e 0%, #0E1117 100%); '
        f'border:1px solid #00D4AA; border-radius:8px; padding:18px 22px; '
        f'margin-bottom:16px; line-height:1.7; color:#E0E0E0;">'
        f'{insights_text}</div>',
        unsafe_allow_html=True,
    )
