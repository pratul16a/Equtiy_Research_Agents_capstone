"""Streamlit UI — AI-Powered Indian Equity Research Analyst."""

from __future__ import annotations

import json
import os
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaLens — Indian Equity Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark Terminal Theme ─────────────────────────
st.markdown("""
<style>
    /* ── Global dark overrides ─────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #0E1117; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stMarkdown, .stText, p, span, div { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #1E2A3A; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1A1F2E; border-radius: 8px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #9AA2B0; border-radius: 6px; font-size: 14px; font-we
            ight: 500; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background-color: #232A3B !important; color: #00D4AA !important; font-weight: 700; }
    hr { border-color: #1E2A3A !important; }

    /* DataFrames */
    [data-testid="stDataFrame"] th { background-color: #1A1F2E !important; color: #A0A8B4 !important; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; padding: 10px 12px !important; }
    [data-testid="stDataFrame"] td { background-color: #0E1117 !important; color: #E0E0E0 !important; border-bottom: 1px solid #1E2A3A !important; padding: 8px 12px !important; font-size: 13px; }

    /* Expanders */
    .streamlit-expanderHeader { background-color: #1A1F2E !important; border: 1px solid #2D3748; border-radius: 8px; color: #E8ECF1 !important; padding: 12px 16px !important; }
    .streamlit-expanderContent { background-color: #141922 !important; border: 1px solid #2D3748; border-top: none; padding: 12px 16px !important; }
    details[data-testid="stExpander"] > summary { background-color: #1A1F2E !important; border-radius: 8px; padding: 12px 16px !important; }

    /* Buttons */
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #00D4AA, #00B894) !important; color: #0E1117 !important; font-weight: 700; border: none !important; border-radius: 8px; }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #00E4BA, #00D4AA) !important; box-shadow: 0 0 20px rgba(0, 212, 170, 0.3); }
    .stButton > button { background: #1A1F2E !important; color: #E0E0E0 !important; border: 1px solid #2D3748 !important; border-radius: 8px; }

    /* Progress bars */
    .stProgress > div > div { background-color: #1A1F2E !important; border-radius: 4px; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }

    /* ── Component classes ──────────────────────── */

    /* Gradient header */
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #4DA6FF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em; margin-bottom: 0;
    }
    .sub-header { font-size: 0.95rem; color: #8892A0; margin-top: 4px; letter-spacing: 0.02em; }

    /* Glassmorphism metric card */
    .metric-card {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(45, 55, 72, 0.6);
        border-radius: 12px; padding: 20px;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.4);
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.1);
        transform: translateY(-2px);
    }
    .metric-card .mc-label {
        font-size: 12px; text-transform: uppercase;
        letter-spacing: 0.08em; color: #9AA2B0; margin-bottom: 8px;
    }
    .metric-card .mc-value { font-size: 1.5rem; font-weight: 700; color: #E8ECF1; line-height: 1.2; }
    .metric-card .mc-delta { font-size: 0.82rem; margin-top: 6px; }
    .mc-delta-up { color: #00D4AA; }
    .mc-delta-down { color: #FF4757; }

    /* Section hero */
    .section-hero {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.05), rgba(77, 166, 255, 0.05));
        border: 1px solid #1E2A3A; border-radius: 12px;
        padding: 24px 28px; margin-bottom: 24px;
    }
    .section-hero h3 { color: #E8ECF1; margin: 0 0 6px 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.01em; }
    .section-hero p { color: #9AA2B0; font-size: 0.9rem; margin: 0; line-height: 1.5; }

    /* Criteria */
    .criteria-group { background: rgba(26, 31, 46, 0.6); border: 1px solid #2D3748; border-radius: 10px; padding: 14px; margin-bottom: 10px; }
    .criteria-item { display: flex; align-items: center; gap: 6px; padding: 2px 0; font-size: 13px; color: #C0C8D4; }
    .tag-yours { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid rgba(0,212,170,0.3); }
    .tag-new { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; background: rgba(77,166,255,0.15); color: #4DA6FF; border: 1px solid rgba(77,166,255,0.3); }
    .tag-deferred { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; background: rgba(136,146,160,0.15); color: #8892A0; border: 1px solid rgba(136,146,160,0.3); }
    .signal-tag { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 4px; margin-bottom: 4px; letter-spacing: 0.03em; }

    /* Warnings */
    .pledge-warning { background: rgba(255, 167, 38, 0.1); border: 1px solid rgba(255, 167, 38, 0.3); border-radius: 8px; padding: 10px; margin: 5px 0; color: #FFA726; }
    .pledge-danger { background: rgba(255, 71, 87, 0.1); border: 1px solid rgba(255, 71, 87, 0.3); border-radius: 8px; padding: 10px; margin: 5px 0; color: #FF4757; }

    /* Regime banner */
    .regime-banner {
        background: rgba(26, 31, 46, 0.8); backdrop-filter: blur(10px);
        border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;
        display: flex; align-items: center; gap: 12px;
    }

    /* Debate card */
    .debate-card {
        background: rgba(26, 31, 46, 0.7); backdrop-filter: blur(8px);
        border: 1px solid #2D3748; border-radius: 12px;
        padding: 20px; margin-bottom: 16px;
    }

    /* Breadth badge */
    .breadth-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 10px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1rem; letter-spacing: 0.03em;
    }
</style>
""", unsafe_allow_html=True)

# ── AlphaLens Logo ───────────────────────────────────────────
st.markdown("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 100" width="300" height="100">
  <circle cx="45" cy="45" r="36" fill="none" stroke="#0D9488" stroke-width="3"/>
  <circle cx="45" cy="45" r="28" fill="none" stroke="#0D9488" stroke-width="1.5" opacity="0.5"/>
  <line x1="73" y1="67" x2="93" y2="87" stroke="#0D9488" stroke-width="3" stroke-linecap="round"/>
  <polyline points="25,57 35,49 45,53 55,37 65,41" fill="none" stroke="#14B8A6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="55" cy="37" r="3" fill="#14B8A6"/>
  <text x="110" y="40" style="font-family: Inter, -apple-system, sans-serif; font-size: 34px; font-weight: 800; fill: #E8ECF1; letter-spacing: -0.5px;">Alpha</text>
  <text x="110" y="70" style="font-family: Inter, -apple-system, sans-serif; font-size: 34px; font-weight: 400; fill: #14B8A6; letter-spacing: 1px;">Lens</text>
</svg>
<div style="color:#9AA2B0; font-size:0.82rem; letter-spacing:0.08em; margin-top:2px; font-style:italic;">
Screen like an Institution. Explain like an Analyst.</div>
""", unsafe_allow_html=True)


# ── Helper: Metric Card ──────────────────────────────────────

def render_metric_card(label: str, value, delta: str = "", delta_up: bool = True, accent: str = "#00D4AA"):
    """Render a glassmorphism metric card replacing st.metric()."""
    delta_html = ""
    if delta:
        cls = "mc-delta-up" if delta_up else "mc-delta-down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="mc-delta {cls}">{arrow} {delta}</div>'
    st.markdown(
        f'<div class="metric-card" style="border-left: 3px solid {accent};">'
        f'  <div class="mc-label">{label}</div>'
        f'  <div class="mc-value">{value}</div>'
        f'  {delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Helper: Dark Chart Layout ────────────────────────────────

DARK_COLORWAY = ["#00D4AA", "#4DA6FF", "#FF4757", "#FFA726", "#B388FF", "#00E5FF", "#FF6E7A"]

def dark_chart_layout(fig, height: int = 350, **overrides):
    """Apply dark terminal theme to a Plotly figure."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0E1117",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color="#E0E0E0", size=12),
        colorway=DARK_COLORWAY,
        xaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
        yaxis=dict(gridcolor="#1E2A3A", zerolinecolor="#1E2A3A"),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892A0")),
    )
    base.update(overrides)
    fig.update_layout(**base)
    return fig


# ── Criteria Data (Momentum & Value Bottom) ──────────────────

from app.tools.screener.gate_defs import MOMENTUM_GATES, VALUE_GATES, MOMENTUM_OVERALL, VALUE_OVERALL, TIER_CONFIG, TIER_ORDER
from app.tools.screener.category_screener import MOMENTUM_CRITERIA, VALUE_CRITERIA

_GATE_TYPE_COLORS = {"hard": "#FF4757", "soft": "#FFA726", "bonus": "#4DA6FF"}
_GATE_TYPE_LABELS = {"hard": "HARD", "soft": "SOFT", "bonus": "BONUS"}


def _build_gate_display(gates, criteria_registry):
    """Build gate-structured display data from gate defs + criteria metadata."""
    display = []
    for gate in gates:
        items = []
        for key in gate["criteria_keys"]:
            meta = criteria_registry.get(key, {"label": key})
            tag = "deferred" if key in gate.get("deferred_keys", []) else (
                "mandatory" if key in gate.get("mandatory_keys", []) else "active"
            )
            items.append((meta.get("label", key), tag, key))
        display.append({
            "gate_name": gate["name"],
            "gate_type": gate["gate_type"],
            "min_pass": gate["min_pass"],
            "total": len([k for k in gate["criteria_keys"] if k not in gate.get("deferred_keys", [])]),
            "items": items,
        })
    return display


MOM_GATE_DISPLAY = _build_gate_display(MOMENTUM_GATES, MOMENTUM_CRITERIA)
VAL_GATE_DISPLAY = _build_gate_display(VALUE_GATES, VALUE_CRITERIA)


def render_criteria_panel(gate_display: list[dict], accent_color: str = "#00D4AA"):
    """Render criteria panel organized by gates with type badges."""
    cols = st.columns(2)
    for i, gate in enumerate(gate_display):
        with cols[i % 2]:
            gtype = gate["gate_type"]
            gcolor = _GATE_TYPE_COLORS.get(gtype, "#8892A0")
            glabel = _GATE_TYPE_LABELS.get(gtype, gtype.upper())
            rule = f"need {gate['min_pass']}/{gate['total']}" if gate["min_pass"] > 0 else "bonus only"

            items_html = ""
            for text, tag, _key in gate["items"]:
                if tag == "deferred":
                    items_html += f'<div class="criteria-item" style="opacity:0.5; font-style:italic;"><span style="color:#8892A0; font-size:12px;">○</span> {text} <span class="tag-deferred">DEFER</span></div>'
                elif tag == "mandatory":
                    items_html += f'<div class="criteria-item"><span style="color:{accent_color}; font-size:12px;">&#10003;</span> {text} <span style="color:#FF4757; font-size:10px; font-weight:600;">MUST</span></div>'
                else:
                    items_html += f'<div class="criteria-item"><span style="color:{accent_color}; font-size:12px;">&#10003;</span> {text}</div>'

            st.markdown(
                f'<div class="criteria-group">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">'
                f'<span style="font-size:12px; font-weight:600; color:{accent_color}; letter-spacing:0.03em;">{gate["gate_name"]}</span>'
                f'<span style="font-size:10px; padding:2px 8px; border-radius:8px; background:{gcolor}20; color:{gcolor}; font-weight:600;">{glabel} · {rule}</span>'
                f'</div>'
                f'{items_html}</div>',
                unsafe_allow_html=True,
            )


# ── Helper Functions ─────────────────────────────────────────
def format_inr_display(val):
    """Format value in Indian Rupee notation (Crores/Lakhs)."""
    if not val or not isinstance(val, (int, float)):
        return "N/A"
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}₹{abs_val / 1e12:,.2f} Lakh Cr"
    if abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    if abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} L"
    return f"{sign}₹{abs_val:,.2f}"


def safe_get(data, key, default="N/A"):
    """Safely get a value from a dict."""
    if not data:
        return default
    val = data.get(key, default)
    return val if val is not None else default


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:16px 0 8px; font-size:10.5px; font-weight:700; '
        'letter-spacing:0.08em; text-transform:uppercase; line-height:2.2;">'
        '<span style="color:#00D4AA;">Gated Screening</span>'
        ' <span style="color:#2D3748;">·</span> '
        '<span style="color:#FFA726;">Premium Indicators</span>'
        '<br/>'
        '<span style="color:#4DA6FF;">AI Debate</span>'
        ' <span style="color:#2D3748;">·</span> '
        '<span style="color:#B388FF;">Deep Dive</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    analysis_mode = st.radio(
        "Mode",
        ["Trend Rider", "Turnaround Hunter", "Stock Deep Dive"],
        key="analysis_mode_radio",
        label_visibility="collapsed",
    )

    test_mode = st.checkbox("Test Mode (50 stocks)", value=True, help="Limit to 50 stocks for faster testing")

    st.divider()

    # Default button states
    run_full = False
    run_data = False
    run_analysis = False
    run_sentiment = False
    run_report = False
    run_breadth = False
    run_screener = False
    run_deep_dive = False
    run_diagnostic = False
    run_debate = False
    run_momentum = False
    run_value = False
    selected_cats = ["Momentum", "ValueBottom"]
    ticker = ""
    exchange = "NSE"

    if analysis_mode == "Trend Rider":
        st.markdown('<span style="color:#00D4AA; font-size:11px; letter-spacing:0.1em; font-weight:700;">MOMENTUM CONTROLS</span>', unsafe_allow_html=True)
        st.caption("5 gates · 17 criteria · Strong stocks in strong sectors")
        run_momentum = st.button("Run Trend Rider", type="primary", use_container_width=True)
        debate_mode = st.radio(
            "AI Debate", ["Off", "Quick (RAG + 1 LLM call, ~20s)", "Deep (multi-round GPT-4o, ~2min)"],
            index=1, horizontal=True, key="mom_debate_mode",
        )
        run_debate = debate_mode != "Off"
        st.session_state["_debate_deep"] = "Deep" in debate_mode

        # ── Gate-organized criteria tree ──
        _cost_badge = {"fast": "", "medium": " ⏱", "slow": " 🐢"}
        with st.expander("Filter Criteria (5 Gates)", expanded=False):
            _mom_selected: set[str] = set()
            for gate in MOMENTUM_GATES:
                gcolor = _GATE_TYPE_COLORS.get(gate["gate_type"], "#8892A0")
                glabel = _GATE_TYPE_LABELS.get(gate["gate_type"], "")
                st.markdown(
                    f'<span style="color:{gcolor}; font-size:11px; font-weight:700;">{glabel}</span> '
                    f'**{gate["name"]}** <span style="color:#8892A0; font-size:11px;">(need {gate["min_pass"]})</span>',
                    unsafe_allow_html=True,
                )
                for k in gate["criteria_keys"]:
                    meta = MOMENTUM_CRITERIA.get(k, {"label": k, "default": False, "cost": "fast"})
                    is_deferred = k in gate.get("deferred_keys", [])
                    is_mandatory = k in gate.get("mandatory_keys", [])
                    suffix = " [MUST]" if is_mandatory else ""
                    suffix += " [DEFER]" if is_deferred else ""
                    checked = st.checkbox(
                        f"{meta['label']}{_cost_badge.get(meta['cost'], '')}{suffix}",
                        value=meta["default"] and not is_deferred,
                        disabled=is_deferred,
                        key=f"mom_crit_{k}",
                    )
                    if checked and not is_deferred:
                        _mom_selected.add(k)

            n_sel = len(_mom_selected)
            st.caption(f"{n_sel}/{len(MOMENTUM_CRITERIA)} enabled · Overall threshold: ≥ {MOMENTUM_OVERALL['threshold']}/{MOMENTUM_OVERALL['active_total']}")

        st.session_state["_mom_criteria"] = _mom_selected

        if st.button("Run Both (Momentum + Value)", use_container_width=True, key="run_both_mom"):
            run_momentum = True
            st.session_state["_run_both"] = True

    elif analysis_mode == "Turnaround Hunter":
        st.markdown('<span style="color:#4DA6FF; font-size:11px; letter-spacing:0.1em; font-weight:700;">VALUE CONTROLS</span>', unsafe_allow_html=True)
        st.caption("28 criteria · Turnaround candidates at valuation floors")
        run_value = st.button("Run Turnaround Hunter", type="primary", use_container_width=True)
        debate_mode = st.radio(
            "AI Debate", ["Off", "Quick (RAG + 1 LLM call, ~20s)", "Deep (multi-round GPT-4o, ~2min)"],
            index=1, horizontal=True, key="val_debate_mode",
        )
        run_debate = debate_mode != "Off"
        st.session_state["_debate_deep"] = "Deep" in debate_mode

        # ── Gate-organized criteria tree ──
        _cost_badge = {"fast": "", "medium": " ⏱", "slow": " 🐢"}
        with st.expander("Filter Criteria (7 Gates)", expanded=False):
            _val_selected: set[str] = set()
            for gate in VALUE_GATES:
                gcolor = _GATE_TYPE_COLORS.get(gate["gate_type"], "#8892A0")
                glabel = _GATE_TYPE_LABELS.get(gate["gate_type"], "")
                st.markdown(
                    f'<span style="color:{gcolor}; font-size:11px; font-weight:700;">{glabel}</span> '
                    f'**{gate["name"]}** <span style="color:#8892A0; font-size:11px;">(need {gate["min_pass"]})</span>',
                    unsafe_allow_html=True,
                )
                for k in gate["criteria_keys"]:
                    meta = VALUE_CRITERIA.get(k, {"label": k, "default": False, "cost": "fast"})
                    is_deferred = k in gate.get("deferred_keys", [])
                    is_mandatory = k in gate.get("mandatory_keys", [])
                    suffix = " [MUST]" if is_mandatory else ""
                    suffix += " [DEFER]" if is_deferred else ""
                    checked = st.checkbox(
                        f"{meta['label']}{_cost_badge.get(meta['cost'], '')}{suffix}",
                        value=meta["default"] and not is_deferred,
                        disabled=is_deferred,
                        key=f"val_crit_{k}",
                    )
                    if checked and not is_deferred:
                        _val_selected.add(k)

            n_sel = len(_val_selected)
            st.caption(f"{n_sel}/{len(VALUE_CRITERIA)} enabled · Overall threshold: ≥ {VALUE_OVERALL['threshold']}/{VALUE_OVERALL['active_total']}")

        st.session_state["_val_criteria"] = _val_selected

        if st.button("Run Both (Momentum + Value)", use_container_width=True, key="run_both_val"):
            run_value = True
            st.session_state["_run_both"] = True

    elif analysis_mode == "Stock Deep Dive":
        st.markdown('<span style="color:#B388FF; font-size:11px; letter-spacing:0.1em; font-weight:700;">STOCK TICKER</span>', unsafe_allow_html=True)
        dd_ticker = st.text_input(
            "Ticker",
            value="RELIANCE.NS",
            max_chars=20,
            help="Enter BSE/NSE ticker (e.g., RELIANCE.NS, TCS.NS)",
            key="dd_ticker_input",
        )
        dd_exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0, key="dd_exchange")
        ticker = dd_ticker.strip().upper()
        exchange = dd_exchange
        if dd_exchange == "NSE" and not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"
        elif dd_exchange == "BSE" and not ticker.endswith(".BO"):
            ticker = f"{ticker}.BO"

        run_deep_dive = st.button("Load Stock Profile", type="primary", use_container_width=True)

        # Quick picks from screener results
        cdata = st.session_state.get("category_screener_data")
        if cdata:
            st.markdown("**QUICK PICKS**")
            cat_results = cdata.get("category_results", {})
            pick_tickers = []
            for cat_stocks in cat_results.values():
                if isinstance(cat_stocks, list):
                    for s in cat_stocks[:5]:
                        t = s.get("ticker", "")
                        if t and t not in pick_tickers:
                            pick_tickers.append(t)
            if pick_tickers:
                cols = st.columns(4)
                for i, t in enumerate(pick_tickers[:8]):
                    with cols[i % 4]:
                        if st.button(t.replace(".NS", "").replace(".BO", ""), key=f"qp_{t}", use_container_width=True):
                            st.session_state["dd_ticker_input"] = t
                            st.rerun()

    st.divider()
    st.caption("v4.0 · Free data sources · No paid APIs")


# ── Session State Init ───────────────────────────────────────
if "research_state" not in st.session_state:
    st.session_state.research_state = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""
if "breadth_data" not in st.session_state:
    st.session_state.breadth_data = None
if "screener_data" not in st.session_state:
    st.session_state.screener_data = None
if "category_screener_data" not in st.session_state:
    st.session_state.category_screener_data = None
if "diagnostic_data" not in st.session_state:
    st.session_state.diagnostic_data = None
if "deep_dive_profile" not in st.session_state:
    st.session_state.deep_dive_profile = None
if "deep_dive_messages" not in st.session_state:
    st.session_state.deep_dive_messages = []
if "deep_dive_ticker" not in st.session_state:
    st.session_state.deep_dive_ticker = ""
# Screener tab deep dive (separate from Stock Deep Dive page)
if "screener_dd_profile" not in st.session_state:
    st.session_state.screener_dd_profile = None
if "screener_dd_messages" not in st.session_state:
    st.session_state.screener_dd_messages = []
if "screener_dd_ticker" not in st.session_state:
    st.session_state.screener_dd_ticker = ""


# ── Market Breadth Render Functions ──────────────────────────

def render_breadth_overview(breadth: dict):
    """Render 52-week breadth overview metrics."""
    st.markdown(
        '<div class="section-hero" style="border-left:4px solid #FFA726;">'
        '<h3>Market Breadth Overview</h3>'
        '<p>52-week high/low analysis across Nifty 500 universe</p>'
        '</div>', unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_card("Stocks Analyzed", breadth["total_stocks"], accent="#4DA6FF")
    with c2:
        render_metric_card("At 52W High", breadth["at_52w_high"],
                           delta=f"{breadth['high_pct']}%", delta_up=True, accent="#00D4AA")
    with c3:
        render_metric_card("At 52W Low", breadth["at_52w_low"],
                           delta=f"{breadth['low_pct']}%", delta_up=False, accent="#FF4757")
    with c4:
        render_metric_card("High/Low Ratio", breadth["high_low_ratio"], accent="#FFA726")
    with c5:
        signal = breadth["breadth_signal"]
        signal_colors = {"Bullish": "#00D4AA", "Bearish": "#FF4757", "Neutral": "#FFA726"}
        color = signal_colors.get(signal, "#FFA726")
        st.markdown(
            f'<div class="breadth-badge" style="background:{color}15; border:2px solid {color}; '
            f'justify-content:center; width:100%;">'
            f'<span style="font-size:1.4rem; color:{color};">{signal}</span></div>',
            unsafe_allow_html=True,
        )

    # Per-sector breakdown
    sector_highs = breadth.get("sector_highs", {})
    sector_lows = breadth.get("sector_lows", {})
    if sector_highs or sector_lows:
        with st.expander("Sector-wise 52W High/Low Breakdown"):
            all_sectors = sorted(set(list(sector_highs.keys()) + list(sector_lows.keys())))
            df = pd.DataFrame({
                "Sector": all_sectors,
                "At 52W High": [sector_highs.get(s, 0) for s in all_sectors],
                "At 52W Low": [sector_lows.get(s, 0) for s in all_sectors],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_sector_rs_table(sector_rs: list[dict]):
    """Render sector relative strength table and bar chart."""
    st.markdown("### Sector Relative Strength vs Nifty 500")

    # Styled DataFrame
    df = pd.DataFrame(sector_rs)
    display_df = pd.DataFrame({
        "Rank": df["rank"],
        "Sector": df["sector"],
        "RS 1M": df["rs_1m"].apply(lambda x: f"{x:+.2f}%"),
        "RS 3M": df["rs_3m"].apply(lambda x: f"{x:+.2f}%"),
        "RS 6M": df["rs_6m"].apply(lambda x: f"{x:+.2f}%"),
        "RS 1Y": df["rs_1y"].apply(lambda x: f"{x:+.2f}%"),
        "Abs Return 3M": df["abs_3m"].apply(lambda x: f"{x:+.2f}%"),
        "Outperforming": df["is_outperforming"].apply(lambda x: "Yes" if x else "No"),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Horizontal bar chart — 3M RS
    fig = go.Figure()
    colors = ["#00D4AA" if r.get("rs_3m", 0) > 0 else "#FF4757" for r in sector_rs]
    fig.add_trace(go.Bar(
        y=[r["sector"] for r in reversed(sector_rs)],
        x=[r["rs_3m"] for r in reversed(sector_rs)],
        orientation="h",
        marker_color=list(reversed(colors)),
        text=[f"{r['rs_3m']:+.1f}%" for r in reversed(sector_rs)],
        textposition="outside",
    ))
    dark_chart_layout(fig,
        height=max(350, len(sector_rs) * 35),
        title="3-Month Relative Strength by Sector",
        xaxis_title="Relative Strength (%)",
        margin=dict(l=20, r=80, t=40, b=20),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


def render_sector_outperformers(outperformers: dict[str, list[dict]], strong_sectors: list[str]):
    """Render top 3 outperformers per strong sector."""
    st.markdown("### Strong Sectors — Top 3 Outperformers")
    if not strong_sectors:
        st.info("No sectors are currently outperforming the benchmark on both 1M and 3M timeframes.")
        return

    for sector in strong_sectors:
        stocks = outperformers.get(sector, [])
        if not stocks:
            continue
        with st.expander(f"{sector} — {len(stocks)} outperformers", expanded=True):
            df = pd.DataFrame(stocks)
            display_df = pd.DataFrame({
                "Ticker": df["ticker"],
                "Name": df["name"],
                "3M Return": df["return_3m"].apply(lambda x: f"{x:+.2f}%"),
                "1M Return": df["return_1m"].apply(lambda x: f"{x:+.2f}%"),
                "Price (INR)": df["current_price"].apply(lambda x: f"₹{x:,.2f}"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_52w_lists(breadth: dict):
    """Render expandable lists of stocks at 52W high and 52W low."""
    st.markdown("### 52-Week High / Low Lists")
    col_left, col_right = st.columns(2)

    with col_left:
        highs = breadth.get("stocks_at_high", [])
        st.markdown(f"**At 52W High ({len(highs)} stocks)**")
        if highs:
            df = pd.DataFrame(highs)
            display_df = pd.DataFrame({
                "Ticker": df["ticker"],
                "Sector": df["sector"],
                "Price": df["price"].apply(lambda x: f"₹{x:,.2f}"),
                "52W High": df["high_52w"].apply(lambda x: f"₹{x:,.2f}"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No stocks at 52-week high")

    with col_right:
        lows = breadth.get("stocks_at_low", [])
        st.markdown(f"**At 52W Low ({len(lows)} stocks)**")
        if lows:
            df = pd.DataFrame(lows)
            display_df = pd.DataFrame({
                "Ticker": df["ticker"],
                "Sector": df["sector"],
                "Price": df["price"].apply(lambda x: f"₹{x:,.2f}"),
                "52W Low": df["low_52w"].apply(lambda x: f"₹{x:,.2f}"),
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)
        else:
            st.info("No stocks at 52-week low")


# ── Stock Screener Render Functions ─────────────────────────

def render_screener_summary(summary: dict):
    """Render screener summary metrics."""
    st.markdown("### Screening Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Stocks Screened", summary.get("total_screened", 0), accent="#4DA6FF")
    with c2:
        render_metric_card("Stocks Passing", summary.get("total_passing", 0), accent="#00D4AA")
    with c3:
        render_metric_card("Min Score", summary.get("min_score", 1), accent="#FFA726")
    with c4:
        render_metric_card("Top Sector", summary.get("top_sector", "N/A"), accent="#B388FF")

    # Criteria hit counts
    criteria_hits = summary.get("criteria_hits", {})
    if criteria_hits:
        with st.expander("Criteria Hit Counts"):
            hit_df = pd.DataFrame([
                {"Criterion": k.replace("_", " ").title(), "Stocks Matched": v}
                for k, v in criteria_hits.items()
            ])
            st.dataframe(hit_df, use_container_width=True, hide_index=True)

    # Sector distribution
    sector_dist = summary.get("sector_distribution", {})
    if sector_dist:
        with st.expander("Sector Distribution of Passing Stocks"):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(sector_dist.values()),
                y=list(sector_dist.keys()),
                orientation="h",
                marker_color="#4DA6FF",
            ))
            dark_chart_layout(fig,
                height=max(250, len(sector_dist) * 30),
                xaxis_title="Number of Stocks",
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_screener_table(stocks: list[dict], criteria_used: list[str]):
    """Render sortable screener results table."""
    st.markdown("### Screener Results")
    if not stocks:
        st.info("No stocks passed the screening criteria.")
        return

    # Build DataFrame
    rows = []
    for s in stocks:
        row = {
            "Ticker": s["ticker"],
            "Sector": s["sector"],
            "Composite": s.get("composite_score", s["score"]),
            "Recommendation": s.get("recommendation", ""),
            "Criteria Hit": s["score"],
            "Categories": s.get("categories_hit", 0),
            "Regime": s.get("regime_used", ""),
        }
        for c in criteria_used:
            label = c.replace("_", " ").title()
            row[label] = "Yes" if c in s.get("criteria_passed", []) else "No"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=min(600, 40 + 35 * len(rows)))


def render_screener_stock_detail(stocks: list[dict]):
    """Render expandable per-stock detail."""
    st.markdown("### Stock Details")
    for stock in stocks[:50]:  # Limit to top 50
        ticker = stock["ticker"]
        score = stock["score"]
        criteria_list = ", ".join(c.replace("_", " ").title() for c in stock.get("criteria_passed", []))
        composite = stock.get("composite_score", score)
        rec = stock.get("recommendation", "")
        with st.expander(f"{ticker} — Composite: {composite}/100 {rec} | {stock['sector']} | {score} criteria"):
            st.markdown(f"**Criteria Passed:** {criteria_list}")
            breakdown = stock.get("dimension_breakdown", {})
            if breakdown:
                st.markdown("**Dimension Scores:** " + " | ".join(f"{d.title()}: {v}" for d, v in breakdown.items()))
            details = stock.get("criteria_details", {})

            for criterion, detail in details.items():
                label = criterion.replace("_", " ").title()
                st.markdown(f"**{label}:**")

                if criterion == "roe":
                    history = detail.get("roe_history", [])
                    if history:
                        hist_str = ", ".join(f"{h['year']}: {h['roe']}%" for h in history)
                        st.write(f"  Current ROE: {detail.get('current_roe', 'N/A')}% | History: {hist_str}")

                elif criterion in ("pb_vs_historical", "ps_vs_historical"):
                    metric = "P/B" if "pb" in criterion else "P/S"
                    current_key = "current_pb" if "pb" in criterion else "current_ps"
                    avg_key = "avg_pb_4yr" if "pb" in criterion else "avg_ps_4yr"
                    st.write(
                        f"  Current {metric}: {detail.get(current_key, 'N/A')} | "
                        f"4Y Avg: {detail.get(avg_key, 'N/A')} | "
                        f"Discount: {detail.get('discount_pct', 'N/A')}%"
                    )

                elif criterion == "insider_buying":
                    st.write(
                        f"  Buys: {detail.get('insider_buys', 0)} | "
                        f"Sells: {detail.get('insider_sells', 0)}"
                    )

                elif criterion == "capacity_utilization":
                    st.write(
                        f"  Asset Turnover: {detail.get('asset_turnover', 'N/A')} | "
                        f"Method: {detail.get('method', 'N/A')}"
                    )

                elif criterion in ("capex", "new_business", "order_booking"):
                    matches = detail.get("matches", [])
                    for m in matches[:3]:
                        st.write(f"  [{m.get('source', '')}] {m.get('headline', '')}")

            st.markdown("---")


def render_screener_export(screener_data: dict):
    """Render CSV, JSON, and PDF export buttons."""
    stocks = screener_data.get("stocks", [])
    if not stocks:
        return

    st.markdown("### Export Results")
    col1, col2, col3 = st.columns(3)

    # CSV export
    with col1:
        rows = []
        for s in stocks:
            row = {"ticker": s["ticker"], "sector": s["sector"], "score": s["score"]}
            for c in s.get("criteria_passed", []):
                row[c] = "Yes"
            rows.append(row)
        csv_df = pd.DataFrame(rows).fillna("No")
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # JSON export
    with col2:
        json_data = json.dumps(screener_data, indent=2, default=str)
        st.download_button(
            "Download JSON",
            data=json_data,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )

    # PDF export
    with col3:
        try:
            from app.sharing import generate_screener_pdf
            cdata = st.session_state.get("category_screener_data")
            if cdata:
                pdf_bytes = generate_screener_pdf(cdata)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"screener_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"PDF export failed: {e}")


# ── Stock Analysis Render Functions ─────────────────────────
def render_company_overview(company_info):
    """Render company overview section with INR formatting."""
    st.markdown("### Company Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        price = safe_get(company_info, "current_price", 0)
        render_metric_card("Current Price", f"₹{price:,.2f}" if isinstance(price, (int, float)) else str(price), accent="#00D4AA")
    with col2:
        mcap = safe_get(company_info, "market_cap", 0)
        render_metric_card("Market Cap", format_inr_display(mcap), accent="#4DA6FF")
    with col3:
        pe = safe_get(company_info, "pe_trailing", 0)
        render_metric_card("P/E (Trailing)", f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe), accent="#FFA726")
    with col4:
        beta = safe_get(company_info, "beta", 0)
        render_metric_card("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else str(beta), accent="#B388FF")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        render_metric_card("Sector", safe_get(company_info, "sector"), accent="#4DA6FF")
    with col6:
        render_metric_card("Industry", safe_get(company_info, "industry"), accent="#B388FF")
    with col7:
        high = safe_get(company_info, "52_week_high", 0)
        render_metric_card("52W High", f"₹{high:,.2f}" if isinstance(high, (int, float)) else str(high), accent="#00D4AA")
    with col8:
        low = safe_get(company_info, "52_week_low", 0)
        render_metric_card("52W Low", f"₹{low:,.2f}" if isinstance(low, (int, float)) else str(low), accent="#FF4757")

    desc = safe_get(company_info, "description", "")
    if desc and desc != "N/A":
        with st.expander("Business Description"):
            st.write(desc)


def render_price_chart(financials):
    """Render stock price chart with INR prices."""
    price_data = financials.get("get_stock_price_history", {})
    recent_prices = price_data.get("recent_prices", [])
    if not recent_prices:
        return

    st.markdown("### Stock Price (Recent)")
    dates = [p["date"] for p in recent_prices]
    closes = [p["close"] for p in recent_prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode="lines+markers",
        name="Close Price", line=dict(color="#00D4AA", width=2),
        marker=dict(size=4),
    ))
    dark_chart_layout(fig, xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ret = safe_get(price_data, 'period_return_pct', 0)
        render_metric_card("Period Return", f"{ret:.2f}%", delta_up=isinstance(ret, (int, float)) and ret >= 0, accent="#00D4AA")
    with col2:
        render_metric_card("Period High", f"₹{safe_get(price_data, 'period_high', 0):,.2f}", accent="#00D4AA")
    with col3:
        render_metric_card("Period Low", f"₹{safe_get(price_data, 'period_low', 0):,.2f}", accent="#FF4757")
    with col4:
        render_metric_card("Avg Volume", f"{safe_get(price_data, 'avg_volume', 0):,.0f}", accent="#4DA6FF")


def render_ratios(ratios):
    """Render financial ratios with Indian Metrics tab."""
    if not ratios:
        return
    st.markdown("### Financial Ratios")

    valuation = {k: v for k, v in ratios.items() if k in [
        "pe_trailing", "pe_forward", "pb_ratio", "ps_ratio",
        "ev_to_ebitda", "ev_to_revenue", "peg_ratio"
    ]}
    profitability = {k: v for k, v in ratios.items() if k in [
        "profit_margin", "operating_margin", "gross_margin", "roe", "roa", "roce"
    ]}
    leverage = {k: v for k, v in ratios.items() if k in [
        "debt_to_equity", "current_ratio", "quick_ratio"
    ]}
    growth = {k: v for k, v in ratios.items() if k in [
        "earnings_growth", "revenue_growth"
    ]}

    tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Profitability & ROCE", "Leverage", "Growth"])

    def _render_ratio_tab(data, fmt_pct=False):
        if not data:
            st.info("No data available")
            return
        cols = st.columns(min(len(data), 4))
        for i, (k, v) in enumerate(data.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                if label == "Roce":
                    label = "ROCE"
                if fmt_pct and isinstance(v, (int, float)):
                    render_metric_card(label, f"{v * 100:.2f}%", accent="#4DA6FF")
                elif isinstance(v, (int, float)):
                    render_metric_card(label, f"{v:.2f}", accent="#4DA6FF")
                else:
                    render_metric_card(label, str(v), accent="#4DA6FF")

    with tab1:
        _render_ratio_tab(valuation)
    with tab2:
        _render_ratio_tab(profitability, fmt_pct=True)
    with tab3:
        _render_ratio_tab(leverage)
    with tab4:
        _render_ratio_tab(growth, fmt_pct=True)


def render_indian_metrics(indian_metrics):
    """Render India-specific metrics: ROCE detail, promoter holding, FII/DII."""
    if not indian_metrics:
        return
    st.markdown("### Indian Market Metrics")

    tab1, tab2, tab3 = st.tabs(["ROCE Analysis", "Shareholding", "FII/DII Activity"])

    with tab1:
        roce_data = indian_metrics.get("roce", {})
        if roce_data and "error" not in roce_data:
            col1, col2, col3 = st.columns(3)
            with col1:
                render_metric_card("Current ROCE", safe_get(roce_data, "roce_pct", "N/A"), accent="#00D4AA")
            with col2:
                render_metric_card("Quality Rating", safe_get(roce_data, "quality_rating", "N/A"), accent="#FFA726")
            with col3:
                render_metric_card("EBIT (Cr)", f"₹{safe_get(roce_data, 'ebit_crores', 0):,.2f}", accent="#4DA6FF")

            historical = roce_data.get("historical_roce", [])
            if historical:
                years = [h["year"][:10] for h in historical]
                values = [h["roce"] * 100 for h in historical]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=years, y=values, marker_color="#FFA726", name="ROCE %"))
                dark_chart_layout(fig, height=300, yaxis_title="ROCE %")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROCE data not available")

    with tab2:
        sh_data = indian_metrics.get("shareholding", {})
        if sh_data:
            major = sh_data.get("major_holders", {})
            if isinstance(major, dict) and major:
                st.markdown("**Major Holders:**")
                for desc, pct in major.items():
                    st.write(f"- {desc}: **{pct}**")
            inst_holders = sh_data.get("top_institutional_holders", [])
            if isinstance(inst_holders, list) and inst_holders:
                st.markdown("**Top Institutional Holders:**")
                import pandas as pd
                df = pd.DataFrame(inst_holders)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Shareholding data not available")

    with tab3:
        fii_data = indian_metrics.get("fii_dii", {})
        if fii_data and "error" not in fii_data:
            inst = fii_data.get("institutional_holders", [])
            mf = fii_data.get("mutual_fund_holders", [])
            if isinstance(inst, list) and inst:
                st.markdown("**Institutional Holders (FII proxy):**")
                import pandas as pd
                df = pd.DataFrame(inst[:10])
                st.dataframe(df, use_container_width=True, hide_index=True)
            if isinstance(mf, list) and mf:
                st.markdown("**Mutual Fund Holders (DII proxy):**")
                import pandas as pd
                df = pd.DataFrame(mf[:10])
                st.dataframe(df, use_container_width=True, hide_index=True)
            note = fii_data.get("analysis_note", "")
            if note:
                st.info(note)
        else:
            st.info("FII/DII data not available")


def render_dcf(dcf):
    """Render DCF valuation with INR and Indian WACC."""
    if not dcf:
        return
    st.markdown("### DCF Valuation (Indian WACC)")

    col1, col2, col3 = st.columns(3)
    with col1:
        intrinsic = safe_get(dcf, "intrinsic_value_per_share", 0)
        render_metric_card("Intrinsic Value", f"₹{intrinsic:,.2f}" if isinstance(intrinsic, (int, float)) else str(intrinsic), accent="#00D4AA")
    with col2:
        current = safe_get(dcf, "current_price", 0)
        render_metric_card("Current Price", f"₹{current:,.2f}" if isinstance(current, (int, float)) else str(current), accent="#4DA6FF")
    with col3:
        upside = safe_get(dcf, "upside_pct", 0)
        if isinstance(upside, (int, float)):
            render_metric_card("Upside/Downside", f"{upside:+.2f}%",
                               delta=f"{upside:+.2f}%", delta_up=upside >= 0, accent="#00D4AA" if upside >= 0 else "#FF4757")
        else:
            render_metric_card("Upside/Downside", str(upside), accent="#FFA726")

    assumptions = dcf.get("assumptions", {})
    if assumptions:
        with st.expander("DCF Assumptions (Indian Parameters)"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                st.write(f"**Growth Rate:** {assumptions.get('growth_rate', 'N/A')}")
                st.write(f"**Risk-Free Rate:** {assumptions.get('risk_free_rate', 'N/A')} (India 10Y bond)")
            with ac2:
                st.write(f"**Discount Rate (WACC):** {assumptions.get('discount_rate_wacc', assumptions.get('discount_rate', 'N/A'))}")
                st.write(f"**Equity Risk Premium:** {assumptions.get('equity_risk_premium', 'N/A')}")
            with ac3:
                st.write(f"**Terminal Growth:** {assumptions.get('terminal_growth', 'N/A')} (India GDP)")
                st.write(f"**Projection Years:** {assumptions.get('projection_years', 'N/A')}")

    projected = dcf.get("projected_fcfs", [])
    if projected:
        # Determine the key name (could be crores or billions)
        fcf_key = "discounted_fcf_crores" if "discounted_fcf_crores" in projected[0] else "discounted_fcf"
        years = [f"Year {p['year']}" for p in projected]
        fcfs = [p.get(fcf_key, 0) for p in projected]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=fcfs, marker_color="#00D4AA", name="Discounted FCF (Cr)"))
        dark_chart_layout(fig, height=300, yaxis_title="Discounted FCF (₹ Crores)")
        st.plotly_chart(fig, use_container_width=True)


def render_peer_comparison(peers):
    """Render peer comparison table with INR Crores."""
    if not peers:
        return
    st.markdown("### Peer Comparison (Indian Sector)")

    import pandas as pd
    df = pd.DataFrame(peers)
    col_map = {
        "ticker": "Ticker", "name": "Company",
        "market_cap_crores": "Mkt Cap (Cr)",
        "pe_trailing": "P/E", "pe_forward": "Fwd P/E",
        "pb_ratio": "P/B", "ev_ebitda": "EV/EBITDA",
        "profit_margin": "Margin", "roe": "ROE",
        "revenue_growth": "Rev Growth", "debt_to_equity": "D/E",
        "dividend_yield": "Div Yield", "current_price": "Price (₹)",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ["Margin", "ROE", "Rev Growth", "Div Yield"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) and x is not None else "N/A"
            )

    if "Mkt Cap (Cr)" in df.columns:
        df["Mkt Cap (Cr)"] = df["Mkt Cap (Cr)"].apply(
            lambda x: f"₹{x:,.0f}" if isinstance(x, (int, float)) else "N/A"
        )

    if "Price (₹)" in df.columns:
        df["Price (₹)"] = df["Price (₹)"].apply(
            lambda x: f"₹{x:,.2f}" if isinstance(x, (int, float)) else "N/A"
        )

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_sentiment(sentiment_scores, news_summaries):
    """Render sentiment analysis."""
    st.markdown("### Sentiment Analysis")

    if sentiment_scores:
        col1, col2 = st.columns([1, 3])
        with col1:
            score = sentiment_scores.get("overall_score", 0)
            label = sentiment_scores.get("overall_label", "neutral")
            color_icon = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "🟡"
            render_metric_card("Sentiment Score", f"{score:+.2f}", accent="#00D4AA" if score >= 0 else "#FF4757")
            st.markdown(f"**{color_icon} {label.upper()}**")
        with col2:
            analysis = sentiment_scores.get("analysis", "")
            if analysis:
                st.markdown(analysis[:800])

    if news_summaries:
        with st.expander(f"Recent Indian News ({len(news_summaries)} articles)"):
            for i, news in enumerate(news_summaries):
                st.markdown(f"**{i+1}.** {news}")
                if i < len(news_summaries) - 1:
                    st.divider()


def render_corporate_actions(corporate_actions):
    """Render BSE/NSE corporate actions and announcements."""
    if not corporate_actions:
        return
    st.markdown("### Corporate Actions & Announcements")
    for i, action in enumerate(corporate_actions[:5]):
        date = action.get("date", "N/A")
        headline = action.get("headline", action.get("title", "N/A"))
        category = action.get("category", "")
        st.markdown(f"**{i+1}. [{category}]** {headline}")
        st.caption(f"Date: {date}")
        if i < min(len(corporate_actions), 5) - 1:
            st.divider()


def render_investment_score(score_data):
    """Render the quantitative investment score with visual breakdown."""
    if not score_data:
        return

    st.markdown("### Quantitative Investment Score")

    composite = score_data.get("composite_score", 0)
    recommendation = score_data.get("recommendation", "HOLD")
    confidence = score_data.get("confidence", "LOW")

    # Recommendation color
    rec_colors = {
        "STRONG BUY": "#00c853",
        "BUY": "#4caf50",
        "HOLD": "#ff9800",
        "SELL": "#f44336",
        "STRONG SELL": "#b71c1c",
    }
    rec_color = rec_colors.get(recommendation, "#ff9800")

    # Top-level score display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div style="text-align:center; padding:20px; background:{rec_color}20; '
            f'border-radius:12px; border:2px solid {rec_color};">'
            f'<div style="font-size:3rem; font-weight:800; color:{rec_color};">{composite}</div>'
            f'<div style="font-size:0.9rem; color:#666;">out of 100</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="text-align:center; padding:20px; background:{rec_color}20; '
            f'border-radius:12px; border:2px solid {rec_color};">'
            f'<div style="font-size:2rem; font-weight:700; color:{rec_color};">{recommendation}</div>'
            f'<div style="font-size:0.9rem; color:#666;">Recommendation</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        conf_color = {"HIGH": "#4caf50", "MEDIUM": "#ff9800", "LOW": "#f44336"}.get(confidence, "#999")
        st.markdown(
            f'<div style="text-align:center; padding:20px; background:{conf_color}20; '
            f'border-radius:12px; border:2px solid {conf_color};">'
            f'<div style="font-size:2rem; font-weight:700; color:{conf_color};">{confidence}</div>'
            f'<div style="font-size:0.9rem; color:#666;">Data Confidence</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Dimension breakdown
    dimensions = score_data.get("dimension_scores", {})
    if dimensions:
        dim_labels = {
            "valuation": ("Valuation", "#1f77b4"),
            "quality": ("Quality", "#ff7f0e"),
            "growth": ("Growth", "#2ca02c"),
            "sentiment": ("Sentiment", "#9467bd"),
            "governance": ("Governance", "#d62728"),
        }

        # Bar chart of dimensions
        dim_names = []
        dim_scores = []
        dim_colors = []
        dim_weights = []
        for dim_name, dim_data in dimensions.items():
            label, color = dim_labels.get(dim_name, (dim_name.title(), "#666"))
            dim_names.append(label)
            dim_scores.append(dim_data.get("score", 0))
            dim_colors.append(color)
            dim_weights.append(dim_data.get("weight", 0))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dim_names,
            y=dim_scores,
            marker_color=dim_colors,
            text=[f"{s}/100" for s in dim_scores],
            textposition="outside",
        ))
        fig.add_hline(y=75, line_dash="dash", line_color="#00D4AA", annotation_text="Strong Buy (75)", annotation_font_color="#00D4AA")
        fig.add_hline(y=60, line_dash="dash", line_color="#4DA6FF", annotation_text="Buy (60)", annotation_font_color="#4DA6FF")
        fig.add_hline(y=40, line_dash="dash", line_color="#FFA726", annotation_text="Hold (40)", annotation_font_color="#FFA726")
        dark_chart_layout(fig,
            yaxis_title="Score",
            yaxis_range=[0, 110],
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed sub-scores in expander
        with st.expander("Score Details & Sub-components"):
            for dim_name, dim_data in dimensions.items():
                label, color = dim_labels.get(dim_name, (dim_name.title(), "#666"))
                score = dim_data.get("score", 0)
                weight = dim_data.get("weight", 0)
                contrib = dim_data.get("weighted_contribution", 0)

                st.markdown(
                    f"**{label}** — Score: **{score}/100** | "
                    f"Weight: {weight}% | Contribution: {contrib} pts"
                )

                sub_scores = dim_data.get("sub_scores", {})
                if sub_scores:
                    sub_cols = st.columns(min(len(sub_scores), 4))
                    for i, (sub_name, sub_val) in enumerate(sub_scores.items()):
                        with sub_cols[i % len(sub_cols)]:
                            display_name = sub_name.replace("_", " ").title()
                            render_metric_card(display_name, f"{sub_val}/100", accent="#B388FF")

                details = dim_data.get("details", {})
                flag = details.get("pledge_flag", "")
                if flag:
                    if "RED FLAG" in flag:
                        st.error(flag)
                    elif "WARNING" in flag:
                        st.warning(flag)

                st.divider()

    # Summary
    summary = score_data.get("summary", "")
    if summary:
        with st.expander("Scoring Summary"):
            st.text(summary)


def _render_sharing_buttons(report: str, ticker: str, score_data: dict, key_suffix: str = ""):
    """Reusable sharing buttons for any report context."""
    recommendation = score_data.get("recommendation", "HOLD")
    composite_score = score_data.get("composite_score", 0)

    st.markdown("#### Share Report")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            label="Download .md",
            data=report,
            file_name=f"report_{ticker}_{datetime.now():%Y%m%d}.md",
            mime="text/markdown",
            key=f"dl_md_{key_suffix}",
        )

    with col2:
        try:
            from app.sharing import generate_report_pdf
            pdf_bytes = generate_report_pdf(report, ticker, recommendation, composite_score)
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"report_{ticker}_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_{key_suffix}",
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    with col3:
        from app.sharing import is_email_configured
        if is_email_configured():
            with st.popover("Email Report", use_container_width=True):
                recipient = st.text_input("Recipient Email", key=f"email_to_{key_suffix}")
                if st.button("Send", key=f"email_send_{key_suffix}"):
                    if recipient:
                        from app.sharing import generate_report_pdf, send_email_with_pdf
                        pdf = generate_report_pdf(report, ticker, recommendation, composite_score)
                        ok, msg = send_email_with_pdf(
                            recipient,
                            f"Equity Research: {ticker} - {recommendation}",
                            f"AI-generated equity research report for {ticker}.\n"
                            f"Recommendation: {recommendation} | Score: {composite_score}/100.\n\n"
                            f"Report generated on {datetime.now().strftime('%d %b %Y')}.",
                            pdf,
                            f"report_{ticker}.pdf",
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Enter a recipient email.")
        else:
            st.button("Email (set SMTP in .env)", disabled=True, key=f"email_dis_{key_suffix}")

    with col4:
        from app.sharing import is_whatsapp_configured, send_whatsapp_report
        if is_whatsapp_configured():
            with st.popover("WhatsApp", use_container_width=True):
                wa_phone = st.text_input(
                    "Phone (e.g. 919876543210)",
                    value=os.getenv("WHATSAPP_DEFAULT_PHONE", ""),
                    key=f"wa_phone_{key_suffix}",
                )
                if st.button("Send on WhatsApp", key=f"wa_send_{key_suffix}"):
                    if wa_phone:
                        ok, msg = send_whatsapp_report(wa_phone, ticker, recommendation, composite_score)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Enter a phone number.")
        else:
            st.button("WhatsApp (set Twilio)", disabled=True, key=f"wa_dis_{key_suffix}")


def render_report(report):
    """Render the final investment report with sharing options."""
    if not report:
        return
    st.markdown("### Investment Research Report")
    st.markdown(report)

    ticker = st.session_state.get("last_ticker", "UNKNOWN")
    score_data = st.session_state.get("research_state", {}).get("investment_score", {})
    _render_sharing_buttons(report, ticker, score_data, key_suffix="main")


# ── Ticker Validation ────────────────────────────────────────
def validate_ticker(ticker_val: str) -> bool:
    """Check if a ticker is valid on yfinance before running the pipeline."""
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker_val)
        info = stock.info
        # yfinance returns a dict with just 'trailingPegRatio' or similar for invalid tickers
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            return False
        return True
    except Exception:
        return False


# ── Agent Runners ────────────────────────────────────────────
def run_single_agent(agent_name: str, node_func, ticker_val: str):
    """Run a single agent and update session state."""
    from app.state import create_initial_state

    state = st.session_state.research_state or create_initial_state(ticker_val)
    state["ticker"] = ticker_val

    st.session_state.agent_status[agent_name] = "running"
    try:
        result = node_func(state)
        for k, v in result.items():
            if k == "errors" and v:
                state.setdefault("errors", []).extend(v)
            elif k == "messages":
                state.setdefault("messages", []).extend(v)
            else:
                state[k] = v
        st.session_state.research_state = state
        st.session_state.agent_status[agent_name] = "done"
    except Exception as e:
        st.session_state.agent_status[agent_name] = "error"
        st.error(f"{agent_name} failed: {e}")


def run_full_pipeline(ticker_val: str):
    """Run the entire LangGraph pipeline."""
    from app.graph import run_research

    agents = ["data_agent", "analysis_agent", "sentiment_agent", "report_agent"]
    for a in agents:
        st.session_state.agent_status[a] = "pending"

    result = run_research(ticker_val)
    st.session_state.research_state = result
    st.session_state.last_ticker = ticker_val
    for a in agents:
        st.session_state.agent_status[a] = "done"


# ── Main Content ─────────────────────────────────────────────
st.divider()

# ── Command Center Mode ─────────────────────────────────────
if analysis_mode == "Command Center":
    # ── KPIs ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_count = len(st.session_state.get('selected_tickers', []))
        st.metric("Selected Stocks", selected_count)
    with col2:
        deep_dive_count = len(st.session_state.get('selected_profiles', {}))
        st.metric("Deep Dives Completed", deep_dive_count)
    with col3:
        debate_count = len(st.session_state.get('debate_results', []))
        st.metric("Debates Completed", debate_count)
    with col4:
        st.metric("Last Run", "00:00:00")  # TODO: track timestamp

    st.divider()

    # ── 3-Column Command Center ──────────────────────────────────
    left_col, mid_col, right_col = st.columns([0.26, 0.42, 0.32])

    # Left: Stock Selection
    with left_col:
        st.subheader("Stock Selection")
        # Get all tickers from screener data
        screener_data = st.session_state.get('category_screener_data', {})
        category_results = screener_data.get('category_results', {}) if isinstance(screener_data, dict) else {}
        momentum_stocks = category_results.get('Momentum', []) if isinstance(category_results, dict) else []
        value_stocks = category_results.get('ValueBottom', []) if isinstance(category_results, dict) else []
        all_tickers = list(set([stock['ticker'] for stock in momentum_stocks + value_stocks if isinstance(stock, dict) and 'ticker' in stock]))

        # Auto-select from last screener run if no selection exists yet
        if not st.session_state.get('selected_tickers') and all_tickers:
            auto_count = min(5, len(all_tickers))
            top_tickers = []
            for pool in (momentum_stocks, value_stocks):
                for s in pool:
                    if not isinstance(s, dict):
                        continue
                    t = s.get('ticker')
                    if t and t not in top_tickers:
                        top_tickers.append(t)
                        if len(top_tickers) >= auto_count:
                            break
                if len(top_tickers) >= auto_count:
                    break
            if not top_tickers:
                top_tickers = all_tickers[:auto_count]
            st.session_state.selected_tickers = top_tickers
            st.info(f"Auto-selected {len(top_tickers)} from screener results in Command Center")

        selected_tickers = st.multiselect("Select Stocks", options=all_tickers, default=st.session_state.get('selected_tickers', []), key="selected_tickers")
        
        if st.button("Run Deep Dive for selected", key="run_deep_dive_selected"):
            if not selected_tickers:
                st.warning("Select stocks first")
            else:
                with st.spinner("Loading profiles for selected stocks..."):
                    from app.tools.stock_deep_dive import build_stock_profile
                    profiles = {}
                    for ticker in selected_tickers:
                        try:
                            profile = build_stock_profile(ticker)
                            profiles[ticker] = profile
                        except Exception as e:
                            st.error(f"Failed to load profile for {ticker}: {e}")
                    st.session_state.selected_profiles = profiles
                    st.success("Profiles loaded!")

        if st.button("Run Deep Dive Top 3", key="run_deep_dive_top3"):
            candidate_tickers = selected_tickers or all_tickers
            if not candidate_tickers:
                st.warning("No screener tickers available")
            else:
                chosen = candidate_tickers[:3]
                with st.spinner(f"Loading top 3 deep-dive profiles: {', '.join(chosen)}..."):
                    from app.tools.stock_deep_dive import build_stock_profile
                    profiles = st.session_state.get('selected_profiles', {}) or {}
                    for ticker in chosen:
                        if ticker not in profiles:
                            try:
                                profiles[ticker] = build_stock_profile(ticker)
                            except Exception as e:
                                st.error(f"Failed to load profile for {ticker}: {e}")
                    st.session_state.selected_profiles = profiles
                    st.session_state.selected_tickers = chosen
                    st.success(f"Loaded top 3 deep dive: {', '.join(chosen)}")
        if st.button("Run Debate for selected", key="run_debate_selected_right"):
            if not selected_tickers:
                st.warning("Select stocks first")
            else:
                with st.spinner("Running category screener on selected stocks..."):
                    from app.tools.screener.category_screener import run_category_screener
                    result = run_category_screener("momentum", tickers=selected_tickers)
                    usp_cards = result["usp_cards"]
                    category_results = result["category_results"]
                with st.spinner("Running debate..."):
                    from app.agents.debate_agents import run_debate
                    debate_results = []
                    for ticker in selected_tickers[:3]:  # limit to 3 for speed
                        res = run_debate(ticker, usp_cards, category_results, rounds=1)
                        debate_results.append(res)
                    st.session_state.debate_results = debate_results
                    st.success("Debate complete!")
        
        # Per-stock actions
        for ticker in selected_tickers:
            with st.container():
                st.markdown(f"### {ticker}")
                # Minimal metrics (next enhancement: real market data)
                st.metric("Price", "₹100")  # placeholder
                if st.button(f"Deep Dive Chat - {ticker}", key=f"dive_{ticker}"):
                    if ticker not in st.session_state.get('selected_profiles', {}):
                        with st.spinner(f"Loading profile for {ticker}..."):
                            from app.tools.stock_deep_dive import build_stock_profile
                            try:
                                profile = build_stock_profile(ticker)
                                st.session_state.selected_profiles = st.session_state.get('selected_profiles', {})
                                st.session_state.selected_profiles[ticker] = profile
                                st.success(f"Profile loaded for {ticker}")
                            except Exception as e:
                                st.error(f"Deep dive profile load failed: {e}")
                                profile = None
                    else:
                        profile = st.session_state.selected_profiles[ticker]

                    if profile:
                        st.session_state.deep_dive_ticker = ticker
                        st.session_state.deep_dive_profile = profile
                        st.session_state.deep_dive_messages = []
                        st.success(f"Deep Dive ready for {ticker}")

                if st.button(f"Run Debate - {ticker}", key=f"debate_{ticker}"):
                    with st.spinner(f"Running debate for {ticker}..."):
                        from app.agents.debate_agents import run_debate
                        cdata = st.session_state.get("category_screener_data", {})
                        cat_results = cdata.get("category_results", {})
                        usp_cards = cdata.get("usp_cards", {})
                        try:
                            result = run_debate(ticker, usp_cards, cat_results if isinstance(cat_results, dict) else [])
                            dr = st.session_state.get("debate_results", [])
                            dr = [d for d in dr if d.get("ticker") != ticker]
                            dr.append(result)
                            st.session_state.debate_results = dr
                            st.success(f"Debate complete for {ticker}")
                        except Exception as e:
                            st.error(f"Debate for {ticker} failed: {e}")

    # Middle: Deep Dive Chat
    with mid_col:
        st.subheader("Deep Dive Chat")
        available_profiles = list(st.session_state.get("selected_profiles", {}).keys())
        if not available_profiles:
            st.info("Run 'Deep Dive for selected' to load profiles first.")
            chat_ticker = None
        else:
            chat_ticker = st.selectbox("Select stock for chat", options=available_profiles, key="chat_ticker")
        
        if chat_ticker:
            if chat_ticker != st.session_state.get("deep_dive_ticker"):
                st.session_state.deep_dive_ticker = chat_ticker
                st.session_state.deep_dive_profile = st.session_state.selected_profiles.get(chat_ticker)
                st.session_state.deep_dive_messages = []
            
            # Display profile metrics
            profile = st.session_state.deep_dive_profile
            if not profile:
                st.warning(f"No deep dive profile data available for {chat_ticker}. Please run profile load again.")
            elif not profile.get("financials_data") and not profile.get("research_state"):
                st.warning("Profile has limited data. Consider running the screener + stock profile pipeline again for richer data.")

            if profile:
                info = profile.get("info", {})
                col1, col2 = st.columns(2)
                with col1:
                    market_cap = info.get("marketCap")
                    if market_cap:
                        market_cap_str = f"₹{market_cap / 1e7:.1f} Cr"
                    else:
                        market_cap_str = "N/A"
                    st.metric("Market Cap", market_cap_str)
                    pe = info.get("trailingPE") or info.get("forwardPE")
                    st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                with col2:
                    # Revenue from latest financials
                    fin_data = profile.get("financials_data", {})
                    revenue = fin_data.get("total_revenue", {}).get("latest", "N/A")
                    if isinstance(revenue, (int, float)):
                        revenue_str = f"₹{revenue / 1e7:.1f} Cr"
                    else:
                        revenue_str = "N/A"
                    st.metric("Revenue", revenue_str)
                    # Net profit
                    net_profit = fin_data.get("net_income", {}).get("latest", "N/A")
                    if isinstance(net_profit, (int, float)):
                        net_profit_str = f"₹{net_profit / 1e7:.1f} Cr"
                    else:
                        net_profit_str = "N/A"
                    st.metric("Net Profit", net_profit_str)
        
        # Chat history
        chat_history = st.session_state.get('deep_dive_messages', [])
        for msg in chat_history:
            st.chat_message(msg['role']).write(msg['content'])
        
        # Chat input
        user_prompt = st.chat_input("Ask analyst question...", key="chat_input")
        if user_prompt:
            if not st.session_state.get('deep_dive_profile'):
                st.error("Please select a stock with loaded profile first.")
            else:
                st.session_state.deep_dive_messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            from app.deep_dive_chat import ask_stock_question
                            response = ask_stock_question(
                                user_prompt,
                                st.session_state.deep_dive_profile,
                                st.session_state.deep_dive_messages[:-1],
                            )
                            st.markdown(response)
                            st.session_state.deep_dive_messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error generating response: {e}"
                            st.error(error_msg)
                            st.session_state.deep_dive_messages.append({"role": "assistant", "content": error_msg})

    # Right: Debate Engine
    with right_col:
        st.subheader("Debate Engine")
        debate_mode = st.radio("Mode", ["Off", "Quick", "Deep"], key="debate_mode")
        if debate_mode != "Off":
            if st.button("Run Debate for selected", key="run_debate_selected"):
                if not selected_tickers:
                    st.warning("Select stocks first")
                else:
                    with st.spinner("Running category screener on selected stocks..."):
                        from app.tools.screener.category_screener import run_category_screener
                        result = run_category_screener("momentum", tickers=selected_tickers)
                        usp_cards = result["usp_cards"]
                        category_results = result["category_results"]
                    with st.spinner("Running debate..."):
                        from app.agents.debate_agents import run_debate
                        debate_results = []
                        for ticker in selected_tickers[:3]:  # limit to 3 for speed
                            res = run_debate(ticker, usp_cards, category_results, rounds=1 if debate_mode == "Quick" else 2)
                            debate_results.append(res)
                        st.session_state.debate_results = debate_results
                        st.success("Debate complete!")
        # Display debate results
        debate_results = st.session_state.get("debate_results", [])
        for result in debate_results:
            ticker = result["ticker"]
            verdict = result["verdict"]
            with st.expander(f"{ticker} Debate - {verdict['recommendation']} (Score: {verdict['conviction_score']}/10)"):
                st.write(f"**Reasoning:** {verdict['reasoning']}")
                st.write(f"**Key Factors:** {', '.join(verdict['key_factors'])}")
                st.write(f"**Bull Strength:** {verdict['bull_strength']}/10 | **Bear Strength:** {verdict['bear_strength']}/10")
                if st.checkbox(f"Show full transcript for {ticker}", key=f"transcript_{ticker}"):
                    for i, (bull, bear) in enumerate(zip(result["bull_arguments"], result["bear_arguments"]), 1):
                        st.markdown(f"**Round {i}:**")
                        st.markdown("**Bull:** " + bull.replace("\n", "  \n"))
                        st.markdown("**Bear:** " + bear.replace("\n", "  \n"))

    st.stop()  # Stop here for Command Center mode

# ── Legacy Mode Content ──────────────────────────────────────

# ── Stock Diagnostic Mode ────────────────────────────────────
if analysis_mode == "Stock Diagnostic":
    st.markdown(
        '<div class="section-hero" style="border-left:4px solid #FFA726;">'
        '<h3>Single Stock Diagnostic</h3>'
        '<p>Run ALL screening criteria on one stock · Gate-by-gate pass/fail · Debug your filters</p>'
        '</div>', unsafe_allow_html=True,
    )

    if run_diagnostic and diag_ticker:
        progress_bar = st.progress(0, text=f"Diagnosing {diag_ticker}...")

        def _diag_progress(fraction: float, message: str):
            progress_bar.progress(min(fraction, 1.0), text=message)

        try:
            import time as _time
            _t0 = _time.time()
            from app.tools.screener.category_screener import run_single_stock_diagnostic
            st.session_state.diagnostic_data = run_single_stock_diagnostic(
                diag_ticker, progress_cb=_diag_progress,
            )
            progress_bar.empty()
            st.success(f"Diagnostic complete in {_time.time() - _t0:.1f}s")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Diagnostic failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    ddata = st.session_state.diagnostic_data
    if ddata:
        ticker_name = ddata["ticker"]
        sector_name = ddata["sector"]
        info = ddata.get("info", {})
        company = info.get("shortName", ticker_name)
        criteria = ddata["criteria_results"]
        passed_set = ddata["passed_set"]

        # Header
        st.markdown(
            f'<div style="background:rgba(255,167,38,0.08); border:1px solid #FFA72640; border-radius:10px; padding:16px; margin-bottom:16px;">'
            f'<span style="font-size:1.4rem; font-weight:700; color:#FFA726;">{company}</span>'
            f'<span style="margin-left:16px; color:#8892A0;">{ticker_name} · {sector_name}</span>'
            f'<span style="margin-left:16px; font-size:1.1rem; color:#00D4AA; font-weight:600;">'
            f'{len(passed_set)} criteria passed</span>'
            f'</div>', unsafe_allow_html=True,
        )

        # Summary cards
        mom = ddata["momentum"]
        val = ddata["value"]
        c1, c2 = st.columns(2)
        with c1:
            mom_color = "#00D4AA" if mom["final_pass"] else "#FF4757"
            mom_label = "PASS" if mom["final_pass"] else "FAIL"
            render_metric_card(
                f"Momentum: {mom_label}",
                f"{mom['score']}/{mom['active_total']}",
                accent=mom_color,
            )
        with c2:
            val_color = "#00D4AA" if val["final_pass"] else "#FF4757"
            val_label = "PASS" if val["final_pass"] else "FAIL"
            render_metric_card(
                f"Value: {val_label}",
                f"{val['score']}/{val['active_total']}",
                accent=val_color,
            )

        st.divider()

        # Gate-by-gate breakdown for both categories
        for cat_label, cat_key, cat_data, gates, accent in [
            ("Momentum", "momentum", mom, MOMENTUM_GATES, "#00D4AA"),
            ("Value Bottom", "value", val, VALUE_GATES, "#4DA6FF"),
        ]:
            pass_label = "PASS" if cat_data["final_pass"] else "FAIL"
            pass_color = "#00D4AA" if cat_data["final_pass"] else "#FF4757"
            st.markdown(
                f'<div style="margin:12px 0 8px 0;">'
                f'<span style="font-size:1.1rem; font-weight:700; color:{accent};">{cat_label}</span>'
                f' <span style="color:{pass_color}; font-weight:700; font-size:0.9rem;">{pass_label}</span>'
                f' <span style="color:#8892A0; font-size:0.85rem;">({cat_data["score"]}/{cat_data["active_total"]}'
                f', need {MOMENTUM_OVERALL["threshold"] if cat_key == "momentum" else VALUE_OVERALL["threshold"]})</span>'
                f'</div>', unsafe_allow_html=True,
            )

            for g in cat_data["gate_details"]:
                gcolor = _GATE_TYPE_COLORS.get(g["gate_type"], "#8892A0")
                glabel = _GATE_TYPE_LABELS.get(g["gate_type"], "")
                status_icon = "✓" if g["passed"] else "✗"
                status_color = "#00D4AA" if g["passed"] else "#FF4757"

                # Build criteria checks with details
                checks_html = ""
                for k in g.get("passed_keys", []):
                    detail = criteria.get(k, {}).get("detail", {})
                    detail_str = ""
                    # Show key metric if available
                    for dkey in ["asset_turnover", "insider_buys", "match_count", "geo_score",
                                 "matched_keywords", "piotroski_score", "altman_z", "roe",
                                 "de_ratio", "rsi", "distance_pct"]:
                        if dkey in detail:
                            detail_str = f' ({dkey}={detail[dkey]})'
                            break
                    checks_html += (
                        f'<div style="margin:2px 0 2px 24px; font-size:12px;">'
                        f'<span style="color:#00D4AA;">✓</span> '
                        f'<span style="color:#C8D0DA;">{k.replace("_"," ").title()}</span>'
                        f'<span style="color:#8892A0; font-size:11px;">{detail_str}</span>'
                        f'</div>'
                    )
                for k in g.get("failed_keys", []):
                    detail = criteria.get(k, {}).get("detail", {})
                    err = detail.get("error", "")
                    err_str = f' ({err})' if err else ""
                    checks_html += (
                        f'<div style="margin:2px 0 2px 24px; font-size:12px;">'
                        f'<span style="color:#FF4757;">✗</span> '
                        f'<span style="color:#8892A0;">{k.replace("_"," ").title()}</span>'
                        f'<span style="color:#FF475780; font-size:11px;">{err_str}</span>'
                        f'</div>'
                    )
                for k in g.get("deferred_keys", []):
                    checks_html += (
                        f'<div style="margin:2px 0 2px 24px; font-size:12px; opacity:0.5; font-style:italic;">'
                        f'<span style="color:#8892A0;">○</span> '
                        f'<span>{k.replace("_"," ").title()}</span>'
                        f' <span style="font-size:10px;">DEFERRED</span>'
                        f'</div>'
                    )

                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03); border:1px solid {gcolor}20; '
                    f'border-radius:8px; padding:10px 12px; margin:6px 0;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="color:{status_color}; font-weight:700; font-size:15px;">{status_icon}</span> '
                    f'<span style="color:{gcolor}; font-weight:600; font-size:13px; flex:1; margin-left:8px;">{g["name"]}</span>'
                    f'<span style="font-size:10px; padding:2px 8px; border-radius:8px; background:{gcolor}20; color:{gcolor};">'
                    f'{glabel} · {g["passed_count"]}/{g["active_count"]} (need {g["min_pass"]})</span>'
                    f'</div>'
                    f'{checks_html}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

        # Raw criteria table
        with st.expander("All Criteria Raw Results", expanded=False):
            import pandas as _pd
            rows = []
            for k, v in sorted(criteria.items()):
                rows.append({
                    "Criterion": k,
                    "Passed": "✓" if v["passed"] else "✗",
                    "Details": str({dk: dv for dk, dv in v.get("detail", {}).items()
                                   if dk not in ("ticker", "sector", "passed")})[:120],
                })
            st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        st.markdown("Enter a ticker in the sidebar and click **Run Diagnostic** to test all screening criteria.")

    st.stop()

# ── Stock Deep Dive Mode ─────────────────────────────────────
if analysis_mode == "Stock Deep Dive":
    if run_deep_dive and ticker:
        # Reset if different ticker
        if ticker != st.session_state.deep_dive_ticker:
            st.session_state.deep_dive_messages = []
            st.session_state.deep_dive_profile = None

        progress_bar = st.progress(0, text="Loading stock profile...")

        def _dd_progress(fraction: float, message: str):
            progress_bar.progress(min(fraction, 1.0), text=message)

        try:
            from app.tools.stock_deep_dive import build_stock_profile
            profile = build_stock_profile(
                ticker, exchange, progress_cb=_dd_progress,
            )
            # Bridge screener data if a screener run has been done
            cdata = st.session_state.get("category_screener_data")
            if cdata:
                _inject_screener_context(profile, ticker, cdata)
            st.session_state.deep_dive_profile = profile
            st.session_state.deep_dive_ticker = ticker
            progress_bar.empty()
            st.success("Stock profile loaded!")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Profile loading failed: {e}")

    profile = st.session_state.deep_dive_profile
    if profile:
        render_stock_profile_scorecard(profile)
        render_deep_dive_chat(profile, "deep_dive_messages", "deep_dive_ticker")

    else:
        st.markdown("""
        ### Stock Deep Dive

        Enter a stock ticker in the sidebar and click **Load Stock Profile** to begin.

        This mode will:
        1. Run the full research pipeline (data, analysis, sentiment, report)
        2. Compute all USP scores (geopolitical, smart money, regulatory, promoter)
        3. Display a comprehensive scorecard
        4. Enable conversational Q&A about the stock

        **Example questions you can ask:**
        - What are the different products and business segments?
        - Revenue and EBITDA contribution from each product
        - Who are the competitors and what's the competitive edge?
        - What are the key strengths and growth potential?
        - What are the risks and red flags?
        - How has the debt position changed over time?
        """)

    st.stop()

# ── Screener Runner (shared by Momentum & Value pages) ────────

def _run_screener_for_category(category_key: str, run_debate_flag: bool):
    """Run the screener for a single or both categories and store results."""
    # Check if "Run Both" was requested
    run_both = st.session_state.pop("_run_both", False)
    categories = ["Momentum", "ValueBottom"] if run_both else [category_key]
    label = "Momentum + Value" if run_both else category_key
    progress_bar = st.progress(0, text=f"Starting {label} screener...")

    def _cat_progress(fraction: float, message: str):
        progress_bar.progress(min(fraction, 1.0), text=message)

    try:
        import time as _time
        _t_start = _time.time()
        _timings: dict[str, float] = {}

        _cat_progress(0.01, "Detecting market regime...")
        try:
            from app.agents.regime_agent import detect_regime
            regime_data = detect_regime(use_llm=False)
        except Exception as e:
            regime_data = {"regime": "mixed", "reasoning": f"Detection failed: {e}", "weights": {"Momentum": 1.0, "ValueBottom": 1.0}}
        _timings["regime"] = _time.time() - _t_start

        _t_screen = _time.time()
        from app.tools.screener.category_screener import run_category_screener

        # Build enabled_criteria and min_criteria from sidebar selections
        _enabled: dict[str, set[str]] = {}
        mom_crit = st.session_state.get("_mom_criteria")
        val_crit = st.session_state.get("_val_criteria")
        if mom_crit is not None:
            _enabled["Momentum"] = mom_crit
        if val_crit is not None:
            _enabled["ValueBottom"] = val_crit

        result = run_category_screener(
            categories=categories,
            progress_cb=_cat_progress,
            max_stocks=50 if test_mode else None,
            enabled_criteria=_enabled if _enabled else None,
        )
        _timings["screening"] = _time.time() - _t_screen
        result["regime_data"] = regime_data
        st.session_state.category_screener_data = result

        # Run debate if requested
        if run_debate_flag and result.get("category_results"):
            _t_debate = _time.time()
            use_deep = st.session_state.get("_debate_deep", False)
            mode_label = "Deep" if use_deep else "Quick"
            progress_bar.progress(0.85, text=f"Running {mode_label} AI debate for top stocks...")
            try:
                from app.agents.debate_agents import _select_top_stocks, MAX_DEBATE_STOCKS
                from app.ui.usp_cards import transform_usp_data

                usp_cards = transform_usp_data(result.get("usp_scores", {}))
                cat_results = result.get("category_results", {})
                if isinstance(cat_results, dict):
                    cat_list = [{"category": k, "stocks": v} for k, v in cat_results.items()]
                else:
                    cat_list = cat_results

                top_tickers = _select_top_stocks(usp_cards, cat_list, max_stocks=MAX_DEBATE_STOCKS)
                debate_results = []

                # Preload deep dive profile for #1 stock in parallel with debate
                dd_thread = None
                if top_tickers:
                    import threading

                    def _preload_deep_dive(t_ticker, cdata_snapshot):
                        try:
                            from app.tools.stock_deep_dive import build_stock_profile
                            dd_prof = build_stock_profile(t_ticker, "NSE")
                            _inject_screener_context(dd_prof, t_ticker, cdata_snapshot)
                            # Store in a thread-safe attribute (Streamlit session_state
                            # is not thread-safe, so we stash on the function object)
                            _preload_deep_dive._result = dd_prof
                            _preload_deep_dive._ticker = t_ticker
                        except Exception:
                            _preload_deep_dive._result = None

                    _preload_deep_dive._result = None
                    _preload_deep_dive._ticker = None
                    dd_thread = threading.Thread(
                        target=_preload_deep_dive,
                        args=(top_tickers[0], {**result, "usp_cards": usp_cards}),
                        daemon=True,
                    )
                    dd_thread.start()

                if use_deep:
                    from app.agents.debate_agents import run_debate as _run_debate_fn
                    for t in top_tickers:
                        dr = _run_debate_fn(ticker=t, usp_cards=usp_cards, category_results=cat_list)
                        debate_results.append(dr)
                else:
                    from app.agents.debate_hybrid import run_hybrid_debate
                    for t in top_tickers:
                        dr = run_hybrid_debate(ticker=t, usp_cards=usp_cards, category_results=cat_list)
                        debate_results.append(dr)

                result["debate_results"] = debate_results
                result["usp_cards"] = usp_cards
                st.session_state.category_screener_data = result

                # Collect preloaded deep dive profile
                if dd_thread is not None:
                    dd_thread.join(timeout=120)
                    if _preload_deep_dive._result is not None:
                        st.session_state.screener_dd_profile = _preload_deep_dive._result
                        st.session_state.screener_dd_ticker = _preload_deep_dive._ticker
                        st.session_state.screener_dd_messages = []
            except Exception as e:
                st.warning(f"Debate phase failed (non-fatal): {e}")
            _timings["debate"] = _time.time() - _t_debate

        _timings["total"] = _time.time() - _t_start
        result["timings"] = _timings

        progress_bar.empty()
        # Show timing breakdown
        parts = []
        for phase in ["regime", "screening", "debate", "total"]:
            if phase in _timings:
                parts.append(f"{phase}: {_timings[phase]:.0f}s")
        st.success(f"{label} complete! ({' | '.join(parts)})")
    except Exception as e:
        progress_bar.empty()
        st.error(f"Screener failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def _fmt_inr_sc(val):
    """Quick INR formatter for scorecard."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        if abs(val) >= 1e12:
            return f"{val/1e12:.1f}L Cr"
        if abs(val) >= 1e7:
            return f"{val/1e7:.1f} Cr"
        if abs(val) >= 1e5:
            return f"{val/1e5:.1f} L"
        return f"{val:,.0f}"
    return str(val)


def _fmt_pct_sc(val):
    """Format a ratio as percentage if small float."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and abs(val) < 10:
        return f"{val:.1%}"
    return f"{val:.2f}" if isinstance(val, float) else str(val)


def render_stock_profile_scorecard(profile: dict):
    """Render the stock profile scorecard with USP expanders (reusable helper)."""
    info = profile.get("info", {})
    research = profile.get("research_state", {})
    company_name = info.get("longName", info.get("shortName", profile.get("ticker", "")))
    inv_score = research.get("investment_score", {})

    # ── Company header (no recommendation badge — screener tier used instead) ──
    # Show screener tier badge if available from screener pipeline
    tier = profile.get("screener_tier", "")
    tier_colors = {"Buy Zone": "#00c853", "Watchlist": "#4DA6FF", "Monitor": "#FFA726", "Near Miss": "#B388FF"}
    tier_color = tier_colors.get(tier, "#8892A0")
    tier_badge = (
        f'<span style="background:{tier_color}25; color:{tier_color}; padding:4px 12px; '
        f'border-radius:6px; font-weight:700; font-size:0.85rem; margin-left:12px;">{tier}</span>'
        if tier else ""
    )
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<div style="display:flex; align-items:center; flex-wrap:wrap; gap:8px;">'
        f'<span style="font-size:1.5rem; font-weight:800; color:#E8ECF1; letter-spacing:-0.01em;">{company_name}</span>'
        f'{tier_badge}'
        f'</div>'
        f'<div style="color:#9AA2B0; font-size:0.85rem; margin-top:4px;">'
        f'{info.get("sector", "")} · {info.get("industry", "")}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Company summary (business description) ───────────────
    desc = info.get("longBusinessSummary", "")
    if desc:
        short_desc = desc[:200] + ("..." if len(desc) > 200 else "")
        st.markdown(
            f'<div style="background:rgba(26,31,46,0.6); border-left:3px solid #4DA6FF; '
            f'border-radius:8px; padding:10px 16px; margin-bottom:12px; '
            f'font-size:0.88rem; color:#B0B8C4; line-height:1.6;">{short_desc}</div>',
            unsafe_allow_html=True,
        )

    # ── Top metric cards (Price, Market Cap, Screener Tier) ──
    c1, c2, c3 = st.columns(3)
    with c1:
        price = info.get("regularMarketPrice", "N/A")
        render_metric_card("Price", f"INR {price}", accent="#00D4AA")
    with c2:
        mcap = info.get("marketCap")
        mcap_display = f"{mcap/1e7:.0f} Cr" if mcap and mcap >= 1e7 else (f"{mcap:,.0f}" if mcap else "N/A")
        render_metric_card("Market Cap", mcap_display, accent="#4DA6FF")
    with c3:
        screener_score = profile.get("screener_score", "")
        if tier:
            render_metric_card("Screener Verdict", tier, delta=f"Score: {screener_score}" if screener_score else "", delta_up=tier in ("Buy Zone", "Watchlist"), accent=tier_color)
        else:
            render_metric_card("Screener Verdict", "Not Screened", accent="#8892A0")

    # ── Key Financials row ───────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        pe_t = info.get("trailingPE")
        pe_f = info.get("forwardPE")
        pe_delta = f"Fwd: {pe_f:.1f}" if pe_f else ""
        render_metric_card("P/E Ratio", f"{pe_t:.1f}" if pe_t else "N/A", delta=pe_delta, accent="#4DA6FF")
    with f2:
        roe = info.get("returnOnEquity")
        opm = info.get("operatingMargins")
        opm_delta = f"OPM: {_fmt_pct_sc(opm)}" if opm else ""
        render_metric_card("ROE", _fmt_pct_sc(roe), delta=opm_delta, delta_up=True, accent="#00D4AA")
    with f3:
        rev = info.get("totalRevenue")
        rev_growth = info.get("revenueGrowth")
        growth_delta = f"YoY: {_fmt_pct_sc(rev_growth)}" if rev_growth else ""
        render_metric_card("Revenue", _fmt_inr_sc(rev), delta=growth_delta,
                           delta_up=rev_growth and rev_growth > 0, accent="#B388FF")
    with f4:
        de = info.get("debtToEquity")
        cr = info.get("currentRatio")
        cr_delta = f"CR: {cr:.1f}" if cr else ""
        render_metric_card("Debt/Equity", f"{de:.1f}" if de else "N/A", delta=cr_delta, accent="#FFA726")

    # ── Quick Take (research report summary) ─────────────────
    report = research.get("final_report", "")
    if report:
        lines = [l.strip() for l in report.split("\n")
                 if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("---")]
        quick_take = " ".join(lines)[:1000]
        if quick_take:
            st.markdown(
                f'<div style="background:rgba(0,212,170,0.05); border:1px solid rgba(0,212,170,0.2); '
                f'border-radius:10px; padding:14px 18px; margin:8px 0 12px 0;">'
                f'<div style="color:#00D4AA; font-weight:700; font-size:0.85rem; margin-bottom:8px; '
                f'text-transform:uppercase; letter-spacing:0.06em;">Quick Take</div>'
                f'<div style="color:#C8CED6; font-size:0.9rem; line-height:1.65;">{quick_take}...</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Research Insights (agent-derived + fallback from yfinance) ──
    dcf = research.get("dcf_valuation", {})
    sentiment = research.get("sentiment_scores", {})
    peers = research.get("peer_comparison", [])
    indian_m = research.get("indian_metrics", {})

    # Fallback: compute DCF estimate from yfinance if agent didn't
    if not dcf or not dcf.get("intrinsic_value"):
        eps = info.get("trailingEps")
        growth = info.get("earningsGrowth") or info.get("revenueGrowth")
        price = info.get("regularMarketPrice")
        if eps and eps > 0 and price:
            # Simple Graham-style intrinsic value: EPS * (8.5 + 2g) where g = growth%
            g_pct = (growth * 100) if growth else 10
            intrinsic = eps * (8.5 + 2 * min(g_pct, 25))
            upside_pct = round(((intrinsic - price) / price) * 100, 1)
            dcf = {"intrinsic_value": round(intrinsic, 1), "upside_pct": upside_pct, "method": "Graham"}

    # Fallback: derive sentiment from news count + price momentum
    if not sentiment or (not sentiment.get("label") and not sentiment.get("overall_score")):
        price_change = info.get("52WeekChange")
        rec_key = info.get("recommendationKey", "")
        if price_change is not None:
            if price_change > 0.15:
                sentiment = {"label": "Bullish", "overall_score": 70, "method": "price_momentum"}
            elif price_change < -0.10:
                sentiment = {"label": "Bearish", "overall_score": 30, "method": "price_momentum"}
            else:
                sentiment = {"label": "Neutral", "overall_score": 50, "method": "price_momentum"}
        elif rec_key:
            rec_map = {"strong_buy": ("Bullish", 80), "buy": ("Bullish", 70), "hold": ("Neutral", 50), "sell": ("Bearish", 30), "strong_sell": ("Bearish", 20)}
            s_label, s_score = rec_map.get(rec_key, ("Neutral", 50))
            sentiment = {"label": s_label, "overall_score": s_score, "method": "analyst_rec"}

    # Fallback: compute ROCE from financials if agent didn't
    if not indian_m.get("roce"):
        fin_data = profile.get("financials_data", {})
        fin_df = fin_data.get("financials")
        bs_df = fin_data.get("balance_sheet")
        if fin_df is not None and bs_df is not None:
            try:
                import pandas as pd
                ebit = None
                for row in ["EBIT", "Operating Income", "Operating Revenue"]:
                    if row in fin_df.index:
                        val = fin_df.loc[row].dropna()
                        if len(val) > 0:
                            ebit = float(val.iloc[0])
                            break
                total_assets = None
                current_liab = None
                for row in ["Total Assets"]:
                    if row in bs_df.index:
                        val = bs_df.loc[row].dropna()
                        if len(val) > 0:
                            total_assets = float(val.iloc[0])
                for row in ["Current Liabilities", "Total Current Liabilities"]:
                    if row in bs_df.index:
                        val = bs_df.loc[row].dropna()
                        if len(val) > 0:
                            current_liab = float(val.iloc[0])
                            break
                if ebit and total_assets and current_liab and (total_assets - current_liab) > 0:
                    roce_val = round((ebit / (total_assets - current_liab)) * 100, 1)
                    rating = "Excellent" if roce_val > 20 else "Good" if roce_val > 12 else "Average" if roce_val > 8 else "Poor"
                    indian_m = {"roce": {"roce_value": f"{roce_val}%", "rating": rating}, "_method": "computed"}
            except Exception:
                pass

    st.markdown(
        '<div style="color:#E8ECF1; font-weight:700; font-size:0.85rem; margin:12px 0 8px 0; '
        'text-transform:uppercase; letter-spacing:0.06em;">Research Insights <span style="color:#9AA2B0; '
        'font-weight:400; font-size:0.75rem; text-transform:none;">(from AI agents)</span></div>',
        unsafe_allow_html=True,
    )
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        if dcf and dcf.get("intrinsic_value"):
            upside = dcf.get("upside_pct", "N/A")
            up_color = "#00D4AA" if isinstance(upside, (int, float)) and upside > 0 else "#FF4757"
            method_tag = f" ({dcf['method']})" if dcf.get("method") else ""
            render_metric_card("DCF Value", f"INR {dcf['intrinsic_value']}", delta=f"Upside: {upside}%{method_tag}", delta_up=isinstance(upside, (int, float)) and upside > 0, accent=up_color)
        else:
            render_metric_card("DCF Value", "N/A", accent="#8892A0")
    with rc2:
        s_label = sentiment.get("label", "")
        s_score = sentiment.get("overall_score")
        if s_label or (s_score is not None and s_score != 0):
            display_label = s_label or ("Bullish" if s_score and s_score > 60 else "Bearish" if s_score and s_score < 40 else "Neutral")
            s_colors = {"Bullish": "#00D4AA", "Bearish": "#FF4757", "Neutral": "#FFA726"}
            render_metric_card("Sentiment", display_label, delta=f"Score: {s_score}" if s_score else "", accent=s_colors.get(display_label, "#9AA2B0"))
        else:
            render_metric_card("Sentiment", "Neutral", accent="#FFA726")
    with rc3:
        roce = indian_m.get("roce", {})
        if roce:
            render_metric_card("ROCE", f"{roce.get('roce_value', 'N/A')}", delta=roce.get("rating", ""), accent="#B388FF")
        else:
            roe = info.get("returnOnEquity")
            if roe:
                roe_pct = f"{roe*100:.1f}%"
                rating = "Excellent" if roe > 0.20 else "Good" if roe > 0.12 else "Average"
                render_metric_card("ROE", roe_pct, delta=rating, accent="#B388FF")
            else:
                render_metric_card("ROCE", "N/A", accent="#8892A0")
    with rc4:
        if isinstance(peers, list) and peers:
            peer_count = len(peers)
            render_metric_card("Peers Compared", f"{peer_count} stocks", accent="#4DA6FF")
        else:
            render_metric_card("Peer Analysis", "N/A", accent="#8892A0")

    # ── USP Commentary (from screener data when available) ────
    screener_commentary = profile.get("screener_commentary", {})
    screener_usp = profile.get("screener_usp", {})

    if screener_commentary or screener_usp:
        # Rich commentary from screener pipeline
        with st.expander("Screening Analysis & Commentary", expanded=True):
            # Investment thesis + catalysts + risks
            thesis = screener_commentary.get("investment_thesis", "")
            catalysts = screener_commentary.get("key_catalysts", [])
            risks = screener_commentary.get("key_risks", [])
            watch = screener_commentary.get("what_to_watch", [])

            if thesis:
                st.markdown(
                    f'<div style="background:rgba(0,212,170,0.05); border-left:3px solid #00D4AA; '
                    f'border-radius:6px; padding:10px 14px; margin-bottom:10px;">'
                    f'<div style="color:#00D4AA; font-weight:700; font-size:0.8rem; text-transform:uppercase; '
                    f'letter-spacing:0.05em; margin-bottom:4px;">Investment Thesis</div>'
                    f'<div style="color:#C8CED6; font-size:0.88rem; line-height:1.6;">{thesis}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
            if catalysts:
                st.markdown(f"**Key Catalysts:** {', '.join(catalysts) if isinstance(catalysts, list) else catalysts}")
            if risks:
                st.markdown(f"**Key Risks:** {', '.join(risks) if isinstance(risks, list) else risks}")
            if watch:
                st.markdown(f"**What to Watch:** {', '.join(watch) if isinstance(watch, list) else watch}")

            # USP dimension summaries (compact)
            if screener_usp:
                st.markdown("---")
                dim_cols = st.columns(5)
                dim_names = [
                    ("geopolitical", "Geo Risk", "#FFA726"),
                    ("smart_money", "Smart Money", "#B388FF"),
                    ("regulatory", "Regulatory", "#4DA6FF"),
                    ("mgmt_credibility", "Mgmt Cred", "#00D4AA"),
                    ("promoter", "Promoter", "#FF6B6B"),
                ]
                for i, (dim_key, dim_label, dim_color) in enumerate(dim_names):
                    dim_data = screener_usp.get(dim_key, {})
                    if dim_data:
                        score = dim_data.get("score", dim_data.get("_composite", "N/A"))
                        level = dim_data.get("level", "")
                        with dim_cols[i]:
                            st.markdown(
                                f'<div style="text-align:center; padding:6px;">'
                                f'<div style="color:{dim_color}; font-size:1.2rem; font-weight:800;">{score}</div>'
                                f'<div style="color:#9AA2B0; font-size:0.72rem; text-transform:uppercase; '
                                f'letter-spacing:0.04em;">{dim_label}</div>'
                                f'{"<div style=&quot;color:#C8CED6; font-size:0.75rem;&quot;>" + level + "</div>" if level else ""}'
                                f'</div>', unsafe_allow_html=True,
                            )
    else:
        # Fallback: show raw USP data from deep dive profile (no screener run)
        geo = profile.get("geopolitical", {})
        reg = profile.get("regulatory", {})
        cred = profile.get("credibility", {})
        buying = profile.get("promoter_buying", {})
        if geo or reg or cred or buying:
            with st.expander("Risk & Governance", expanded=False):
                ucol1, ucol2 = st.columns(2)
                with ucol1:
                    if geo:
                        st.markdown("**Geopolitical Risk**")
                        st.write(f"- Overall: {geo.get('overall_score', 'N/A')}/100")
                        st.write(f"- Risk Level: {geo.get('risk_level', 'N/A')}")
                    if cred:
                        st.markdown("**Management Credibility**")
                        st.write(f"- Score: {cred.get('score', 'N/A')}/100 ({cred.get('method', '')})")
                with ucol2:
                    if reg:
                        st.markdown("**Regulatory**")
                        st.write(f"- Signal: {reg.get('net_signal', 'N/A')} (Score: {reg.get('score', 'N/A')})")
                    if buying:
                        st.markdown("**Promoter**")
                        st.write(f"- Signal: {buying.get('signal', 'N/A')}")


def render_deep_dive_chat(profile: dict, messages_key: str, ticker_key: str):
    """Render the deep dive chat interface (reusable helper).

    Args:
        profile: StockProfile dict.
        messages_key: Session state key for chat messages list.
        ticker_key: Session state key for ticker string.
    """
    info = profile.get("info", {})
    company_name = info.get("longName", info.get("shortName", profile.get("ticker", "")))

    st.markdown("")  # spacer
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(0,212,170,0.06),rgba(77,166,255,0.06)); '
        'border:1px solid #1E2A3A; border-radius:10px; padding:16px 20px; margin:12px 0;">'
        '<div style="color:#E8ECF1; font-size:1.1rem; font-weight:700; margin-bottom:4px;">Ask Questions About This Stock</div>'
        '<div style="color:#9AA2B0; font-size:0.82rem;">Products offered · Revenue by segment · Competitors · Competitive edge · Strengths · Risks · Red flags</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Display chat history
    messages = st.session_state.get(messages_key, [])
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask about {company_name}...", key=f"chat_input_{messages_key}"):
        if messages_key not in st.session_state:
            st.session_state[messages_key] = []
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from app.deep_dive_chat import ask_stock_question
                    response = ask_stock_question(
                        prompt,
                        profile,
                        st.session_state[messages_key][:-1],
                    )
                    st.markdown(response)
                    st.session_state[messages_key].append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {e}"
                    st.error(error_msg)
                    st.session_state[messages_key].append({"role": "assistant", "content": error_msg})


def _render_regime_banner(cdata: dict):
    """Render the market regime banner."""
    regime = cdata.get("regime_data", {})
    if regime:
        regime_name = regime.get("regime", "mixed").upper()
        regime_colors = {"BULL": "#2ecc71", "BEAR": "#e74c3c", "ROTATION": "#f39c12", "MIXED": "#95a5a6"}
        color = regime_colors.get(regime_name, "#95a5a6")
        vix = regime.get("vix")
        vix_text = f" | VIX: {vix:.1f}" if vix else ""
        st.markdown(
            f'<div style="background:{color}20; border:2px solid {color}; border-radius:10px; padding:12px; margin-bottom:16px;">'
            f'<span style="font-size:1.3rem; font-weight:700; color:{color};">Market Regime: {regime_name}</span>'
            f'<span style="margin-left:20px; color:#9AA2B0; font-size:0.9rem;">{regime.get("reasoning", "")}{vix_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _inject_screener_context(profile: dict, ticker: str, cdata: dict) -> dict:
    """Inject screener USP data, tier info, and commentary into a deep dive profile.

    Mutates and returns the profile dict with screener_usp, screener_tier,
    screener_score, and screener_commentary keys.
    """
    if not cdata:
        return profile

    # Inject USP cards
    usp_cards = cdata.get("usp_cards", {})
    clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
    if clean_ticker in usp_cards:
        usp_data = usp_cards[clean_ticker]
        profile["screener_usp"] = usp_data
        # Extract commentary from USP card if present
        commentary = {}
        if usp_data.get("investment_thesis"):
            commentary["investment_thesis"] = usp_data["investment_thesis"]
        if usp_data.get("key_catalysts"):
            commentary["key_catalysts"] = usp_data["key_catalysts"]
        if usp_data.get("key_risks"):
            commentary["key_risks"] = usp_data["key_risks"]
        if usp_data.get("what_to_watch"):
            commentary["what_to_watch"] = usp_data["what_to_watch"]
        if commentary:
            profile["screener_commentary"] = commentary

    # Inject tier info from category results
    cat_results = cdata.get("category_results", {})
    for cat_key, cat_stocks in cat_results.items():
        stock_list = cat_stocks if isinstance(cat_stocks, list) else cat_stocks.get("stocks", [])
        for s in stock_list:
            s_ticker = (s.get("ticker") or "").replace(".NS", "").replace(".BO", "")
            if s_ticker == clean_ticker:
                profile["screener_tier"] = s.get("tier_label", s.get("tier", ""))
                profile["screener_score"] = f"{s.get('score', 0)}/{s.get('active_total', '?')}"
                break

    return profile


def _render_screener_results(cdata: dict, category_key: str, criteria_groups: list, accent_color: str):
    """Render screener results with criteria panel and stock table."""
    _render_regime_banner(cdata)

    # Summary metrics
    summary = cdata.get("summary", {})
    if summary:
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            render_metric_card("Universe", summary.get("universe_size", "?"), accent="#4DA6FF")
        with sc2:
            render_metric_card("After Tech Filter", summary.get("after_tech_filter", "?"), accent="#FFA726")
        with sc3:
            render_metric_card("With Fundamentals", summary.get("with_fundamentals", "?"), accent="#B388FF")
        with sc4:
            render_metric_card("Total Survivors", summary.get("total_survivors", "?"), accent="#00D4AA")
        st.divider()

    # ── Extract stocks and bucket by tier (hoisted above tabs) ──
    cat_results = cdata.get("category_results", {})
    stocks = []
    if isinstance(cat_results, dict):
        stocks = cat_results.get(category_key, [])
        if isinstance(stocks, dict):
            stocks = stocks.get("stocks", [])
    elif isinstance(cat_results, list):
        for cr in cat_results:
            if cr.get("category") == category_key:
                stocks = cr.get("stocks", [])
                break

    tier_buckets: dict[str, list] = {t: [] for t in TIER_ORDER}
    for s in stocks:
        tier = s.get("tier", "failed")
        tier_buckets.setdefault(tier, []).append(s)

    # 4 Tabs: Results | USP | Debate | Deep Dive
    tab1, tab2, tab3, tab4 = st.tabs(["Screener Results", "USP Analysis", "AI Debate", "Stock Deep Dive"])

    with tab1:
        # Criteria panel
        with st.expander("Active Screening Criteria", expanded=False):
            render_criteria_panel(criteria_groups, accent_color)

        st.divider()

        if stocks:
            # ── Tier category cards ──────────────────────────────
            # Summary count cards
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                render_metric_card("★ Buy Zone", len(tier_buckets.get("buy_zone", [])),
                                   accent=TIER_CONFIG["buy_zone"]["color"])
            with tc2:
                render_metric_card("◉ Watchlist", len(tier_buckets.get("watchlist", [])),
                                   accent=TIER_CONFIG["watchlist"]["color"])
            with tc3:
                render_metric_card("◎ Monitor", len(tier_buckets.get("monitor", [])),
                                   accent=TIER_CONFIG["monitor"]["color"])
            with tc4:
                render_metric_card("⊘ Near Miss", len(tier_buckets.get("near_miss", [])),
                                   accent=TIER_CONFIG["near_miss"]["color"])

            st.divider()

            # ── Expandable tier cards with stock details + action buttons ──
            for tier_key in TIER_ORDER:
                tier_stocks = tier_buckets.get(tier_key, [])
                if not tier_stocks:
                    continue

                tc = TIER_CONFIG[tier_key]
                is_failed = tier_key == "failed"

                # Tier header card
                st.markdown(
                    f'<div style="background:rgba(26,31,46,0.8); backdrop-filter:blur(10px); '
                    f'border-left:4px solid {tc["color"]}; border-radius:10px; '
                    f'padding:12px 20px; margin:16px 0 8px 0; '
                    f'display:flex; align-items:center; gap:10px;">'
                    f'<span style="font-size:1.3rem;">{tc["icon"]}</span>'
                    f'<span style="font-size:1.05rem; font-weight:700; color:{tc["color"]}; letter-spacing:0.02em;">{tc["label"]}</span>'
                    f'<span style="color:#9AA2B0; font-size:0.85rem; font-weight:500;">— {len(tier_stocks)} stock{"s" if len(tier_stocks) != 1 else ""}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                if is_failed:
                    with st.expander(f"Show {len(tier_stocks)} failed stocks", expanded=False):
                        fail_rows = []
                        for s in tier_stocks[:50]:
                            gr = s.get("gate_results", [])
                            failed_gates = [g["name"] for g in gr if not g["passed"]]
                            fail_rows.append({
                                "Ticker": s.get("ticker", ""),
                                "Sector": s.get("sector", ""),
                                "Score": f"{s.get('score', 0)}/{s.get('active_total', '?')}",
                                "Failed Gates": ", ".join(failed_gates),
                            })
                        st.dataframe(pd.DataFrame(fail_rows), use_container_width=True, hide_index=True)
                    continue

                # Summary table
                rows = []
                for s in tier_stocks:
                    gate_cols = {}
                    for g in s.get("gate_results", []):
                        gate_cols[g["name"]] = "✓" if g["passed"] else "✗"
                    row = {
                        "Ticker": s.get("ticker", ""),
                        "Sector": s.get("sector", ""),
                        "Score": f"{s.get('score', 0)}/{s.get('active_total', '?')}",
                        **gate_cols,
                    }
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Per-stock expanders with criteria + action buttons
                for s in tier_stocks[:30]:
                    score = s.get("score", 0)
                    active_total = s.get("active_total", "?")
                    gate_results = s.get("gate_results", [])
                    sticker = s.get("ticker", "")

                    with st.expander(f"{sticker} — {score}/{active_total} | {s.get('sector', '')}"):
                        # Near miss reason
                        if tier_key == "near_miss" and s.get("near_miss_reason"):
                            st.markdown(
                                f'<div style="background:#FFA72615; border:1px solid #FFA72640; '
                                f'border-radius:8px; padding:10px 14px; margin-bottom:10px; font-size:13px; color:#FFA726; line-height:1.5;">'
                                f'{s["near_miss_reason"]}</div>',
                                unsafe_allow_html=True,
                            )

                        # Gate-by-gate breakdown
                        for g in gate_results:
                            gcolor = _GATE_TYPE_COLORS.get(g["gate_type"], "#8892A0")
                            status_icon = "✓" if g["passed"] else "✗"
                            status_color = "#00D4AA" if g["passed"] else "#FF4757"
                            passed_keys = g.get("passed_keys", [])
                            failed_keys = g.get("failed_keys", [])
                            checks = " ".join(
                                f'<span style="color:#00D4AA;">✓{k.replace("_"," ").title()}</span>' for k in passed_keys
                            ) + " " + " ".join(
                                f'<span style="color:#FF4757;">✗{k.replace("_"," ").title()}</span>' for k in failed_keys
                            )
                            st.markdown(
                                f'<div style="margin:4px 0; font-size:13px;">'
                                f'<span style="color:{status_color}; font-weight:700;">{status_icon}</span> '
                                f'<span style="color:{gcolor}; font-weight:600;">{g["name"]}</span> '
                                f'<span style="color:#8892A0;">({g["passed_count"]}/{g["active_count"]})</span> '
                                f'{checks}</div>',
                                unsafe_allow_html=True,
                            )

                        # ── Action buttons per stock ──
                        st.markdown("")  # spacer
                        btn1, btn2, _ = st.columns([1, 1, 2])
                        with btn1:
                            if st.button("Run AI Debate", key=f"debate_{category_key}_{tier_key}_{sticker}"):
                                with st.spinner(f"Running debate for {sticker}..."):
                                    try:
                                        from app.agents.debate_hybrid import run_hybrid_debate
                                        usp_cards_data = cdata.get("usp_cards", {})
                                        cat_results_list = cdata.get("category_results", [])
                                        if isinstance(cat_results_list, dict):
                                            cat_results_list = [{"category": k, "stocks": v if isinstance(v, list) else v.get("stocks", [])}
                                                                for k, v in cat_results_list.items()]
                                        debate_result = run_hybrid_debate(sticker, usp_cards_data, cat_results_list)
                                        if "debate_results" not in cdata:
                                            cdata["debate_results"] = []
                                        cdata["debate_results"].append(debate_result)
                                        st.success(f"Debate complete for {sticker}! Check AI Debate tab.")
                                    except Exception as e:
                                        st.error(f"Debate failed: {e}")
                        with btn2:
                            if st.button("Deep Dive", key=f"dd_{category_key}_{tier_key}_{sticker}"):
                                with st.spinner(f"Loading {sticker} profile..."):
                                    try:
                                        from app.tools.stock_deep_dive import build_stock_profile
                                        dd_prof = build_stock_profile(sticker, "NSE")
                                        _inject_screener_context(dd_prof, sticker, cdata)
                                        st.session_state.screener_dd_profile = dd_prof
                                        st.session_state.screener_dd_ticker = sticker
                                        st.session_state.screener_dd_messages = []
                                        st.success(f"{sticker} loaded! Switch to **Stock Deep Dive** tab.")
                                    except Exception as e:
                                        st.error(f"Failed to load {sticker}: {e}")
        else:
            st.info("No stocks found. Try running the screener.")

        # Export buttons
        st.divider()
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            json_data = json.dumps(cdata, indent=2, default=str)
            st.download_button("Download JSON", data=json_data,
                               file_name=f"{category_key}_{datetime.now():%Y%m%d}.json",
                               mime="application/json", key=f"{category_key}_json")
        with ex2:
            try:
                from app.sharing import generate_screener_pdf
                pdf_bytes = generate_screener_pdf(cdata)
                st.download_button("Download PDF", data=pdf_bytes,
                                   file_name=f"{category_key}_{datetime.now():%Y%m%d}.pdf",
                                   mime="application/pdf", key=f"{category_key}_pdf")
            except Exception as e:
                st.button("PDF unavailable", disabled=True, key=f"{category_key}_pdf_dis")
        with ex3:
            try:
                # Flat raw data columns per criteria
                all_criteria = MOMENTUM_CRITERIA if category_key == "Momentum" else VALUE_CRITERIA
                csv_rows = []
                for s in stocks:
                    row = {
                        "Ticker": s.get("ticker", ""),
                        "Sector": s.get("sector", ""),
                        "Tier": s.get("tier_label", s.get("tier", "")),
                        "Score": s.get("score", 0),
                        "Active_Total": s.get("active_total", ""),
                        "All_Gates_Pass": s.get("all_gates_pass", ""),
                    }
                    # Gate results
                    for g in s.get("gate_results", []):
                        row[f"Gate:{g['name']}"] = "PASS" if g["passed"] else "FAIL"
                        row[f"Gate:{g['name']}:detail"] = f"{g['passed_count']}/{g['active_count']}"
                    # Per-criteria raw data — auto-capture all fields from screen results
                    details = s.get("criteria_details", {})
                    passed_set = set(s.get("criteria_passed", []))
                    for ckey in all_criteria:
                        row[f"{ckey}"] = "PASS" if ckey in passed_set else ("FAIL" if ckey in details else "")
                        if ckey in details:
                            d = details[ckey]
                            for dk, dv in d.items():
                                if dk in ("ticker", "passed", "sector"):
                                    continue
                                row[f"{ckey}:{dk}"] = str(dv) if isinstance(dv, (list, dict)) else dv
                    csv_rows.append(row)
                if csv_rows:
                    raw_df = pd.DataFrame(csv_rows)
                    raw_csv = raw_df.to_csv(index=False)
                    st.download_button("Download Raw Data CSV", data=raw_csv,
                                       file_name=f"{category_key}_raw_data_{datetime.now():%Y%m%d}.csv",
                                       mime="text/csv", key=f"{category_key}_raw_csv")
                else:
                    st.button("No data to export", disabled=True, key=f"{category_key}_raw_dis")
            except Exception as e:
                st.button(f"CSV error: {e}", disabled=True, key=f"{category_key}_raw_err")

    # Tab 2: USP Analysis
    with tab2:
        try:
            from app.ui.usp_cards import (
                render_usp_heatmap, render_usp_card, render_usp_radar,
                render_analyst_panel,
                render_contradiction_alerts, transform_usp_data,
                render_portfolio_insights,
            )

            usp_cards = cdata.get("usp_cards", {})
            if not usp_cards and cdata.get("usp_scores"):
                usp_cards = transform_usp_data(cdata["usp_scores"])

            if usp_cards:
                st.markdown("### USP Heatmap")
                render_usp_heatmap(usp_cards)

                cat_results_list = cdata.get("category_results", [])
                if isinstance(cat_results_list, dict):
                    cat_results_list = [{"category": k, "stocks": v} if isinstance(v, list) else v
                                       for k, v in cat_results_list.items()]
                render_contradiction_alerts(
                    {c.get("category", ""): c.get("stocks", []) for c in cat_results_list},
                    usp_cards,
                )

                # Portfolio-level insights (cross-stock comparison)
                usp_raw = cdata.get("usp_scores", {})
                portfolio_data = usp_raw.get("_portfolio_insights", {})
                if portfolio_data:
                    render_portfolio_insights(portfolio_data.get("text", ""))

                st.markdown("### Per-Stock USP Analysis")
                for tkr in sorted(
                    [t for t in usp_cards.keys() if not t.startswith("_")],
                    key=lambda t: usp_cards[t].get("_composite", 0),
                    reverse=True,
                ):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        render_usp_card(tkr, usp_cards[tkr])
                    with c2:
                        render_usp_radar(tkr, usp_cards[tkr])
                        render_analyst_panel(tkr, usp_cards[tkr])
            else:
                st.info("No USP data available. Run the screener first.")
        except Exception as e:
            st.error(f"USP rendering failed: {e}")

    # Tab 3: AI Debate
    with tab3:
        debate_results = cdata.get("debate_results", [])
        if debate_results:
            st.markdown("### Bull vs Bear Debate")
            for debate in debate_results:
                tkr = debate.get("ticker", "?")
                verdict = debate.get("verdict", {})
                conviction = verdict.get("conviction_score", 5)
                rec = verdict.get("recommendation", "HOLD")

                conv_color = "#00D4AA" if conviction >= 7 else "#FFA726" if conviction >= 5 else "#FF4757"
                pct = conviction * 10
                # SVG conviction gauge
                dash = 2 * 3.14159 * 40  # circumference
                fill = dash * pct / 100

                with st.expander(f"{tkr} — {rec} (Conviction: {conviction}/10)", expanded=True):
                    gc1, gc2 = st.columns([1, 3])
                    with gc1:
                        st.markdown(
                            f'<div style="text-align:center;">'
                            f'<svg width="100" height="100" viewBox="0 0 100 100">'
                            f'<circle cx="50" cy="50" r="40" fill="none" stroke="#1E2A3A" stroke-width="8"/>'
                            f'<circle cx="50" cy="50" r="40" fill="none" stroke="{conv_color}" stroke-width="8" '
                            f'stroke-dasharray="{fill:.1f} {dash:.1f}" stroke-linecap="round" '
                            f'transform="rotate(-90 50 50)"/>'
                            f'<text x="50" y="46" text-anchor="middle" fill="{conv_color}" '
                            f'font-size="20" font-weight="800">{conviction}</text>'
                            f'<text x="50" y="62" text-anchor="middle" fill="#8892A0" font-size="9">/10</text>'
                            f'</svg>'
                            f'<div style="color:{conv_color}; font-weight:700; font-size:0.9rem;">{rec}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with gc2:
                        st.markdown(
                            f'<div class="debate-card" style="border-left:3px solid {conv_color};">'
                            f'<div style="color:#E8ECF1; font-size:0.92rem; line-height:1.6;">{verdict.get("reasoning", "")}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    factors = verdict.get("key_factors", [])
                    if factors:
                        tags = "".join(
                            f'<span class="signal-tag" style="background:rgba(0,212,170,0.1); '
                            f'color:#00D4AA; border:1px solid rgba(0,212,170,0.3);">{f}</span>'
                            for f in factors
                        )
                        st.markdown(f'<div style="margin:8px 0;">{tags}</div>', unsafe_allow_html=True)

                    col_bull, col_bear = st.columns(2)
                    with col_bull:
                        st.markdown(
                            '<div class="debate-card" style="border-left:3px solid #00D4AA;">'
                            '<div style="color:#00D4AA; font-weight:700; margin-bottom:8px;">BULL CASE</div>',
                            unsafe_allow_html=True,
                        )
                        for i, arg in enumerate(debate.get("bull_arguments", []), 1):
                            st.markdown(f"**Round {i}:** {arg}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col_bear:
                        st.markdown(
                            '<div class="debate-card" style="border-left:3px solid #FF4757;">'
                            '<div style="color:#FF4757; font-weight:700; margin-bottom:8px;">BEAR CASE</div>',
                            unsafe_allow_html=True,
                        )
                        for i, arg in enumerate(debate.get("bear_arguments", []), 1):
                            st.markdown(f"**Round {i}:** {arg}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    bull_str = verdict.get("bull_strength", 5)
                    bear_str = verdict.get("bear_strength", 5)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            f'<div style="margin:6px 0;"><span style="color:#00D4AA; font-size:12px; font-weight:600; letter-spacing:0.04em;">BULL STRENGTH {bull_str}/10</span>'
                            f'<div style="background:#1A1F2E; border-radius:5px; height:10px; margin-top:6px;">'
                            f'<div style="background:linear-gradient(90deg,#00D4AA,#00E4BA); width:{bull_str*10}%; height:100%; border-radius:5px;"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f'<div style="margin:6px 0;"><span style="color:#FF4757; font-size:12px; font-weight:600; letter-spacing:0.04em;">BEAR STRENGTH {bear_str}/10</span>'
                            f'<div style="background:#1A1F2E; border-radius:5px; height:10px; margin-top:6px;">'
                            f'<div style="background:linear-gradient(90deg,#FF4757,#FF6E7A); width:{bear_str*10}%; height:100%; border-radius:5px;"></div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No debate results. Enable 'Run AI Debate' in the sidebar and run the screener.")

    # Tab 4: Stock Deep Dive (auto-picks top stock from screener)
    with tab4:
        # Determine top stock from Buy Zone > Watchlist > Monitor
        top_stock_ticker = None
        for tk in ["buy_zone", "watchlist", "monitor"]:
            bucket = tier_buckets.get(tk, [])
            if bucket:
                top_stock_ticker = bucket[0].get("ticker")
                break

        if not top_stock_ticker:
            st.info("No qualifying stocks for deep dive. Run the screener first.")
        else:
            # If user clicked "Deep Dive" on a specific stock, use that instead
            if st.session_state.screener_dd_ticker:
                top_stock_ticker = st.session_state.screener_dd_ticker

            st.markdown(
                f'<div style="background:rgba(26,31,46,0.8); border-left:4px solid #B388FF; '
                f'border-radius:10px; padding:12px 20px; margin-bottom:16px;">'
                f'<span style="font-size:1rem; font-weight:700; color:#B388FF;">Deep Dive Target: </span>'
                f'<span style="font-size:1.1rem; font-weight:800; color:#E8ECF1;">{top_stock_ticker}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Allow switching to a different stock from screener results
            all_tickers = [s.get("ticker") for s in stocks if s.get("ticker")]
            if all_tickers:
                selected = st.selectbox(
                    "Select stock for deep dive",
                    all_tickers,
                    index=all_tickers.index(top_stock_ticker) if top_stock_ticker in all_tickers else 0,
                    key=f"dd_select_{category_key}",
                )
                if selected != top_stock_ticker:
                    top_stock_ticker = selected
                    st.session_state.screener_dd_ticker = selected
                    st.session_state.screener_dd_profile = None
                    st.session_state.screener_dd_messages = []

            # Load profile
            dd_profile = st.session_state.screener_dd_profile
            if dd_profile and st.session_state.screener_dd_ticker == top_stock_ticker:
                render_stock_profile_scorecard(dd_profile)
                render_deep_dive_chat(dd_profile, "screener_dd_messages", "screener_dd_ticker")
            else:
                if st.button(f"Load Deep Dive for {top_stock_ticker}", key=f"load_dd_{category_key}", type="primary"):
                    progress = st.progress(0, text=f"Loading {top_stock_ticker} profile...")

                    def _dd_prog(fraction: float, message: str):
                        progress.progress(min(fraction, 1.0), text=message)

                    try:
                        from app.tools.stock_deep_dive import build_stock_profile
                        profile = build_stock_profile(top_stock_ticker, "NSE", progress_cb=_dd_prog)
                        # Bridge screener data into the profile
                        _inject_screener_context(profile, top_stock_ticker, cdata)
                        st.session_state.screener_dd_profile = profile
                        st.session_state.screener_dd_ticker = top_stock_ticker
                        st.session_state.screener_dd_messages = []
                        progress.empty()
                        st.rerun()
                    except Exception as e:
                        progress.empty()
                        st.error(f"Failed to load profile: {e}")


# ── Trend Rider Page ────────────────────────────────────
if analysis_mode == "Trend Rider":
    st.markdown(
        '<div class="section-hero" style="border-left:4px solid #00D4AA;">'
        '<h3>Trend Rider</h3>'
        '<p>5 gates · 17 criteria · Strong stocks in strong sectors · RSI 55-75 · Volume confirmed</p>'
        '</div>', unsafe_allow_html=True,
    )

    if run_momentum:
        _run_screener_for_category("Momentum", run_debate)

    cdata = st.session_state.category_screener_data
    if cdata and "Momentum" in (cdata.get("category_results") or {}):
        _render_screener_results(cdata, "Momentum", MOM_GATE_DISPLAY, "#00D4AA")
    else:
        st.markdown("Click **Run Trend Rider** in the sidebar to begin.")
        st.divider()
        st.markdown("#### Active Screening Criteria (5 Gates, 17 Criteria)")
        render_criteria_panel(MOM_GATE_DISPLAY, "#00D4AA")

    st.stop()

# ── Turnaround Hunter Page ───────────────────────────────────────
if analysis_mode == "Turnaround Hunter":
    st.markdown(
        '<div class="section-hero" style="border-left:4px solid #4DA6FF;">'
        '<h3>Turnaround Hunter</h3>'
        '<p>Turnaround candidates at valuation floors · 28 criteria · RSI &lt;40 · F-Score ≥ 5</p>'
        '</div>', unsafe_allow_html=True,
    )

    if run_value:
        _run_screener_for_category("ValueBottom", run_debate)

    cdata = st.session_state.category_screener_data
    if cdata and "ValueBottom" in (cdata.get("category_results") or {}):
        _render_screener_results(cdata, "ValueBottom", VAL_GATE_DISPLAY, "#4DA6FF")
    else:
        st.markdown("Click **Run Turnaround Hunter** in the sidebar to begin.")
        st.divider()
        st.markdown("#### Active Screening Criteria (7 Gates, 33 Criteria)")
        render_criteria_panel(VAL_GATE_DISPLAY, "#4DA6FF")

    st.stop()

# ── Market Breadth Mode ──────────────────────────────────────
if analysis_mode == "Market Breadth":
    if run_breadth:
        progress_bar = st.progress(0, text="Starting Nifty 500 breadth analysis...")

        def _update_progress(fraction: float, message: str):
            progress_bar.progress(min(fraction, 1.0), text=message)

        try:
            from app.tools.market_breadth import run_market_breadth_analysis
            st.session_state.breadth_data = run_market_breadth_analysis(
                progress_cb=_update_progress,
                max_stocks=50 if test_mode else None,
            )
            progress_bar.empty()
            st.success("Market breadth analysis complete!")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Breadth analysis failed: {e}")

    bdata = st.session_state.breadth_data
    if bdata:
        render_breadth_overview(bdata["breadth"])
        st.divider()
        render_sector_rs_table(bdata["sector_rs"])
        st.divider()
        render_sector_outperformers(bdata["outperformers"], bdata["strong_sectors"])
        st.divider()
        render_52w_lists(bdata["breadth"])
    else:
        st.markdown("""
        ### Market Breadth Analysis

        Click **Run Breadth Analysis** in the sidebar to:
        - Count stocks at **52-week highs / lows** across ~500 Nifty constituents
        - Calculate **sector relative strength** vs Nifty 500 over 1M, 3M, 6M, 1Y
        - Identify **strong sectors** and **top 3 outperformers** in each

        First run takes ~45-60 seconds (fetching data for 500 stocks in batches). Subsequent runs are near-instant (cached for 10 minutes).
        """)

    st.stop()  # Don't render stock analysis UI below

# Handle button actions
if run_full:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        st.session_state.last_ticker = ticker
        with st.spinner(f"Validating ticker **{ticker}**..."):
            if not validate_ticker(ticker):
                st.error(
                    f"Ticker **{ticker}** not found on Yahoo Finance. "
                    f"Please check the symbol. Examples: RELIANCE.NS, TCS.NS, HDFCBANK.NS"
                )
            else:
                with st.spinner(f"Running full research pipeline for **{ticker}**... This may take 2-3 minutes."):
                    try:
                        run_full_pipeline(ticker)
                        st.success(f"Research complete for {ticker}!")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

elif run_data:
    if ticker:
        from app.agents.data_agent import data_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"Gathering financial data for {ticker}..."):
            run_single_agent("data_agent", data_node, ticker)
            st.success("Data collection complete!")

elif run_analysis:
    if ticker:
        from app.agents.analysis_agent import analysis_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"Running financial analysis for {ticker}..."):
            run_single_agent("analysis_agent", analysis_node, ticker)
            st.success("Analysis complete!")

elif run_sentiment:
    if ticker:
        from app.agents.sentiment_agent import sentiment_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"Analyzing sentiment for {ticker}..."):
            run_single_agent("sentiment_agent", sentiment_node, ticker)
            st.success("Sentiment analysis complete!")

elif run_report:
    if ticker:
        from app.agents.report_agent import report_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"Generating investment report for {ticker}..."):
            run_single_agent("report_agent", report_node, ticker)
            st.success("Report generated!")

# ── Display Results ──────────────────────────────────────────
state = st.session_state.research_state

if state:
    # Agent status bar
    status_map = st.session_state.agent_status
    if status_map:
        st.markdown("#### Agent Status")
        scols = st.columns(4)
        agent_labels = {
            "data_agent": "Data",
            "analysis_agent": "Analysis",
            "sentiment_agent": "Sentiment",
            "report_agent": "Report",
        }
        for i, (agent, label) in enumerate(agent_labels.items()):
            with scols[i]:
                s = status_map.get(agent, "")
                if s == "done":
                    st.success(f"{label} Done")
                elif s == "running":
                    st.info(f"{label} Running...")
                elif s == "error":
                    st.error(f"{label} Error")
                elif s == "pending":
                    st.warning(f"{label} Pending")
        st.divider()

    # Company Overview
    company_info = state.get("company_info", {})
    if company_info:
        render_company_overview(company_info)
        st.divider()

    # Price Chart
    financials = state.get("financials", {})
    if financials:
        render_price_chart(financials)
        st.divider()

    # Financial Ratios
    ratios = state.get("ratios", {})
    if ratios:
        render_ratios(ratios)
        st.divider()

    # Indian Metrics (ROCE, Shareholding, FII/DII)
    indian_metrics = state.get("indian_metrics", {})
    if indian_metrics:
        render_indian_metrics(indian_metrics)
        st.divider()

    # DCF Valuation
    dcf = state.get("dcf_valuation", {})
    if dcf:
        render_dcf(dcf)
        st.divider()

    # Peer Comparison
    peers = state.get("peer_comparison", [])
    if peers:
        render_peer_comparison(peers)
        st.divider()

    # Sentiment
    sentiment = state.get("sentiment_scores", {})
    news = state.get("news_summaries", [])
    if sentiment or news:
        render_sentiment(sentiment, news)
        st.divider()

    # Corporate Actions
    actions = state.get("corporate_actions", [])
    if actions:
        render_corporate_actions(actions)
        st.divider()

    # Investment Score
    investment_score = state.get("investment_score", {})
    if investment_score:
        render_investment_score(investment_score)
        st.divider()

    # Final Report
    report = state.get("final_report", "")
    if report:
        render_report(report)
        st.divider()

    # Errors
    errors = state.get("errors", [])
    if errors:
        with st.expander("Errors / Warnings"):
            for err in errors:
                st.warning(err)

    # Raw data
    with st.expander("Raw State Data (Debug)"):
        display_state = {k: v for k, v in state.items() if k != "messages"}
        st.json(json.loads(json.dumps(display_state, default=str)))

else:
    # Welcome screen
    st.markdown("""
    ### Welcome!

    Enter an **Indian stock ticker** in the sidebar and click **Run Full Analysis** to generate
    a comprehensive AI-powered equity research report.

    **What you'll get:**
    - Financial data (income statement, balance sheet, cash flows in INR)
    - Quantitative analysis (ratios, ROCE, DCF with Indian WACC, peer comparison)
    - Indian-specific metrics (promoter holding, FII/DII activity, pledge status)
    - Sentiment analysis from Indian news sources (MoneyControl, ET, Google News India)
    - Professional investment memo with bull/bear cases

    You can also run individual agents using the buttons in the sidebar.
    """)

    st.markdown("#### Popular Indian Stocks")
    qcols = st.columns(6)
    popular = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS"]
    for i, t in enumerate(popular):
        with qcols[i]:
            st.code(t)
