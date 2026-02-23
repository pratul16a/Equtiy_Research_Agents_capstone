"""Streamlit UI — AI-Powered Equity Research Analyst."""

from __future__ import annotations

import json
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Equity Research Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #1f77b4;
    }
    .agent-status {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    .status-running { background: #fff3cd; color: #856404; }
    .status-done { background: #d4edda; color: #155724; }
    .status-error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=64)
    st.markdown("## AI Equity Research")
    st.markdown("Multi-agent system powered by LangGraph")
    st.divider()

    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        max_chars=10,
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Enter a US stock ticker symbol",
    ).upper().strip()

    run_full = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Run Individual Agents")
    col1, col2 = st.columns(2)
    with col1:
        run_data = st.button("📥 Data", use_container_width=True)
        run_sentiment = st.button("💬 Sentiment", use_container_width=True)
    with col2:
        run_analysis = st.button("📈 Analysis", use_container_width=True)
        run_report = st.button("📝 Report", use_container_width=True)

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **Agents:**
    - 📥 **Data Agent** — Fetches financials & SEC filings
    - 📈 **Analysis Agent** — Ratios, DCF, peer comparison
    - 💬 **Sentiment Agent** — News & sentiment scoring
    - 📝 **Report Agent** — Investment memo with RAG context
    """)


# ── Session State Init ───────────────────────────────────────
if "research_state" not in st.session_state:
    st.session_state.research_state = None
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""


# ── Helper Functions ─────────────────────────────────────────
def format_market_cap(val):
    """Format market cap to human-readable string."""
    if not val:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def safe_get(data, key, default="N/A"):
    """Safely get a value from a dict."""
    if not data:
        return default
    val = data.get(key, default)
    return val if val is not None else default


def render_company_overview(company_info):
    """Render company overview section."""
    st.markdown("### 🏢 Company Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${safe_get(company_info, 'current_price', 0):.2f}")
    with col2:
        st.metric("Market Cap", format_market_cap(safe_get(company_info, "market_cap", 0)))
    with col3:
        st.metric("P/E (Trailing)", f"{safe_get(company_info, 'pe_trailing', 0):.2f}")
    with col4:
        st.metric("Beta", f"{safe_get(company_info, 'beta', 0):.2f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Sector", safe_get(company_info, "sector"))
    with col6:
        st.metric("Industry", safe_get(company_info, "industry"))
    with col7:
        st.metric("52W High", f"${safe_get(company_info, '52_week_high', 0):.2f}")
    with col8:
        st.metric("52W Low", f"${safe_get(company_info, '52_week_low', 0):.2f}")

    desc = safe_get(company_info, "description", "")
    if desc and desc != "N/A":
        with st.expander("📋 Business Description"):
            st.write(desc)


def render_price_chart(financials):
    """Render stock price chart from historical data."""
    price_data = financials.get("get_stock_price_history", {})
    recent_prices = price_data.get("recent_prices", [])
    if not recent_prices:
        return

    st.markdown("### 📉 Stock Price (Recent)")
    dates = [p["date"] for p in recent_prices]
    closes = [p["close"] for p in recent_prices]
    volumes = [p["volume"] for p in recent_prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode="lines+markers",
        name="Close Price", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Price summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Period Return", f"{safe_get(price_data, 'period_return_pct', 0):.2f}%")
    with col2:
        st.metric("Period High", f"${safe_get(price_data, 'period_high', 0):.2f}")
    with col3:
        st.metric("Period Low", f"${safe_get(price_data, 'period_low', 0):.2f}")
    with col4:
        st.metric("Avg Volume", f"{safe_get(price_data, 'avg_volume', 0):,.0f}")


def render_ratios(ratios):
    """Render financial ratios."""
    if not ratios:
        return
    st.markdown("### 📊 Financial Ratios")

    # Group ratios
    valuation = {k: v for k, v in ratios.items() if k in [
        "pe_trailing", "pe_forward", "pb_ratio", "ps_ratio",
        "ev_to_ebitda", "ev_to_revenue", "peg_ratio"
    ]}
    profitability = {k: v for k, v in ratios.items() if k in [
        "profit_margin", "operating_margin", "gross_margin", "roe", "roa"
    ]}
    leverage = {k: v for k, v in ratios.items() if k in [
        "debt_to_equity", "current_ratio", "quick_ratio"
    ]}
    growth = {k: v for k, v in ratios.items() if k in [
        "earnings_growth", "revenue_growth"
    ]}

    tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Profitability", "Leverage", "Growth"])

    def _ratio_table(data, fmt_pct=False):
        for k, v in data.items():
            label = k.replace("_", " ").title()
            if fmt_pct and isinstance(v, (int, float)):
                st.metric(label, f"{v * 100:.2f}%")
            elif isinstance(v, (int, float)):
                st.metric(label, f"{v:.2f}")
            else:
                st.metric(label, str(v))

    with tab1:
        cols = st.columns(min(len(valuation), 4)) if valuation else [st.container()]
        for i, (k, v) in enumerate(valuation.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                st.metric(label, f"{v:.2f}" if isinstance(v, (int, float)) else str(v))

    with tab2:
        cols = st.columns(min(len(profitability), 4)) if profitability else [st.container()]
        for i, (k, v) in enumerate(profitability.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                if isinstance(v, (int, float)):
                    st.metric(label, f"{v * 100:.2f}%")
                else:
                    st.metric(label, str(v))

    with tab3:
        cols = st.columns(min(len(leverage), 4)) if leverage else [st.container()]
        for i, (k, v) in enumerate(leverage.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                st.metric(label, f"{v:.2f}" if isinstance(v, (int, float)) else str(v))

    with tab4:
        cols = st.columns(min(len(growth), 4)) if growth else [st.container()]
        for i, (k, v) in enumerate(growth.items()):
            with cols[i % len(cols)]:
                label = k.replace("_", " ").title()
                if isinstance(v, (int, float)):
                    st.metric(label, f"{v * 100:.2f}%")
                else:
                    st.metric(label, str(v))


def render_dcf(dcf):
    """Render DCF valuation results."""
    if not dcf:
        return
    st.markdown("### 💰 DCF Valuation")

    col1, col2, col3 = st.columns(3)
    with col1:
        intrinsic = safe_get(dcf, "intrinsic_value_per_share", 0)
        st.metric("Intrinsic Value", f"${intrinsic:.2f}" if isinstance(intrinsic, (int, float)) else str(intrinsic))
    with col2:
        current = safe_get(dcf, "current_price", 0)
        st.metric("Current Price", f"${current:.2f}" if isinstance(current, (int, float)) else str(current))
    with col3:
        upside = safe_get(dcf, "upside_pct", 0)
        if isinstance(upside, (int, float)):
            st.metric("Upside/Downside", f"{upside:+.2f}%",
                       delta=f"{upside:+.2f}%",
                       delta_color="normal")
        else:
            st.metric("Upside/Downside", str(upside))

    # Assumptions
    assumptions = dcf.get("assumptions", {})
    if assumptions:
        with st.expander("📐 DCF Assumptions"):
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                st.write(f"**Growth Rate:** {assumptions.get('growth_rate', 'N/A')}")
            with ac2:
                st.write(f"**Discount Rate:** {assumptions.get('discount_rate', 'N/A')}")
            with ac3:
                st.write(f"**Terminal Growth:** {assumptions.get('terminal_growth', 'N/A')}")
            with ac4:
                st.write(f"**Years:** {assumptions.get('projection_years', 'N/A')}")

    # Projected FCFs chart
    projected = dcf.get("projected_fcfs", [])
    if projected:
        years = [f"Year {p['year']}" for p in projected]
        fcfs = [p["discounted_fcf"] for p in projected]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=years, y=fcfs, marker_color="#2ca02c", name="Discounted FCF ($B)"))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Discounted FCF ($B)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_peer_comparison(peers):
    """Render peer comparison table."""
    if not peers:
        return
    st.markdown("### 🏆 Peer Comparison")

    import pandas as pd
    df = pd.DataFrame(peers)
    # Rename columns for display
    col_map = {
        "ticker": "Ticker", "name": "Company", "market_cap_b": "Mkt Cap ($B)",
        "pe_trailing": "P/E", "pe_forward": "Fwd P/E", "ev_ebitda": "EV/EBITDA",
        "profit_margin": "Margin", "roe": "ROE", "revenue_growth": "Rev Growth",
        "debt_to_equity": "D/E",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Format percentage columns
    for col in ["Margin", "ROE", "Rev Growth"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) and x is not None else "N/A")

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_sentiment(sentiment_scores, news_summaries):
    """Render sentiment analysis."""
    st.markdown("### 💬 Sentiment Analysis")

    if sentiment_scores:
        col1, col2 = st.columns([1, 3])
        with col1:
            score = sentiment_scores.get("overall_score", 0)
            label = sentiment_scores.get("overall_label", "neutral")
            color = "🟢" if label == "bullish" else "🔴" if label == "bearish" else "🟡"
            st.metric("Sentiment Score", f"{score:+.2f}")
            st.markdown(f"**{color} {label.upper()}**")
        with col2:
            analysis = sentiment_scores.get("analysis", "")
            if analysis:
                st.markdown(analysis[:800])

    if news_summaries:
        with st.expander(f"📰 Recent News ({len(news_summaries)} articles)"):
            for i, news in enumerate(news_summaries):
                st.markdown(f"**{i+1}.** {news}")
                if i < len(news_summaries) - 1:
                    st.divider()


def render_report(report):
    """Render the final investment report."""
    if not report:
        return
    st.markdown("### 📝 Investment Research Report")
    st.markdown(report)

    # Download button
    st.download_button(
        label="📥 Download Report as Markdown",
        data=report,
        file_name=f"research_report_{st.session_state.last_ticker}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )


# ── Agent Runners ────────────────────────────────────────────
def run_single_agent(agent_name: str, node_func, ticker: str):
    """Run a single agent and update session state."""
    from app.state import create_initial_state

    state = st.session_state.research_state or create_initial_state(ticker)
    state["ticker"] = ticker

    st.session_state.agent_status[agent_name] = "running"
    try:
        result = node_func(state)
        # Merge result into state
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


def run_full_pipeline(ticker: str):
    """Run the entire LangGraph pipeline."""
    from app.graph import run_research

    agents = ["data_agent", "analysis_agent", "sentiment_agent", "report_agent"]
    for a in agents:
        st.session_state.agent_status[a] = "pending"

    result = run_research(ticker)
    st.session_state.research_state = result
    st.session_state.last_ticker = ticker
    for a in agents:
        st.session_state.agent_status[a] = "done"


# ── Main Content ─────────────────────────────────────────────
st.markdown('<p class="main-header">📊 AI Equity Research Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-agent investment research powered by LangGraph + LLM</p>', unsafe_allow_html=True)
st.divider()

# Handle button actions
if run_full:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        st.session_state.last_ticker = ticker
        with st.spinner(f"🔄 Running full research pipeline for **{ticker}**... This may take 2-3 minutes."):
            try:
                run_full_pipeline(ticker)
                st.success(f"✅ Research complete for {ticker}!")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

elif run_data:
    if ticker:
        from app.agents.data_agent import data_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"📥 Gathering financial data for {ticker}..."):
            run_single_agent("data_agent", data_node, ticker)
            st.success("✅ Data collection complete!")

elif run_analysis:
    if ticker:
        from app.agents.analysis_agent import analysis_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"📈 Running financial analysis for {ticker}..."):
            run_single_agent("analysis_agent", analysis_node, ticker)
            st.success("✅ Analysis complete!")

elif run_sentiment:
    if ticker:
        from app.agents.sentiment_agent import sentiment_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"💬 Analyzing sentiment for {ticker}..."):
            run_single_agent("sentiment_agent", sentiment_node, ticker)
            st.success("✅ Sentiment analysis complete!")

elif run_report:
    if ticker:
        from app.agents.report_agent import report_node
        st.session_state.last_ticker = ticker
        with st.spinner(f"📝 Generating investment report for {ticker}..."):
            run_single_agent("report_agent", report_node, ticker)
            st.success("✅ Report generated!")

# ── Display Results ──────────────────────────────────────────
state = st.session_state.research_state

if state:
    # Agent status bar
    status_map = st.session_state.agent_status
    if status_map:
        st.markdown("#### Agent Status")
        scols = st.columns(4)
        agent_labels = {
            "data_agent": "📥 Data",
            "analysis_agent": "📈 Analysis",
            "sentiment_agent": "💬 Sentiment",
            "report_agent": "📝 Report",
        }
        for i, (agent, label) in enumerate(agent_labels.items()):
            with scols[i]:
                s = status_map.get(agent, "")
                if s == "done":
                    st.success(f"{label} ✅")
                elif s == "running":
                    st.info(f"{label} ⏳")
                elif s == "error":
                    st.error(f"{label} ❌")
                elif s == "pending":
                    st.warning(f"{label} ⏳")
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

    # Final Report
    report = state.get("final_report", "")
    if report:
        render_report(report)
        st.divider()

    # Errors
    errors = state.get("errors", [])
    if errors:
        with st.expander("⚠️ Errors / Warnings"):
            for err in errors:
                st.warning(err)

    # Raw data expander
    with st.expander("🔍 Raw State Data (Debug)"):
        # Filter out messages for display
        display_state = {k: v for k, v in state.items() if k != "messages"}
        st.json(json.loads(json.dumps(display_state, default=str)))

else:
    # Welcome screen
    st.markdown("""
    ### 👋 Welcome!

    Enter a **stock ticker** in the sidebar and click **Run Full Analysis** to generate
    a comprehensive AI-powered equity research report.

    **What you'll get:**
    - 📥 Financial data (income statement, balance sheet, cash flows)
    - 📈 Quantitative analysis (ratios, DCF valuation, peer comparison)
    - 💬 Sentiment analysis (news, market mood)
    - 📝 Professional investment memo with bull/bear cases

    You can also run individual agents using the buttons in the sidebar.
    """)

    # Quick ticker suggestions
    st.markdown("#### 🔥 Popular Tickers")
    qcols = st.columns(6)
    popular = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"]
    for i, t in enumerate(popular):
        with qcols[i]:
            st.code(t)
