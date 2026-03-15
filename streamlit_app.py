"""Streamlit UI — AI-Powered Indian Equity Research Analyst."""

from __future__ import annotations

import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Equity Research Analyst",
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
    .pledge-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .pledge-danger {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## Indian Equity Research")
    st.markdown("Multi-agent system powered by LangGraph")
    st.divider()

    analysis_mode = st.radio(
        "Mode",
        ["Stock Analysis", "Market Breadth", "Stock Screener", "Stock Deep Dive"],
        horizontal=True,
        key="analysis_mode_radio",
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
    ticker = ""
    exchange = "NSE"

    if analysis_mode == "Stock Analysis":
        ticker = st.text_input(
            "Stock Ticker",
            value="RELIANCE.NS",
            max_chars=20,
            placeholder="e.g., RELIANCE.NS, TCS.NS",
            help="Enter an Indian stock ticker with .NS (NSE) or .BO (BSE) suffix",
        ).upper().strip()

        exchange = st.selectbox("Exchange", ["NSE", "BSE"], index=0)

        # Auto-append suffix if missing
        if ticker and not ticker.endswith((".NS", ".BO")):
            suffix = ".NS" if exchange == "NSE" else ".BO"
            ticker = ticker + suffix
            st.caption(f"Using: **{ticker}**")

        run_full = st.button("Run Full Analysis", type="primary", use_container_width=True)

        st.divider()
        st.markdown("### Run Individual Agents")
        col1, col2 = st.columns(2)
        with col1:
            run_data = st.button("Data", use_container_width=True)
            run_sentiment = st.button("Sentiment", use_container_width=True)
        with col2:
            run_analysis = st.button("Analysis", use_container_width=True)
            run_report = st.button("Report", use_container_width=True)

    elif analysis_mode == "Market Breadth":
        st.markdown("Analyze Nifty 500 market breadth, sector relative strength, and top outperformers.")
        run_breadth = st.button("Run Breadth Analysis", type="primary", use_container_width=True)

    elif analysis_mode == "Stock Screener":
        st.markdown("Screen Nifty 500 stocks using fundamental and qualitative criteria.")

        screener_universe = st.selectbox(
            "Universe",
            ["Full Nifty 500", "Strong Sectors from Breadth"],
            index=0,
            help="'Strong Sectors' requires running Market Breadth first.",
        )

        st.markdown("**Quantitative:**")
        criteria_options = {
            "roe": st.checkbox("ROE < 10% (2 years)", value=True),
            "pb_vs_historical": st.checkbox("P/B below 4yr avg", value=True),
            "ps_vs_historical": st.checkbox("P/S below 4yr avg", value=True),
        }

        st.markdown("**Financial Quality:**")
        criteria_options.update({
            "debt_reduction": st.checkbox("Debt reduction trend", value=True),
            "opm_improvement": st.checkbox("Improving operating margins", value=True),
            "cf_turnaround": st.checkbox("Cash flow turnaround", value=True),
            "dividend_initiation": st.checkbox("Dividend initiation/resumption", value=False),
        })

        st.markdown("**Technical:**")
        criteria_options.update({
            "above_200dma": st.checkbox("Price above 200 DMA", value=True),
            "volume_breakout": st.checkbox("Volume breakout (2x)", value=False),
        })

        st.markdown("**Governance:**")
        criteria_options.update({
            "low_pledge": st.checkbox("Strong promoter holding", value=True),
            "increasing_institutional": st.checkbox("Institutional interest", value=False),
        })

        st.markdown("**Qualitative:**")
        criteria_options.update({
            "capacity_utilization": st.checkbox("Low capacity utilization", value=True),
            "insider_buying": st.checkbox("Insider buying activity", value=True),
            "capex": st.checkbox("New capex announcements", value=True),
            "new_business": st.checkbox("New business / diversification", value=True),
            "order_booking": st.checkbox("Order booking announcements", value=True),
        })

        st.markdown("**USP Screens:**")
        criteria_options.update({
            "geopolitical": st.checkbox("Geopolitical resilience", value=True),
            "smart_money_lag": st.checkbox("Smart money lag", value=False),
            "regulatory_tailwind": st.checkbox("Regulatory tailwind", value=True),
            "promoter_anomaly": st.checkbox("Promoter behavior", value=False),
        })

        screener_min_score = st.slider("Minimum criteria score", 1, 20, 3)
        run_screener = st.button("Run Screener", type="primary", use_container_width=True)

    elif analysis_mode == "Stock Deep Dive":
        st.markdown("### Stock Deep Dive")
        dd_ticker = st.text_input(
            "Stock Ticker",
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

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **Agents:**
    - **Data Agent** -- Fetches financials & BSE/NSE filings
    - **Analysis Agent** -- Ratios, ROCE, DCF (Indian WACC), peers
    - **Sentiment Agent** -- Indian news & sentiment scoring
    - **Report Agent** -- Investment memo with RAG context
    """)


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
if "deep_dive_profile" not in st.session_state:
    st.session_state.deep_dive_profile = None
if "deep_dive_messages" not in st.session_state:
    st.session_state.deep_dive_messages = []
if "deep_dive_ticker" not in st.session_state:
    st.session_state.deep_dive_ticker = ""


# ── Market Breadth Render Functions ──────────────────────────

def render_breadth_overview(breadth: dict):
    """Render 52-week breadth overview metrics."""
    st.markdown("### Market Breadth Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Stocks Analyzed", breadth["total_stocks"])
    with c2:
        st.metric("At 52W High", breadth["at_52w_high"],
                   delta=f"{breadth['high_pct']}%", delta_color="normal")
    with c3:
        st.metric("At 52W Low", breadth["at_52w_low"],
                   delta=f"-{breadth['low_pct']}%", delta_color="inverse")
    with c4:
        st.metric("High/Low Ratio", breadth["high_low_ratio"])
    with c5:
        signal = breadth["breadth_signal"]
        signal_colors = {"Bullish": "#00c853", "Bearish": "#f44336", "Neutral": "#ff9800"}
        color = signal_colors.get(signal, "#ff9800")
        st.markdown(
            f'<div style="text-align:center; padding:10px; background:{color}20; '
            f'border-radius:8px; border:2px solid {color};">'
            f'<div style="font-size:1.4rem; font-weight:700; color:{color};">{signal}</div>'
            f'<div style="font-size:0.8rem; color:#666;">Breadth Signal</div></div>',
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
    colors = ["#2ca02c" if r.get("rs_3m", 0) > 0 else "#d62728" for r in sector_rs]
    fig.add_trace(go.Bar(
        y=[r["sector"] for r in reversed(sector_rs)],
        x=[r["rs_3m"] for r in reversed(sector_rs)],
        orientation="h",
        marker_color=list(reversed(colors)),
        text=[f"{r['rs_3m']:+.1f}%" for r in reversed(sector_rs)],
        textposition="outside",
    ))
    fig.update_layout(
        title="3-Month Relative Strength by Sector",
        xaxis_title="Relative Strength (%)",
        height=max(350, len(sector_rs) * 35),
        margin=dict(l=20, r=80, t=40, b=20),
        template="plotly_white",
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
        st.metric("Stocks Screened", summary.get("total_screened", 0))
    with c2:
        st.metric("Stocks Passing", summary.get("total_passing", 0))
    with c3:
        st.metric("Min Score", summary.get("min_score", 1))
    with c4:
        st.metric("Top Sector", summary.get("top_sector", "N/A"))

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
                marker_color="#1f77b4",
            ))
            fig.update_layout(
                height=max(250, len(sector_dist) * 30),
                xaxis_title="Number of Stocks",
                margin=dict(l=20, r=20, t=10, b=20),
                template="plotly_white",
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
    """Render CSV and JSON export buttons."""
    stocks = screener_data.get("stocks", [])
    if not stocks:
        return

    st.markdown("### Export Results")
    col1, col2 = st.columns(2)

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


# ── Stock Analysis Render Functions ─────────────────────────
def render_company_overview(company_info):
    """Render company overview section with INR formatting."""
    st.markdown("### Company Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        price = safe_get(company_info, "current_price", 0)
        st.metric("Current Price", f"₹{price:,.2f}" if isinstance(price, (int, float)) else str(price))
    with col2:
        mcap = safe_get(company_info, "market_cap", 0)
        st.metric("Market Cap", format_inr_display(mcap))
    with col3:
        pe = safe_get(company_info, "pe_trailing", 0)
        st.metric("P/E (Trailing)", f"{pe:.2f}" if isinstance(pe, (int, float)) else str(pe))
    with col4:
        beta = safe_get(company_info, "beta", 0)
        st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else str(beta))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Sector", safe_get(company_info, "sector"))
    with col6:
        st.metric("Industry", safe_get(company_info, "industry"))
    with col7:
        high = safe_get(company_info, "52_week_high", 0)
        st.metric("52W High", f"₹{high:,.2f}" if isinstance(high, (int, float)) else str(high))
    with col8:
        low = safe_get(company_info, "52_week_low", 0)
        st.metric("52W Low", f"₹{low:,.2f}" if isinstance(low, (int, float)) else str(low))

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
        name="Close Price", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Period Return", f"{safe_get(price_data, 'period_return_pct', 0):.2f}%")
    with col2:
        st.metric("Period High", f"₹{safe_get(price_data, 'period_high', 0):,.2f}")
    with col3:
        st.metric("Period Low", f"₹{safe_get(price_data, 'period_low', 0):,.2f}")
    with col4:
        st.metric("Avg Volume", f"{safe_get(price_data, 'avg_volume', 0):,.0f}")


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
                    st.metric(label, f"{v * 100:.2f}%")
                elif isinstance(v, (int, float)):
                    st.metric(label, f"{v:.2f}")
                else:
                    st.metric(label, str(v))

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
                st.metric("Current ROCE", safe_get(roce_data, "roce_pct", "N/A"))
            with col2:
                st.metric("Quality Rating", safe_get(roce_data, "quality_rating", "N/A"))
            with col3:
                st.metric("EBIT (Cr)", f"₹{safe_get(roce_data, 'ebit_crores', 0):,.2f}")

            historical = roce_data.get("historical_roce", [])
            if historical:
                years = [h["year"][:10] for h in historical]
                values = [h["roce"] * 100 for h in historical]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=years, y=values, marker_color="#ff7f0e", name="ROCE %"))
                fig.update_layout(height=300, yaxis_title="ROCE %", template="plotly_white")
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
        st.metric("Intrinsic Value", f"₹{intrinsic:,.2f}" if isinstance(intrinsic, (int, float)) else str(intrinsic))
    with col2:
        current = safe_get(dcf, "current_price", 0)
        st.metric("Current Price", f"₹{current:,.2f}" if isinstance(current, (int, float)) else str(current))
    with col3:
        upside = safe_get(dcf, "upside_pct", 0)
        if isinstance(upside, (int, float)):
            st.metric("Upside/Downside", f"{upside:+.2f}%",
                       delta=f"{upside:+.2f}%", delta_color="normal")
        else:
            st.metric("Upside/Downside", str(upside))

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
        fig.add_trace(go.Bar(x=years, y=fcfs, marker_color="#2ca02c", name="Discounted FCF (Cr)"))
        fig.update_layout(
            height=300, yaxis_title="Discounted FCF (₹ Crores)", template="plotly_white",
        )
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
            st.metric("Sentiment Score", f"{score:+.2f}")
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
        fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Strong Buy (75)")
        fig.add_hline(y=60, line_dash="dash", line_color="lightgreen", annotation_text="Buy (60)")
        fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Hold (40)")
        fig.update_layout(
            height=350,
            yaxis_title="Score",
            yaxis_range=[0, 110],
            template="plotly_white",
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
                            st.metric(display_name, f"{sub_val}/100")

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


def render_report(report):
    """Render the final investment report."""
    if not report:
        return
    st.markdown("### Investment Research Report")
    st.markdown(report)

    st.download_button(
        label="Download Report as Markdown",
        data=report,
        file_name=f"indian_equity_report_{st.session_state.last_ticker}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown",
    )


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
st.markdown('<p class="main-header">Indian Equity Research Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-agent investment research for NSE/BSE stocks powered by LangGraph + LLM</p>', unsafe_allow_html=True)
st.divider()

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
            st.session_state.deep_dive_profile = build_stock_profile(
                ticker, exchange, progress_cb=_dd_progress,
            )
            st.session_state.deep_dive_ticker = ticker
            progress_bar.empty()
            st.success("Stock profile loaded!")
        except Exception as e:
            progress_bar.empty()
            st.error(f"Profile loading failed: {e}")

    profile = st.session_state.deep_dive_profile
    if profile:
        info = profile.get("info", {})
        research = profile.get("research_state", {})
        company_name = info.get("longName", info.get("shortName", profile.get("ticker", "")))

        # ── Scorecard ─────────────────────────────────────────
        st.markdown(f"### {company_name}")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            price = info.get("regularMarketPrice", "N/A")
            st.metric("Price", f"INR {price}")
        with c2:
            mcap = info.get("marketCap")
            if mcap:
                mcap_display = f"{mcap/1e7:.0f} Cr" if mcap >= 1e7 else f"{mcap:,.0f}"
            else:
                mcap_display = "N/A"
            st.metric("Market Cap", mcap_display)
        with c3:
            inv_score = research.get("investment_score", {})
            comp_score = inv_score.get("composite_score", "N/A")
            rec = inv_score.get("recommendation", "")
            st.metric("Investment Score", f"{comp_score}/100", delta=rec)
        with c4:
            geo = profile.get("geopolitical", {})
            st.metric("Geo Risk", geo.get("risk_level", "N/A"), delta=f"{geo.get('overall_score', 'N/A')}/100")
        with c5:
            lag = profile.get("smart_money_lag", 0)
            lag_label = "Opportunity" if lag > 20 else "Neutral" if lag > -10 else "Crowded"
            st.metric("Smart Money Lag", f"{lag}", delta=lag_label)

        # ── USP Analysis Expander ─────────────────────────────
        with st.expander("USP Analysis Details", expanded=False):
            ucol1, ucol2 = st.columns(2)
            with ucol1:
                st.markdown("**Geopolitical Risk**")
                geo = profile.get("geopolitical", {})
                st.write(f"- Trade Risk: {geo.get('trade_risk_score', 'N/A')}/100")
                st.write(f"- Policy Risk: {geo.get('policy_risk_score', 'N/A')}/100")
                st.write(f"- Commodity Risk: {geo.get('commodity_risk_score', 'N/A')}/100")
                st.write(f"- Event Risk: {geo.get('event_risk_score', 'N/A')}/100")

                st.markdown("**Management Credibility**")
                cred = profile.get("credibility", {})
                st.write(f"- Score: {cred.get('score', 'N/A')}/100 ({cred.get('method', '')})")
                st.write(f"- {cred.get('reasoning', 'N/A')}")

            with ucol2:
                st.markdown("**Regulatory Environment**")
                reg = profile.get("regulatory", {})
                st.write(f"- Net Signal: {reg.get('net_signal', 'N/A')} (Score: {reg.get('score', 'N/A')})")
                tailwinds = reg.get("tailwind_policies", [])
                if tailwinds:
                    st.write(f"- Tailwinds: {', '.join(tailwinds)}")
                headwinds = reg.get("headwind_policies", [])
                if headwinds:
                    st.write(f"- Headwinds: {', '.join(headwinds)}")

                st.markdown("**Promoter Behavior**")
                buying = profile.get("promoter_buying", {})
                st.write(f"- Signal: {buying.get('signal', 'N/A')}")
                st.write(f"- Buys: {buying.get('buys', 0)} | Sells: {buying.get('sells', 0)}")
                rpt = profile.get("related_party", {})
                st.write(f"- RPT Anomaly: {'Yes' if rpt.get('has_anomaly') else 'No'}")

        # ── Investment Score Expander ──────────────────────────
        if inv_score and "dimension_scores" in inv_score:
            with st.expander("Investment Score Breakdown", expanded=False):
                render_investment_score(inv_score)

        # ── Chat Interface ────────────────────────────────────
        st.divider()
        st.markdown("### Ask Questions About This Stock")
        st.caption("Examples: Products offered, revenue by segment, competitors, competitive edge, strengths, risks, red flags")

        # Display chat history
        for msg in st.session_state.deep_dive_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input(f"Ask about {company_name}..."):
            st.session_state.deep_dive_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        from app.deep_dive_chat import ask_stock_question
                        response = ask_stock_question(
                            prompt,
                            profile,
                            st.session_state.deep_dive_messages[:-1],
                        )
                        st.markdown(response)
                        st.session_state.deep_dive_messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.deep_dive_messages.append({"role": "assistant", "content": error_msg})

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

# ── Stock Screener Mode ──────────────────────────────────────
if analysis_mode == "Stock Screener":
    if run_screener:
        # Determine universe
        universe_key = "nifty500"
        breadth_for_screener = None
        if screener_universe == "Strong Sectors from Breadth":
            if st.session_state.breadth_data:
                universe_key = "strong_sectors"
                breadth_for_screener = st.session_state.breadth_data
            else:
                st.warning("Run Market Breadth analysis first to use Strong Sectors mode. Falling back to Full Nifty 500.")

        selected_criteria = [k for k, v in criteria_options.items() if v]
        if not selected_criteria:
            st.warning("Please select at least one screening criterion.")
        else:
            progress_bar = st.progress(0, text="Starting stock screener...")

            def _screener_progress(fraction: float, message: str):
                progress_bar.progress(min(fraction, 1.0), text=message)

            try:
                from app.tools.screener import run_stock_screener
                st.session_state.screener_data = run_stock_screener(
                    universe=universe_key,
                    criteria=selected_criteria,
                    min_score=screener_min_score,
                    breadth_data=breadth_for_screener,
                    progress_cb=_screener_progress,
                    max_stocks=50 if test_mode else None,
                )
                progress_bar.empty()
                st.success("Stock screening complete!")
            except Exception as e:
                progress_bar.empty()
                st.error(f"Screener failed: {e}")

    sdata = st.session_state.screener_data
    if sdata:
        render_screener_summary(sdata["summary"])
        st.divider()
        render_screener_table(sdata["stocks"], sdata.get("criteria_used", []))
        st.divider()
        render_screener_stock_detail(sdata["stocks"])
        st.divider()
        render_screener_export(sdata)
    else:
        st.markdown("""
        ### Fundamental Stock Screener

        Click **Run Screener** in the sidebar to screen Nifty 500 stocks using:

        **Quantitative Criteria:**
        - ROE < 10% for last 2 years (turnaround candidates)
        - P/B below 4-year average (value opportunity)
        - P/S below 4-year average (revenue discount)
        - Low capacity utilization (asset turnover proxy)

        **Qualitative Criteria:**
        - Insider buying activity
        - New capex / capacity expansion announcements
        - New business / diversification signals
        - Order booking announcements

        First run takes ~3-5 minutes for Full Nifty 500. Subsequent runs use cached data.
        """)

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
