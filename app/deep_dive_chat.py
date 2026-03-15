"""Deep Dive Chat — Context serializer and conversational Q&A engine.

Converts a StockProfile into a structured text block for LLM context injection,
then handles conversational Q&A with full stock data available.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.config import get_llm
from app.scoring import format_score_for_prompt

logger = logging.getLogger(__name__)

DEEP_DIVE_SYSTEM_PROMPT = """You are an expert Indian equity research analyst (CFA, 15+ years experience) with deep knowledge of {company_name} ({ticker}).

You have access to comprehensive stock data below. Use it to provide DETAILED, THOROUGH analysis. Think like a sell-side analyst writing a research note — be exhaustive, not superficial.

## RESPONSE RULES — FOLLOW STRICTLY

1. **Always cite specific numbers**: Don't say "revenue grew" — say "Revenue grew 18.2% YoY from ₹1,42,300 Cr to ₹1,68,100 Cr". Pull exact figures from the financial data.

2. **Use Indian financial notation**: Crores (Cr), Lakhs (L). Format large numbers as ₹X,XXX Cr.

3. **Structure every answer with clear sections**: Use markdown headings (##, ###), bullet points, and tables where appropriate.

4. **Provide multi-dimensional analysis**: For every topic, cover:
   - Current state (with numbers)
   - Historical trend (YoY or QoQ changes from the financial data)
   - Peer/industry comparison (from peer data if available)
   - Forward outlook / implications
   - Key risks to the thesis

5. **For product/segment questions**:
   - Use the business description to identify all segments/verticals
   - Cross-reference revenue and operating income trends to infer segment performance
   - If exact segment breakdown isn't available, clearly state so but provide analysis using total revenue, margin trends, and sector knowledge
   - Mention market share where known

6. **For competitor/moat questions**:
   - Use peer comparison data for direct metrics comparison (P/E, margins, growth)
   - Analyze competitive advantages: brand, scale, cost leadership, network effects, switching costs, regulatory barriers
   - Discuss market positioning and market share dynamics

7. **For risk/red flag questions**:
   - Cross-reference ALL risk signals: governance score, promoter pledge %, related party transactions, debt/equity ratio, geopolitical exposure, regulatory headwinds, smart money lag
   - Rate severity: Critical / Moderate / Low for each risk
   - Mention both financial AND non-financial risks (regulatory, ESG, key-man, concentration)

8. **For valuation questions**:
   - Use DCF intrinsic value, current multiples (P/E, P/B, EV/EBITDA), and peer multiples
   - Show upside/downside math
   - Discuss what assumptions would need to change for a different outcome

9. **For financial analysis**:
   - Show multi-year trends (compute YoY growth rates from the data)
   - Highlight inflection points or concerning patterns
   - Compare margins and ratios to industry averages where possible
   - Comment on cash flow quality vs. reported earnings

10. **Be balanced but decisive**: Present bull AND bear cases, but end with your weighted view based on the data. Don't sit on the fence.

11. **Length**: Provide comprehensive answers. A good answer is typically 400-800 words with specific data points. Don't be brief — the user wants deep analysis, not summaries.

=== COMPREHENSIVE STOCK DATA ===
{stock_context}
=== END STOCK DATA ==="""


def _format_number_inr(val: float | None) -> str:
    """Format a number in Indian notation."""
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"{val/1e12:.1f}L Cr"
    if abs(val) >= 1e7:
        return f"{val/1e7:.1f} Cr"
    if abs(val) >= 1e5:
        return f"{val/1e5:.1f} L"
    return f"{val:,.0f}"


def _safe_get(d: dict | None, *keys, default="N/A") -> Any:
    """Safely traverse nested dicts."""
    val = d
    for key in keys:
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default


def _df_to_text(df: pd.DataFrame | None, label: str, max_rows: int = 20) -> str:
    """Convert a DataFrame to a text table."""
    if df is None or df.empty:
        return f"\n[{label}: No data available]\n"

    lines = [f"\n--- {label} ---"]
    # Use first 4 columns (years) and important rows
    cols = df.columns[:4]
    col_labels = [str(c)[:10] for c in cols]
    lines.append(f"{'Item':<40} " + " ".join(f"{c:>15}" for c in col_labels))
    lines.append("-" * (40 + 16 * len(col_labels)))

    for idx in df.index[:max_rows]:
        row_name = str(idx)[:40]
        values = []
        for c in cols:
            val = df.loc[idx, c]
            if pd.notna(val):
                try:
                    values.append(f"{float(val):>15,.0f}")
                except (ValueError, TypeError):
                    values.append(f"{str(val):>15}")
            else:
                values.append(f"{'N/A':>15}")
        lines.append(f"{row_name:<40} " + " ".join(values))

    return "\n".join(lines)


def _compute_yoy_growth(df: pd.DataFrame, label: str) -> str:
    """Compute YoY growth percentages for key rows in a financial statement."""
    if df is None or df.empty or len(df.columns) < 2:
        return ""

    lines = [f"\n--- {label} ---"]
    cols = df.columns[:4]  # Most recent 4 years

    for idx in df.index:
        row_name = str(idx)[:40]
        growths = []
        for i in range(len(cols) - 1):
            try:
                curr = float(df.loc[idx, cols[i]])
                prev = float(df.loc[idx, cols[i + 1]])
                if prev != 0 and pd.notna(curr) and pd.notna(prev):
                    pct = ((curr - prev) / abs(prev)) * 100
                    growths.append(f"{str(cols[i])[:10]}: {pct:+.1f}%")
                else:
                    growths.append(f"{str(cols[i])[:10]}: N/A")
            except (ValueError, TypeError):
                growths.append(f"{str(cols[i])[:10]}: N/A")
        if growths:
            lines.append(f"  {row_name}: {' | '.join(growths)}")

    return "\n".join(lines) if len(lines) > 1 else ""


def serialize_stock_context(profile: dict) -> str:
    """Convert a StockProfile dict into a structured text block for LLM context."""
    sections: list[str] = []
    info = profile.get("info", {})
    research = profile.get("research_state", {})
    fin_data = profile.get("financials_data", {})

    # ── Company Overview ──────────────────────────────────────
    sections.append("## COMPANY OVERVIEW")
    sections.append(f"Ticker: {profile.get('ticker', 'N/A')}")
    sections.append(f"Name: {info.get('longName', info.get('shortName', 'N/A'))}")
    sections.append(f"Sector: {info.get('sector', 'N/A')}")
    sections.append(f"Industry: {info.get('industry', 'N/A')}")
    sections.append(f"Current Price: INR {info.get('regularMarketPrice', 'N/A')}")
    sections.append(f"Market Cap: {_format_number_inr(info.get('marketCap'))}")
    sections.append(f"52W High: INR {info.get('fiftyTwoWeekHigh', 'N/A')}")
    sections.append(f"52W Low: INR {info.get('fiftyTwoWeekLow', 'N/A')}")
    sections.append(f"Beta: {info.get('beta', 'N/A')}")

    # Company description
    desc = info.get("longBusinessSummary", "")
    if desc:
        sections.append(f"\nBusiness Description:\n{desc}")

    # ── Key Financial Ratios ──────────────────────────────────
    sections.append("\n## KEY FINANCIAL RATIOS")
    ratios = research.get("ratios", {})
    company_info = research.get("company_info", {})

    ratio_items = [
        ("P/E (Trailing)", info.get("trailingPE")),
        ("P/E (Forward)", info.get("forwardPE")),
        ("P/B", info.get("priceToBook")),
        ("P/S", info.get("priceToSalesTrailing12Months")),
        ("EV/EBITDA", info.get("enterpriseToEbitda")),
        ("ROE", info.get("returnOnEquity")),
        ("ROA", info.get("returnOnAssets")),
        ("Operating Margin", info.get("operatingMargins")),
        ("Profit Margin", info.get("profitMargins")),
        ("Debt/Equity", info.get("debtToEquity")),
        ("Current Ratio", info.get("currentRatio")),
        ("Revenue Growth", info.get("revenueGrowth")),
        ("Earnings Growth", info.get("earningsGrowth")),
        ("Dividend Yield", info.get("dividendYield")),
    ]
    for label, val in ratio_items:
        if val is not None:
            if isinstance(val, float) and abs(val) < 10:
                sections.append(f"  {label}: {val:.2%}" if "Margin" in label or "Growth" in label or "Yield" in label or "ROE" in label or "ROA" in label else f"  {label}: {val:.2f}")
            else:
                sections.append(f"  {label}: {val}")

    # Additional ratios from analysis agent
    if ratios:
        sections.append("\nDetailed Ratios from Analysis:")
        for k, v in ratios.items():
            if v is not None and k not in ("error",):
                sections.append(f"  {k}: {v}")

    # ── Additional Key Metrics ─────────────────────────────────
    sections.append("\n## ADDITIONAL KEY METRICS")
    extra_metrics = [
        ("Enterprise Value", info.get("enterpriseValue")),
        ("Total Revenue (TTM)", info.get("totalRevenue")),
        ("EBITDA (TTM)", info.get("ebitda")),
        ("Net Income (TTM)", info.get("netIncomeToCommon")),
        ("Total Cash", info.get("totalCash")),
        ("Total Debt", info.get("totalDebt")),
        ("Free Cash Flow", info.get("freeCashflow")),
        ("Operating Cash Flow", info.get("operatingCashflow")),
        ("Revenue Per Share", info.get("revenuePerShare")),
        ("Book Value Per Share", info.get("bookValue")),
        ("Earnings Per Share (TTM)", info.get("trailingEps")),
        ("Earnings Per Share (Forward)", info.get("forwardEps")),
        ("PEG Ratio", info.get("pegRatio")),
        ("Shares Outstanding", info.get("sharesOutstanding")),
        ("Float Shares", info.get("floatShares")),
        ("Held by Insiders %", info.get("heldPercentInsiders")),
        ("Held by Institutions %", info.get("heldPercentInstitutions")),
    ]
    for label, val in extra_metrics:
        if val is not None:
            if isinstance(val, float) and abs(val) >= 1e7:
                sections.append(f"  {label}: {_format_number_inr(val)}")
            elif isinstance(val, float) and abs(val) < 1:
                sections.append(f"  {label}: {val:.2%}")
            else:
                sections.append(f"  {label}: {val}")

    # ── Financial Statements ──────────────────────────────────
    sections.append("\n## FINANCIAL STATEMENTS")

    # Income statement
    fin_df = fin_data.get("financials")
    sections.append(_df_to_text(fin_df, "INCOME STATEMENT (Annual)"))

    # Compute YoY growth rates from income statement
    if fin_df is not None and not fin_df.empty:
        sections.append(_compute_yoy_growth(fin_df, "INCOME STATEMENT YoY GROWTH %"))

    # Balance sheet
    bs_df = fin_data.get("balance_sheet")
    sections.append(_df_to_text(bs_df, "BALANCE SHEET (Annual)"))

    # Cash flow
    cf_df = fin_data.get("cashflow")
    sections.append(_df_to_text(cf_df, "CASH FLOW STATEMENT (Annual)"))

    if cf_df is not None and not cf_df.empty:
        sections.append(_compute_yoy_growth(cf_df, "CASH FLOW YoY GROWTH %"))

    # Dividends
    dividends = fin_data.get("dividends")
    if dividends is not None and hasattr(dividends, '__len__') and len(dividends) > 0:
        sections.append("\n--- DIVIDEND HISTORY ---")
        if hasattr(dividends, 'tail'):
            recent = dividends.tail(10)
            for date_idx, val in recent.items():
                sections.append(f"  {str(date_idx)[:10]}: INR {val:.2f}")

    # ── DCF Valuation ─────────────────────────────────────────
    dcf = research.get("dcf_valuation", {})
    if dcf:
        sections.append("\n## DCF VALUATION")
        sections.append(f"  Intrinsic Value: INR {dcf.get('intrinsic_value', 'N/A')}")
        sections.append(f"  Upside/Downside: {dcf.get('upside_pct', 'N/A')}%")
        assumptions = dcf.get("assumptions", {})
        if assumptions:
            sections.append(f"  WACC: {assumptions.get('wacc', 'N/A')}")
            sections.append(f"  Terminal Growth: {assumptions.get('terminal_growth', 'N/A')}")

    # ── Peer Comparison ───────────────────────────────────────
    peers = research.get("peer_comparison", [])
    if peers:
        sections.append("\n## PEER COMPARISON")
        if isinstance(peers, str):
            sections.append(peers)
        elif isinstance(peers, list):
            for p in peers:
                if isinstance(p, dict):
                    name = p.get("name", p.get("ticker", "Unknown"))
                    sections.append(f"\n  {name}:")
                    for k, v in p.items():
                        if k not in ("name", "ticker") and v is not None:
                            sections.append(f"    {k}: {v}")

    # ── Investment Score ──────────────────────────────────────
    inv_score = research.get("investment_score", {})
    if inv_score and "composite_score" in inv_score:
        sections.append("\n## INVESTMENT SCORE")
        sections.append(format_score_for_prompt(inv_score))

    # ── Indian Market Metrics ─────────────────────────────────
    indian = research.get("indian_metrics", {})
    if indian:
        sections.append("\n## INDIAN MARKET METRICS")
        roce = indian.get("roce", {})
        if roce:
            sections.append(f"  ROCE: {roce.get('roce_value', 'N/A')}")
            sections.append(f"  ROCE Rating: {roce.get('rating', 'N/A')}")

        shareholding = indian.get("shareholding", {})
        if shareholding:
            sections.append(f"  Promoter Holding: {shareholding.get('promoter_pct', 'N/A')}%")

        pledge = indian.get("promoter_pledge", {})
        if pledge:
            sections.append(f"  Promoter Pledge: {pledge.get('pledge_pct', 'N/A')}%")
            sections.append(f"  Pledge Risk: {pledge.get('risk_level', 'N/A')}")

    # ── Sentiment & News ──────────────────────────────────────
    sentiment = research.get("sentiment_scores", {})
    if sentiment:
        sections.append("\n## SENTIMENT ANALYSIS")
        sections.append(f"  Overall Sentiment: {sentiment.get('overall_score', 'N/A')}")
        sections.append(f"  Sentiment Label: {sentiment.get('label', 'N/A')}")

    news = research.get("news_summaries", [])
    if news:
        sections.append("\n  Recent News Headlines:")
        for article in news[:10]:
            sections.append(f"  - {article}")

    # ── Geopolitical Risk ─────────────────────────────────────
    geo = profile.get("geopolitical", {})
    if geo:
        sections.append("\n## GEOPOLITICAL RISK PROFILE")
        sections.append(f"  Overall Score: {geo.get('overall_score', 'N/A')}/100 (higher = more resilient)")
        sections.append(f"  Risk Level: {geo.get('risk_level', 'N/A')}")
        sections.append(f"  Trade Risk: {geo.get('trade_risk_score', 'N/A')}/100")
        sections.append(f"  Policy Risk: {geo.get('policy_risk_score', 'N/A')}/100")
        sections.append(f"  Commodity Risk: {geo.get('commodity_risk_score', 'N/A')}/100")
        sections.append(f"  Event Risk: {geo.get('event_risk_score', 'N/A')}/100")

    # ── Smart Money Lag ───────────────────────────────────────
    sections.append("\n## SMART MONEY LAG ANALYSIS")
    imp = profile.get("fundamental_improvement", {})
    flow = profile.get("institutional_flow", {})
    lag = profile.get("smart_money_lag", 0)
    sections.append(f"  Fundamental Improvement Score: {imp.get('score', 'N/A')}/100")
    sections.append(f"  Institutional Flow Score: {flow.get('score', 'N/A')}/100")
    sections.append(f"  Smart Money Lag: {lag} (positive = institutions haven't caught up)")
    imp_details = imp.get("details", {})
    if imp_details:
        for k, v in imp_details.items():
            sections.append(f"    {k}: {v}")

    # ── Management Credibility ────────────────────────────────
    cred = profile.get("credibility", {})
    if cred:
        sections.append("\n## MANAGEMENT CREDIBILITY")
        sections.append(f"  Score: {cred.get('score', 'N/A')}/100")
        sections.append(f"  Method: {cred.get('method', 'N/A')}")
        sections.append(f"  Reasoning: {cred.get('reasoning', 'N/A')}")

    # ── Regulatory Environment ────────────────────────────────
    reg = profile.get("regulatory", {})
    if reg:
        sections.append("\n## REGULATORY ENVIRONMENT")
        sections.append(f"  Score: {reg.get('score', 'N/A')}/100")
        sections.append(f"  Net Signal: {reg.get('net_signal', 'N/A')}")
        tailwinds = reg.get("tailwind_policies", [])
        headwinds = reg.get("headwind_policies", [])
        if tailwinds:
            sections.append(f"  Tailwind Policies: {', '.join(tailwinds)}")
        if headwinds:
            sections.append(f"  Headwind Policies: {', '.join(headwinds)}")

    # ── Promoter Behavior ─────────────────────────────────────
    buying = profile.get("promoter_buying", {})
    rpt = profile.get("related_party", {})
    if buying.get("has_data"):
        sections.append("\n## PROMOTER BEHAVIOR")
        sections.append(f"  Buying Signal: {buying.get('signal', 'N/A')}")
        sections.append(f"  Promoter Buys: {buying.get('buys', 0)}")
        sections.append(f"  Promoter Sells: {buying.get('sells', 0)}")
    if rpt:
        sections.append(f"  Related Party Transactions: {rpt.get('rpt_count', 0)} filings")
        sections.append(f"  RPT Anomaly: {'Yes' if rpt.get('has_anomaly') else 'No'}")

    # ── Full Research Report ──────────────────────────────────
    report = research.get("final_report", "")
    if report:
        sections.append("\n## FULL RESEARCH REPORT")
        sections.append(report[:8000])  # Limit to avoid token overflow

    return "\n".join(sections)


def ask_stock_question(
    question: str,
    profile: dict,
    chat_history: list[dict],
) -> str:
    """Answer a question about a stock using the full profile as context.

    Args:
        question: The user's question.
        profile: The StockProfile dict from build_stock_profile().
        chat_history: List of {"role": "user"/"assistant", "content": str} dicts.

    Returns the LLM's response string.
    """
    info = profile.get("info", {})
    company_name = info.get("longName", info.get("shortName", profile.get("ticker", "Unknown")))
    ticker = profile.get("ticker", "Unknown")

    # Serialize context
    context = serialize_stock_context(profile)

    # Build system prompt
    system_prompt = DEEP_DIVE_SYSTEM_PROMPT.format(
        company_name=company_name,
        ticker=ticker,
        stock_context=context,
    )

    # Build messages
    messages = [SystemMessage(content=system_prompt)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))

    # Call LLM with higher max_tokens for detailed responses
    llm = get_llm()
    response = llm.invoke(messages, max_tokens=4096)

    return response.content if hasattr(response, "content") else str(response)
