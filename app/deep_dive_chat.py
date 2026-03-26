"""Deep Dive Chat — Context serializer and conversational Q&A engine.

Converts a StockProfile into a structured text block for LLM context injection,
then handles conversational Q&A with full stock data available.

v2: Question-aware context filtering + tiered response lengths.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.config import get_llm
from app.scoring import format_score_for_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

_THESIS_PATTERNS = re.compile(
    r"\b(should i buy|should i sell|should i hold|investment thesis|bull case|bear case|"
    r"outlook|recommendation|target price|worth buying|accumulate|avoid)\b", re.I
)
_ANALYSIS_PATTERNS = re.compile(
    r"\b(why|how has|compare|trend|risk|red flag|competitor|moat|growth driver|"
    r"margin pressure|debt concern|peer|valuation expensive|undervalued|overvalued|"
    r"what are the risks|what are the strengths|weakness|opportunity|threat|swot|"
    r"explain|analyze|analysis|assess|evaluate|implications|impact|concern)\b", re.I
)


def classify_question(question: str) -> str:
    """Classify a user question into 'factual', 'analysis', or 'thesis'.

    Priority: thesis > analysis > factual (avoids misclassifying
    "what are the risks?" as factual).
    """
    if _THESIS_PATTERNS.search(question):
        return "thesis"
    if _ANALYSIS_PATTERNS.search(question):
        return "analysis"
    return "factual"


# ---------------------------------------------------------------------------
# System prompts — one per question type
# ---------------------------------------------------------------------------

_BASE_RULES = """You are an expert Indian equity research analyst with deep knowledge of {company_name} ({ticker}).

## Rules
- Always cite specific numbers from the data (e.g. "Revenue grew 18.2% YoY from 1,42,300 Cr to 1,68,100 Cr").
- Use Indian financial notation: Crores (Cr), Lakhs (L). Format as INR X,XXX Cr.
- If data is missing, say "Data not available" explicitly.
- Do NOT repeat information already given in earlier messages.
- Be balanced but decisive — give your weighted view, don't sit on the fence."""

FACTUAL_PROMPT = _BASE_RULES + """

## Response style
Answer in **1-3 sentences**. Cite the exact number(s) the user is asking about. No multi-section analysis, no headings, no bullet lists unless the user asks for a list.

=== STOCK DATA ===
{stock_context}
=== END ==="""

ANALYSIS_PROMPT = _BASE_RULES + """

## Response style
Provide focused analysis in **200-400 words**. Cover only the specific topic asked. Use markdown headings sparingly (one ### at most). Include relevant numbers and peer context where available.

**Topic-specific guidance:**
- For competitor/peer questions: compare key metrics (P/E, margins, growth, market cap), discuss competitive advantages (brand, scale, cost leadership, network effects, moat), and market positioning.
- For risk questions: cross-reference governance, debt, promoter pledge, geopolitical exposure. Rate severity.
- For growth questions: cite YoY trends, segment drivers, and forward guidance.

=== STOCK DATA ===
{stock_context}
=== END ==="""

THESIS_PROMPT = _BASE_RULES + """

## Response style
Provide comprehensive analysis in **300-500 words**. Structure with clear sections:
- Bull case (with numbers)
- Bear case / risks
- Valuation view
- Your weighted verdict

=== STOCK DATA ===
{stock_context}
=== END ==="""

_PROMPT_MAP = {
    "factual": FACTUAL_PROMPT,
    "analysis": ANALYSIS_PROMPT,
    "thesis": THESIS_PROMPT,
}

_MAX_TOKENS_MAP = {
    "factual": 400,
    "analysis": 1000,
    "thesis": 1200,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _df_to_summary(df: pd.DataFrame | None, label: str) -> str:
    """Extract only key rows from a financial DataFrame for concise context."""
    if df is None or df.empty:
        return f"\n[{label}: No data]\n"

    key_rows = [
        "Total Revenue", "Operating Revenue", "Net Income",
        "Operating Income", "EBITDA", "Gross Profit",
        "Total Assets", "Total Debt", "Total Liabilities Net Minority Interest",
        "Free Cash Flow", "Operating Cash Flow", "Capital Expenditure",
    ]
    cols = df.columns[:4]
    col_labels = [str(c)[:10] for c in cols]

    lines = [f"\n--- {label} (Key Items) ---"]
    lines.append(f"{'Item':<40} " + " ".join(f"{c:>15}" for c in col_labels))
    lines.append("-" * (40 + 16 * len(col_labels)))

    matched = 0
    for idx in df.index:
        idx_str = str(idx)
        if any(k.lower() in idx_str.lower() for k in key_rows):
            row_name = idx_str[:40]
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
            matched += 1

    if matched == 0:
        # Fallback: show first 5 rows
        for idx in df.index[:5]:
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
    cols = df.columns[:4]

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


# ---------------------------------------------------------------------------
# Context serialization — section builders
# ---------------------------------------------------------------------------

def _build_overview(profile: dict) -> str:
    """Company overview + key ratios (compact, always included)."""
    sections: list[str] = []
    info = profile.get("info", {})

    sections.append("## COMPANY OVERVIEW")
    sections.append(f"Ticker: {profile.get('ticker', 'N/A')}")
    sections.append(f"Name: {info.get('longName', info.get('shortName', 'N/A'))}")
    sections.append(f"Sector: {info.get('sector', 'N/A')}  |  Industry: {info.get('industry', 'N/A')}")
    sections.append(f"Price: INR {info.get('regularMarketPrice', 'N/A')}  |  Mkt Cap: {_format_number_inr(info.get('marketCap'))}")
    sections.append(f"52W: {info.get('fiftyTwoWeekLow', 'N/A')} – {info.get('fiftyTwoWeekHigh', 'N/A')}  |  Beta: {info.get('beta', 'N/A')}")

    desc = info.get("longBusinessSummary", "")
    if desc:
        sections.append(f"\n{desc[:800]}")

    sections.append("\n## KEY RATIOS")
    ratio_items = [
        ("P/E(T)", info.get("trailingPE")), ("P/E(F)", info.get("forwardPE")),
        ("P/B", info.get("priceToBook")), ("EV/EBITDA", info.get("enterpriseToEbitda")),
        ("ROE", info.get("returnOnEquity")), ("ROA", info.get("returnOnAssets")),
        ("OPM", info.get("operatingMargins")), ("NPM", info.get("profitMargins")),
        ("D/E", info.get("debtToEquity")), ("CR", info.get("currentRatio")),
        ("Rev Growth", info.get("revenueGrowth")), ("Earn Growth", info.get("earningsGrowth")),
        ("Div Yield", info.get("dividendYield")),
    ]
    ratio_parts = []
    for label, val in ratio_items:
        if val is not None:
            if isinstance(val, float) and abs(val) < 10 and ("Margin" in label or "Growth" in label or "Yield" in label or "ROE" in label or "ROA" in label or "OPM" in label or "NPM" in label):
                ratio_parts.append(f"{label}: {val:.1%}")
            else:
                ratio_parts.append(f"{label}: {val:.2f}" if isinstance(val, float) else f"{label}: {val}")
    sections.append("  " + "  |  ".join(ratio_parts))

    # Additional key metrics (compact)
    extra = [
        ("Total Revenue", info.get("totalRevenue")), ("EBITDA", info.get("ebitda")),
        ("Net Income", info.get("netIncomeToCommon")), ("FCF", info.get("freeCashflow")),
        ("Total Debt", info.get("totalDebt")), ("Total Cash", info.get("totalCash")),
        ("EPS(T)", info.get("trailingEps")), ("EPS(F)", info.get("forwardEps")),
        ("PEG", info.get("pegRatio")),
    ]
    extra_parts = []
    for label, val in extra:
        if val is not None:
            if isinstance(val, float) and abs(val) >= 1e7:
                extra_parts.append(f"{label}: {_format_number_inr(val)}")
            else:
                extra_parts.append(f"{label}: {val}")
    if extra_parts:
        sections.append("  " + "  |  ".join(extra_parts))

    return "\n".join(sections)


def _build_financials(profile: dict, summarize: bool = False) -> str:
    """Financial statements + growth rates."""
    sections: list[str] = []
    fin_data = profile.get("financials_data", {})
    research = profile.get("research_state", {})

    sections.append("\n## FINANCIAL STATEMENTS")
    convert = _df_to_summary if summarize else _df_to_text

    fin_df = fin_data.get("financials")
    sections.append(convert(fin_df, "INCOME STATEMENT"))
    if fin_df is not None and not fin_df.empty and not summarize:
        sections.append(_compute_yoy_growth(fin_df, "INCOME STATEMENT YoY %"))

    bs_df = fin_data.get("balance_sheet")
    sections.append(convert(bs_df, "BALANCE SHEET"))

    cf_df = fin_data.get("cashflow")
    sections.append(convert(cf_df, "CASH FLOW"))
    if cf_df is not None and not cf_df.empty and not summarize:
        sections.append(_compute_yoy_growth(cf_df, "CASH FLOW YoY %"))

    # Detailed ratios from analysis agent
    ratios = research.get("ratios", {})
    if ratios:
        sections.append("\nDetailed Ratios:")
        for k, v in ratios.items():
            if v is not None and k not in ("error",):
                sections.append(f"  {k}: {v}")

    return "\n".join(sections)


def _build_valuation(profile: dict) -> str:
    """DCF + peer comparison."""
    sections: list[str] = []
    research = profile.get("research_state", {})

    dcf = research.get("dcf_valuation", {})
    if dcf:
        sections.append("\n## DCF VALUATION")
        sections.append(f"  Intrinsic Value: INR {dcf.get('intrinsic_value', 'N/A')}")
        sections.append(f"  Upside/Downside: {dcf.get('upside_pct', 'N/A')}%")
        assumptions = dcf.get("assumptions", {})
        if assumptions:
            sections.append(f"  WACC: {assumptions.get('wacc', 'N/A')}  |  Terminal Growth: {assumptions.get('terminal_growth', 'N/A')}")

    peers = research.get("peer_comparison", [])
    if peers:
        sections.append("\n## PEER COMPARISON")
        if isinstance(peers, str):
            sections.append(peers)
        elif isinstance(peers, list):
            for p in peers:
                if isinstance(p, dict):
                    name = p.get("name", p.get("ticker", "Unknown"))
                    items = [f"{k}: {v}" for k, v in p.items() if k not in ("name", "ticker") and v is not None]
                    sections.append(f"  {name}: {' | '.join(items)}")

    return "\n".join(sections)


def _build_risk_governance(profile: dict) -> str:
    """Geopolitical, regulatory, promoter, management — risk/governance data."""
    sections: list[str] = []

    geo = profile.get("geopolitical", {})
    if geo:
        sections.append("\n## GEOPOLITICAL RISK")
        sections.append(f"  Overall: {geo.get('overall_score', 'N/A')}/100 | Trade: {geo.get('trade_risk_score', 'N/A')} | Policy: {geo.get('policy_risk_score', 'N/A')} | Commodity: {geo.get('commodity_risk_score', 'N/A')} | Event: {geo.get('event_risk_score', 'N/A')}")

    reg = profile.get("regulatory", {})
    if reg:
        sections.append("\n## REGULATORY")
        sections.append(f"  Score: {reg.get('score', 'N/A')}/100 | Signal: {reg.get('net_signal', 'N/A')}")
        tw = reg.get("tailwind_policies", [])
        hw = reg.get("headwind_policies", [])
        if tw:
            sections.append(f"  Tailwinds: {', '.join(tw)}")
        if hw:
            sections.append(f"  Headwinds: {', '.join(hw)}")

    cred = profile.get("credibility", {})
    if cred:
        sections.append("\n## MANAGEMENT CREDIBILITY")
        sections.append(f"  Score: {cred.get('score', 'N/A')}/100 | {cred.get('reasoning', '')}")

    buying = profile.get("promoter_buying", {})
    rpt = profile.get("related_party", {})
    if buying.get("has_data") or rpt:
        sections.append("\n## PROMOTER BEHAVIOR")
        if buying.get("has_data"):
            sections.append(f"  Signal: {buying.get('signal', 'N/A')} | Buys: {buying.get('buys', 0)} | Sells: {buying.get('sells', 0)}")
        if rpt:
            sections.append(f"  RPT: {rpt.get('rpt_count', 0)} filings | Anomaly: {'Yes' if rpt.get('has_anomaly') else 'No'}")

    # Smart money
    imp = profile.get("fundamental_improvement", {})
    flow = profile.get("institutional_flow", {})
    lag = profile.get("smart_money_lag", 0)
    if imp or flow:
        sections.append("\n## SMART MONEY LAG")
        sections.append(f"  Fundamental: {imp.get('score', 'N/A')}/100 | Institutional: {flow.get('score', 'N/A')}/100 | Lag: {lag}")

    # Indian metrics
    indian = profile.get("research_state", {}).get("indian_metrics", {})
    if indian:
        roce = indian.get("roce", {})
        sh = indian.get("shareholding", {})
        pledge = indian.get("promoter_pledge", {})
        parts = []
        if roce:
            parts.append(f"ROCE: {roce.get('roce_value', 'N/A')} ({roce.get('rating', '')})")
        if sh:
            parts.append(f"Promoter: {sh.get('promoter_pct', 'N/A')}%")
        if pledge:
            parts.append(f"Pledge: {pledge.get('pledge_pct', 'N/A')}% ({pledge.get('risk_level', '')})")
        if parts:
            sections.append("\n## INDIAN METRICS")
            sections.append("  " + " | ".join(parts))

    return "\n".join(sections)


def _build_sentiment(profile: dict) -> str:
    """Sentiment + news."""
    sections: list[str] = []
    research = profile.get("research_state", {})

    sentiment = research.get("sentiment_scores", {})
    if sentiment:
        sections.append("\n## SENTIMENT")
        sections.append(f"  Score: {sentiment.get('overall_score', 'N/A')} | Label: {sentiment.get('label', 'N/A')}")

    news = research.get("news_summaries", [])
    if news:
        sections.append("  Headlines:")
        for article in news[:8]:
            sections.append(f"  - {article}")

    return "\n".join(sections)


def _build_report(profile: dict) -> str:
    """Full research report (truncated)."""
    report = profile.get("research_state", {}).get("final_report", "")
    if report:
        return f"\n## RESEARCH REPORT\n{report[:4000]}"
    return ""


def _build_score(profile: dict) -> str:
    """Investment score section."""
    inv_score = profile.get("research_state", {}).get("investment_score", {})
    if inv_score and "composite_score" in inv_score:
        return "\n## INVESTMENT SCORE\n" + format_score_for_prompt(inv_score)
    return ""


# ---------------------------------------------------------------------------
# Full context (backward-compatible)
# ---------------------------------------------------------------------------

def serialize_stock_context(profile: dict) -> str:
    """Convert a StockProfile dict into a structured text block for LLM context.

    This is the full dump — used for thesis-level questions and
    by external callers that need the complete context.
    """
    parts = [
        _build_overview(profile),
        _build_financials(profile, summarize=False),
        _build_valuation(profile),
        _build_risk_governance(profile),
        _build_sentiment(profile),
        _build_score(profile),
        _build_report(profile),
    ]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Question-aware context filter
# ---------------------------------------------------------------------------

def filter_context_for_question(profile: dict, q_type: str) -> str:
    """Return only sections relevant to the question type.

    factual  -> overview + ratios only (~500 tokens)
    analysis -> overview + financials(summary) + valuation + relevant risk (~2K tokens)
    thesis   -> everything with summarized tables (~3K tokens)
    """
    if q_type == "factual":
        parts = [
            _build_overview(profile),
            _build_score(profile),
        ]
    elif q_type == "analysis":
        # Include report snippet for richer context (competitive landscape, etc.)
        report_snippet = _build_report(profile)
        if len(report_snippet) > 2000:
            report_snippet = report_snippet[:2000] + "\n[...truncated]"
        parts = [
            _build_overview(profile),
            _build_financials(profile, summarize=True),
            _build_valuation(profile),
            _build_risk_governance(profile),
            _build_sentiment(profile),
            _build_score(profile),
            report_snippet,
        ]
    else:  # thesis
        parts = [
            _build_overview(profile),
            _build_financials(profile, summarize=True),
            _build_valuation(profile),
            _build_risk_governance(profile),
            _build_sentiment(profile),
            _build_score(profile),
            _build_report(profile),
        ]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Main Q&A entry point
# ---------------------------------------------------------------------------

def ask_stock_question(
    question: str,
    profile: dict,
    chat_history: list[dict],
) -> str:
    """Answer a question about a stock using filtered profile context.

    Args:
        question: The user's question.
        profile: The StockProfile dict from build_stock_profile().
        chat_history: List of {"role": "user"/"assistant", "content": str} dicts.

    Returns the LLM's response string.
    """
    info = profile.get("info", {})
    company_name = info.get("longName", info.get("shortName", profile.get("ticker", "Unknown")))
    ticker = profile.get("ticker", "Unknown")

    # Classify and filter
    q_type = classify_question(question)
    context = filter_context_for_question(profile, q_type)
    max_tokens = _MAX_TOKENS_MAP[q_type]

    # Build system prompt
    prompt_template = _PROMPT_MAP[q_type]
    system_prompt = prompt_template.format(
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

    # Call LLM
    llm = get_llm()
    response = llm.invoke(messages, max_tokens=max_tokens, temperature=0.3)

    return response.content if hasattr(response, "content") else str(response)
