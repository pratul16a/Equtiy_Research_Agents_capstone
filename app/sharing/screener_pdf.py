"""Screener Summary PDF — multi-stock report with regime, categories, USP, debate.

Generates a multi-page branded PDF summarizing screening results.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from app.sharing.pdf_generator import (
    BRAND_BLUE, DARK_TEXT, GREEN, GREY, LIGHT_BG, ORANGE, RED, WHITE,
    EquityReportPDF, _sanitize,
)


def _score_rgb(score: float) -> tuple[int, int, int]:
    """Map a 0-100 score to an RGB color for PDF cells."""
    if score >= 70:
        return (46, 204, 113)   # green
    elif score >= 50:
        return (243, 156, 18)   # orange/yellow
    elif score >= 30:
        return (230, 126, 34)   # dark orange
    else:
        return (231, 76, 60)    # red


def generate_screener_pdf(
    screener_data: dict[str, Any],
    include_usp: bool = True,
    include_debates: bool = True,
) -> bytes:
    """Generate a multi-page PDF summarizing screener results.

    Args:
        screener_data: The full category_screener_data dict from session state.
        include_usp: Include USP score heatmap page.
        include_debates: Include debate verdict page.

    Returns:
        PDF file content as bytes.
    """
    regime = screener_data.get("regime_data", {})
    regime_name = regime.get("regime", "mixed").upper()
    cat_results = screener_data.get("category_results", {})
    summary = screener_data.get("summary", {})
    usp_scores = screener_data.get("usp_scores", {})
    usp_cards = screener_data.get("usp_cards", {})
    debate_results = screener_data.get("debate_results", [])

    pdf = EquityReportPDF(ticker="Screener Summary", recommendation=regime_name, score=0)

    # ── Cover Page ────────────────────────────────────────────
    pdf.add_page()
    pdf.alias_nb_pages()

    # Blue banner
    pdf.set_fill_color(*BRAND_BLUE)
    pdf.rect(0, 0, 210, 70, "F")
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(15, 15)
    pdf.cell(0, 12, "Stock Screener Report")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_xy(15, 32)
    pdf.cell(0, 8, f"Market Regime: {regime_name}")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_xy(15, 45)
    pdf.cell(0, 6, _sanitize(regime.get("reasoning", "")))
    pdf.set_xy(15, 55)
    pdf.cell(0, 6, datetime.now().strftime("%d %B %Y"))
    pdf.set_text_color(*DARK_TEXT)

    # Summary metrics
    pdf.set_xy(15, 80)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if summary:
        for key in ["universe_size", "after_tech_filter", "with_fundamentals", "total_survivors"]:
            val = summary.get(key, "?")
            label = key.replace("_", " ").title()
            pdf.cell(0, 6, f"{label}: {val}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Category Tables ───────────────────────────────────────
    cat_labels = {"A": "Category A: Strengthening Industries", "B": "Category B: Momentum", "C": "Category C: Value Bottoms"}

    for cat_key, cat_label in cat_labels.items():
        stocks = []
        if isinstance(cat_results, dict):
            stocks = cat_results.get(cat_key, [])
        elif isinstance(cat_results, list):
            for entry in cat_results:
                if entry.get("category") == cat_key:
                    stocks = entry.get("stocks", [])
                    break

        if not stocks:
            continue

        pdf.add_page()
        pdf.section_heading(cat_label, level=1)

        headers = ["Ticker", "Sector", "Score", "Criteria Passed"]
        rows = []
        for s in stocks[:20]:  # cap at 20 per category
            criteria = s.get("criteria_passed", [])
            if isinstance(criteria, list):
                criteria_str = ", ".join(criteria[:3])
                if len(criteria) > 3:
                    criteria_str += f" +{len(criteria) - 3}"
            else:
                criteria_str = str(criteria)
            rows.append([
                s.get("ticker", ""),
                s.get("sector", ""),
                str(s.get("composite_score", s.get("score", ""))),
                criteria_str,
            ])
        pdf.add_table(headers, rows)

    # ── USP Heatmap ───────────────────────────────────────────
    if include_usp and (usp_scores or usp_cards):
        pdf.add_page()
        pdf.section_heading("USP Analysis Heatmap", level=1)

        dimensions = ["geopolitical", "smart_money", "regulatory", "mgmt_credibility", "promoter"]
        dim_labels = ["Geo", "Smart$", "Regulatory", "Mgmt", "Promoter", "Composite"]
        headers = ["Ticker"] + dim_labels

        source = usp_cards if usp_cards else usp_scores
        rows = []
        for ticker, data in source.items():
            row = [ticker]
            for dim in dimensions:
                dim_data = data.get(dim, {})
                score = dim_data.get("score", 0) if isinstance(dim_data, dict) else 0
                row.append(str(int(score)))
            # Composite
            composite = data.get("composite_score", 0)
            if isinstance(composite, dict):
                composite = composite.get("score", 0)
            row.append(str(int(composite)))
            rows.append(row)

        if rows:
            pdf.add_table(headers, rows)

    # ── Debate Verdicts ───────────────────────────────────────
    if include_debates and debate_results:
        pdf.add_page()
        pdf.section_heading("AI Debate Verdicts", level=1)

        for debate in debate_results:
            ticker = debate.get("ticker", "Unknown")
            verdict = debate.get("verdict", debate.get("judge", {}))
            if isinstance(verdict, dict):
                recommendation = verdict.get("recommendation", "N/A")
                conviction = verdict.get("conviction_score", verdict.get("conviction", "?"))
                reasoning = verdict.get("reasoning", verdict.get("summary", ""))
            else:
                recommendation = str(verdict)
                conviction = "?"
                reasoning = ""

            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*BRAND_BLUE)
            pdf.cell(0, 7, _sanitize(f"{ticker}  -  {recommendation}  (Conviction: {conviction}/10)"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*DARK_TEXT)
            if reasoning:
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, _sanitize(reasoning[:300]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # ── Disclaimer ────────────────────────────────────────────
    pdf.disclaimer_block(
        "Disclaimer: This report is generated by an AI system for educational and informational purposes only. "
        "It does not constitute investment advice. Past performance is not indicative of future results. "
        "Please consult a qualified financial advisor before making investment decisions."
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
