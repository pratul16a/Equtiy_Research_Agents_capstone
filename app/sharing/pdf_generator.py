"""PDF Report Generator — converts Markdown reports to branded PDFs.

Uses fpdf2 for pure-Python PDF generation with no system dependencies.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from fpdf import FPDF


# ── Brand colors (match Streamlit CSS) ────────────────────────
BRAND_BLUE = (31, 119, 180)       # #1f77b4
DARK_TEXT = (33, 37, 41)          # #212529
LIGHT_BG = (248, 249, 250)       # #f8f9fa
WHITE = (255, 255, 255)
GREEN = (40, 167, 69)            # BUY
RED = (220, 53, 69)              # SELL
ORANGE = (255, 193, 7)           # HOLD
GREY = (108, 117, 125)

REC_COLORS = {"BUY": GREEN, "SELL": RED, "HOLD": ORANGE, "STRONG BUY": GREEN, "STRONG SELL": RED}

# Unicode → ASCII replacements for Helvetica compatibility
_UNICODE_SUBS = {
    "\u2014": "-",   # em-dash
    "\u2013": "-",   # en-dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2022": "-",   # bullet
    "\u2026": "...", # ellipsis
    "\u20b9": "Rs.", # rupee sign
    "\u2192": "->",  # right arrow
    "\u2190": "<-",  # left arrow
    "\u2713": "[x]", # check mark
    "\u2717": "[ ]", # cross mark
}


def _sanitize(text: str) -> str:
    """Replace Unicode characters that Helvetica cannot render."""
    for char, replacement in _UNICODE_SUBS.items():
        text = text.replace(char, replacement)
    # Fallback: replace any remaining non-latin1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


class EquityReportPDF(FPDF):
    """Custom FPDF subclass with Indian Equity Research branding."""

    def __init__(self, ticker: str = "", recommendation: str = "", score: int = 0):
        super().__init__()
        self.ticker = ticker
        self.recommendation = recommendation.upper()
        self.score = score
        self.set_auto_page_break(auto=True, margin=20)

    # ── Header / Footer ──────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return  # Cover page has custom header
        self.set_fill_color(*BRAND_BLUE)
        self.rect(0, 0, 210, 12, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        self.set_xy(10, 3)
        self.cell(0, 6, f"Indian Equity Research  |  {self.ticker}", align="L")
        self.set_xy(10, 3)
        self.cell(0, 6, datetime.now().strftime("%d %b %Y"), align="R")
        self.set_text_color(*DARK_TEXT)
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  AI-Generated Report - Not Investment Advice", align="C")

    # ── Cover Page ────────────────────────────────────────────

    def add_cover_page(self):
        self.add_page()
        self.alias_nb_pages()

        # Blue banner
        self.set_fill_color(*BRAND_BLUE)
        self.rect(0, 0, 210, 80, "F")

        # Title
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*WHITE)
        self.set_xy(15, 18)
        self.cell(0, 12, "Investment Research Report")

        # Ticker
        self.set_font("Helvetica", "B", 22)
        self.set_xy(15, 35)
        self.cell(0, 10, self.ticker)

        # Date
        self.set_font("Helvetica", "", 12)
        self.set_xy(15, 50)
        self.cell(0, 8, datetime.now().strftime("%d %B %Y"))

        # Subtitle
        self.set_font("Helvetica", "I", 10)
        self.set_xy(15, 62)
        self.cell(0, 6, "AI Indian Equity Research System  |  NSE / BSE")

        # Recommendation badge
        if self.recommendation:
            rec_color = REC_COLORS.get(self.recommendation, GREY)
            self.set_fill_color(*rec_color)
            self.set_font("Helvetica", "B", 14)
            badge_w = self.get_string_width(f"  {self.recommendation}  ") + 8
            self.set_xy(15, 90)
            self.cell(badge_w, 12, f"  {self.recommendation}  ", fill=True, align="C")
            self.set_text_color(*DARK_TEXT)

        # Score
        if self.score:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*DARK_TEXT)
            self.set_xy(15 + (badge_w + 10 if self.recommendation else 0), 90)
            self.cell(50, 12, f"Score: {self.score}/100")

        self.set_text_color(*DARK_TEXT)
        self.ln(30)

    # ── Content Renderers ─────────────────────────────────────

    def section_heading(self, text: str, level: int = 2):
        """Render a section heading."""
        text = _sanitize(text)
        self.ln(4)
        if level == 1:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*BRAND_BLUE)
        elif level == 2:
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*BRAND_BLUE)
        else:
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*DARK_TEXT)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        if level <= 2:
            self.set_draw_color(*BRAND_BLUE)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)
        self.set_text_color(*DARK_TEXT)

    def paragraph(self, text: str):
        """Render a paragraph with basic bold support."""
        text = _sanitize(text)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK_TEXT)
        # Handle **bold** inline
        parts = re.split(r"(\*\*.*?\*\*)", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font("Helvetica", "B", 10)
                self.write(5, part[2:-2])
                self.set_font("Helvetica", "", 10)
            else:
                self.write(5, part)
        self.ln(6)

    def bullet_item(self, text: str):
        """Render a bullet point."""
        text = _sanitize(text)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK_TEXT)
        x = self.get_x()
        self.cell(8, 5, "-")  # bullet character
        self.multi_cell(0, 5, text.strip(), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def add_table(self, headers: list[str], rows: list[list[str]]):
        """Render a simple table with alternating row colors."""
        headers = [_sanitize(h) for h in headers]
        rows = [[_sanitize(c) for c in row] for row in rows]
        if not headers:
            return
        col_count = len(headers)
        usable_w = 190
        col_w = usable_w / col_count

        # Header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*BRAND_BLUE)
        self.set_text_color(*WHITE)
        for h in headers:
            self.cell(col_w, 7, h.strip(), border=1, fill=True, align="C")
        self.ln()

        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK_TEXT)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(*LIGHT_BG)
            else:
                self.set_fill_color(*WHITE)
            for j, cell in enumerate(row):
                self.cell(col_w, 6, cell.strip(), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)

    def add_divider(self):
        self.set_draw_color(*GREY)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def disclaimer_block(self, text: str):
        """Render a disclaimer in small italic grey text."""
        text = _sanitize(text)
        self.ln(6)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4, text)
        self.set_text_color(*DARK_TEXT)


# ── Markdown Parser ───────────────────────────────────────────

def _parse_table_block(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse a Markdown table block into headers and rows."""
    if len(lines) < 2:
        return [], []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip separator line
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells:
            rows.append(cells)
    return headers, rows


def generate_report_pdf(
    markdown_report: str,
    ticker: str = "UNKNOWN",
    recommendation: str = "",
    composite_score: int = 0,
) -> bytes:
    """Convert a Markdown investment report to a styled PDF.

    Returns PDF content as bytes (in-memory, no temp files).
    """
    pdf = EquityReportPDF(ticker=ticker, recommendation=recommendation, score=composite_score)
    pdf.add_cover_page()
    pdf.add_page()

    lines = markdown_report.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Headings
        if stripped.startswith("### "):
            pdf.section_heading(stripped[4:], level=3)
            i += 1
            continue
        if stripped.startswith("## "):
            pdf.section_heading(stripped[3:], level=2)
            i += 1
            continue
        if stripped.startswith("# "):
            pdf.section_heading(stripped[2:], level=1)
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            pdf.add_divider()
            i += 1
            continue

        # Table block
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\|[\s\-:|]+\|$", lines[i + 1].strip()):
            table_lines = []
            while i < len(lines) and "|" in lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            headers, rows = _parse_table_block(table_lines)
            if headers:
                pdf.add_table(headers, rows)
            continue

        # Bullet points
        if re.match(r"^\s*[-*]\s+", stripped):
            text = re.sub(r"^\s*[-*]\s+", "", stripped)
            pdf.bullet_item(text)
            i += 1
            continue

        # Numbered list
        if re.match(r"^\s*\d+\.\s+", stripped):
            text = re.sub(r"^\s*\d+\.\s+", "", stripped)
            pdf.bullet_item(text)
            i += 1
            continue

        # Disclaimer detection
        if stripped.lower().startswith("*disclaimer") or stripped.lower().startswith("disclaimer"):
            # Collect full disclaimer block
            disclaimer_text = stripped.strip("*").strip()
            i += 1
            while i < len(lines) and lines[i].strip():
                disclaimer_text += " " + lines[i].strip().strip("*")
                i += 1
            pdf.disclaimer_block(disclaimer_text)
            continue

        # Regular paragraph
        pdf.paragraph(stripped)
        i += 1

    # Output as bytes
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
