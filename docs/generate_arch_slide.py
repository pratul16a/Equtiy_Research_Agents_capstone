"""Generate architecture slide for Indian Equity Analyst — single .pptx slide."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ── Colors ──
BG = RGBColor(0x0E, 0x11, 0x17)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xC0, 0xC0, 0xC0)
MED_GRAY = RGBColor(0x88, 0x92, 0xA0)
DARK_CARD = RGBColor(0x1A, 0x1F, 0x2E)
DARKER_CARD = RGBColor(0x11, 0x18, 0x20)

# Accent colors
CYAN = RGBColor(0x00, 0xD4, 0xAA)
ORANGE = RGBColor(0xFF, 0xA7, 0x26)
GREEN = RGBColor(0x4E, 0xC9, 0xB0)
YELLOW = RGBColor(0xFF, 0xD7, 0x00)
PURPLE = RGBColor(0xB3, 0x88, 0xFF)
RED = RGBColor(0xFF, 0x47, 0x57)
BLUE = RGBColor(0x56, 0x9C, 0xD6)

# Slide dimensions (widescreen 16:9)
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def add_rounded_box(slide, left, top, width, height, fill_color, border_color=None, border_width=Pt(1)):
    """Add a rounded rectangle shape."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = border_width
    else:
        shape.line.fill.background()
    # Smaller corner radius
    shape.adjustments[0] = 0.05
    return shape


def add_text_box(slide, left, top, width, height, text, font_size=10,
                 color=WHITE, bold=False, alignment=PP_ALIGN.LEFT, font_name="Segoe UI"):
    """Add a text box with styled text."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_arrow(slide, start_left, start_top, end_left, end_top, color=MED_GRAY):
    """Add a connector arrow."""
    connector = slide.shapes.add_connector(
        1,  # straight connector
        start_left, start_top, end_left, end_top
    )
    connector.line.color.rgb = color
    connector.line.width = Pt(1.5)
    return connector


def add_pipeline_node(slide, left, top, width, height, title, subtitle, accent_color, border=False):
    """Add a pipeline node box with title and subtitle."""
    box = add_rounded_box(slide, left, top, width, height,
                          DARK_CARD, accent_color if border else None, Pt(2))

    # Title
    txBox = slide.shapes.add_textbox(left + Pt(4), top + Pt(4), width - Pt(8), Pt(18))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(9)
    p.font.color.rgb = accent_color
    p.font.bold = True
    p.font.name = "Segoe UI"
    p.alignment = PP_ALIGN.CENTER

    # Subtitle
    if subtitle:
        txBox2 = slide.shapes.add_textbox(left + Pt(4), top + Pt(22), width - Pt(8), height - Pt(26))
        tf2 = txBox2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(7)
        p2.font.color.rgb = LIGHT_GRAY
        p2.font.name = "Segoe UI"
        p2.alignment = PP_ALIGN.CENTER

    return box


def add_info_box(slide, left, top, width, height, title, body, accent_color):
    """Add an info box with colored left border effect."""
    # Main box
    box = add_rounded_box(slide, left, top, width, height, DARKER_CARD, accent_color, Pt(1.5))

    # Title
    txBox = slide.shapes.add_textbox(left + Pt(8), top + Pt(6), width - Pt(16), Pt(16))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(9)
    p.font.color.rgb = accent_color
    p.font.bold = True
    p.font.name = "Segoe UI"

    # Body
    txBox2 = slide.shapes.add_textbox(left + Pt(8), top + Pt(24), width - Pt(16), height - Pt(30))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = body
    p2.font.size = Pt(7)
    p2.font.color.rgb = LIGHT_GRAY
    p2.font.name = "Segoe UI"
    p2.line_spacing = Pt(11)


def add_arrow_text(slide, left, top, text="→", color=MED_GRAY, size=14):
    """Add arrow text between nodes."""
    add_text_box(slide, left, top, Pt(20), Pt(20), text, size, color, alignment=PP_ALIGN.CENTER)


def generate_slide():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # Blank layout
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)

    # Set background
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG

    # ── Title ──
    add_text_box(slide, Inches(0.4), Inches(0.2), Inches(8), Inches(0.5),
                 "Two Pipelines, One Platform", 26, WHITE, bold=True)

    # ── Pipeline 1 Label ──
    add_text_box(slide, Inches(0.4), Inches(0.8), Inches(2), Inches(0.3),
                 "PIPELINE 1: SCREENER", 10, CYAN, bold=True)
    add_text_box(slide, Inches(2.8), Inches(0.8), Inches(5), Inches(0.3),
                 "Fan-out/fan-in · 8 nodes · Parallel screening", 8, MED_GRAY)

    # ── Pipeline 1 Nodes ──
    node_y = Inches(1.2)
    node_h = Pt(48)
    node_w = Pt(100)
    gap = Pt(12)

    # Node positions (left to right)
    x_start = Inches(0.4)

    # 1. Universe
    x = x_start
    add_pipeline_node(slide, x, node_y, node_w, node_h,
                      "Universe", "Nifty 500", WHITE, border=True)

    # Arrow
    x_arrow = x + node_w + Pt(2)
    add_arrow_text(slide, x_arrow, node_y + Pt(14), "→")

    # 2. Regime
    x = x_arrow + Pt(22)
    add_pipeline_node(slide, x, node_y, node_w, node_h,
                      "Regime", "Bull/Bear/Mixed", BLUE, border=True)

    # Arrow
    x_arrow = x + node_w + Pt(2)
    add_arrow_text(slide, x_arrow, node_y + Pt(14), "→")

    # 3. Fan-out: Momentum + Value (stacked)
    x_fanout = x_arrow + Pt(22)
    fanout_w = Pt(120)

    # Fan-out bracket box
    bracket = add_rounded_box(slide, x_fanout - Pt(4), node_y - Pt(8),
                              fanout_w + Pt(8), node_h + Pt(56), BG, MED_GRAY, Pt(1))

    # Momentum (top)
    add_pipeline_node(slide, x_fanout, node_y - Pt(2), fanout_w, Pt(40),
                      "Momentum", "17 crit / 5 gates", CYAN, border=True)

    # Value (bottom)
    add_pipeline_node(slide, x_fanout, node_y + Pt(44), fanout_w, Pt(40),
                      "Value", "28+ crit / 7 gates", ORANGE, border=True)

    # Arrow after fan-out
    x_arrow = x_fanout + fanout_w + Pt(8)
    add_arrow_text(slide, x_arrow, node_y + Pt(20), "→")

    # 4. Merge
    x = x_arrow + Pt(22)
    add_pipeline_node(slide, x, node_y + Pt(6), node_w, node_h,
                      "Merge", "Deduplicate", WHITE, border=True)

    # Arrow
    x_arrow = x + node_w + Pt(2)
    add_arrow_text(slide, x_arrow, node_y + Pt(20), "→")

    # 5. Fan-out: Validation + RAG (stacked)
    x_fanout2 = x_arrow + Pt(22)
    fanout_w2 = Pt(110)

    bracket2 = add_rounded_box(slide, x_fanout2 - Pt(4), node_y - Pt(8),
                               fanout_w2 + Pt(8), node_h + Pt(56), BG, MED_GRAY, Pt(1))

    add_pipeline_node(slide, x_fanout2, node_y - Pt(2), fanout_w2, Pt(40),
                      "Validation", "USP scoring", GREEN, border=True)

    add_pipeline_node(slide, x_fanout2, node_y + Pt(44), fanout_w2, Pt(40),
                      "RAG Ingest", "FAISS index", YELLOW, border=True)

    # Arrow
    x_arrow = x_fanout2 + fanout_w2 + Pt(8)
    add_arrow_text(slide, x_arrow, node_y + Pt(20), "→")

    # 6. AI Debate
    x = x_arrow + Pt(22)
    add_pipeline_node(slide, x, node_y + Pt(6), Pt(110), node_h,
                      "AI Debate", "Bull/Bear/Judge", PURPLE, border=True)

    # ── Pipeline 2 Label ──
    p2_y = Inches(3.0)
    add_text_box(slide, Inches(0.4), p2_y, Inches(2), Inches(0.3),
                 "PIPELINE 2: DEEP DIVE", 10, ORANGE, bold=True)
    add_text_box(slide, Inches(2.8), p2_y, Inches(5), Inches(0.3),
                 "Sequential per-stock · 4 nodes · Full investment memo", 8, MED_GRAY)

    # ── Pipeline 2 Nodes ──
    p2_node_y = p2_y + Inches(0.35)
    p2_w = Pt(140)
    p2_h = Pt(48)

    agents = [
        ("Data Agent", "yfinance + BSE\nfilings + shareholding", BLUE),
        ("Analysis Agent", "Ratios, DCF, ROCE\npeer comparison", GREEN),
        ("Sentiment Agent", "News RSS + NLP\nmanagement tone", ORANGE),
        ("Report Agent", "5-dim scoring\n+ RAG retrieval", PURPLE),
    ]

    x = Inches(0.4)
    for i, (title, sub, color) in enumerate(agents):
        add_pipeline_node(slide, x, p2_node_y, p2_w, p2_h, title, sub, color, border=True)
        if i < len(agents) - 1:
            add_arrow_text(slide, x + p2_w + Pt(4), p2_node_y + Pt(14), "→")
        x += p2_w + Pt(30)

    # ── Bottom Section: 4 Info Boxes ──
    bottom_y = Inches(4.6)
    add_text_box(slide, Inches(0.4), bottom_y - Inches(0.35), Inches(6), Inches(0.3),
                 "HOW AGENTS COMMUNICATE", 11, WHITE, bold=True)

    box_w = Inches(3.0)
    box_h = Inches(1.1)
    box_gap = Inches(0.2)

    info_boxes = [
        ("Shared State", "TypedDict — each agent reads what\nit needs, writes what it knows.\nFan-out via operator.add", CYAN),
        ("USP Dimensions", "Geopolitical · Smart Money Lag\nRegulatory · Mgmt Credibility\nPromoter Behavior", GREEN),
        ("Data Sources", "yfinance · BSE APIs\nGoogle News RSS · Screener.in\nFAISS RAG (local)", YELLOW),
        ("Resilience", "SQLite cache (4hr) · Circuit breaker\n(3 fails → 5min cooldown)\nRate limiter · Retry backoff", ORANGE),
    ]

    x = Inches(0.4)
    for title, body, color in info_boxes:
        add_info_box(slide, x, bottom_y, box_w, box_h, title, body, color)
        x += box_w + box_gap

    # ── LLM Commentary annotation ──
    ann_y = Inches(6.0)
    add_text_box(slide, Inches(0.4), ann_y, Inches(12), Inches(0.25),
                 "LLM Commentary Layer: Investment Thesis · Key Catalysts · Key Risks · What to Watch — grounded in Google News RSS",
                 8, MED_GRAY)

    # ── Footer ──
    add_text_box(slide, Inches(0.4), Inches(6.9), Inches(6), Inches(0.3),
                 "Indian Equity Analyst · LangGraph + Streamlit + Gemini 2.0 Flash", 8, MED_GRAY)

    # Save
    output_path = "docs/architecture_slide.pptx"
    prs.save(output_path)
    print(f"Slide saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_slide()
