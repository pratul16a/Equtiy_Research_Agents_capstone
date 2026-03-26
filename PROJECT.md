# Indian Equity Analyst

A multi-agent AI system for Indian equity research — built with LangGraph, Streamlit, and LLM-powered analysis.

## What It Does

Screens 500+ NSE/BSE stocks through a gate-based filtering system, scores them across 5 proprietary USP dimensions, runs a Bull vs Bear AI debate on top picks, and generates investment-ready research notes.

### Key Features

- **Momentum Screener** — 17 criteria across 5 gates (sector tide, trend structure, volume conviction, smart money, fundamentals)
- **Value Bottom Screener** — 28+ criteria across 7 gates (valuation floor, turnaround signals, trap avoidance, technical timing, re-rating catalysts, insider conviction, macro tailwinds)
- **4-Tier Conviction System** — Buy Zone / Watchlist / Monitor / Near Miss based on gate pass/fail + absolute score thresholds
- **5-Dimension USP Scoring** — Geopolitical Risk, Smart Money Lag, Regulatory Tailwind, Management Credibility, Promoter Behavior
- **LLM-Powered Commentary** — Per-stock investment thesis, catalysts, risks, and actionable triggers grounded in Google News RSS
- **Bull vs Bear Debate** — Multi-round AI debate with Judge agent assigning conviction scores and BUY/HOLD/SELL ratings
- **Stock Deep Dive** — Full pipeline: data fetch, ratio analysis, DCF valuation, sentiment scoring, investment memo generation
- **RAG Integration** — FAISS-indexed research frameworks (Buffett, Lynch, Indian market guides) + Screener.in data
- **PDF/CSV Export** — Downloadable screener results, raw data for validation, and deep dive reports
- **Sharing** — Email and WhatsApp integration for distributing reports

### Data Sources

| Source | What | Cost |
|--------|------|------|
| Yahoo Finance (yfinance) | Price, volume, financials, institutional holdings | Free |
| BSE APIs | Corporate announcements, board meetings, filings | Free |
| Google News RSS | Recent headlines for sentiment + commentary grounding | Free |
| Screener.in | Shareholding patterns, quarterly results (web scrape) | Free |
| FAISS (local) | RAG retrieval over indexed research documents | Free |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (multi-agent fan-out/fan-in) |
| LLM | Gemini 2.0 Flash (default) via Google AI, or GPT-4o via OpenRouter/OpenAI |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) or Google/OpenAI |
| Vector Store | FAISS (local, file-based) |
| UI | Streamlit (dark theme) |
| Charts | Plotly (radar charts, sector heatmaps) |
| Caching | In-memory TTL + SQLite disk cache (4hr TTL) |
| Deployment | Docker + GCP Cloud Run |

---

## Quick Start

### Prerequisites

- Python 3.11+
- At least one LLM API key (Google AI, OpenRouter, or OpenAI)

### Setup

```bash
# Clone and navigate
git clone https://github.com/pratul16a/Equtiy_Research_Agents_capstone.git
cd indian_equity_analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API key(s)

# Run
streamlit run streamlit_app.py
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | One of three | Google AI API key (for Gemini models) |
| `OPENROUTER_API_KEY` | One of three | OpenRouter API key (access to multiple models) |
| `OPENAI_API_KEY` | One of three | OpenAI API key (GPT-4o) |
| `LLM_MODEL` | No | Model override (default: `gemini-2.0-flash`) |
| `NEWSAPI_KEY` | No | NewsAPI key for additional news sources |
| `LANGSMITH_API_KEY` | No | LangSmith tracing for debugging |
| `SMTP_EMAIL` | No | Gmail address for email sharing |
| `SMTP_PASSWORD` | No | Gmail app password |
| `TWILIO_ACCOUNT_SID` | No | Twilio SID for WhatsApp sharing |
| `TWILIO_AUTH_TOKEN` | No | Twilio auth token |

---

## Architecture Overview

### Two LangGraph Pipelines

**Pipeline 1: Screener** (multi-agent fan-out/fan-in)
```
load_universe → regime → [momentum | value_bottom] → merge → [validation | rag_ingest] → debate → END
```

**Pipeline 2: Deep Dive** (sequential per-stock)
```
data_agent → analysis_agent → sentiment_agent → report_agent → END
```

### Agent Communication

Agents communicate via LangGraph shared state (TypedDict), not direct calls:
- Fan-out: Momentum and Value nodes run in parallel, both append to `category_results`
- Fan-in: Merge node deduplicates, outputs `recommended_tickers`
- Validation reads tickers, writes `usp_scores` → Debate agents read both

### USP Scoring Modules

| Dimension | Source | What It Measures |
|-----------|--------|-----------------|
| Geopolitical Risk | Config-driven (sector profiles) | Trade/policy/commodity/event risk exposure |
| Smart Money Lag | yfinance + Screener.in | Gap between fundamental improvement and institutional positioning |
| Regulatory | Policy mapping JSON | Government policy tailwinds/headwinds per sector |
| Management Credibility | News + announcements | Track record assessment via NLP |
| Promoter Behavior | yfinance + Screener.in | Holding %, pledge %, insider activity, related party txns |

### Caching Strategy

| Layer | TTL | Purpose |
|-------|-----|---------|
| In-memory dict | 5 min | Hot path: repeated UI renders |
| SQLite disk cache | 4 hours | Survives restarts: yfinance data, computed scores |
| Rate limiter | 10 calls/sec | Yahoo Finance API throttling |
| Circuit breaker | 3 failures → 5 min cooldown | BSE API resilience |

**Performance**: Cold run ~18 min (500+ stock universe). Cached run ~22 seconds.

---

## Deployment

### GCP Cloud Run

```bash
# Authenticate
gcloud auth login

# Deploy (uses Dockerfile)
bash deploy.sh
```

Deployed at: `https://indian-equity-analyst-<project-id>.asia-south1.run.app`

**Cloud Run config**: 4 vCPU / 4 GB RAM / 300s timeout / min 0 instances

### Docker (local)

```bash
docker build -t indian-equity-analyst .
docker run -p 8501:8501 --env-file .env indian-equity-analyst
```

---

## Project Structure

```
indian_equity_analyst/
├── streamlit_app.py              # Main UI (Streamlit)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container build
├── deploy.sh                     # GCP Cloud Run deploy script
├── .env.example                  # Environment template
├── ARCHITECTURE.md               # Detailed architecture reference
├── PROJECT.md                    # This file
│
├── app/
│   ├── config.py                 # LLM factory, market constants
│   ├── state.py                  # LangGraph state schemas
│   ├── graph.py                  # LangGraph workflow definitions
│   ├── scoring.py                # 5-dimensional quantitative scoring
│   ├── deep_dive_chat.py         # Conversational stock chat
│   ├── prompts/                  # LLM prompt templates
│   ├── agents/                   # LangGraph agent nodes (10 agents)
│   ├── rag/                      # FAISS ingestion + retrieval
│   ├── tools/                    # Financial data + screening logic
│   │   └── screener/             # 50+ screening criteria + USP modules
│   ├── ui/                       # Streamlit rendering components
│   ├── utils/                    # Cache, currency, fiscal year helpers
│   └── sharing/                  # PDF, email, WhatsApp export
│
├── data/
│   ├── faiss_index/              # FAISS vector store
│   ├── sample_reports/           # RAG source documents
│   ├── policy_mapping.json       # Regulatory tailwind/headwind config
│   ├── geopolitical_config.json  # Sector risk profiles
│   └── cache.db                  # SQLite disk cache (auto-generated)
│
├── mcp_servers/                  # MCP tool servers (financial data, news, filings)
└── venv/                         # Virtual environment (not committed)
```

---

## Versioning

| Branch | Status | Description |
|--------|--------|-------------|
| `Equity_Reserach_v3` | Stable / GCP deployed | Production branch |
| `Equity_Research_v4` | Active development | USP commentary, Screener.in fallback, regulatory scoring, UI redesign |

---

## Key Design Decisions

1. **Gate-based screening over scoring-only** — Hard/soft/bonus gates ensure minimum quality bars before scoring, preventing high-scoring stocks with fatal flaws from passing
2. **Deterministic USP scoring + LLM commentary** — Scores are rule-based (reproducible, fast). Commentary is LLM-generated (contextual, opinionated). Commentary never modifies scores.
3. **Web grounding via Google News RSS** — Free, no API key required. Injects 8 recent headlines per stock into LLM prompt for real-world context.
4. **Screener.in fallback** — yfinance returns None for many Indian PSU stocks' shareholding data. Screener.in web scraping fills the gap.
5. **SQLite disk cache** — Survives app restarts. 4-hour TTL balances freshness vs API rate limits. Cold run = 18 min, cached = 22 sec.
6. **Multi-round debate** — Bull/Bear agents argue with evidence, Judge assigns conviction. Surfaces risks that screening alone misses.
