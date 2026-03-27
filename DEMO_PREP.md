# AlphaLens — Demo Prep Document

## One-Line Pitch
**AlphaLens screens 500 Indian stocks like an institution and explains them like an analyst — powered by 10 AI agents, gated screening, RAG, and multi-round debate.**

---

## Architecture Overview

```
                          ┌─────────────────────────────────┐
                          │         AlphaLens App            │
                          │      (Streamlit Frontend)        │
                          └──────────┬──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
            ┌───────▼────────┐              ┌─────────▼────────┐
            │  SCREENER GRAPH │              │  DEEP DIVE GRAPH  │
            │  (6 agents)     │              │  (4 agents)       │
            └───────┬─────────┘              └─────────┬────────┘
                    │                                  │
        ┌───────────┴──────────┐                Sequential
        │   LangGraph Engine   │              data → analysis
        │   Fan-out/Fan-in     │              → sentiment → report
        └───────────┬──────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
  FAISS RAG    SQLite Cache    Free APIs
  (local)      (2-layer)      (yfinance, BSE,
                               Google News)
```

---

## The 10 Agents

### Screener Pipeline (6 agents)

| # | Agent | Role | Key Output |
|---|-------|------|------------|
| 1 | **Regime Agent** | Detects market mood (Bull/Bear/Rotation/Mixed) from VIX + Nifty data | Regime weights that adjust scoring |
| 2 | **Momentum Agent** | Screens stocks through 5 gates / 17 criteria for trend strength | Scored + tiered stock list |
| 3 | **Value Agent** | Screens through 7 gates / 30 criteria for turnaround candidates | Scored + tiered stock list |
| 4 | **Merge Agent** | Deduplicates across categories, produces final pick list | recommended_tickers |
| 5 | **Validation/USP Agent** | Runs 5-dimensional USP scoring + contradiction detection | USP cards + alerts |
| 6 | **Debate Agent** | Bull vs Bear vs Judge on top 3 stocks | Conviction score + recommendation |

### Deep Dive Pipeline (4 agents)

| # | Agent | Role | Key Output |
|---|-------|------|------------|
| 7 | **Data Agent** | Fetches financials, filings, corporate actions | Raw company data |
| 8 | **Analysis Agent** | Computes ROCE, DCF, peer comparison, Indian metrics | Ratios + valuation |
| 9 | **Sentiment Agent** | Analyzes Indian news sentiment + management mood | Sentiment scores |
| 10 | **Report Agent** | Generates investment memo with quantitative scoring | Score/100 + report |

---

## LangGraph Workflow — Fan-In / Fan-Out

### Screener Graph (the interesting one for demo)

```
                    Load Universe (Nifty 500)
                           │
                    ┌──────▼──────┐
                    │ Regime Agent │  ← VIX + Nifty analysis
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼────────┐     ┌─────────▼─────────┐
     │ Momentum Agent  │     │  Value Bot Agent   │   ◄── FAN-OUT
     │ (5 gates, 17    │     │  (7 gates, 30      │       (parallel)
     │  criteria)      │     │   criteria)        │
     └────────┬────────┘     └─────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                    ┌──────▼──────┐
                    │ Merge Agent │                    ◄── FAN-IN
                    └──────┬──────┘                        (collect + dedup)
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼────────┐     ┌─────────▼─────────┐
     │ USP Validation  │     │   RAG Ingest       │   ◄── FAN-OUT #2
     │ (5 dimensions)  │     │   (FAISS index)    │       (parallel)
     └────────┬────────┘     └─────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                    ┌──────▼──────┐
                    │ Debate Agent│                    ◄── FAN-IN #2
                    │ (Bull/Bear/ │                        (waits for both)
                    │  Judge)     │
                    └──────┬──────┘
                           │
                          END
```

**Why Fan-Out matters:** Momentum and Value screening run simultaneously — cuts total time nearly in half. Same for USP scoring and RAG ingestion.

### Deep Dive Graph (sequential)

```
    Data Agent → Analysis Agent → Sentiment Agent → Report Agent → END
```

Sequential because each agent needs the previous one's output (can't analyze without data, can't score sentiment without context, can't write report without everything).

---

## Gate-Based Screening — How It Works

### The 3 Gate Types

```
  ┌─────────────────────────────────────────────────┐
  │  HARD GATE   │  Must pass or stock eliminated    │  ← No exceptions
  ├──────────────┼──────────────────────────────────│
  │  SOFT GATE   │  Must pass, but lower threshold   │  ← Some flexibility
  ├──────────────┼──────────────────────────────────│
  │  BONUS GATE  │  Adds to score, never blocks      │  ← Extra credit
  └─────────────────────────────────────────────────┘
```

### Example: Momentum (Trend Rider)

```
Gate 1: Sector Tide [HARD] — need 2/3
  ├─ Sector RS positive?
  ├─ Sector highs > lows?
  └─ Sector in top 4?

Gate 2: Trend Structure [HARD] — need 3/5
  ├─ Above 200 DMA? [MUST]
  ├─ Golden alignment (50>100>200)?
  ├─ RSI 55-75?
  ├─ Near 52W high?
  └─ MACD bullish?

Gate 3: Volume Conviction [SOFT] — need 1/2
  ├─ Volume breakout?
  └─ OBV rising?

Gate 4: Smart Money [SOFT] — need 1/2
  ├─ Promoter holding stable?
  └─ Institutional interest?

Gate 5: Fundamental Backbone [SOFT] — need 1/3
  ├─ OPM improving?
  ├─ Order booking?
  └─ ROE > 12%?
```

### 4-Tier Conviction System

```
  ┌──────────────────────────────────────────────────────┐
  │  BUY ZONE    │  Score ≥ 10/15  +  ALL gates pass     │  ★
  ├──────────────┼──────────────────────────────────────│
  │  WATCHLIST   │  Score ≥ 9/15   +  ALL gates pass     │  ◉
  ├──────────────┼──────────────────────────────────────│
  │  MONITOR     │  Score ≥ 8/15   +  ALL gates pass     │  ◎
  ├──────────────┼──────────────────────────────────────│
  │  NEAR MISS   │  Score ≥ 9  +  0 hard fail + 1 soft   │  ⊘
  └──────────────────────────────────────────────────────┘
```

---

## RAG (Retrieval-Augmented Generation)

### How it works in AlphaLens

```
  1. INGEST                    2. RETRIEVE                  3. GENERATE
  ┌──────────────┐            ┌──────────────┐            ┌──────────────┐
  │ Screener.in  │            │ "RELIANCE    │            │ LLM (GPT-4o) │
  │ data for top │──chunk──►  │  revenue     │──search──► │ + retrieved  │
  │ 15 stocks    │  1000      │  growth"     │  top 5     │   context    │
  │              │  tokens    │              │  chunks    │              │
  │ + PDF reports│  each      │  FAISS index │            │ = grounded   │
  └──────────────┘            └──────────────┘            │   response   │
                                                          └──────────────┘
```

### Key Details
- **Embeddings:** all-MiniLM-L6-v2 (local, free — no API cost)
- **Vector Store:** FAISS (in-memory, fast similarity search)
- **Chunk Size:** 1000 tokens, 200 overlap
- **Retrieval:** Top 5 chunks per query, filtered by stock symbol
- **Data Sources:** Screener.in (quarterly results, balance sheet, ratios, shareholding, pros/cons)

### Where RAG is used
1. **Debate agents** — Bull fetches growth evidence, Bear fetches risk evidence
2. **Report agent** — Retrieves analyst context for memo generation
3. **Deep Dive chat** — Context-aware Q&A grounded in real data

---

## Cache Strategy

### 2-Layer Architecture

```
  Request
    │
    ▼
  ┌──────────────────┐     HIT
  │  Layer 1:        │────────────► Return instantly
  │  In-Memory Dict  │
  │  TTL: 1 hour     │
  └────────┬─────────┘
           │ MISS
           ▼
  ┌──────────────────┐     HIT
  │  Layer 2:        │────────────► Return + promote
  │  SQLite on Disk  │              to Layer 1
  │  TTL: 12 hours   │
  └────────┬─────────┘
           │ MISS
           ▼
  ┌──────────────────┐
  │  Fetch from API  │────────────► Save to BOTH layers
  │  (yfinance, BSE, │
  │   Google News)   │
  └──────────────────┘
```

### What Gets Cached

| Data | TTL | Why |
|------|-----|-----|
| Stock fundamentals (.info) | 12 hours | Rarely changes intraday |
| Financial statements | 12 hours | Quarterly data |
| Price/volume data | 10 min | Needs freshness for technical screens |
| News articles | 1-2 hours | News cycle |
| Shareholding data | 1 hour | Updated quarterly |

### Protection Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| **Rate Limiter** (token bucket) | 10 req/sec to yfinance — prevents IP ban |
| **Circuit Breaker** | After 3 failures → stop calling BSE API for 5 min |
| **Disk persistence** | Cache survives app restarts |

---

## AI Debate System

### Multi-Round Bull vs Bear

```
  ROUND 1 (parallel)
  ┌─────────────┐         ┌─────────────┐
  │  BULL AGENT  │         │  BEAR AGENT  │
  │  "Why buy?"  │         │  "Why avoid?"│
  │  + RAG data  │         │  + RAG data  │
  └──────┬───────┘         └──────┬───────┘
         │                        │
         ▼                        ▼
  ROUND 2 (sequential — needs rebuttals)
  ┌─────────────┐         ┌─────────────┐
  │  BULL rebuts │◄────────│  Bear said:  │
  │  Bear's pts  │         │  "..."       │
  └──────┬───────┘         └──────┬───────┘
         │                        │
         │  BEAR rebuts   ◄───────┘
         │  Bull's pts
         ▼
  ┌──────────────────┐
  │   JUDGE AGENT    │
  │                  │
  │  Conviction: 7/10│
  │  Rec: BUY        │
  │  Bull str: 8/10  │
  │  Bear str: 5/10  │
  └──────────────────┘
```

### Models & Fallback
- **Primary:** GPT-4o via OpenRouter
- **Fallback:** Gemini 2.0 Flash (free, auto-switches on failure)
- **Temperature:** 0.5 (balanced creativity)

---

## USP Scoring — 5 Dimensions

```
  ┌────────────────────────────────────────────────────┐
  │                  USP COMPOSITE                      │
  │                                                    │
  │   Geopolitical  Smart Money  Regulatory  Mgmt  Promoter │
  │   Risk          Lag          Signal      Cred  Behavior │
  │   ─────────     ──────────   ─────────  ─────  ──────── │
  │   Trade Risk    FII flow     Tailwinds  Score  Signal   │
  │   Policy Risk   DII flow     Headwinds  Method Buys/Sells│
  │   Commodity     Lag score    Net signal         RPT      │
  │   Event Risk                                   Anomaly  │
  └────────────────────────────────────────────────────┘
```

Each dimension scored 0-100, combined into composite for ranking.

---

## Tech Stack (All Free)

| Component | Tool | Cost |
|-----------|------|------|
| Orchestration | LangGraph | Free |
| Frontend | Streamlit (dark theme) | Free |
| Market Data | yfinance, BSE API | Free |
| News | Google News RSS | Free |
| Fundamental Data | Screener.in scraping | Free |
| Embeddings | all-MiniLM-L6-v2 (local) | Free |
| Vector Store | FAISS (in-memory) | Free |
| Cache | SQLite (local) | Free |
| Charts | Plotly | Free |
| LLM | OpenRouter (GPT-4o + Gemini Flash fallback) | ~$0.01/query |

---

## Demo Flow (Suggested)

### 1. Show Trend Rider (2 min)
- Click "Run Trend Rider" → show progress
- Results: 4 tier cards (Buy Zone, Watchlist, Monitor, Near Miss)
- Expand a Buy Zone stock → show gate-by-gate breakdown
- Switch to USP Analysis tab → heatmap + radar charts

### 2. Show AI Debate (1 min)
- Switch to AI Debate tab
- Show conviction gauge, bull/bear arguments, strength bars
- Highlight: "This ran automatically on top 3 stocks"

### 3. Show Stock Deep Dive (2 min)
- Switch to Deep Dive tab or use sidebar
- Show scorecard: business summary, key financials, quick take
- Ask in chat: "Who are the competitors?" → focused analysis answer
- Ask: "What's the P/E?" → short factual answer
- Highlight: "Different question types get different depth answers"

### 4. Architecture Talking Points (1 min)
- "10 agents orchestrated by LangGraph"
- "Fan-out: Momentum + Value screen in parallel"
- "RAG grounds every debate argument in real financial data"
- "2-layer cache: first run ~3 min, repeat runs instant"
- "100% free data sources — no Bloomberg, no paid APIs"

---

## Key Numbers for Demo

- **500+** stocks screened from Nifty 500 universe
- **50+** screening criteria across 12 gates
- **10** specialized AI agents
- **5** USP dimensions per stock
- **2** parallel fan-out stages in the graph
- **3** debate rounds (1 parallel + 1 sequential + judge)
- **0** paid data APIs (all free sources)
