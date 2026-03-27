# AlphaLens — Demo Prep Document (v4)

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
| 5 | **Validation/USP Agent** | Runs 5-dimensional USP scoring + contradiction detection | USP cards + commentary |
| 6 | **Debate Agent** | Bull vs Bear vs Judge on top 3 stocks | Conviction score + recommendation |

### Deep Dive Pipeline (4 agents)

| # | Agent | Role | Key Output |
|---|-------|------|------------|
| 7 | **Data Agent** | Fetches financials, filings, corporate actions | Raw company data |
| 8 | **Analysis Agent** | Computes ROCE, DCF, peer comparison, Indian metrics | DCF Value + ROCE + peer cards |
| 9 | **Sentiment Agent** | Analyzes Indian news sentiment + management mood | Sentiment label + score |
| 10 | **Report Agent** | Generates investment memo (narrative, not score) | Quick Take + full report |

### v4 Deep Dive Design — Scores → Commentary + RAG

**Problem solved:** In v3, the deep dive produced an "Investment Score" (48/100 = HOLD) that often contradicted the screener tier (Buy Zone). Different methodology (5-dimension continuous averaging vs gate-based pass/fail) confused users.

**v4 approach:** Deep dive agents no longer produce a conflicting composite score. Instead, their outputs surface as:

```
  ┌─────────────────────────────────────────────────────────────┐
  │  SCORECARD LAYOUT (v4)                                       │
  │                                                              │
  │  [Company Name]  [SCREENER TIER BADGE]                       │
  │  [Sector · Industry]                                         │
  │  [Business description — 200 chars]                          │
  │                                                              │
  │  ┌─────────┐ ┌─────────┐ ┌──────────────┐                   │
  │  │  Price   │ │Mkt Cap  │ │Screener Tier │  ← from screener  │
  │  └─────────┘ └─────────┘ └──────────────┘                   │
  │                                                              │
  │  ┌─────────┐ ┌─────────┐ ┌──────────────┐ ┌─────────┐      │
  │  │  P/E    │ │  ROE    │ │  Revenue     │ │  D/E    │      │
  │  └─────────┘ └─────────┘ └──────────────┘ └─────────┘      │
  │                                                              │
  │  RESEARCH INSIGHTS (from AI agents)                          │
  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐     │
  │  │DCF Value│ │Sentiment │ │  ROCE   │ │Peers Compared│     │
  │  │INR 2450 │ │Bullish   │ │  18.5%  │ │  5 stocks    │     │
  │  │+12% up  │ │Score: 72 │ │  Good   │ │              │     │
  │  └─────────┘ └──────────┘ └─────────┘ └──────────────┘     │
  │                                                              │
  │  ┌─── Quick Take (Report Agent narrative) ──────────────┐   │
  │  │ First 1000 chars of research report...                │   │
  │  └──────────────────────────────────────────────────────┘   │
  │                                                              │
  │  ┌─── Screening Analysis & Commentary ──────────────────┐   │
  │  │ Investment Thesis: ...  (from USP Agent)              │   │
  │  │ Key Catalysts: ...                                    │   │
  │  │ Key Risks: ...                                        │   │
  │  │ What to Watch: ...                                    │   │
  │  │                                                       │   │
  │  │ Geo: 72  Smart$: 45  Reg: 68  Mgmt: 55  Promo: 80   │   │
  │  └──────────────────────────────────────────────────────┘   │
  │                                                              │
  │  ┌─── RAG-Powered Chat ─────────────────────────────────┐   │
  │  │ Ask: "Who are the competitors?"                       │   │
  │  │ → 300-500 word answer grounded in Screener.in data    │   │
  │  └──────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
```

**Key design decision:** Screener tier (gate-based) is the primary verdict shown on the scorecard. Deep dive agents provide supporting evidence (DCF, sentiment, ROCE) without a competing score.

---

## Screener → Deep Dive Data Bridge

```
  SCREENER RUN                         DEEP DIVE PROFILE
  ┌────────────────┐                  ┌────────────────────┐
  │ USP Cards      │──inject──────►   │ screener_usp       │
  │ (5 dimensions) │                  │ screener_commentary │
  ├────────────────┤                  │ (thesis, catalysts, │
  │ Tier + Score   │──inject──────►   │  risks, watch)      │
  │ (Buy Zone 12/15│                  │ screener_tier       │
  ├────────────────┤                  │ screener_score      │
  │ RAG Index      │──query───────►   │                     │
  │ (FAISS chunks) │  at chat time    │ Chat context        │
  └────────────────┘                  └────────────────────┘
```

When screener data exists, it automatically flows into the deep dive profile via `_inject_screener_context()`. This means:
- Scorecard shows the screener tier instead of a conflicting score
- USP commentary (thesis, catalysts, risks) appears in the scorecard
- Chat answers are grounded in Screener.in financial data via RAG

---

## RAG-Powered Deep Dive Chat (v4)

### Question Classification → Context Filtering

```
  User Question
       │
       ▼
  ┌──────────────┐
  │  CLASSIFIER  │
  │              │
  │ "What's P/E?"│──► FACTUAL   → overview only      → 600 tokens  → 1-3 sentences
  │ "Competition"│──► ANALYSIS  → full data + RAG(4)  → 1500 tokens → 300-500 words
  │ "Should buy?"│──► THESIS    → everything + RAG(6) → 2000 tokens → 500-800 words
  └──────────────┘
```

### Context Sources per Question Type

| Source | Factual | Analysis | Thesis |
|--------|---------|----------|--------|
| Company overview + ratios | Yes | Yes | Yes |
| Financial statements | - | Summary | Summary |
| DCF + peer comparison | - | Yes | Yes |
| Risk & governance | - | Yes | Yes |
| Sentiment + news | - | Yes | Yes |
| Screener insights (tier, USP) | - | Yes | Yes |
| RAG chunks (Screener.in) | - | k=4 | k=6 |
| Research report | - | 2000 chars | Full |

### Analysis keywords that trigger RAG
`competitor, competition, competitive, moat, growth driver, risk, red flag, peer, strength, advantage, market share, segment, product, catalyst, headwind, tailwind, explain, analyze, assess, evaluate, impact, concern`

---

## Parallel Deep Dive Preloading

```
  SCREENER COMPLETE
       │
       ├──────────────────────────────────┐
       │                                  │
  ┌────▼─────────┐              ┌─────────▼────────────┐
  │  AI DEBATE   │              │  DEEP DIVE PRELOAD   │
  │  (top 3      │              │  (top 1 stock)       │
  │   stocks)    │              │  build_stock_profile()│
  │              │   PARALLEL   │  + inject screener    │
  │  Bull/Bear/  │              │    context            │
  │  Judge       │              │                       │
  └──────┬───────┘              └─────────┬────────────┘
         │                                │
         └────────────┬───────────────────┘
                      │
              BOTH COMPLETE
                      │
              User opens Tab 4
              → Profile already loaded!
              → No "Load Deep Dive" button needed
              → Instant scorecard + chat
```

**Why:** Previously, debate finished → user had to click "Load Deep Dive" → wait 30-60s. Now the top stock's profile loads in the background during debate. When the user switches to the Deep Dive tab, everything is ready.

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
                    ┌──────▼──────────┐
                    │ Deep Dive       │               ◄── PARALLEL with debate
                    │ Preload (bg)    │
                    └─────────────────┘
```

**Why Fan-Out matters:** Momentum and Value screening run simultaneously — cuts total time nearly in half. Same for USP scoring and RAG ingestion. Deep dive preloads in parallel with debate.

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
  │  BUY ZONE    │  Score >= 10/15  +  ALL gates pass     │  ★
  ├──────────────┼──────────────────────────────────────│
  │  WATCHLIST   │  Score >= 9/15   +  ALL gates pass     │  ◉
  ├──────────────┼──────────────────────────────────────│
  │  MONITOR     │  Score >= 8/15   +  ALL gates pass     │  ◎
  ├──────────────┼──────────────────────────────────────│
  │  NEAR MISS   │  Score >= 9  +  0 hard fail + 1 soft   │  ⊘
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
- **Retrieval:** Top 4-6 chunks per query, filtered by stock symbol
- **Data Sources:** Screener.in (quarterly results, balance sheet, ratios, shareholding, pros/cons)

### Where RAG is used
1. **Debate agents** — Bull fetches growth evidence, Bear fetches risk evidence
2. **Report agent** — Retrieves analyst context for memo generation
3. **Deep Dive chat (v4)** — Analysis questions get k=4 chunks, thesis questions get k=6 chunks, factual questions skip RAG for speed

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

**v4:** USP scores + commentary (investment thesis, catalysts, risks) are bridged into deep dive scorecard when available.

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
- Switch to Deep Dive tab — **profile is already loaded** (preloaded during debate)
- Show new scorecard layout:
  - Screener Tier badge (Buy Zone) — **no conflicting Investment Score**
  - Research Insights: DCF Value, Sentiment, ROCE, Peers Compared
  - Quick Take narrative (1000 chars from Report Agent)
  - Screening Analysis & Commentary (thesis, catalysts, risks from USP Agent)
- Ask in chat: **"Who are the competitors and what's the competitive edge?"** → 300-500 word answer grounded in Screener.in data + financial context
- Ask: **"What are the key risks?"** → cross-references governance, debt, promoter, geopolitical
- Ask: **"Should I buy this stock?"** → full thesis with bull/bear/verdict (500-800 words)
- Highlight: "Different question types get different depth — factual stays short, analysis uses RAG"

### 4. Architecture Talking Points (1 min)
- "10 agents orchestrated by LangGraph"
- "Fan-out: Momentum + Value screen in parallel"
- "RAG grounds every debate argument AND deep dive chat in real financial data"
- "2-layer cache: first run ~3 min, repeat runs instant"
- "Deep dive preloads in parallel with debate — zero wait time"
- "Screener tier is the single source of truth — no conflicting scores"
- "100% free data sources — no Bloomberg, no paid APIs"

---

## Key Numbers for Demo

- **500+** stocks screened from Nifty 500 universe
- **50+** screening criteria across 12 gates
- **10** specialized AI agents
- **5** USP dimensions per stock
- **3** parallel execution points (fan-out x2 + deep dive preload)
- **3** debate rounds (1 parallel + 1 sequential + judge)
- **3** question types with adaptive context (factual/analysis/thesis)
- **4** research insight cards from AI agents (DCF, Sentiment, ROCE, Peers)
- **0** paid data APIs (all free sources)
- **0** conflicting scores (screener tier = single verdict)
