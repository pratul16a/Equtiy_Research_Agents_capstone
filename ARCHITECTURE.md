# Indian Equity Analyst — Architecture Reference

## Folder Structure

```
indian_equity_analyst/
├── .env / .env.example              # API keys (OpenRouter, Google, etc.)
├── .streamlit/config.toml           # Streamlit dark theme config
├── requirements.txt                 # Dependencies
├── streamlit_app.py                 # Main UI (2,400+ lines)
├── data/
│   ├── faiss_index/                 # FAISS vector store for RAG
│   ├── sample_reports/              # Input docs for RAG ingestion
│   └── cache.db                     # SQLite disk cache
│
├── app/
│   ├── config.py                    # LLM factory, Indian market constants
│   ├── state.py                     # LangGraph state schemas
│   ├── graph.py                     # LangGraph workflow definitions
│   ├── scoring.py                   # 5-dimensional quantitative scoring
│   ├── deep_dive_chat.py            # Conversational stock chat
│   │
│   ├── prompts/
│   │   └── templates.py             # LLM prompt templates per agent
│   │
│   ├── agents/                      # 10 LangGraph agent nodes
│   │   ├── regime_agent.py          # Market regime (Bull/Bear/Rotation/Mixed)
│   │   ├── category_agents.py       # Momentum & Value screening nodes
│   │   ├── validation_agent.py      # USP scoring + contradiction detection
│   │   ├── debate_agents.py         # Bull/Bear/Judge multi-round debate
│   │   ├── debate_hybrid.py         # Quick debate (rule-based + LLM)
│   │   ├── data_agent.py            # Financial data fetcher
│   │   ├── analysis_agent.py        # Ratios, DCF, ROCE, peer comparison
│   │   ├── sentiment_agent.py       # News sentiment analyzer
│   │   └── report_agent.py          # Investment memo generator
│   │
│   ├── rag/
│   │   ├── ingest.py                # FAISS indexing (PDF + Screener.in)
│   │   ├── retriever.py             # Similarity search + symbol filtering
│   │   └── screener_scraper.py      # Screener.in web scraper
│   │
│   ├── tools/
│   │   ├── financials.py            # yfinance + BSE APIs
│   │   ├── filings.py               # BSE announcements, board meetings
│   │   ├── indian_metrics.py        # ROCE, FII/DII, promoter pledge
│   │   ├── market_breadth.py        # Nifty 500, sector RS, breadth signals
│   │   ├── news.py                  # Google News RSS
│   │   ├── peer_data.py             # Peer comparison
│   │   ├── stock_deep_dive.py       # Per-stock analysis
│   │   └── screener/                # 50+ screening criteria
│   │       ├── gate_defs.py         # Gate definitions + tier config
│   │       ├── category_screener.py # Orchestrator (17 + 28 criteria)
│   │       ├── screener_engine.py   # 4-phase pipeline
│   │       ├── batch_fundamentals.py# Bulk yfinance fetching
│   │       ├── technical_screens.py # 200DMA, RSI, MACD, OBV, volume
│   │       ├── quality_screens.py   # Debt, OPM, CF, Piotroski, Altman Z
│   │       ├── quantitative.py      # ROE, P/B, P/S, EV/EBITDA, FCF
│   │       ├── qualitative.py       # Insider, announcements, capacity
│   │       ├── ownership_screens.py # Promoter, institutional holdings
│   │       ├── geopolitical.py      # Supply chain, export/PLI
│   │       ├── smart_money.py       # Fundamental vs institutional lag
│   │       ├── regulatory.py        # Regulatory tailwind scoring
│   │       ├── mgmt_credibility.py  # Management track record
│   │       └── promoter_anomaly.py  # Suspicious promoter behavior
│   │
│   ├── ui/
│   │   └── usp_cards.py            # USP dimension card transforms
│   │
│   ├── utils/
│   │   ├── cache.py                # TTL cache + rate limiter + circuit breaker
│   │   ├── currency.py             # INR formatting (Cr/Lakh)
│   │   ├── fiscal_year.py          # Indian fiscal year (Apr-Mar)
│   │   └── ticker_resolver.py      # NSE/BSE suffix normalization
│   │
│   └── sharing/
│       ├── pdf_generator.py        # Deep dive PDF
│       ├── screener_pdf.py         # Screener results PDF
│       ├── email_sender.py         # Email sharing
│       └── whatsapp_share.py       # WhatsApp sharing
```

---

## Two LangGraph Pipelines

### Pipeline 1: Screener (multi-agent fan-out/fan-in)

```
load_universe
       │
    regime ──────── detect Bull/Bear/Rotation/Mixed, set weights
       │
   ┌───┴───┐
momentum  value_bottom ── fan-out: parallel 2-category screening
   └───┬───┘
   merge_categories ───── fan-in: deduplicate tickers
       │
   ┌───┴───┐
validation  rag_ingest ── parallel: USP scoring + FAISS indexing
   └───┬───┘
     debate ─────────── multi-round Bull/Bear/Judge for top 3
       │
      END
```

### Pipeline 2: Deep Dive (sequential per-stock)

```
data_agent ──→ analysis_agent ──→ sentiment_agent ──→ report_agent ──→ END
(fetch data)   (ratios, DCF)     (news sentiment)    (investment memo)
```

---

## Agent Details

### Screener Pipeline Agents

| Agent | Role | Key Logic |
|-------|------|-----------|
| **Regime Agent** | Classify market as Bull/Bear/Rotation/Mixed | Rule-based: VIX levels + Nifty trends. Sets category weight multipliers (Bull: Mom 1.3x/Val 0.7x) |
| **Momentum Node** | Screen trending stocks in strong sectors | 17 criteria across 5 gates. Gate-based pass/fail + score. Applies regime weight |
| **Value Node** | Screen cheap stocks with turnaround signals | 28+ criteria across 7 gates. Bonus gate for macro tailwinds. Applies regime weight |
| **Merge Node** | Combine & deduplicate results | Outputs `recommended_tickers` + `active_categories` |
| **Validation Agent** | Score 5 USP dimensions + detect contradictions | Deterministic scoring (geo, smart money, regulatory, mgmt, promoter). LLM explains top 10 |
| **Debate Agents** | Multi-round Bull/Bear/Judge | Round 1: parallel. Round 2+: sequential rebuttals. Judge assigns conviction 1-10 + BUY/HOLD/SELL |

### Deep Dive Pipeline Agents

| Agent | Tools Used | Output |
|-------|-----------|--------|
| **Data Agent** | yfinance, BSE APIs, filings | financials, company_info, shareholding, corporate_actions |
| **Analysis Agent** | compute_ratios, DCF, ROCE, peer_comparison | ratios, indian_metrics, dcf_valuation, peer_comparison |
| **Sentiment Agent** | Google News RSS, yfinance news | sentiment_scores (-1 to +1), news_summaries, management_commentary |
| **Report Agent** | 5-dim scoring + RAG retrieval + LLM | investment_score, rag_context, final_report (HTML memo) |

---

## How Agents Communicate

Agents communicate via **LangGraph shared state** (TypedDict), not direct calls:

1. Each agent is a **node** in the graph that reads from and writes to the shared state
2. **Fan-out**: Momentum and Value nodes run in parallel, both append to `category_results` (using `Annotated[list, operator.add]`)
3. **Fan-in**: Merge node reads `category_results` from both, outputs `recommended_tickers`
4. **Handoff**: Validation agent reads `recommended_tickers`, writes `usp_scores` → Debate agents read both
5. **Deep dive**: Sequential — each agent reads previous agent's output fields from state

```
ScreenerState (shared dict):
  regime_agent writes → regime{}, weights{}
  momentum/value write → category_results[]  (accumulates via operator.add)
  merge reads both → recommended_tickers[]
  validation reads tickers → usp_scores{}, contradictions[]
  debate reads usp + categories → debate_results[]
```

---

## RAG System

### Ingestion
1. **Sources**: PDF/TXT files in `data/sample_reports/` + Screener.in web scraper (quarterly results, ratios, pros/cons)
2. **Processing**: RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
3. **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (local) or OpenAI/Google
4. **Storage**: FAISS index at `data/faiss_index/`
5. **Metadata**: Each chunk has `symbol`, `section`, `source` for filtering

### Retrieval
- **General search**: `retriever.search(query, k=5)` — similarity search across all chunks
- **Symbol-filtered**: `retriever.search_by_symbol("RELIANCE", query)` — fetches k*3, post-filters by symbol metadata
- **Used by**: Bull/Bear debate agents (stock-specific evidence), Report agent (historical analyst frameworks)

### When RAG runs
- **Screener pipeline**: `rag_ingest` node runs in parallel with `validation` after merge. Scrapes Screener.in for top 15 tickers, indexes into FAISS
- **Deep dive**: Report agent calls retriever as a tool during memo generation
- **Fallback**: If FAISS index doesn't exist, RAG returns None (system works without it)

---

## Gate-Based Screening

### Gate Types
- **Hard**: Stock MUST pass or it's eliminated
- **Soft**: Should pass (lower bar), failure doesn't block if hard gates pass
- **Bonus**: Never blocks, only adds to score

### Momentum (5 gates, 17 criteria)

| Gate | Type | Min Pass | Criteria |
|------|------|----------|----------|
| Sector Tide | HARD | 2/3 | sector_rs_positive, sector_highs_gt_lows, sector_top4 |
| Trend Structure | HARD | 3/5 | above_200dma*, golden_alignment, rsi_55_75, near_52w_high, macd_bullish |
| Volume Conviction | SOFT | 1/2 | volume_breakout, obv_rising |
| Smart Money Flow | SOFT | 1/2 | promoter_holding, institutional_interest |
| Fundamental Backbone | SOFT | 1/3 | opm_improvement, order_booking, roe_above_12 |

*above_200dma is mandatory in Trend Structure

### Value Bottom (7 gates, 33 criteria)

| Gate | Type | Min Pass | Criteria |
|------|------|----------|----------|
| Valuation Floor | HARD | 2/5 | low_roe, pb_below_avg, ps_below_avg, ev_ebitda_below_median, fcf_yield_above_sector |
| Turnaround Signals | HARD | 2/5 | debt_reduction, opm_improvement, cf_turnaround, dividend_initiation, working_capital |
| Trap Avoidance | HARD | 2/3 | piotroski, altman_z, no_sebi_red_flags |
| Technical Timing | SOFT | 1/3 | rsi_oversold, above_200dma, sector_rotation |
| Re-rating Catalysts | SOFT | 2/5 | capacity_util, capex, new_business, order_booking, capex_going_live |
| Insider Conviction | SOFT | 2/4 | insider_buying, promoter_holding, mgmt_credibility, promoter_behavior |
| Macro Tailwinds | BONUS | 0 | geopolitical, smart_money, regulatory, supply_chain, export_pli |

---

## 4-Tier Conviction System

Results are classified into tiers using **absolute score thresholds**:

| Tier | Rule | Momentum Threshold |
|------|------|--------------------|
| **★ Buy Zone** | All gates pass + score >= buy_zone | >= 10 |
| **◉ Watchlist** | All gates pass + score >= watchlist | >= 9 |
| **◎ Monitor** | All gates pass + score >= monitor | >= 8 |
| **⊘ Near Miss** | 0 hard failures + exactly 1 soft failure + score >= near_miss | >= 9 |
| **✗ Failed** | Everything else | — |

---

## 5-Dimensional Scoring (Deep Dive)

| Dimension | Weight | Sub-metrics |
|-----------|--------|-------------|
| Valuation | 25 pts | DCF upside (40%), P/E vs peers (35%), P/B ratio (25%) |
| Quality | 25 pts | ROCE (35%), profit margin (25%), D/E (20%), ROE (20%) |
| Growth | 20 pts | Revenue growth (45%), earnings growth (45%), peer-relative (10%) |
| Sentiment | 15 pts | News sentiment (60%), institutional trend (40%) |
| Governance | 15 pts | Promoter holding (50%), pledge % (30%), institutional trust (20%) |

**Recommendation**: >= 75 STRONG BUY · 60-74 BUY · 40-59 HOLD · 25-39 SELL · <25 STRONG SELL

---

## Caching & Performance

| Layer | TTL | Purpose |
|-------|-----|---------|
| In-memory dict | 5 min | Hot path: repeated UI renders |
| SQLite disk cache | 4 hours | Survives restarts: yfinance .info, financials |
| Bulk cache | 4 hours | Full ticker set cached by MD5 hash |
| Rate limiter | 10 calls/sec | yfinance API throttling |
| Circuit breaker | 3 failures → 5 min cooldown | BSE API resilience |
| Retry decorator | Exponential backoff (1s → 30s) | Transient error recovery |

---

## UI Pages (Streamlit)

1. **Market Breadth** — Nifty 500 sector RS heatmap, 52W high/low ratio, breadth signal
2. **Momentum Screener** — Run 17-criteria gate-based screen, 4-tier results, debate
3. **Value Screener** — Run 28-criteria gate-based screen, 4-tier results, debate
4. **Stock Diagnostic** — Single stock: run ALL criteria, show gate-by-gate pass/fail
5. **Stock Deep Dive** — Full LangGraph pipeline: data → analysis → sentiment → report
