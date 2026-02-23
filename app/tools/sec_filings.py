"""SEC EDGAR filing tools — fetch 10-K, 10-Q summaries from SEC's free API."""

from __future__ import annotations

import json
import requests
from langchain_core.tools import tool


SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index?q={query}&dateRange=custom&startdt={start}&enddt={end}&forms={form}"

# SEC requires a User-Agent header
HEADERS = {
    "User-Agent": "EquityResearchAnalyst research@example.com",
    "Accept": "application/json",
}


def _get_cik(ticker: str) -> str | None:
    """Look up the CIK number for a ticker using SEC's company tickers JSON."""
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass
    return None


@tool
def get_sec_filings(ticker: str, form_type: str = "10-K", count: int = 3) -> str:
    """Fetch recent SEC filings metadata for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        form_type: Filing type — '10-K' (annual), '10-Q' (quarterly), '8-K' (events)
        count: Number of recent filings to retrieve (default 3)
    """
    try:
        cik = _get_cik(ticker)
        if not cik:
            return json.dumps({"error": f"Could not find CIK for ticker {ticker}"})

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        company_name = data.get("name", ticker)
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        filings = []
        for i in range(len(forms)):
            if forms[i] == form_type:
                accession_clean = accessions[i].replace("-", "")
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}"
                filings.append({
                    "form": forms[i],
                    "filing_date": dates[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": filing_url,
                    "accession_number": accessions[i],
                })
                if len(filings) >= count:
                    break

        result = {
            "company": company_name,
            "ticker": ticker.upper(),
            "cik": cik,
            "form_type": form_type,
            "filings_found": len(filings),
            "filings": filings,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"SEC filing lookup failed for {ticker}: {e}"})


@tool
def search_sec_full_text(query: str, ticker: str = "", forms: str = "10-K,10-Q") -> str:
    """Search SEC EDGAR full-text search for specific topics in filings.

    Args:
        query: Search terms (e.g., 'revenue growth risk factors')
        ticker: Optional ticker to narrow results
        forms: Comma-separated form types to search (default '10-K,10-Q')
    """
    try:
        search_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{ticker}" {query}' if ticker else query,
            "forms": forms,
            "dateRange": "custom",
            "startdt": "2024-01-01",
            "enddt": "2026-12-31",
        }
        resp = requests.get(search_url, params=params, headers=HEADERS, timeout=15)

        # If full-text search fails, return a helpful message
        if resp.status_code != 200:
            return json.dumps({
                "note": "SEC full-text search is rate-limited. Use get_sec_filings for direct filing access.",
                "suggestion": f"Try accessing filings directly for {ticker} using get_sec_filings tool.",
            })

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])[:5]

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "entity": source.get("entity_name", ""),
                "form": source.get("form_type", ""),
                "date": source.get("file_date", ""),
                "description": source.get("display_names", [""])[0] if source.get("display_names") else "",
            })

        return json.dumps({"query": query, "results_count": len(results), "results": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"SEC search failed: {e}"})


SEC_TOOLS = [get_sec_filings, search_sec_full_text]
