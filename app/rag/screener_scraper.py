"""Screener.in Scraper — fetches company financial data for RAG ingestion.

Scrapes the public company page and extracts structured sections:
- Quarterly Results (13 quarters)
- Annual P&L (12 years)
- Balance Sheet
- Cash Flows
- Ratios (ROCE, working capital, etc.)
- Shareholding Pattern
- Key Metrics
- Pros/Cons

Uses requests + BeautifulSoup for HTML parsing.
Includes disk caching to avoid re-scraping.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "screener_cache")
_BASE_URL = "https://www.screener.in/company/{symbol}/consolidated/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-IN,en;q=0.9",
}
_REQUEST_DELAY = 2.0  # seconds between requests (respect rate limits)


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str) -> str:
    return os.path.join(_CACHE_DIR, f"{symbol.upper()}.json")


def _load_from_cache(symbol: str, max_age_hours: int = 24) -> dict | None:
    """Load cached data if it exists and is fresh enough."""
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None

    try:
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours > max_age_hours:
            return None

        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_to_cache(symbol: str, data: dict) -> None:
    """Save scraped data to disk cache."""
    _ensure_cache_dir()
    path = _cache_path(symbol)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to cache %s: %s", symbol, e)


def _parse_table(table_element) -> list[list[str]]:
    """Parse an HTML table into a list of rows (list of cell strings)."""
    rows = []
    for tr in table_element.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            cells.append(td.get_text(strip=True))
        if cells:
            rows.append(cells)
    return rows


def _table_to_text(section_name: str, rows: list[list[str]]) -> str:
    """Convert parsed table rows to readable text for RAG chunking."""
    if not rows:
        return ""

    lines = [f"## {section_name}"]
    # First row is header
    if rows:
        lines.append(" | ".join(rows[0]))
        lines.append("-" * 60)
    for row in rows[1:]:
        lines.append(" | ".join(row))

    return "\n".join(lines)


def scrape_company_page(symbol: str, use_cache: bool = True) -> dict[str, Any]:
    """Scrape Screener.in company page and extract all data sections.

    Args:
        symbol: Stock symbol without exchange suffix (e.g., "RELIANCE", "ZOMATO")
        use_cache: Whether to use disk cache (default True)

    Returns:
        Dict with keys: quarterly_results, annual_pl, balance_sheet, cash_flows,
        ratios, shareholding, key_metrics, pros_cons, about, raw_text
    """
    symbol = symbol.upper().replace(".NS", "").replace(".BO", "")

    # Check cache
    if use_cache:
        cached = _load_from_cache(symbol)
        if cached:
            logger.info("Screener cache hit for %s", symbol)
            return cached

    # Fetch page
    url = _BASE_URL.format(symbol=symbol)
    logger.info("Scraping Screener.in for %s: %s", symbol, url)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to fetch Screener.in page for %s: %s", symbol, e)
        return {"error": str(e), "symbol": symbol}

    soup = BeautifulSoup(resp.text, "html.parser")

    sections: dict[str, Any] = {"symbol": symbol, "url": url}

    # Extract each data section
    section_cards = soup.find_all("section", class_="card")

    section_names = [
        "Quarterly Results", "Profit & Loss", "Balance Sheet",
        "Cash Flows", "Ratios", "Shareholding Pattern",
    ]
    section_keys = [
        "quarterly_results", "annual_pl", "balance_sheet",
        "cash_flows", "ratios", "shareholding",
    ]

    for card in section_cards:
        header = card.find(["h2", "h3"])
        if not header:
            continue
        header_text = header.get_text(strip=True)

        for name, key in zip(section_names, section_keys):
            if name.lower() in header_text.lower():
                table = card.find("table")
                if table:
                    rows = _parse_table(table)
                    sections[key] = {
                        "header": header_text,
                        "rows": rows,
                        "text": _table_to_text(header_text, rows),
                    }
                break

    # Extract key metrics (top summary box)
    ratios_list = soup.find("ul", id="top-ratios")
    if ratios_list:
        metrics = {}
        for li in ratios_list.find_all("li"):
            name_el = li.find("span", class_="name")
            value_el = li.find("span", class_="number")
            if name_el and value_el:
                metrics[name_el.get_text(strip=True)] = value_el.get_text(strip=True)
        sections["key_metrics"] = metrics

    # Extract Pros/Cons
    pros_cons = {"pros": [], "cons": []}
    for div in soup.find_all("div", class_="pros-cons"):
        for li in div.find_all("li"):
            text = li.get_text(strip=True)
            # Determine if pro or con based on parent class
            parent_class = div.get("class", [])
            if "pros" in str(parent_class):
                pros_cons["pros"].append(text)
            else:
                pros_cons["cons"].append(text)

    # Alternative: look for specific sections
    for section in soup.find_all("div"):
        section_class = " ".join(section.get("class", []))
        if "pros" in section_class:
            for li in section.find_all("li"):
                text = li.get_text(strip=True)
                if text and text not in pros_cons["pros"]:
                    pros_cons["pros"].append(text)
        elif "cons" in section_class:
            for li in section.find_all("li"):
                text = li.get_text(strip=True)
                if text and text not in pros_cons["cons"]:
                    pros_cons["cons"].append(text)

    sections["pros_cons"] = pros_cons

    # Extract about section
    about_section = soup.find("div", class_="about")
    if about_section:
        sections["about"] = about_section.get_text(strip=True)

    # Build full raw text for RAG (all sections concatenated)
    text_parts = [f"# {symbol} — Screener.in Data\n"]
    for key in section_keys:
        if key in sections and isinstance(sections[key], dict) and "text" in sections[key]:
            text_parts.append(sections[key]["text"])

    if sections.get("key_metrics"):
        text_parts.append("\n## Key Metrics")
        for k, v in sections["key_metrics"].items():
            text_parts.append(f"- {k}: {v}")

    if pros_cons["pros"] or pros_cons["cons"]:
        text_parts.append("\n## Pros")
        for p in pros_cons["pros"]:
            text_parts.append(f"- {p}")
        text_parts.append("\n## Cons")
        for c in pros_cons["cons"]:
            text_parts.append(f"- {c}")

    sections["raw_text"] = "\n\n".join(text_parts)

    # Cache result
    if use_cache:
        _save_to_cache(symbol, sections)

    return sections


def scrape_multiple(symbols: list[str], delay: float = _REQUEST_DELAY) -> dict[str, dict]:
    """Scrape Screener.in for multiple symbols with rate limiting.

    Args:
        symbols: List of stock symbols (e.g., ["RELIANCE", "ZOMATO"])
        delay: Seconds between requests.

    Returns:
        {symbol: scraped_data}
    """
    results = {}
    for i, symbol in enumerate(symbols):
        try:
            data = scrape_company_page(symbol)
            if "error" not in data:
                results[symbol] = data
                logger.info("Scraped %d/%d: %s", i + 1, len(symbols), symbol)
            else:
                logger.warning("Scrape failed for %s: %s", symbol, data.get("error"))
        except Exception as e:
            logger.warning("Scrape error for %s: %s", symbol, e)

        # Rate limiting
        if i < len(symbols) - 1:
            time.sleep(delay)

    logger.info("Scraped %d/%d symbols successfully", len(results), len(symbols))
    return results
