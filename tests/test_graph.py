"""Tests for graph pipeline — requires LLM API key and network access."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_state_creation():
    """Test that initial state is created correctly."""
    from app.state import create_initial_state

    state = create_initial_state("RELIANCE")
    assert state["ticker"] == "RELIANCE.NS"
    assert state["exchange"] == "NSE"
    assert state["financials"] == {}
    assert state["errors"] == []
    print("  State creation: PASSED")

    state_bse = create_initial_state("RELIANCE", exchange="BSE")
    assert state_bse["ticker"] == "RELIANCE.BO"
    assert state_bse["exchange"] == "BSE"
    print("  BSE state creation: PASSED")

    state_suffix = create_initial_state("TCS.NS")
    assert state_suffix["ticker"] == "TCS.NS"
    print("  Suffix preservation: PASSED")


def test_graph_builds():
    """Test that the LangGraph compiles without errors."""
    from app.graph import build_graph

    graph = build_graph()
    assert graph is not None
    print("  Graph compilation: PASSED")


def test_fiscal_year_utils():
    """Test Indian fiscal year utilities."""
    from datetime import date
    from app.utils.fiscal_year import get_indian_fy, get_fy_quarter, get_fy_date_range

    assert get_indian_fy(date(2025, 1, 15)) == "FY25"
    assert get_indian_fy(date(2025, 5, 1)) == "FY26"
    assert get_indian_fy(date(2024, 3, 31)) == "FY24"
    assert get_indian_fy(date(2024, 4, 1)) == "FY25"
    print("  Indian FY: PASSED")

    assert get_fy_quarter(date(2025, 5, 1)) == "Q1 FY26"
    assert get_fy_quarter(date(2025, 8, 1)) == "Q2 FY26"
    assert get_fy_quarter(date(2025, 11, 1)) == "Q3 FY26"
    assert get_fy_quarter(date(2025, 2, 1)) == "Q4 FY25"
    print("  FY quarters: PASSED")

    start, end = get_fy_date_range("FY25")
    assert start == date(2024, 4, 1)
    assert end == date(2025, 3, 31)
    print("  FY date range: PASSED")


def test_currency_formatting():
    """Test INR currency formatting."""
    from app.utils.currency import format_inr, format_market_cap_inr

    assert "Cr" in format_inr(1_50_00_000)
    assert "L" in format_inr(5_00_000)
    assert "₹" in format_inr(45000)
    print("  INR formatting: PASSED")

    assert "Lakh Cr" in format_market_cap_inr(15_00_000_00_00_000)
    assert "Cr" in format_market_cap_inr(50_000_00_00_000)
    print("  Market cap formatting: PASSED")


if __name__ == "__main__":
    tests = [
        ("state_creation", test_state_creation),
        ("graph_builds", test_graph_builds),
        ("fiscal_year_utils", test_fiscal_year_utils),
        ("currency_formatting", test_currency_formatting),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            print(f"Testing {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
