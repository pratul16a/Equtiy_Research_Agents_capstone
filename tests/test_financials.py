"""Tests for financial tools — requires network access to yfinance."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.financials import (
    get_company_info,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_stock_price_history,
    compute_financial_ratios,
    compute_dcf_valuation,
)

TEST_TICKER = "RELIANCE.NS"


def test_get_company_info():
    result = json.loads(get_company_info.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert result["name"] != "N/A"
    assert result["market_cap"] > 0
    assert result["currency"] == "INR"
    print(f"  Company: {result['name']}, Market Cap: {result['market_cap_crores']} Cr")


def test_get_income_statement():
    result = json.loads(get_income_statement.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert len(result) > 0
    print(f"  Income statement fields: {len(result)}")


def test_get_balance_sheet():
    result = json.loads(get_balance_sheet.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert len(result) > 0
    print(f"  Balance sheet fields: {len(result)}")


def test_get_cash_flow():
    result = json.loads(get_cash_flow.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert len(result) > 0
    print(f"  Cash flow fields: {len(result)}")


def test_get_stock_price_history():
    result = json.loads(get_stock_price_history.invoke({"ticker": TEST_TICKER, "period": "3mo"}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert result["data_points"] > 0
    assert result["currency"] == "INR"
    print(f"  Price points: {result['data_points']}, Latest: INR {result['latest_close']}")


def test_compute_financial_ratios():
    result = json.loads(compute_financial_ratios.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert "pe_trailing" in result or "pb_ratio" in result
    print(f"  Ratios computed: {list(result.keys())}")


def test_compute_dcf_valuation():
    result = json.loads(compute_dcf_valuation.invoke({"ticker": TEST_TICKER}))
    assert "error" not in result, f"Error: {result.get('error')}"
    assert result["currency"] == "INR"
    assert "intrinsic_value_per_share" in result
    print(f"  DCF intrinsic value: INR {result['intrinsic_value_per_share']}, Upside: {result['upside_pct']}%")


if __name__ == "__main__":
    tests = [
        ("get_company_info", test_get_company_info),
        ("get_income_statement", test_get_income_statement),
        ("get_balance_sheet", test_get_balance_sheet),
        ("get_cash_flow", test_get_cash_flow),
        ("get_stock_price_history", test_get_stock_price_history),
        ("compute_financial_ratios", test_compute_financial_ratios),
        ("compute_dcf_valuation", test_compute_dcf_valuation),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            print(f"Testing {name}...")
            test_func()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
