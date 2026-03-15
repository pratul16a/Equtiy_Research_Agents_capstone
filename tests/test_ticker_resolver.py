"""Tests for ticker resolver utility."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.ticker_resolver import resolve_ticker


def test_already_has_ns_suffix():
    assert resolve_ticker("RELIANCE.NS") == "RELIANCE.NS"


def test_already_has_bo_suffix():
    assert resolve_ticker("RELIANCE.BO") == "RELIANCE.BO"


def test_case_insensitive():
    assert resolve_ticker("reliance.ns") == "RELIANCE.NS"


def test_alias_lookup():
    assert resolve_ticker("RELIANCE") == "RELIANCE.NS"
    assert resolve_ticker("INFOSYS") == "INFY.NS"
    assert resolve_ticker("INFY") == "INFY.NS"
    assert resolve_ticker("HUL") == "HINDUNILVR.NS"
    assert resolve_ticker("SBI") == "SBIN.NS"
    assert resolve_ticker("AIRTEL") == "BHARTIARTL.NS"


def test_alias_bse_exchange():
    assert resolve_ticker("RELIANCE", exchange="BSE") == "RELIANCE.BO"
    assert resolve_ticker("INFOSYS", exchange="BSE") == "INFY.BO"


def test_unknown_ticker_appends_suffix():
    assert resolve_ticker("UNKNOWN") == "UNKNOWN.NS"
    assert resolve_ticker("UNKNOWN", exchange="BSE") == "UNKNOWN.BO"


def test_whitespace_handling():
    assert resolve_ticker("  RELIANCE  ") == "RELIANCE.NS"
    assert resolve_ticker(" TCS ") == "TCS.NS"


if __name__ == "__main__":
    test_already_has_ns_suffix()
    test_already_has_bo_suffix()
    test_case_insensitive()
    test_alias_lookup()
    test_alias_bse_exchange()
    test_unknown_ticker_appends_suffix()
    test_whitespace_handling()
    print("All ticker resolver tests passed!")
