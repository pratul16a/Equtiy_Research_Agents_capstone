"""Ticker resolver — maps company names and bare symbols to valid yfinance Indian tickers."""

# Common Indian stock aliases → yfinance NSE tickers
TICKER_ALIASES: dict[str, str] = {
    # IT Services
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HCL": "HCLTECH.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "LTI": "LTIM.NS",
    "LTIM": "LTIM.NS",
    "LTIMINDTREE": "LTIM.NS",
    # Banking
    "HDFC BANK": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBI": "SBIN.NS",
    "SBIN": "SBIN.NS",
    "AXIS BANK": "AXISBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "INDUSIND": "INDUSINDBK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    # Energy & Conglomerates
    "RELIANCE": "RELIANCE.NS",
    "RIL": "RELIANCE.NS",
    "ONGC": "ONGC.NS",
    "IOC": "IOC.NS",
    "BPCL": "BPCL.NS",
    "GAIL": "GAIL.NS",
    "NTPC": "NTPC.NS",
    "POWERGRID": "POWERGRID.NS",
    "ADANI ENT": "ADANIENT.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    # FMCG
    "HUL": "HINDUNILVR.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "NESTLE": "NESTLEIND.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "DABUR": "DABUR.NS",
    "MARICO": "MARICO.NS",
    # Automobiles
    "MARUTI": "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "HERO": "HEROMOTOCO.NS",
    "EICHER": "EICHERMOT.NS",
    "EICHERMOT": "EICHERMOT.NS",
    # Pharma
    "SUNPHARMA": "SUNPHARMA.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "DRREDDY": "DRREDDY.NS",
    "DR REDDY": "DRREDDY.NS",
    "CIPLA": "CIPLA.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "BIOCON": "BIOCON.NS",
    # Metals
    "TATASTEEL": "TATASTEEL.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "HINDALCO": "HINDALCO.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "VEDL": "VEDL.NS",
    "VEDANTA": "VEDL.NS",
    "COALINDIA": "COALINDIA.NS",
    "COAL INDIA": "COALINDIA.NS",
    # Telecom
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    # Financial Services
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "SBILIFE": "SBILIFE.NS",
    # Infrastructure & Others
    "LT": "LT.NS",
    "L&T": "LT.NS",
    "LARSEN": "LT.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "TITAN": "TITAN.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "DMART": "DMART.NS",
    "AVENUE SUPERMARTS": "DMART.NS",
    "TATACONSUM": "TATACONSUM.NS",
    # EV / New Economy
    "OLA": "OLAELEC.NS",
    "OLA ELECTRIC": "OLAELEC.NS",
    "OLAELECTRIC": "OLAELEC.NS",
    "OLAELEC": "OLAELEC.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "NYKAA": "NYKAA.NS",
    "POLICYBZR": "POLICYBZR.NS",
    "POLICYBAZAAR": "POLICYBZR.NS",
    "TRENT": "TRENT.NS",
    "HAL": "HAL.NS",
    "IRFC": "IRFC.NS",
    "JIOFIN": "JIOFIN.NS",
    "JIO FINANCIAL": "JIOFIN.NS",
    "JIOFINANCIAL": "JIOFIN.NS",
}


def resolve_ticker(user_input: str, exchange: str = "NSE") -> str:
    """Resolve user input to a valid yfinance ticker for Indian stocks.

    Rules:
    1. If input already has .NS or .BO suffix, return as-is (uppercased).
    2. Check TICKER_ALIASES dictionary for common names.
    3. Otherwise, append .NS or .BO based on the exchange parameter.
    """
    cleaned = user_input.upper().strip()

    if cleaned.endswith((".NS", ".BO")):
        return cleaned

    if cleaned in TICKER_ALIASES:
        ticker = TICKER_ALIASES[cleaned]
        if exchange.upper() == "BSE":
            ticker = ticker.replace(".NS", ".BO")
        return ticker

    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return cleaned + suffix
