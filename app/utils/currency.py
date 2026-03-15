"""Indian currency formatting utilities — Crores, Lakhs, and INR notation."""


def format_inr(value: float) -> str:
    """Format a number in Indian Rupee notation.

    Examples:
        format_inr(1_50_00_000) → '₹1.50 Cr'
        format_inr(5_00_000)    → '₹5.00 L'
        format_inr(45000)       → '₹45,000.00'
    """
    if value is None:
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    elif abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} L"
    else:
        return f"{sign}₹{abs_val:,.2f}"


def format_market_cap_inr(value: float) -> str:
    """Format market cap in Lakh Crores or Crores.

    Examples:
        format_market_cap_inr(15_00_000_00_00_000) → '₹15.00 Lakh Cr'
        format_market_cap_inr(50_000_00_00_000)    → '₹50,000.00 Cr'
    """
    if value is None:
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}₹{abs_val / 1e12:,.2f} Lakh Cr"
    elif abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    elif abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} L"
    else:
        return f"{sign}₹{abs_val:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal or percentage value as a display string.

    If |value| < 1, assumes it's a ratio and multiplies by 100.
    """
    if value is None:
        return "N/A"
    if abs(value) < 1:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}%"
