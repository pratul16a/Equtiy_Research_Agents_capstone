"""Indian fiscal year utilities — April to March convention."""

from datetime import date, datetime


def get_indian_fy(dt: date | datetime | None = None) -> str:
    """Get the Indian fiscal year label.

    Indian FY runs April to March. FY25 = April 2024 - March 2025.

    Examples:
        get_indian_fy(date(2025, 1, 15)) → 'FY25'  (Jan 2025 is in FY25)
        get_indian_fy(date(2025, 5, 1))  → 'FY26'  (May 2025 is in FY26)
    """
    if dt is None:
        dt = date.today()
    if isinstance(dt, datetime):
        dt = dt.date()

    if dt.month >= 4:
        fy_end_year = dt.year + 1
    else:
        fy_end_year = dt.year

    return f"FY{fy_end_year % 100:02d}"


def get_fy_quarter(dt: date | datetime | None = None) -> str:
    """Get Indian FY quarter label.

    Q1: Apr-Jun, Q2: Jul-Sep, Q3: Oct-Dec, Q4: Jan-Mar.

    Examples:
        get_fy_quarter(date(2025, 5, 1))  → 'Q1 FY26'
        get_fy_quarter(date(2025, 11, 1)) → 'Q3 FY26'
        get_fy_quarter(date(2025, 2, 1))  → 'Q4 FY25'
    """
    if dt is None:
        dt = date.today()
    if isinstance(dt, datetime):
        dt = dt.date()

    month = dt.month
    if 4 <= month <= 6:
        quarter = "Q1"
    elif 7 <= month <= 9:
        quarter = "Q2"
    elif 10 <= month <= 12:
        quarter = "Q3"
    else:
        quarter = "Q4"

    fy = get_indian_fy(dt)
    return f"{quarter} {fy}"


def get_fy_date_range(fy_label: str) -> tuple[date, date]:
    """Convert a FY label to (start_date, end_date).

    Example:
        get_fy_date_range('FY25') → (date(2024, 4, 1), date(2025, 3, 31))
    """
    fy_num = int(fy_label.replace("FY", ""))
    if fy_num < 100:
        end_year = 2000 + fy_num
    else:
        end_year = fy_num

    start = date(end_year - 1, 4, 1)
    end = date(end_year, 3, 31)
    return start, end
