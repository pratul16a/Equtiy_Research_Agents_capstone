"""WhatsApp Sharing — sends reports via Twilio WhatsApp API.

Requires a Twilio account (free sandbox available for testing).
Set these in .env:
    TWILIO_ACCOUNT_SID   — from Twilio console
    TWILIO_AUTH_TOKEN     — from Twilio console
    TWILIO_WHATSAPP_FROM  — sandbox number, e.g. whatsapp:+14155238886
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def is_whatsapp_configured() -> bool:
    """Check whether Twilio WhatsApp credentials are set."""
    return bool(
        os.getenv("TWILIO_ACCOUNT_SID")
        and os.getenv("TWILIO_AUTH_TOKEN")
        and os.getenv("TWILIO_WHATSAPP_FROM")
    )


def _build_report_message(
    ticker: str,
    recommendation: str = "",
    composite_score: int = 0,
    summary_lines: list[str] | None = None,
) -> str:
    """Build a WhatsApp message for a stock report."""
    lines = [
        f"*Indian Equity Research: {ticker}*",
    ]
    if recommendation:
        lines.append(f"Recommendation: *{recommendation.upper()}*")
    if composite_score:
        lines.append(f"Score: *{composite_score}/100*")
    if summary_lines:
        lines.append("")
        lines.extend(summary_lines)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%d %b %Y')}")
    lines.append("_AI Indian Equity Research Analyst_")
    return "\n".join(lines)


def _build_screener_message(
    total_stocks: int,
    regime: str = "",
    top_picks: list[str] | None = None,
) -> str:
    """Build a WhatsApp message for screener results."""
    lines = [
        "*Indian Equity Screener Results*",
        f"Stocks Found: *{total_stocks}*",
    ]
    if regime:
        lines.append(f"Market Regime: *{regime.upper()}*")
    if top_picks:
        lines.append("")
        lines.append("Top Picks:")
        for t in top_picks[:5]:
            lines.append(f"  - {t}")
    lines.append("")
    lines.append(f"Screened: {datetime.now().strftime('%d %b %Y')}")
    lines.append("_AI Indian Equity Research Analyst_")
    return "\n".join(lines)


def send_whatsapp_report(
    phone_number: str,
    ticker: str,
    recommendation: str = "",
    composite_score: int = 0,
    summary_lines: list[str] | None = None,
    pdf_bytes: bytes | None = None,
) -> tuple[bool, str]:
    """Send a stock report via WhatsApp using Twilio.

    Args:
        phone_number: Recipient phone with country code (e.g. '919876543210').
        ticker: Stock ticker.
        recommendation: BUY/HOLD/SELL.
        composite_score: Score 0-100.
        summary_lines: Extra message lines.
        pdf_bytes: Optional PDF attachment (Twilio sandbox doesn't support media).

    Returns:
        (success, message) tuple.
    """
    if not is_whatsapp_configured():
        return False, "Twilio WhatsApp not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM in .env"

    if not phone_number or len(phone_number) < 10:
        return False, "Invalid phone number. Use format: 919876543210 (country code + number)"

    # Normalize phone
    phone_number = phone_number.strip().replace("+", "").replace(" ", "").replace("-", "")

    body = _build_report_message(ticker, recommendation, composite_score, summary_lines)

    return _send_twilio_message(phone_number, body)


def send_whatsapp_screener(
    phone_number: str,
    total_stocks: int,
    regime: str = "",
    top_picks: list[str] | None = None,
) -> tuple[bool, str]:
    """Send screener results via WhatsApp using Twilio.

    Args:
        phone_number: Recipient phone (e.g. '919876543210').
        total_stocks: Number of stocks found.
        regime: Market regime string.
        top_picks: Top ticker names.

    Returns:
        (success, message) tuple.
    """
    if not is_whatsapp_configured():
        return False, "Twilio WhatsApp not configured."

    if not phone_number or len(phone_number) < 10:
        return False, "Invalid phone number."

    phone_number = phone_number.strip().replace("+", "").replace(" ", "").replace("-", "")
    body = _build_screener_message(total_stocks, regime, top_picks)

    return _send_twilio_message(phone_number, body)


def _send_twilio_message(phone_number: str, body: str) -> tuple[bool, str]:
    """Send a WhatsApp message via Twilio API.

    Returns:
        (success, message) tuple.
    """
    try:
        from twilio.rest import Client

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_FROM")

        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body=body,
            from_=from_number,
            to=f"whatsapp:+{phone_number}",
        )

        logger.info("WhatsApp sent to %s: SID=%s", phone_number, message.sid)
        return True, f"Sent to +{phone_number} (SID: {message.sid})"

    except Exception as e:
        logger.warning("WhatsApp send failed: %s", e)
        error_msg = str(e)
        if "unverified" in error_msg.lower():
            return False, f"Phone +{phone_number} not verified in Twilio sandbox. Send 'join <sandbox-code>' to the Twilio number first."
        return False, f"WhatsApp failed: {e}"
