"""Email Sender — sends PDF report as email attachment via SMTP.

Reads credentials from environment variables:
    SMTP_EMAIL    — sender Gmail address
    SMTP_PASSWORD — Gmail App Password (NOT regular password)
    SMTP_HOST     — defaults to smtp.gmail.com
    SMTP_PORT     — defaults to 587
"""

from __future__ import annotations

import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def is_email_configured() -> bool:
    """Check whether SMTP credentials are set in environment."""
    return bool(os.getenv("SMTP_EMAIL")) and bool(os.getenv("SMTP_PASSWORD"))


def send_email_with_pdf(
    recipient_email: str,
    subject: str,
    body_text: str,
    pdf_bytes: bytes,
    pdf_filename: str,
) -> tuple[bool, str]:
    """Send an email with a PDF attachment via SMTP.

    Returns:
        (success, message) tuple for UI feedback.
    """
    sender = os.getenv("SMTP_EMAIL")
    password = os.getenv("SMTP_PASSWORD")
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))

    if not sender or not password:
        return False, "SMTP not configured. Set SMTP_EMAIL and SMTP_PASSWORD in .env"

    if not recipient_email or "@" not in recipient_email:
        return False, "Invalid recipient email address."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient_email
        msg["Subject"] = subject

        # Body
        msg.attach(MIMEText(body_text, "plain"))

        # PDF attachment
        part = MIMEBase("application", "pdf")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{pdf_filename}"')
        msg.attach(part)

        # Send
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        logger.info("Email sent to %s: %s", recipient_email, subject)
        return True, f"Report sent to {recipient_email}"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check SMTP_EMAIL and SMTP_PASSWORD (use Gmail App Password)."
    except smtplib.SMTPException as e:
        logger.warning("SMTP error: %s", e)
        return False, f"Email failed: {e}"
    except Exception as e:
        logger.error("Email send error: %s", e)
        return False, f"Email failed: {e}"
