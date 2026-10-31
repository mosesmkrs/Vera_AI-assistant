import asyncio
import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from html import escape

from app.config import settings

logger = logging.getLogger(__name__)


class EmailError(RuntimeError):
    """Raised when a resolution notification cannot be delivered."""


async def send_resolution_email(recipient: str, question: str, answer: str) -> None:
    if (
        not settings.email_smtp_username
        or not settings.email_smtp_password
        or not settings.email_from
    ):
        raise EmailError(
            "EMAIL_SMTP_USERNAME, EMAIL_SMTP_PASSWORD, and EMAIL_FROM must be configured"
        )

    message = EmailMessage()
    message["From"] = formataddr((settings.email_from_name, settings.email_from))
    message["To"] = recipient
    message["Reply-To"] = settings.email_reply_to or settings.email_from
    message["Subject"] = "Your Vera question has been answered"
    message.set_content(
        "VERA | STUDENT KNOWLEDGE ASSISTANT\n\n"
        "Your question has been answered.\n\n"
        f"YOUR QUESTION\n{question}\n\n"
        f"RESPONSE\n{answer}\n\n"
        "This is an automated notification from Vera. This inbox is not monitored. "
        "Please do not reply to this email."
    )
    message.add_alternative(
        _render_branded_html(question, answer),
        subtype="html",
    )
    try:
        await asyncio.to_thread(
            _send_smtp_message,
            message,
            settings.email_smtp_host,
            settings.email_smtp_port,
            settings.email_smtp_username,
            settings.email_smtp_password,
            settings.email_smtp_use_ssl,
        )
    except smtplib.SMTPResponseException as exc:
        logger.warning("SMTP server rejected a resolution email with code %s", exc.smtp_code)
        raise EmailError(
            f"SMTP server rejected the notification (code {exc.smtp_code})"
        ) from exc
    except (smtplib.SMTPException, OSError, TimeoutError) as exc:
        logger.warning("SMTP request failed (%s)", type(exc).__name__)
        raise EmailError(f"SMTP request failed ({type(exc).__name__})") from exc


def _render_branded_html(question: str, answer: str) -> str:
    safe_question = escape(question).replace("\n", "<br>")
    safe_answer = escape(answer).replace("\n", "<br>")
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Your question has been answered</title>
  </head>
  <body style="margin:0;padding:0;background-color:#f2f5f9;font-family:Arial,Helvetica,sans-serif;color:#17243b;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f2f5f9;padding:32px 12px;">
      <tr><td align="center">
        <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="width:100%;max-width:600px;background-color:#ffffff;border-radius:16px;overflow:hidden;">
          <tr>
            <td style="padding:26px 32px;background-color:#12304a;color:#ffffff;">
              <table role="presentation" cellspacing="0" cellpadding="0"><tr>
                <td style="width:38px;height:38px;border-radius:11px;background-color:#43c6ac;color:#12304a;text-align:center;vertical-align:middle;font-size:22px;font-weight:700;">V</td>
                <td style="padding-left:12px;font-size:17px;font-weight:700;letter-spacing:2px;">VERA</td>
              </tr></table>
              <p style="margin:22px 0 0;color:#c8d8e5;font-size:13px;letter-spacing:1px;text-transform:uppercase;">Student Knowledge Assistant</p>
            </td>
          </tr>
          <tr><td style="padding:34px 32px 10px;">
            <h1 style="margin:0;color:#12304a;font-size:25px;line-height:1.3;">Your question has been answered</h1>
            <p style="margin:12px 0 0;color:#526277;font-size:15px;line-height:1.6;">A response from university staff is ready for you.</p>
          </td></tr>
          <tr><td style="padding:18px 32px 8px;">
            <p style="margin:0 0 8px;color:#526277;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">Your question</p>
            <div style="padding:15px 17px;border:1px solid #e1e8ef;border-radius:10px;background-color:#f8fafc;color:#34445a;font-size:15px;line-height:1.6;">{safe_question}</div>
          </td></tr>
          <tr><td style="padding:18px 32px 34px;">
            <p style="margin:0 0 8px;color:#087f75;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;">Staff response</p>
            <div style="padding:18px 20px;border-left:4px solid #43c6ac;border-radius:8px;background-color:#effaf8;color:#17243b;font-size:16px;line-height:1.7;">{safe_answer}</div>
          </td></tr>
          <tr><td style="padding:20px 32px;background-color:#f8fafc;border-top:1px solid #e7edf3;text-align:center;">
            <p style="margin:0;color:#68788b;font-size:12px;line-height:1.6;">This is an automated notification from Vera.<br>This inbox is not monitored. Please do not reply to this email.</p>
          </td></tr>
        </table>
        <p style="margin:16px 0 0;color:#91a0b0;font-size:11px;">Vera · Student Knowledge Assistant</p>
      </td></tr>
    </table>
  </body>
</html>"""


def _send_smtp_message(
    message: EmailMessage,
    host: str,
    port: int,
    username: str,
    password: str,
    use_ssl: bool,
) -> None:
    context = ssl.create_default_context()
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30, context=context) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls(context=context)
        smtp.ehlo()
        smtp.login(username, password)
        smtp.send_message(message)
