from html import escape

import httpx

from app.config import settings


class EmailError(RuntimeError):
    """Raised when a resolution notification cannot be delivered."""


async def send_resolution_email(recipient: str, question: str, answer: str) -> None:
    if not settings.email_api_key or not settings.email_from:
        raise EmailError("EMAIL_API_KEY and EMAIL_FROM must be configured")

    payload = {
        "from": settings.email_from,
        "to": [recipient],
        "subject": "Your Vera question has been answered",
        "text": (
            "Your question has been answered by university staff.\n\n"
            f"Question: {question}\n\nAnswer: {answer}"
        ),
        "html": (
            "<p>Your question has been answered by university staff.</p>"
            f"<p><strong>Question:</strong> {escape(question)}</p>"
            f"<p><strong>Answer:</strong> {escape(answer)}</p>"
        ),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {settings.email_api_key}"},
                json=payload,
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise EmailError("The email provider could not deliver the notification") from exc
