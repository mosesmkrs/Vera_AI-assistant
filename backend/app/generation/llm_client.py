import logging
from collections.abc import Callable

import httpx

from app.config import settings
from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.base import ScoredChunk

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the configured language model cannot return an answer."""


async def generate_answer(
    question: str,
    chunks: tuple[ScoredChunk, ...],
    *,
    client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
) -> str:
    if not settings.llm_api_key:
        raise LLMError("LLM_API_KEY is not configured")

    provider = settings.llm_provider.strip().casefold()
    headers, payload, url = _request_for_provider(provider, question, chunks)
    if provider == "anthropic":
        headers["x-api-key"] = settings.llm_api_key
    else:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    try:
        async with client_factory(timeout=45.0) as client:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        content = _extract_content(provider, body)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("LLM provider %s returned HTTP %s", provider, status_code)
        if status_code in (401, 403):
            message = f"{_provider_label(provider)} rejected the API credentials (HTTP {status_code})"
        elif status_code == 404:
            message = f"{_provider_label(provider)} could not find the configured model (HTTP 404)"
        elif status_code == 429:
            message = f"{_provider_label(provider)} rate limit or quota was reached (HTTP 429)"
        else:
            message = f"{_provider_label(provider)} returned HTTP {status_code}"
        raise LLMError(message) from exc
    except httpx.RequestError as exc:
        logger.warning("Could not connect to LLM provider %s (%s)", provider, type(exc).__name__)
        raise LLMError(
            f"Could not connect to {_provider_label(provider)}. Check network or proxy settings."
        ) from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("LLM provider %s returned an unreadable response", provider)
        raise LLMError(f"{_provider_label(provider)} returned an invalid response") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError("The language model returned an empty answer")
    return content.strip()


def _request_for_provider(
    provider: str,
    question: str,
    chunks: tuple[ScoredChunk, ...],
) -> tuple[dict[str, str], dict, str]:
    user_prompt = build_user_prompt(question, chunks)
    if provider == "openai":
        model = settings.llm_model
        request_body = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        if model == "gpt-6-luna":
            request_body["reasoning_effort"] = "none"
        return (
            {"Content-Type": "application/json"},
            request_body,
            "https://api.openai.com/v1/chat/completions",
        )
    if provider == "anthropic":
        return (
            {"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
            {
                "model": settings.llm_model,
                "max_tokens": 700,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            "https://api.anthropic.com/v1/messages",
        )
    raise LLMError("LLM_PROVIDER must be 'openai' or 'anthropic'")


def _extract_content(provider: str, body: dict) -> str:
    if provider == "openai":
        return body["choices"][0]["message"]["content"]
    return "\n".join(
        item["text"] for item in body["content"] if item.get("type") == "text"
    )


def _provider_label(provider: str) -> str:
    return {"openai": "OpenAI", "anthropic": "Anthropic"}.get(provider, "LLM provider")
