import asyncio
import json

import httpx
import pytest
from app.config import settings
from app.generation.llm_client import LLMError, generate_answer
from app.retrieval.base import ScoredChunk


def test_openai_generation_uses_zero_temperature_and_returns_model_text(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-6-luna")

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer unit-test-key"
        assert body["model"] == "gpt-6-luna"
        assert body["temperature"] == 0
        assert body["reasoning_effort"] == "none"
        assert "only the supplied approved source excerpts" in body["messages"][0]["content"]
        assert "[S1]" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Submit the form [S1]."}}]},
        )

    transport = httpx.MockTransport(handle_request)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    answer = asyncio.run(generate_answer(_question(), (_chunk(),), client_factory=client_factory))
    assert answer == "Submit the form [S1]."


def test_openai_generation_explains_authentication_http_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "openai")

    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {}}))

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    with pytest.raises(LLMError, match="rejected the API credentials \\(HTTP 401\\)"):
        asyncio.run(generate_answer(_question(), (_chunk(),), client_factory=client_factory))


def test_openai_generation_explains_connection_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "openai")

    def fail_to_connect(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(fail_to_connect)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    with pytest.raises(LLMError, match="Could not connect to OpenAI"):
        asyncio.run(generate_answer(_question(), (_chunk(),), client_factory=client_factory))


def _question() -> str:
    return "What form should I submit?"


def _chunk() -> ScoredChunk:
    from datetime import date
    from uuid import uuid4

    return ScoredChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Test policy",
        authority_office="Registrar",
        source_type="manual",
        publication_date=None,
        effective_date=date(2025, 1, 1),
        expiry_date=None,
        content="Submit the test form.",
        score=1.0,
        rank=1,
        retrieval_method="hybrid",
    )
