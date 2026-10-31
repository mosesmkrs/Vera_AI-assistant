import asyncio
import json

import httpx
import pytest

from app.config import settings
from app.generation.llm_client import LLMError, generate_answer, verify_answer_support
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


def test_answer_support_verifier_checks_claim_citations(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "llm_model", "gpt-6-luna")

    def handle_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "independent evidence verifier" in body["messages"][0]["content"]
        assert "Submit the test form [S1]." in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "fully_answered": True,
                                    "support_score": 0.91,
                                    "claims": [
                                        {
                                            "text": "Submit the test form",
                                            "supported": True,
                                            "source_ids": ["S1"],
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handle_request)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    result = asyncio.run(
        verify_answer_support(
            _question(), "Submit the test form [S1].", (_chunk(),), client_factory=client_factory
        )
    )
    assert result.supported is True
    assert result.support_score == 0.91


def test_answer_support_verifier_rejects_unknown_citation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_api_key", "unit-test-key")
    monkeypatch.setattr(settings, "llm_provider", "openai")

    content = json.dumps(
        {
            "fully_answered": True,
            "support_score": 0.99,
            "claims": [
                {"text": "Claim", "supported": True, "source_ids": ["S2"]}
            ],
        }
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )
    )

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=transport, **kwargs)

    result = asyncio.run(
        verify_answer_support(
            _question(),
            "Claim [S2].",
            (_chunk(),),
            client_factory=client_factory,
        )
    )
    assert result.supported is False
    assert result.support_score == 0.0


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
