import asyncio
from datetime import date
from uuid import uuid4

from app.retrieval.base import ScoredChunk
from app.services.answer_pipeline import answer_question
from app.validation.evidence import EvidenceAssessment


def test_answer_pipeline_returns_grounded_citation_and_logs_answer() -> None:
    asyncio.run(_exercise_sufficient_evidence())


async def _exercise_sufficient_evidence() -> None:
    session = _FakeSession()
    chunk = _chunk()
    generated: list[tuple[str, tuple[ScoredChunk, ...]]] = []

    async def retriever(_session, question, top_k):
        assert question == "How do I request a registration extension?"
        assert top_k == 5
        return [chunk]

    async def validator(_session, candidates, *, question):
        assert question == "How do I request a registration extension?"
        assert candidates == [chunk]
        return EvidenceAssessment("sufficient", None, 0.87, (chunk,))

    async def generator(question, evidence):
        generated.append((question, evidence))
        return "Submit the extension form to the Registrar [S1]."

    result = await answer_question(
        session,
        "  How do I request a registration extension?  ",
        "student@example.test",
        retriever=retriever,
        validator=validator,
        generator=generator,
    )

    assert result.evidence_status == "sufficient"
    assert result.confidence_status == "high"
    assert result.citations[0].source_id == "S1"
    assert result.citations[0].title == "Registration policy"
    assert len(generated) == 1
    assert generated[0][1] == (chunk,)
    assert session.rows[0].student_contact == "student@example.test"
    assert session.rows[0].escalated is False
    assert session.rows[0].answer_document_id == chunk.document_id
    assert session.commit_count == 1


def test_answer_pipeline_escalates_without_calling_llm_when_evidence_is_missing() -> None:
    asyncio.run(_exercise_no_evidence())


async def _exercise_no_evidence() -> None:
    session = _FakeSession()
    generated = False

    async def retriever(_session, _question, _top_k):
        return []

    async def validator(_session, candidates, *, question):
        assert question == "Where can I find an undocumented exception?"
        assert not candidates
        return EvidenceAssessment("insufficient", "no_evidence", 0.0, ())

    async def generator(_question, _evidence):
        nonlocal generated
        generated = True
        raise AssertionError("generation must not run when evidence is insufficient")

    result = await answer_question(
        session,
        "Where can I find an undocumented exception?",
        "student@example.test",
        retriever=retriever,
        validator=validator,
        generator=generator,
    )

    assert result.evidence_status == "insufficient"
    assert result.escalation_reason == "no_evidence"
    assert result.escalated is True
    assert result.confidence_status == "low"
    assert not result.citations
    assert "sent for staff review" in result.answer
    assert generated is False
    assert session.rows[0].escalated is True
    assert session.rows[0].answer_document_id is None
    assert session.commit_count == 1


class _FakeSession:
    def __init__(self) -> None:
        self.rows = []
        self.commit_count = 0

    def add(self, row) -> None:
        self.rows.append(row)

    async def commit(self) -> None:
        self.commit_count += 1


def _chunk() -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Registration policy",
        authority_office="Registrar",
        source_type="manual",
        publication_date=None,
        effective_date=date(2025, 1, 1),
        expiry_date=None,
        content="Submit the extension form to the Registrar within five days.",
        score=0.93,
        rank=1,
        retrieval_method="hybrid",
    )
