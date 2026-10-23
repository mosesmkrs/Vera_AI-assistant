from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import QuestionLog
from app.generation.llm_client import generate_answer
from app.retrieval.base import ScoredChunk
from app.retrieval.hybrid import hybrid_search
from app.validation.evidence import EvidenceAssessment, validate_evidence

Retriever = Callable[..., Awaitable[list[ScoredChunk]]]
Validator = Callable[..., Awaitable[EvidenceAssessment]]
Generator = Callable[[str, tuple[ScoredChunk, ...]], Awaitable[str]]


@dataclass(frozen=True)
class Citation:
    source_id: str
    document_id: str
    title: str
    authority_office: str
    effective_date: str
    excerpt: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    evidence_status: str
    escalation_reason: str | None
    escalated: bool
    confidence_score: float
    confidence_status: str
    citations: tuple[Citation, ...]


async def answer_question(
    session: AsyncSession,
    question: str,
    student_contact: str | None = None,
    *,
    top_k: int = 5,
    retriever: Retriever = hybrid_search,
    validator: Validator = validate_evidence,
    generator: Generator = generate_answer,
) -> AnswerResult:
    normalized_question = question.strip()
    candidates = await retriever(session, normalized_question, top_k)
    assessment = await validator(session, candidates)
    citations = tuple(_citation(index, chunk) for index, chunk in enumerate(assessment.chunks, 1))

    if not assessment.can_answer:
        answer = _abstention_message(assessment.evidence_status)
        await _record_question(
            session,
            normalized_question,
            student_contact,
            assessment,
            answer=None,
        )
        return AnswerResult(
            answer=answer,
            evidence_status=assessment.evidence_status,
            escalation_reason=assessment.escalation_reason,
            escalated=True,
            confidence_score=assessment.confidence_score,
            confidence_status="low",
            citations=citations,
        )

    # This is the only path that calls the model. The evidence gate above must pass.
    answer = await generator(normalized_question, assessment.chunks)
    await _record_question(
        session,
        normalized_question,
        student_contact,
        assessment,
        answer=answer,
    )
    if assessment.confidence_score >= 0.8:
        confidence_status = "high"
    elif assessment.confidence_score >= settings.evidence_confidence_threshold:
        confidence_status = "medium"
    else:
        confidence_status = "low"
    return AnswerResult(
        answer=answer,
        evidence_status=assessment.evidence_status,
        escalation_reason=None,
        escalated=False,
        confidence_score=assessment.confidence_score,
        confidence_status=confidence_status,
        citations=citations,
    )


def _citation(index: int, chunk: ScoredChunk) -> Citation:
    return Citation(
        source_id=f"S{index}",
        document_id=str(chunk.document_id),
        title=chunk.title,
        authority_office=chunk.authority_office,
        effective_date=chunk.effective_date.isoformat(),
        excerpt=chunk.content,
    )


def _abstention_message(evidence_status: str) -> str:
    if evidence_status == "conflicting":
        return (
            "I can’t provide a reliable answer because approved sources from different "
            "offices conflict. The question has been sent for staff review."
        )
    return (
        "I couldn’t find enough approved information to answer this reliably. "
        "The question has been sent for staff review."
    )


async def _record_question(
    session: AsyncSession,
    question: str,
    student_contact: str | None,
    assessment: EvidenceAssessment,
    *,
    answer: str | None,
) -> None:
    session.add(
        QuestionLog(
            query_text=question,
            student_contact=student_contact,
            retrieval_confidence=assessment.confidence_score,
            evidence_status=assessment.evidence_status,
            escalation_reason=assessment.escalation_reason,
            escalated=not assessment.can_answer,
            answer_document_id=(assessment.chunks[0].document_id if answer is not None else None),
        )
    )
    await session.commit()
