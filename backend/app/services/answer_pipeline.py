from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import QuestionLog
from app.generation.llm_client import (
    AnswerSupport,
    LLMError,
    generate_answer,
    verify_answer_support,
)
from app.retrieval.base import ScoredChunk
from app.retrieval.hybrid import hybrid_search
from app.validation.evidence import EvidenceAssessment, validate_evidence

Retriever = Callable[..., Awaitable[list[ScoredChunk]]]
Validator = Callable[..., Awaitable[EvidenceAssessment]]
Generator = Callable[[str, tuple[ScoredChunk, ...]], Awaitable[str]]
Verifier = Callable[[str, str, tuple[ScoredChunk, ...]], Awaitable[AnswerSupport]]


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
    verifier: Verifier = verify_answer_support,
) -> AnswerResult:
    normalized_question = question.strip()
    candidates = await retriever(session, normalized_question, top_k)
    assessment = await validator(session, candidates, question=normalized_question)
    citations = tuple(_citation(index, chunk) for index, chunk in enumerate(assessment.chunks, 1))

    if not assessment.can_answer:
        answer = _abstention_message(assessment.evidence_status)
        await _record_question(
            session,
            normalized_question,
            student_contact,
            assessment,
            answer=None,
            answer_confidence=0.0,
        )
        return AnswerResult(
            answer=answer,
            evidence_status=assessment.evidence_status,
            escalation_reason=assessment.escalation_reason,
            escalated=True,
            confidence_score=0.0,
            confidence_status="low",
            citations=citations,
        )

    # Retrieval is only a candidate gate. The generated answer must also pass
    # an independent claim-support check before it can be returned as reliable.
    answer = await generator(normalized_question, assessment.chunks)
    try:
        support = await verifier(normalized_question, answer, assessment.chunks)
    except LLMError:
        support = AnswerSupport(supported=False, support_score=0.0)

    if (
        not support.supported
        or support.support_score < settings.answer_confidence_medium_threshold
    ):
        failed_assessment = replace(
            assessment,
            evidence_status="insufficient",
            escalation_reason="no_evidence",
        )
        answer = _abstention_message("insufficient")
        await _record_question(
            session,
            normalized_question,
            student_contact,
            failed_assessment,
            answer=None,
            answer_confidence=0.0,
        )
        return AnswerResult(
            answer=answer,
            evidence_status="insufficient",
            escalation_reason="no_evidence",
            escalated=True,
            confidence_score=0.0,
            confidence_status="low",
            citations=(),
        )

    await _record_question(
        session,
        normalized_question,
        student_contact,
        assessment,
        answer=answer,
        answer_confidence=support.support_score,
    )
    if support.support_score >= settings.answer_confidence_high_threshold:
        confidence_status = "high"
    elif support.support_score >= settings.answer_confidence_medium_threshold:
        confidence_status = "medium"
    else:
        confidence_status = "low"
    return AnswerResult(
        answer=answer,
        evidence_status=assessment.evidence_status,
        escalation_reason=None,
        escalated=False,
        confidence_score=support.support_score,
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
    answer_confidence: float,
) -> None:
    session.add(
        QuestionLog(
            query_text=question,
            student_contact=student_contact,
            retrieval_confidence=assessment.retrieval_score,
            answer_confidence=answer_confidence,
            evidence_status=assessment.evidence_status,
            escalation_reason=assessment.escalation_reason,
            escalated=not assessment.can_answer,
            answer_document_id=(assessment.chunks[0].document_id if answer is not None else None),
        )
    )
    await session.commit()
