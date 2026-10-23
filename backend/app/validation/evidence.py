from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.retrieval.base import ScoredChunk, scored_chunk

EvidenceStatus = Literal["sufficient", "insufficient", "conflicting"]
EscalationReason = Literal["no_evidence", "conflicting_evidence"] | None


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_status: EvidenceStatus
    escalation_reason: EscalationReason
    confidence_score: float
    chunks: tuple[ScoredChunk, ...]

    @property
    def can_answer(self) -> bool:
        return self.evidence_status == "sufficient"


async def validate_evidence(
    session: AsyncSession,
    candidates: list[ScoredChunk],
    *,
    today: date | None = None,
    confidence_threshold: float | None = None,
    rrf_k: int = 60,
) -> EvidenceAssessment:
    """Revalidate retrieved chunks and decide whether evidence can support an answer.

    This is an operational retrieval gate, not a measure of factual correctness.
    The caller must not generate an answer unless ``can_answer`` is true.
    """
    threshold = (
        settings.evidence_confidence_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if not candidates:
        return _insufficient("no_evidence")

    unique_ids = {candidate.chunk_id for candidate in candidates}
    result = await session.execute(
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id.in_(unique_ids))
    )
    stored = {chunk.id: (chunk, document) for chunk, document in result.all()}
    as_of = today or date.today()

    valid: list[ScoredChunk] = []
    documents: dict[UUID, Document] = {}
    for candidate in candidates:
        stored_pair = stored.get(candidate.chunk_id)
        if stored_pair is None:
            continue
        chunk, document = stored_pair
        if chunk.document_id != candidate.document_id:
            continue
        if document.status != "active":
            continue
        if document.effective_date > as_of:
            continue
        if document.expiry_date is not None and document.expiry_date < as_of:
            continue

        documents[document.id] = document
        valid.append(
            scored_chunk(
                chunk,
                document,
                candidate.score,
                candidate.rank,
                candidate.retrieval_method,
            )
        )

    if not valid:
        return _insufficient("no_evidence")

    confidence = _confidence_score(valid, rrf_k)
    if _has_cross_office_conflict(valid, documents):
        return EvidenceAssessment(
            evidence_status="conflicting",
            escalation_reason="conflicting_evidence",
            confidence_score=confidence,
            chunks=tuple(valid),
        )
    if confidence < threshold:
        return EvidenceAssessment(
            evidence_status="insufficient",
            escalation_reason="no_evidence",
            confidence_score=confidence,
            chunks=tuple(valid),
        )
    return EvidenceAssessment(
        evidence_status="sufficient",
        escalation_reason=None,
        confidence_score=confidence,
        chunks=tuple(valid),
    )


def _confidence_score(chunks: list[ScoredChunk], rrf_k: int) -> float:
    top_strength = _normalize_score(chunks[0], rrf_k)
    second_strength = _normalize_score(chunks[1], rrf_k) if len(chunks) > 1 else 0.0
    margin = max(0.0, top_strength - second_strength)
    score = 0.5 * top_strength + 0.4 * margin + 0.1
    return max(0.0, min(1.0, score))


def _normalize_score(chunk: ScoredChunk, rrf_k: int) -> float:
    if chunk.retrieval_method == "embedding":
        return max(0.0, min(1.0, (chunk.score + 1.0) / 2.0))
    if chunk.retrieval_method == "hybrid":
        theoretical_maximum = 2.0 / (rrf_k + 1)
        return max(0.0, min(1.0, chunk.score / theoretical_maximum))
    if chunk.score <= 0:
        return 0.0
    return chunk.score / (1.0 + chunk.score)


def _has_cross_office_conflict(
    chunks: list[ScoredChunk],
    documents: dict[UUID, Document],
) -> bool:
    top_by_document: list[ScoredChunk] = []
    seen_document_ids = set()
    for chunk in chunks:
        if chunk.document_id not in seen_document_ids:
            top_by_document.append(chunk)
            seen_document_ids.add(chunk.document_id)
        if len(top_by_document) == 2:
            break
    if len(top_by_document) < 2:
        return False

    first, second = top_by_document
    first_document = documents[first.document_id]
    second_document = documents[second.document_id]
    offices_differ = first.authority_office.strip().casefold() != second.authority_office.strip().casefold()
    documents_linked = (
        first_document.supersedes_id == second_document.id
        or second_document.supersedes_id == first_document.id
    )
    return offices_differ and not documents_linked


def _insufficient(reason: Literal["no_evidence"]) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_status="insufficient",
        escalation_reason=reason,
        confidence_score=0.0,
        chunks=(),
    )
