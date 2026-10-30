import re
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
    question: str | None = None,
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

    selected = _select_evidence_chunks(valid)
    confidence = _confidence_score(selected, rrf_k)
    if _has_cross_office_conflict(selected, documents, question=question or ""):
        return EvidenceAssessment(
            evidence_status="conflicting",
            escalation_reason="conflicting_evidence",
            confidence_score=confidence,
            chunks=tuple(selected),
        )
    if confidence < threshold:
        return EvidenceAssessment(
            evidence_status="insufficient",
            escalation_reason="no_evidence",
            confidence_score=confidence,
            chunks=tuple(selected),
        )
    return EvidenceAssessment(
        evidence_status="sufficient",
        escalation_reason=None,
        confidence_score=confidence,
        chunks=tuple(selected),
    )


def _confidence_score(chunks: list[ScoredChunk], rrf_k: int) -> float:
    best_by_document: dict[UUID, ScoredChunk] = {}
    for chunk in chunks:
        best_by_document.setdefault(chunk.document_id, chunk)
    document_chunks = list(best_by_document.values())
    top_strength = _normalize_score(document_chunks[0], rrf_k)
    second_strength = _normalize_score(document_chunks[1], rrf_k) if len(document_chunks) > 1 else 0.0
    margin = max(0.0, top_strength - second_strength)
    score = 0.5 * top_strength + 0.4 * margin + 0.1
    return max(0.0, min(1.0, score))


def _select_evidence_chunks(chunks: list[ScoredChunk], *, per_document_limit: int = 3) -> list[ScoredChunk]:
    """Drop identical passages and cap repeated chunks from a single source."""
    selected = []
    document_counts: dict[UUID, int] = {}
    seen_content: set[tuple[UUID, str]] = set()
    for chunk in chunks:
        identity = (chunk.document_id, " ".join(chunk.content.casefold().split()))
        if identity in seen_content:
            continue
        if document_counts.get(chunk.document_id, 0) >= per_document_limit:
            continue
        seen_content.add(identity)
        document_counts[chunk.document_id] = document_counts.get(chunk.document_id, 0) + 1
        selected.append(chunk)
    return selected


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
    *,
    question: str,
) -> bool:
    """Flag explicit, question-relevant numeric contradictions across offices.

    Different issuing offices alone do not make sources contradictory. Require
    both sources to state a different value for the same unit and near-identical
    subject wording that is relevant to the student's question.
    """
    query_terms = _canonical_terms(question)
    if not query_terms:
        return False
    claims_by_document: dict[UUID, list[tuple[str, float, frozenset[str]]]] = {}
    for chunk in chunks:
        claims_by_document.setdefault(chunk.document_id, []).extend(
            _numeric_claims(chunk.content, query_terms)
        )

    document_ids = list(claims_by_document)
    for index, first_id in enumerate(document_ids):
        first_document = documents[first_id]
        for second_id in document_ids[index + 1 :]:
            second_document = documents[second_id]
            if first_document.authority_office.strip().casefold() == second_document.authority_office.strip().casefold():
                continue
            if (
                first_document.supersedes_id == second_document.id
                or second_document.supersedes_id == first_document.id
            ):
                continue
            for first_unit, first_value, first_terms in claims_by_document[first_id]:
                for second_unit, second_value, second_terms in claims_by_document[second_id]:
                    if first_unit != second_unit or first_value == second_value:
                        continue
                    shared_terms = first_terms & second_terms
                    union_terms = first_terms | second_terms
                    if len(shared_terms) >= 2 and len(shared_terms) / len(union_terms) >= 0.6:
                        return True
    return False


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "the", "to", "what", "when", "where", "which", "who", "with",
}
_TERM_ALIASES = {
    "examination": "exam",
    "examinations": "exam",
    "exams": "exam",
    "resit": "supplementary",
    "retake": "supplementary",
    "failed": "fail",
    "failing": "fail",
    "cats": "cat",
    "percentage": "percent",
    "percentages": "percent",
}
_NUMERIC_VALUE = re.compile(r"(?:(KES)\s*)?(\d[\d,]*(?:\.\d+)?)\s*(%)?", re.IGNORECASE)


def _canonical_terms(text: str) -> frozenset[str]:
    terms = set()
    for token in re.findall(r"[a-z]+", text.casefold()):
        if token in _STOPWORDS or len(token) < 3:
            continue
        term = _TERM_ALIASES.get(token, token)
        if term.endswith("s") and len(term) > 4:
            term = term[:-1]
        terms.add(term)
    return frozenset(terms)


def _numeric_claims(
    text: str,
    query_terms: frozenset[str],
) -> list[tuple[str, float, frozenset[str]]]:
    claims = []
    for match in _NUMERIC_VALUE.finditer(text):
        currency, raw_value, percent = match.groups()
        unit = "currency" if currency else "percent" if percent else None
        if unit is None:
            continue
        value = float(raw_value.replace(",", ""))
        start = max(0, match.start() - 55)
        end = min(len(text), match.end() + 55)
        context_terms = _canonical_terms(text[start:end])
        query_overlap = context_terms & query_terms
        if len(query_overlap) < min(2, len(query_terms)):
            continue
        claims.append((unit, value, context_terms))
    return claims


def _insufficient(reason: Literal["no_evidence"]) -> EvidenceAssessment:
    return EvidenceAssessment(
        evidence_status="insufficient",
        escalation_reason=reason,
        confidence_score=0.0,
        chunks=(),
    )
