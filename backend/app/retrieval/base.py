from dataclasses import dataclass
from datetime import date
from typing import Literal
from uuid import UUID

from sqlalchemy import and_, or_, text

from app.db.models import Document, DocumentChunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: UUID
    document_id: UUID
    title: str
    authority_office: str
    source_type: str
    publication_date: date | None
    effective_date: date
    expiry_date: date | None
    content: str
    score: float
    rank: int
    retrieval_method: Literal["keyword", "embedding", "hybrid"] = "hybrid"


def current_document_filters():
    """SQL predicates that keep non-current source material out of retrieval."""
    today = text("CURRENT_DATE")
    return and_(
        Document.status == "active",
        Document.effective_date <= today,
        or_(Document.expiry_date.is_(None), Document.expiry_date >= today),
    )


def scored_chunk(
    chunk: DocumentChunk,
    document: Document,
    score: float,
    rank: int,
    retrieval_method: Literal["keyword", "embedding", "hybrid"],
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        document_id=document.id,
        title=document.title,
        authority_office=document.authority_office,
        source_type=document.source_type,
        publication_date=document.publication_date,
        effective_date=document.effective_date,
        expiry_date=document.expiry_date,
        content=chunk.content,
        score=score,
        rank=rank,
        retrieval_method=retrieval_method,
    )


def validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
