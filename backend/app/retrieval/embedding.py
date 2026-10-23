from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import EMBEDDING_DIMENSION, Document, DocumentChunk
from app.ingestion.embedder import embed_query
from app.retrieval.base import (
    ScoredChunk,
    current_document_filters,
    scored_chunk,
    validate_top_k,
)


async def embedding_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    *,
    query_embedding: list[float] | None = None,
    embedder: Callable[[str, str | None], list[float]] = embed_query,
) -> list[ScoredChunk]:
    """Rank current document chunks by pgvector cosine similarity."""
    validate_top_k(top_k)
    if not query.strip():
        return []

    vector = (
        query_embedding
        if query_embedding is not None
        else embedder(query.strip(), settings.embedding_model_name)
    )
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Query embedding must contain {EMBEDDING_DIMENSION} dimensions.")

    distance = DocumentChunk.embedding.cosine_distance(vector)
    statement = (
        select(DocumentChunk, Document, distance.label("cosine_distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(current_document_filters())
        .order_by(distance, DocumentChunk.id)
        .limit(top_k)
    )
    result = await session.execute(statement)
    rows = result.all()
    return [
        scored_chunk(chunk, document, 1.0 - float(cosine_distance), rank_index)
        for rank_index, (chunk, document, cosine_distance) in enumerate(rows, start=1)
    ]
