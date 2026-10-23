from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.base import ScoredChunk, validate_top_k
from app.retrieval.embedding import embedding_search
from app.retrieval.keyword import keyword_search


async def hybrid_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    *,
    rrf_k: int = 60,
    query_embedding: list[float] | None = None,
    keyword_retriever: Callable[..., Awaitable[list[ScoredChunk]]] = keyword_search,
    embedding_retriever: Callable[..., Awaitable[list[ScoredChunk]]] = embedding_search,
) -> list[ScoredChunk]:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion."""
    validate_top_k(top_k)
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    if not query.strip():
        return []

    candidate_k = max(top_k, top_k * 2)
    # AsyncSession is not safe for concurrent operations, so run both searches
    # sequentially while keeping each retrieval strategy independently callable.
    keyword_results = await keyword_retriever(session, query, candidate_k)
    vector_results = await embedding_retriever(
        session,
        query,
        candidate_k,
        query_embedding=query_embedding,
    )

    fused: dict = {}
    for results in (keyword_results, vector_results):
        for item in results:
            existing = fused.get(item.chunk_id)
            rrf_score = 1.0 / (rrf_k + item.rank)
            if existing is None:
                fused[item.chunk_id] = [item, rrf_score]
            else:
                existing[1] += rrf_score

    ordered = sorted(fused.values(), key=lambda value: (-value[1], str(value[0].chunk_id)))[:top_k]
    return [
        ScoredChunk(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            title=item.title,
            authority_office=item.authority_office,
            source_type=item.source_type,
            publication_date=item.publication_date,
            effective_date=item.effective_date,
            expiry_date=item.expiry_date,
            content=item.content,
            score=score,
            rank=rank,
        )
        for rank, (item, score) in enumerate(ordered, start=1)
    ]
