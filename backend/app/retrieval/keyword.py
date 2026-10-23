import re

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.retrieval.base import (
    ScoredChunk,
    current_document_filters,
    scored_chunk,
    validate_top_k,
)

_ENGLISH_STOP_WORDS = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"]
)


async def keyword_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
) -> list[ScoredChunk]:
    """Rank current document chunks with PostgreSQL full-text search."""
    validate_top_k(top_k)
    if not query.strip():
        return []

    query_terms = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
    terms = [term for term in query_terms if term not in _ENGLISH_STOP_WORDS]
    if not terms:
        terms = query_terms
    if not terms:
        return []

    text_vector = func.to_tsvector("simple", DocumentChunk.content)
    # Natural questions often include words absent from the source passage.
    # OR keeps keyword retrieval useful for partial lexical matches.
    text_query = func.to_tsquery("simple", " | ".join(terms))
    rank = func.ts_rank_cd(text_vector, text_query)
    statement = (
        select(DocumentChunk, Document, rank.label("search_rank"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(current_document_filters(), text_vector.bool_op("@@")(text_query))
        .order_by(desc("search_rank"), DocumentChunk.id)
        .limit(top_k)
    )
    result = await session.execute(statement)
    rows = result.all()
    return [
        scored_chunk(chunk, document, float(search_rank), rank_index)
        for rank_index, (chunk, document, search_rank) in enumerate(rows, start=1)
    ]
