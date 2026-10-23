import math
from datetime import date
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import EMBEDDING_DIMENSION, Document, DocumentChunk
from app.ingestion.chunker import Tokenizer, chunk_text
from app.ingestion.embedder import embed_passages, load_tokenizer
from app.ingestion.parser import parse_uploaded_file


async def create_pending_uploaded_document(
    session: AsyncSession,
    *,
    title: str,
    authority_office: str,
    authored_by: UUID,
    filename: str,
    content: bytes,
    storage_url: str | None = None,
    publication_date: date | None = None,
    effective_date: date | None = None,
    expiry_date: date | None = None,
) -> Document:
    """Parse an upload and retain its extracted text for approval review."""
    extracted_text = parse_uploaded_file(filename, content)
    document = Document(
        title=title.strip(),
        authority_office=authority_office.strip(),
        source_type="uploaded",
        storage_url=storage_url,
        pending_content=extracted_text,
        authored_by=authored_by,
        publication_date=publication_date,
        effective_date=effective_date or date.today(),
        expiry_date=expiry_date,
        status="pending",
    )
    session.add(document)
    await session.flush()
    return document


async def ingest_approved_document(
    session: AsyncSession,
    document_id: UUID,
    *,
    tokenizer: Tokenizer | None = None,
    embedder=embed_passages,
) -> list[DocumentChunk]:
    """Create searchable chunks only for an approved, active document.

    The caller owns the surrounding transaction and should set approval fields
    before calling this function. On success, pending source text is cleared.
    """
    result = await session.execute(
        select(Document).where(Document.id == document_id).with_for_update()
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise ValueError(f"Document {document_id} does not exist.")
    if document.status != "active" or document.approved_by is None or document.approved_at is None:
        raise ValueError("Only approved active documents can be ingested.")
    if not document.pending_content:
        raise ValueError("The document has no pending source text to ingest.")

    active_tokenizer = tokenizer or load_tokenizer(settings.embedding_model_name)
    passages = chunk_text(document.pending_content, active_tokenizer)
    if not passages:
        raise ValueError("The document contains no text to index.")

    vectors = embedder(passages, settings.embedding_model_name)
    if len(vectors) != len(passages):
        raise ValueError("The embedder returned a different number of vectors than passages.")
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"Each embedding must contain {EMBEDDING_DIMENSION} finite values.")

    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    chunks = [
        DocumentChunk(
            document_id=document.id,
            chunk_index=index,
            content=passage,
            embedding=vector,
        )
        for index, (passage, vector) in enumerate(zip(passages, vectors, strict=True))
    ]
    session.add_all(chunks)
    document.pending_content = None
    await session.flush()
    return chunks
