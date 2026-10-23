from datetime import date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document
from app.ingestion.parser import clean_text


async def create_pending_manual_document(
    session: AsyncSession,
    *,
    title: str,
    authority_office: str,
    authored_by: UUID,
    content: str,
    effective_date: date | None = None,
    publication_date: date | None = None,
    expiry_date: date | None = None,
) -> Document:
    """Store typed source text as pending; approval is required before indexing."""
    normalized = clean_text(content)
    if not normalized:
        raise ValueError("Manual document content cannot be empty.")

    document = Document(
        title=title.strip(),
        authority_office=authority_office.strip(),
        source_type="manual",
        pending_content=normalized,
        authored_by=authored_by,
        effective_date=effective_date or date.today(),
        publication_date=publication_date,
        expiry_date=expiry_date,
        status="pending",
    )
    session.add(document)
    await session.flush()
    return document
