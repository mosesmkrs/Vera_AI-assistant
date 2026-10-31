import asyncio
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser, Document, DocumentChunk
from app.retrieval.base import ScoredChunk
from app.validation.evidence import validate_evidence
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def test_evidence_validation_checks_dates_conflicts_and_confidence() -> None:
    asyncio.run(_exercise_validation())


async def _exercise_validation() -> None:
    schema = f"test_{uuid4().hex}"
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    test_engine = engine.execution_options(schema_translate_map={None: schema})

    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection.execution_options(schema_translate_map={None: schema})
                )
            )

        session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
        today = date.today()
        admin_id = uuid4()
        document_ids = {
            label: uuid4()
            for label in ("registrar", "finance", "conflict", "expired", "future")
        }
        chunk_ids = {
            label: uuid4()
            for label in (
                "registrar_top", "registrar_second", "finance", "conflict", "expired", "future"
            )
        }

        async with session_factory() as session:
            session.add(
                AdminUser(
                    id=admin_id,
                    name="Test Admin",
                    email=f"admin-{admin_id}@example.test",
                    role="admin",
                )
            )
            await session.flush()

            registrar = _document(
                document_ids["registrar"], admin_id, "Registrar", today - timedelta(days=10)
            )
            finance = _document(
                document_ids["finance"], admin_id, "Finance", today - timedelta(days=10)
            )
            conflict_document = _document(
                document_ids["conflict"], admin_id, "Examinations Office", today - timedelta(days=10)
            )
            expired = _document(
                document_ids["expired"],
                admin_id,
                "Registrar",
                today - timedelta(days=10),
                expiry_date=today - timedelta(days=1),
            )
            future = _document(
                document_ids["future"],
                admin_id,
                "Registrar",
                today + timedelta(days=1),
            )
            session.add_all([registrar, finance, conflict_document, expired, future])
            await session.flush()

            chunks = [
                DocumentChunk(
                    id=chunk_ids["registrar_top"],
                    document_id=registrar.id,
                    chunk_index=0,
                    content="Supplementary examination fee: KES 2,500 per paper.",
                    embedding=[0.0] * 768,
                ),
                DocumentChunk(
                    id=chunk_ids["registrar_second"],
                    document_id=registrar.id,
                    chunk_index=1,
                    content="A student with 35-39% may sit one supplementary examination.",
                    embedding=[0.0] * 768,
                ),
                DocumentChunk(
                    id=chunk_ids["finance"],
                    document_id=finance.id,
                    chunk_index=0,
                    content="Supplementary examination costs KES 2,500 per paper.",
                    embedding=[0.0] * 768,
                ),
                DocumentChunk(
                    id=chunk_ids["conflict"],
                    document_id=conflict_document.id,
                    chunk_index=0,
                    content="Supplementary examination fee: KES 3,000 per paper.",
                    embedding=[0.0] * 768,
                ),
                DocumentChunk(
                    id=chunk_ids["expired"],
                    document_id=expired.id,
                    chunk_index=0,
                    content="Expired registration policy.",
                    embedding=[0.0] * 768,
                ),
                DocumentChunk(
                    id=chunk_ids["future"],
                    document_id=future.id,
                    chunk_index=0,
                    content="Future registration policy.",
                    embedding=[0.0] * 768,
                ),
            ]
            session.add_all(chunks)
            await session.commit()

            db_chunks = (await session.scalars(select(DocumentChunk))).all()
            chunk_by_id = {chunk.id: chunk for chunk in db_chunks}

            sufficient = await validate_evidence(
                session,
                [
                    _candidate(chunk_by_id[chunk_ids["registrar_top"]], 0.95, 1),
                    _candidate(chunk_by_id[chunk_ids["registrar_second"]], 0.20, 2),
                ],
                today=today,
            )
            assert sufficient.evidence_status == "sufficient"
            assert sufficient.can_answer
            assert sufficient.retrieval_score >= settings.evidence_confidence_threshold

            low_confidence = await validate_evidence(
                session,
                [
                    _candidate(chunk_by_id[chunk_ids["registrar_top"]], 0.20, 1),
                    _candidate(chunk_by_id[chunk_ids["finance"]], 0.19, 2),
                ],
                today=today,
            )
            assert low_confidence.evidence_status == "insufficient"
            assert low_confidence.escalation_reason == "no_evidence"

            rejected_dates = await validate_evidence(
                session,
                [
                    _candidate(chunk_by_id[chunk_ids["expired"]], 0.99, 1),
                    _candidate(chunk_by_id[chunk_ids["future"]], 0.98, 2),
                ],
                today=today,
            )
            assert rejected_dates.evidence_status == "insufficient"
            assert rejected_dates.retrieval_score == 0.0
            assert not rejected_dates.chunks

            conflict = await validate_evidence(
                session,
                [
                    _candidate(chunk_by_id[chunk_ids["registrar_top"]], 0.95, 1),
                    _candidate(chunk_by_id[chunk_ids["finance"]], 0.94, 2),
                ],
                question="How much does a supplementary examination cost?",
                today=today,
            )
            assert conflict.evidence_status == "sufficient"
            assert conflict.escalation_reason is None

            conflict = await validate_evidence(
                session,
                [
                    _candidate(chunk_by_id[chunk_ids["registrar_top"]], 0.95, 1),
                    _candidate(chunk_by_id[chunk_ids["conflict"]], 0.94, 2),
                ],
                question="How much does a supplementary examination cost?",
                today=today,
            )
            assert conflict.evidence_status == "conflicting"
            assert conflict.escalation_reason == "conflicting_evidence"
            assert not conflict.can_answer
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


def _document(
    document_id,
    admin_id,
    authority_office: str,
    effective_date: date,
    *,
    expiry_date: date | None = None,
) -> Document:
    return Document(
        id=document_id,
        title=f"{authority_office} policy",
        authority_office=authority_office,
        source_type="manual",
        effective_date=effective_date,
        expiry_date=expiry_date,
        status="active",
        authored_by=admin_id,
        approved_by=admin_id,
        approved_at=datetime.now(UTC),
    )


def _candidate(chunk: DocumentChunk, score: float, rank: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        title="Candidate title",
        authority_office="Candidate office",
        source_type="manual",
        publication_date=None,
        effective_date=date(2000, 1, 1),
        expiry_date=None,
        content=chunk.content,
        score=score,
        rank=rank,
        retrieval_method="embedding",
    )
