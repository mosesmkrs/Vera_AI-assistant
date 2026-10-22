import asyncio
from datetime import UTC, date, datetime
from uuid import uuid4

from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser, Document, DocumentChunk, QuestionLog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def test_model_round_trip() -> None:
    asyncio.run(_exercise_models())


async def _exercise_models() -> None:
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
        admin_id = uuid4()
        assistant_id = uuid4()
        document_id = uuid4()
        question_id = uuid4()
        now = datetime.now(UTC)

        async with session_factory() as session:
            session.add_all(
                [
                    AdminUser(
                        id=admin_id,
                        name="Senior Admin",
                        email=f"admin-{admin_id}@example.test",
                        role="admin",
                    ),
                    AdminUser(
                        id=assistant_id,
                        name="Office Assistant",
                        email=f"assistant-{assistant_id}@example.test",
                        role="assistant",
                    ),
                ]
            )
            await session.flush()

            session.add(
                Document(
                    id=document_id,
                    title="Registration clarification",
                    authority_office="Registrar",
                    source_type="manual",
                    publication_date=date(2026, 10, 1),
                    effective_date=date(2026, 10, 1),
                    status="active",
                    authored_by=assistant_id,
                    approved_by=admin_id,
                    approved_at=now,
                )
            )
            session.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=0,
                    content="Late registration is reviewed by the Registrar.",
                    embedding=[0.0] * 768,
                )
            )
            session.add(
                QuestionLog(
                    id=question_id,
                    query_text="Can I register late?",
                    student_contact="student@example.test",
                    retrieval_confidence=0.82,
                    evidence_status="insufficient",
                    escalation_reason="no_evidence",
                    escalated=True,
                    answered_by=assistant_id,
                    answer_document_id=document_id,
                    resolved_by=admin_id,
                    notification_status="sent",
                    notification_sent_at=now,
                    answered_at=now,
                    resolved_at=now,
                )
            )
            await session.commit()

        async with session_factory() as session:
            stored_admin = await session.get(AdminUser, admin_id)
            stored_document = await session.get(Document, document_id)
            stored_chunk = await session.scalar(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            stored_question = await session.get(QuestionLog, question_id)

            assert stored_admin is not None and stored_admin.role == "admin"
            assert stored_document is not None and stored_document.status == "active"
            assert stored_document.approver is not None and stored_document.approver.id == admin_id
            assert stored_chunk is not None and len(stored_chunk.embedding) == 768
            assert stored_question is not None and stored_question.notification_status == "sent"
            assert stored_question.answer_document_id == document_id
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
