"""Seed or remove a clearly labeled, local Stage 6 answer-path test document."""

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.models import AdminUser, Document, DocumentChunk, QuestionLog
from app.ingestion.pipeline import ingest_approved_document

TEST_TITLE = "Vera Stage 6 Answer Path Test Document"
TEST_ADMIN_EMAIL = "vera-stage6-test-admin@example.test"
TEST_QUERY = "What form must a student submit to request a registration extension?"
TEST_CONTENT = (
    "For the Vera Stage 6 answer-path test only: A student requesting a registration "
    "extension must submit the Registration Extension Form to the Registrar's Office. "
    "The student must submit the form within five calendar days of the missed deadline. "
    "The Registrar's Office reviews the request and sends the student its decision by email."
)


async def seed() -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            admin_result = await session.execute(
                select(AdminUser).where(AdminUser.email == TEST_ADMIN_EMAIL)
            )
            admin = admin_result.scalar_one_or_none()
            if admin is None:
                admin = AdminUser(
                    name="Vera Stage 6 Test Admin",
                    email=TEST_ADMIN_EMAIL,
                    role="admin",
                )
                session.add(admin)
                await session.flush()

            document_result = await session.execute(
                select(Document).where(Document.title == TEST_TITLE)
            )
            document = document_result.scalar_one_or_none()
            if document is None:
                document = Document(
                    title=TEST_TITLE,
                    authority_office="Stage 6 Test Registrar",
                    source_type="manual",
                    pending_content=TEST_CONTENT,
                    effective_date=date.today(),
                    status="active",
                    authored_by=admin.id,
                    approved_by=admin.id,
                    approved_at=datetime.now(UTC),
                )
                session.add(document)
                await session.flush()
            else:
                document.authority_office = "Stage 6 Test Registrar"
                document.source_type = "manual"
                document.pending_content = TEST_CONTENT
                document.effective_date = date.today()
                document.expiry_date = None
                document.status = "active"
                document.authored_by = admin.id
                document.approved_by = admin.id
                document.approved_at = datetime.now(UTC)

            chunks = await ingest_approved_document(session, document.id)
            await session.commit()
            print(f"Seeded {len(chunks)} chunk(s) in the configured database.")
            print(f"Test question: {TEST_QUERY}")
            print("The source is explicitly labeled as test-only; run this script with 'cleanup' afterward.")
    except Exception:
        await session.rollback()
        raise
    finally:
        await engine.dispose()


async def cleanup() -> None:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            documents = (
                await session.scalars(select(Document).where(Document.title == TEST_TITLE))
            ).all()
            document_ids = [document.id for document in documents]
            if document_ids:
                await session.execute(
                    delete(QuestionLog).where(QuestionLog.answer_document_id.in_(document_ids))
                )
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id.in_(document_ids))
                )
                await session.execute(delete(Document).where(Document.id.in_(document_ids)))

            admin = await session.scalar(
                select(AdminUser).where(AdminUser.email == TEST_ADMIN_EMAIL)
            )
            if admin is not None:
                document_count = await session.scalar(
                    select(func.count())
                    .select_from(Document)
                    .where(or_(Document.authored_by == admin.id, Document.approved_by == admin.id))
                )
                question_count = await session.scalar(
                    select(func.count())
                    .select_from(QuestionLog)
                    .where(
                        or_(
                            QuestionLog.answered_by == admin.id,
                            QuestionLog.resolved_by == admin.id,
                        )
                    )
                )
                if not document_count and not question_count:
                    await session.execute(delete(AdminUser).where(AdminUser.id == admin.id))

            await session.commit()
            print(f"Removed {len(document_ids)} Stage 6 test document(s) and their answer logs.")
    except Exception:
        await session.rollback()
        raise
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seed", "cleanup"))
    action = parser.parse_args().action
    asyncio.run(seed() if action == "seed" else cleanup())


if __name__ == "__main__":
    main()
