"""Add original questions to already-published staff answer documents.

Run without --apply to preview the number of documents affected. Re-embedding
requires --apply because it replaces the existing chunks for each document.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.models import Document, DocumentChunk, QuestionLog
from app.ingestion.pipeline import ingest_approved_document


async def backfill(*, apply: bool) -> int:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(QuestionLog, Document)
                    .join(Document, Document.id == QuestionLog.answer_document_id)
                    .where(
                        Document.status == "active",
                        Document.title.startswith("Staff clarification for question "),
                        QuestionLog.resolved_at.is_not(None),
                    )
                    .order_by(QuestionLog.created_at)
                )
            ).all()
            if not apply:
                print(f"Found {len(rows)} published staff answer(s) to update.")
                print("Run again with --apply to add the original questions and rebuild embeddings.")
                return len(rows)

            updated = 0
            for question, document in rows:
                chunks = (
                    await session.scalars(
                        select(DocumentChunk)
                        .where(DocumentChunk.document_id == document.id)
                        .order_by(DocumentChunk.chunk_index)
                    )
                ).all()
                old_answer = " ".join(chunk.content for chunk in chunks).strip()
                if not old_answer:
                    continue

                document.title = f"Staff answer: {question.query_text[:480]}"
                document.pending_content = (
                    f"Question: {question.query_text}\n\n"
                    f"Approved staff answer: {old_answer}"
                )
                await ingest_approved_document(session, document.id)
                updated += 1

            await session.commit()
            print(f"Updated {updated} published staff answer(s).")
            return updated
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace old chunks and regenerate embeddings for matching answers",
    )
    args = parser.parse_args()
    asyncio.run(backfill(apply=args.apply))


if __name__ == "__main__":
    main()
