"""Run the three retrieval strategies against real local PDF/text documents."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser
from app.ingestion.pipeline import (
    create_pending_uploaded_document,
    ingest_approved_document,
)
from app.retrieval.embedding import embedding_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.keyword import keyword_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily index local PDF/.txt files with the real embedding model, "
            "then compare keyword, vector, and hybrid results."
        )
    )
    parser.add_argument("--files", nargs="+", required=True, type=Path, help="PDF or UTF-8 .txt files")
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="A real question to search; repeat --query to try several questions.",
    )
    parser.add_argument("--authority", default="Smoke test source", help="Authority-office label")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def print_results(method: str, query: str, results) -> None:
    print(f"\n{method.upper()} — {query}")
    if not results:
        print("  No matching chunks.")
        return
    for result in results:
        print(
            f"  #{result.rank} score={result.score:.5f} | {result.title} "
            f"| {result.authority_office} | effective {result.effective_date}"
        )
        print(f"     {result.content[:500]}")


async def run(files: list[Path], queries: list[str], authority: str, top_k: int) -> None:
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(map(str, missing)))
    if top_k < 1:
        raise ValueError("--top-k must be at least 1")

    schema = f"retrieval_smoke_{uuid4().hex}"
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    scoped_engine = engine.execution_options(schema_translate_map={None: schema})
    session_factory = async_sessionmaker(scoped_engine, expire_on_commit=False)

    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection.execution_options(schema_translate_map={None: schema})
                )
            )

        async with session_factory() as session:
            admin = AdminUser(
                name="Temporary Retrieval Smoke Test",
                email=f"retrieval-smoke-{uuid4()}@example.test",
                role="admin",
            )
            session.add(admin)
            await session.flush()

            for path in files:
                document = await create_pending_uploaded_document(
                    session,
                    title=path.stem,
                    authority_office=authority,
                    authored_by=admin.id,
                    filename=path.name,
                    content=path.read_bytes(),
                )
                # The admin approval API is a later stage. This harness activates
                # the temporary test document to exercise the ingestion pipeline.
                document.status = "active"
                document.approved_by = admin.id
                document.approved_at = datetime.now(UTC)
                await ingest_approved_document(session, document.id)

            await session.commit()
            print(f"Indexed {len(files)} file(s) in temporary schema {schema}.")
            print("The schema and all test rows will be deleted when this command exits.")

            for query in queries:
                keyword_results = await keyword_search(session, query, top_k)
                vector_results = await embedding_search(session, query, top_k)
                hybrid_results = await hybrid_search(session, query, top_k)
                print_results("keyword", query, keyword_results)
                print_results("embedding", query, vector_results)
                print_results("hybrid", query, hybrid_results)
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(run(arguments.files, arguments.query, arguments.authority, arguments.top_k))
