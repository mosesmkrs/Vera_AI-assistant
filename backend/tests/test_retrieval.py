import asyncio
from datetime import UTC, date, datetime
from uuid import uuid4

from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser, Document, DocumentChunk
from app.retrieval.base import ScoredChunk
from app.retrieval.embedding import embedding_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.keyword import keyword_search
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def test_keyword_embedding_and_hybrid_retrieval_filter_noncurrent_documents() -> None:
    asyncio.run(_exercise_retrieval_methods())


async def _exercise_retrieval_methods() -> None:
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
        now = datetime.now(UTC)
        active_id, expired_id, future_id, pending_id = (uuid4() for _ in range(4))
        query_vector = [1.0] + [0.0] * 767
        nearby_vector = [0.9, 0.1] + [0.0] * 766

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

            documents = [
                Document(
                    id=active_id,
                    title="Current registration policy",
                    authority_office="Registrar",
                    source_type="manual",
                    effective_date=date(2000, 1, 1),
                    status="active",
                    authored_by=admin_id,
                    approved_by=admin_id,
                    approved_at=now,
                ),
                Document(
                    id=expired_id,
                    title="Expired registration policy",
                    authority_office="Registrar",
                    source_type="manual",
                    effective_date=date(2000, 1, 1),
                    expiry_date=date(2000, 1, 2),
                    status="active",
                    authored_by=admin_id,
                    approved_by=admin_id,
                    approved_at=now,
                ),
                Document(
                    id=future_id,
                    title="Future registration policy",
                    authority_office="Registrar",
                    source_type="manual",
                    effective_date=date(9999, 1, 1),
                    status="active",
                    authored_by=admin_id,
                    approved_by=admin_id,
                    approved_at=now,
                ),
                Document(
                    id=pending_id,
                    title="Pending registration policy",
                    authority_office="Registrar",
                    source_type="manual",
                    effective_date=date(2000, 1, 1),
                    status="pending",
                    authored_by=admin_id,
                ),
            ]
            session.add_all(documents)
            await session.flush()

            session.add_all(
                [
                    DocumentChunk(
                        document_id=document.id,
                        chunk_index=0,
                        content="Registration extensions require Registrar review.",
                        embedding=query_vector if document.id == active_id else nearby_vector,
                    )
                    for document in documents
                ]
            )
            await session.commit()

        async with session_factory() as session:
            keyword_results = await keyword_search(
                session, "What is the registration policy for extensions?", top_k=10
            )
            vector_results = await embedding_search(
                session,
                "late enrollment review",
                top_k=10,
                query_embedding=query_vector,
            )
            hybrid_results = await hybrid_search(
                session,
                "What is the registration policy for extensions?",
                top_k=10,
                query_embedding=query_vector,
            )

            assert [item.document_id for item in keyword_results] == [active_id]
            assert [item.document_id for item in vector_results] == [active_id]
            assert [item.document_id for item in hybrid_results] == [active_id]
            assert all(item.title == "Current registration policy" for item in hybrid_results)
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


def test_hybrid_rrf_can_rank_a_shared_result_above_each_strategy_leader() -> None:
    async def _exercise_rrf() -> None:
        keyword_leader = _scored_chunk("keyword leader", rank=1)
        vector_leader = _scored_chunk("vector leader", rank=1)
        shared = _scored_chunk("shared result", rank=2)

        async def fake_keyword(_session, _query: str, _top_k: int):
            return [keyword_leader, shared]

        async def fake_embedding(
            _session,
            _query: str,
            _top_k: int,
            *,
            query_embedding: list[float] | None = None,
        ):
            return [vector_leader, shared]

        result = await hybrid_search(
            None,
            "example query",
            top_k=3,
            keyword_retriever=fake_keyword,
            embedding_retriever=fake_embedding,
        )
        assert result[0].chunk_id == shared.chunk_id
        assert result[0].content == "shared result"

    asyncio.run(_exercise_rrf())


def _scored_chunk(content: str, *, rank: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        title="Test document",
        authority_office="Registrar",
        source_type="manual",
        publication_date=None,
        effective_date=date(2000, 1, 1),
        expiry_date=None,
        content=content,
        score=1.0,
        rank=rank,
    )
