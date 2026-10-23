import asyncio
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

import pytest
from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser, Document, DocumentChunk
from app.ingestion.chunker import chunk_text
from app.ingestion.direct_entry import create_pending_manual_document
from app.ingestion.parser import (
    EmptyDocumentError,
    UnsupportedDocumentError,
    parse_uploaded_file,
)
from app.ingestion.pipeline import (
    create_pending_uploaded_document,
    ingest_approved_document,
)
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


class CharacterTokenizer:
    def encode(self, value: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in value]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def make_pdf_with_text(value: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=100)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    resources = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    page[NameObject("/Resources")] = resources

    stream = DecodedStreamObject()
    escaped_value = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 10 50 Td ({escaped_value}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_parser_extracts_pdf_and_plain_text() -> None:
    assert parse_uploaded_file("policy.txt", b"  Registration   opens Monday. ") == (
        "Registration opens Monday."
    )
    assert parse_uploaded_file("policy.pdf", make_pdf_with_text("Registration opens Monday.")) == (
        "Registration opens Monday."
    )


def test_parser_rejects_unsupported_and_empty_documents() -> None:
    with pytest.raises(UnsupportedDocumentError):
        parse_uploaded_file("policy.docx", b"not supported")
    with pytest.raises(EmptyDocumentError):
        parse_uploaded_file("empty.txt", b"  \n ")


def test_chunker_creates_overlapping_token_windows() -> None:
    chunks = chunk_text("abcdefghij", CharacterTokenizer(), chunk_size=6, overlap=2)

    assert chunks == ["abcdef", "efghij"]


def test_ingestion_requires_approval_and_stores_chunks() -> None:
    asyncio.run(_exercise_ingestion())


async def _exercise_ingestion() -> None:
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
        tokenizer = CharacterTokenizer()

        def fake_embedder(passages: list[str], _model_name: str) -> list[list[float]]:
            return [[0.1] * 768 for _ in passages]

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

            manual = await create_pending_manual_document(
                session,
                title="Manual policy",
                authority_office="Registrar",
                authored_by=admin_id,
                content="Students may request a registration extension.",
            )
            uploaded = await create_pending_uploaded_document(
                session,
                title="Uploaded policy",
                authority_office="Registrar",
                authored_by=admin_id,
                filename="registration.txt",
                content=b"Registration extensions are reviewed by the Registrar.",
            )

            assert manual.status == uploaded.status == "pending"
            assert manual.pending_content is not None
            assert uploaded.pending_content is not None
            assert await session.scalar(select(func.count()).select_from(DocumentChunk)) == 0

            with pytest.raises(ValueError, match="Only approved active documents"):
                await ingest_approved_document(
                    session, manual.id, tokenizer=tokenizer, embedder=fake_embedder
                )

            for document in (manual, uploaded):
                document.status = "active"
                document.approved_by = admin_id
                document.approved_at = now
                chunks = await ingest_approved_document(
                    session, document.id, tokenizer=tokenizer, embedder=fake_embedder
                )
                assert chunks
                assert document.pending_content is None

            await session.commit()

        async with session_factory() as session:
            stored_docs = (await session.scalars(select(Document))).all()
            stored_chunks = (await session.scalars(select(DocumentChunk))).all()
            assert len(stored_docs) == 2
            assert len(stored_chunks) == 2
            assert {chunk.document_id for chunk in stored_chunks} == {
                document.id for document in stored_docs
            }
            assert all(len(chunk.embedding) == 768 for chunk in stored_chunks)
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
