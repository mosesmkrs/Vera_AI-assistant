import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from app.auth.security import create_access_token, hash_password
from app.config import settings
from app.db.base import Base
from app.db.models import AdminUser, Document, DocumentChunk, QuestionLog
from app.db.session import get_db
from app.main import app
from app.routers import admin as admin_router
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


def test_admin_approval_resolves_assistant_answer_and_sends_email(monkeypatch) -> None:
    asyncio.run(_exercise_admin_workflow(monkeypatch))


async def _exercise_admin_workflow(monkeypatch) -> None:
    schema = f"test_{uuid4().hex}"
    engine = create_async_engine(settings.database_url, poolclass=NullPool, pool_pre_ping=True)
    test_engine = engine.execution_options(schema_translate_map={None: schema})
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    admin_id = uuid4()
    assistant_id = uuid4()
    question_id = uuid4()
    email_calls: list[tuple[str, str, str]] = []

    async def send_email(recipient: str, question: str, answer: str) -> None:
        email_calls.append((recipient, question, answer))

    async def fake_ingest(session, document_id):
        document = await session.get(Document, document_id)
        assert document is not None and document.status == "active"
        if document.title.startswith("Staff answer:"):
            assert document.pending_content == (
                "Question: What is the registration extension process? "
                "Approved staff answer: Submit the Registration Extension Form to the Registrar."
            )
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content=document.pending_content,
            embedding=[0.0] * 768,
        )
        session.add(chunk)
        document.pending_content = None
        await session.flush()
        return [chunk]

    monkeypatch.setattr(admin_router, "send_resolution_email", send_email)
    monkeypatch.setattr(admin_router, "ingest_approved_document", fake_ingest)

    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection.execution_options(schema_translate_map={None: schema})
                )
            )

        async with session_factory() as session:
            admin = AdminUser(
                id=admin_id,
                name="Test Admin",
                email="admin@example.test",
                password_hash=hash_password("a-long-test-password"),
                role="admin",
            )
            assistant = AdminUser(
                id=assistant_id,
                name="Test Assistant",
                email="assistant@example.test",
                password_hash=hash_password("another-long-password"),
                role="assistant",
            )
            question = QuestionLog(
                id=question_id,
                query_text="What is the registration extension process?",
                student_contact="student@example.test",
                evidence_status="insufficient",
                escalation_reason="no_evidence",
                escalated=True,
            )
            session.add_all([admin, assistant, question])
            await session.commit()

        async def override_db():
            async with session_factory() as session:
                yield session

        app.dependency_overrides[get_db] = override_db
        admin_token = create_access_token(admin_id)
        assistant_token = create_access_token(assistant_id)
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        assistant_headers = {"Authorization": f"Bearer {assistant_token}"}

        with TestClient(app) as client:
            assert client.get("/admin/questions/unanswered").status_code == 401

            login = client.post(
                "/admin/auth/login",
                json={"email": "admin@example.test", "password": "a-long-test-password"},
            )
            assert login.status_code == 200
            assert login.json()["token_type"] == "bearer"

            content = client.post(
                "/admin/content",
                headers=assistant_headers,
                json={
                    "title": "Assistant draft",
                    "authority_office": "Registrar",
                    "content": "Assistant submitted policy text awaiting approval.",
                },
            )
            assert content.status_code == 201
            assert content.json()["status"] == "pending"
            assert content.json()["chunk_count"] == 0
            assert client.post(
                f"/admin/content/{content.json()['id']}/approve",
                headers=assistant_headers,
                json={},
            ).status_code == 403

            answer = client.post(
                f"/admin/questions/{question_id}/answer",
                headers=assistant_headers,
                json={
                    "answer": "Submit the Registration Extension Form to the Registrar.",
                    "authority_office": "Registrar",
                },
            )
            assert answer.status_code == 200
            assert answer.json()["resolved"] is False
            assert answer.json()["document_status"] == "pending"
            assert not email_calls

            queue = client.get("/admin/questions/unanswered", headers=admin_headers)
            assert queue.status_code == 200
            assert len(queue.json()) == 1
            assert "student_contact" not in queue.json()[0]

            linked_document = answer.json()["document_id"]
            approval = client.post(
                f"/admin/content/{linked_document}/approve",
                headers=admin_headers,
                json={},
            )
            assert approval.status_code == 200
            assert approval.json()["status"] == "active"
            assert approval.json()["notification_status"] == "sent"
            assert approval.json()["chunk_count"] == 1
            assert email_calls == [
                (
                    "student@example.test",
                    "What is the registration extension process?",
                    "Submit the Registration Extension Form to the Registrar.",
                )
            ]

            queue_after_approval = client.get(
                "/admin/questions/unanswered", headers=admin_headers
            )
            assert queue_after_approval.json() == []

        async with session_factory() as session:
            saved_question = await session.get(QuestionLog, question_id)
            saved_document = await session.get(Document, linked_document)
            assert saved_question is not None and saved_question.escalated is False
            assert saved_question.resolved_by == admin_id
            assert saved_question.notification_status == "sent"
            assert saved_document is not None and saved_document.pending_content is None
    finally:
        app.dependency_overrides.pop(get_db, None)
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()


def test_assistant_upload_stays_pending_and_admin_direct_entry_publishes(monkeypatch) -> None:
    asyncio.run(_exercise_content_routes(monkeypatch))


async def _exercise_content_routes(monkeypatch) -> None:
    schema = f"test_{uuid4().hex}"
    engine = create_async_engine(settings.database_url, poolclass=NullPool, pool_pre_ping=True)
    test_engine = engine.execution_options(schema_translate_map={None: schema})
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    admin_id = uuid4()
    assistant_id = uuid4()
    old_document_id = uuid4()

    async def fake_ingest(session, document_id):
        document = await session.get(Document, document_id)
        assert document is not None and document.status == "active"
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content=document.pending_content,
            embedding=[0.0] * 768,
        )
        session.add(chunk)
        document.pending_content = None
        await session.flush()
        return [chunk]

    monkeypatch.setattr(admin_router, "ingest_approved_document", fake_ingest)
    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection.execution_options(schema_translate_map={None: schema})
                )
            )
        async with session_factory() as session:
            session.add_all(
                [
                    AdminUser(
                        id=admin_id,
                        name="Test Admin",
                        email="admin@example.test",
                        role="admin",
                    ),
                    AdminUser(
                        id=assistant_id,
                        name="Test Assistant",
                        email="assistant@example.test",
                        role="assistant",
                    ),
                ]
            )
            session.add(
                Document(
                    id=old_document_id,
                    title="Previous registration policy",
                    authority_office="Registrar",
                    source_type="manual",
                    status="active",
                    authored_by=admin_id,
                    approved_by=admin_id,
                    approved_at=datetime.now(UTC),
                )
            )
            await session.commit()

        async def override_db():
            async with session_factory() as session:
                yield session

        app.dependency_overrides[get_db] = override_db
        admin_headers = {"Authorization": f"Bearer {create_access_token(admin_id)}"}
        assistant_headers = {"Authorization": f"Bearer {create_access_token(assistant_id)}"}
        with TestClient(app) as client:
            invalid_supersedes = client.post(
                "/admin/content",
                headers=admin_headers,
                json={
                    "title": "Invalid supersedes reference",
                    "authority_office": "Registrar",
                    "content": "This should produce a validation response.",
                    "supersedes_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                },
            )
            assert invalid_supersedes.status_code == 422
            assert "existing active document" in invalid_supersedes.json()["detail"]

            uploaded = client.post(
                "/admin/content/upload",
                params={"title": "Uploaded assistant draft", "authority_office": "Registrar"},
                headers={**assistant_headers, "X-Filename": "policy.txt"},
                content=b"Uploaded content remains pending for admin review.",
            )
            assert uploaded.status_code == 201
            assert uploaded.json()["status"] == "pending"
            assert uploaded.json()["source_type"] == "uploaded"

            published = client.post(
                "/admin/content",
                headers=admin_headers,
                json={
                    "title": "Admin direct entry",
                    "authority_office": "Registrar",
                    "content": "Admin-authored content is approved and indexed immediately.",
                    "supersedes_id": str(old_document_id),
                },
            )
            assert published.status_code == 201
            assert published.json()["status"] == "active"
            assert published.json()["chunk_count"] == 1
            assert published.json()["supersedes_id"] == str(old_document_id)

        async with session_factory() as session:
            old_document = await session.get(Document, old_document_id)
            assert old_document is not None and old_document.status == "superseded"
    finally:
        app.dependency_overrides.pop(get_db, None)
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
