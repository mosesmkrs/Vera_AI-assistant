from datetime import UTC, date, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import (
    create_access_token,
    get_current_admin,
    hash_password,
    require_admin,
    verify_password,
)
from app.db.models import AdminUser, Document, DocumentChunk, QuestionLog
from app.db.session import get_db
from app.ingestion.direct_entry import create_pending_manual_document
from app.ingestion.parser import UnsupportedDocumentError
from app.ingestion.pipeline import (
    create_pending_uploaded_document,
    ingest_approved_document,
)
from app.notifications.email import EmailError, send_resolution_email

router = APIRouter(prefix="/admin", tags=["admin"])
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
EmailContact = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
CurrentAdmin = Annotated[AdminUser, Depends(get_current_admin)]
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


class LoginRequest(BaseModel):
    email: EmailContact
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = 8 * 60 * 60


class UserCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: EmailContact
    password: str = Field(min_length=12, max_length=256)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must contain non-whitespace text")
        return value


class UserResponse(BaseModel):
    id: UUID
    name: str
    email: str
    role: Literal["admin", "assistant"]


class ContentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    authority_office: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    publication_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    supersedes_id: UUID | None = None

    @model_validator(mode="after")
    def dates_are_ordered(self):
        if self.expiry_date and self.expiry_date < (self.effective_date or date.today()):
            raise ValueError("expiry_date must be on or after effective_date")
        return self


class ApprovalRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=500)
    authority_office: str | None = Field(default=None, min_length=1, max_length=200)
    publication_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    supersedes_id: UUID | None = None

    @model_validator(mode="after")
    def dates_are_ordered(self):
        if self.expiry_date and self.effective_date and self.expiry_date < self.effective_date:
            raise ValueError("expiry_date must be on or after effective_date")
        return self


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    authority_office: str
    source_type: str
    status: str
    publication_date: date | None
    effective_date: date
    expiry_date: date | None
    authored_by: UUID
    approved_by: UUID | None
    supersedes_id: UUID | None
    pending_content: str | None = None
    chunk_count: int = 0
    linked_question_id: UUID | None = None
    notification_status: str | None = None


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=50_000)
    authority_office: str = Field(min_length=1, max_length=200)

    @field_validator("answer", "authority_office")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must contain non-whitespace text")
        return value


class QuestionResponse(BaseModel):
    id: UUID
    query_text: str
    evidence_status: str
    escalation_reason: str | None
    created_at: datetime
    pending_answer_document_id: UUID | None = None


class AnswerSubmissionResponse(BaseModel):
    question_id: UUID
    document_id: UUID
    document_status: str
    resolved: bool
    notification_status: str | None


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, session: DatabaseSession) -> TokenResponse:
    normalized_email = str(request.email).casefold()
    user = await session.scalar(select(AdminUser).where(AdminUser.email == normalized_email))
    if user is None or user.role not in ("admin", "assistant"):
        raise _invalid_credentials()
    if not verify_password(request.password, user.password_hash):
        raise _invalid_credentials()
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_assistant(
    request: UserCreateRequest,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
) -> UserResponse:
    require_admin(current_admin)
    email = str(request.email).casefold()
    existing = await session.scalar(select(AdminUser).where(AdminUser.email == email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="An account with that email already exists")
    user = AdminUser(
        name=request.name,
        email=email,
        password_hash=hash_password(request.password),
        role="assistant",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return UserResponse(id=user.id, name=user.name, email=user.email, role=user.role)


@router.post("/content", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def submit_content(
    request: ContentCreateRequest,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
) -> DocumentResponse:
    await _validate_supersedes_id(session, request.supersedes_id)
    document = await create_pending_manual_document(
        session,
        title=request.title,
        authority_office=request.authority_office,
        authored_by=current_admin.id,
        content=request.content,
        effective_date=request.effective_date,
        publication_date=request.publication_date,
        expiry_date=request.expiry_date,
    )
    if current_admin.role != "admin":
        document.supersedes_id = request.supersedes_id
    chunk_count = 0
    if current_admin.role == "admin":
        chunk_count = await _publish_document(
            session, document, current_admin, supersedes_id=request.supersedes_id
        )
    await session.commit()
    await session.refresh(document)
    return _document_response(document, chunk_count=chunk_count)


@router.post("/content/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def submit_upload(
    request: Request,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
    title: Annotated[str, Query(min_length=1, max_length=500)],
    authority_office: Annotated[str, Query(min_length=1, max_length=200)],
    filename: Annotated[str, Header(alias="X-Filename")],
    publication_date: date | None = None,
    effective_date: date | None = None,
    expiry_date: date | None = None,
    supersedes_id: UUID | None = None,
) -> DocumentResponse:
    content_buffer = bytearray()
    async for part in request.stream():
        content_buffer.extend(part)
        if len(content_buffer) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Maximum upload size is 10 MiB")
    content = bytes(content_buffer)
    await _validate_supersedes_id(session, supersedes_id)
    try:
        document = await create_pending_uploaded_document(
            session,
            title=title,
            authority_office=authority_office,
            authored_by=current_admin.id,
            filename=filename,
            content=content,
        )
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    document.publication_date = publication_date
    if effective_date is not None:
        document.effective_date = effective_date
    document.expiry_date = expiry_date
    document.supersedes_id = supersedes_id
    _validate_dates(document.effective_date, document.expiry_date)
    chunk_count = 0
    if current_admin.role == "admin":
        chunk_count = await _publish_document(
            session, document, current_admin, supersedes_id=supersedes_id
        )
    await session.commit()
    await session.refresh(document)
    return _document_response(document, chunk_count=chunk_count)


@router.get("/content/pending", response_model=list[DocumentResponse])
async def pending_content(session: DatabaseSession, current_admin: CurrentAdmin):
    require_admin(current_admin)
    documents = (
        await session.scalars(
            select(Document).where(Document.status == "pending").order_by(Document.created_at)
        )
    ).all()
    linked = await _linked_question_ids(session, [item.id for item in documents])
    return [
        _document_response(item, linked_question_id=linked.get(item.id)) for item in documents
    ]


@router.post("/content/{document_id}/approve", response_model=DocumentResponse)
async def approve_content(
    document_id: UUID,
    request: ApprovalRequest,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
) -> DocumentResponse:
    require_admin(current_admin)
    document = await _locked_pending_document(session, document_id)
    if request.content is not None:
        document.pending_content = request.content.strip()
    if request.title is not None:
        document.title = request.title.strip()
    if request.authority_office is not None:
        document.authority_office = request.authority_office.strip()
    if request.publication_date is not None:
        document.publication_date = request.publication_date
    if request.effective_date is not None:
        document.effective_date = request.effective_date
    if request.expiry_date is not None:
        document.expiry_date = request.expiry_date
    if request.supersedes_id is not None:
        await _validate_supersedes_id(session, request.supersedes_id)
        document.supersedes_id = request.supersedes_id
    _validate_dates(document.effective_date, document.expiry_date)

    chunk_count = await _publish_document(
        session, document, current_admin, supersedes_id=document.supersedes_id
    )
    linked_questions = await _mark_linked_questions_resolved(session, document.id, current_admin)
    await session.commit()
    notification_status = await _notify_linked_questions(session, linked_questions)
    await session.refresh(document)
    return _document_response(
        document,
        chunk_count=chunk_count,
        linked_question_id=linked_questions[0].id if linked_questions else None,
        notification_status=notification_status,
    )


@router.post("/content/{document_id}/reject", response_model=DocumentResponse)
async def reject_content(
    document_id: UUID,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
) -> DocumentResponse:
    require_admin(current_admin)
    document = await _locked_pending_document(session, document_id)
    linked_questions = (
        await session.scalars(
            select(QuestionLog)
            .where(QuestionLog.answer_document_id == document.id)
            .with_for_update()
        )
    ).all()
    for question in linked_questions:
        question.answer_document_id = None
        question.answered_by = None
        question.answered_at = None
    document.status = "rejected"
    await session.commit()
    await session.refresh(document)
    return _document_response(document, linked_question_id=linked_questions[0].id if linked_questions else None)


@router.get("/questions/unanswered", response_model=list[QuestionResponse])
async def unanswered_questions(session: DatabaseSession, current_admin: CurrentAdmin):
    questions = (
        await session.scalars(
            select(QuestionLog)
            .where(QuestionLog.escalated.is_(True), QuestionLog.resolved_at.is_(None))
            .order_by(QuestionLog.created_at)
        )
    ).all()
    pending_doc_ids = [item.answer_document_id for item in questions if item.answer_document_id]
    linked = await _linked_question_ids(session, pending_doc_ids)
    document_to_question = {doc_id: q_id for doc_id, q_id in linked.items()}
    return [
        QuestionResponse(
            id=item.id,
            query_text=item.query_text,
            evidence_status=item.evidence_status,
            escalation_reason=item.escalation_reason,
            created_at=item.created_at,
            pending_answer_document_id=(
                item.answer_document_id
                if item.answer_document_id in document_to_question
                else None
            ),
        )
        for item in questions
    ]


@router.post("/questions/{question_id}/answer", response_model=AnswerSubmissionResponse)
async def answer_escalated_question(
    question_id: UUID,
    request: AnswerRequest,
    session: DatabaseSession,
    current_admin: CurrentAdmin,
) -> AnswerSubmissionResponse:
    question = await session.scalar(
        select(QuestionLog).where(QuestionLog.id == question_id).with_for_update()
    )
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")
    if not question.escalated or question.resolved_at is not None:
        raise HTTPException(status_code=409, detail="Question is not awaiting resolution")
    if question.answer_document_id is not None:
        raise HTTPException(status_code=409, detail="A staff answer is already awaiting review")

    document = await create_pending_manual_document(
        session,
        title=f"Staff answer: {question.query_text[:480]}",
        authority_office=request.authority_office,
        authored_by=current_admin.id,
        content=(
            f"Question: {question.query_text}\n\n"
            f"Approved staff answer: {request.answer}"
        ),
    )
    question.answer_document_id = document.id
    question.answered_by = current_admin.id
    question.answered_at = datetime.now(UTC)
    if current_admin.role == "admin":
        await _publish_document(session, document, current_admin)
        question.resolved_by = current_admin.id
        question.resolved_at = datetime.now(UTC)
        question.escalated = False
        question.notification_status = "pending" if question.student_contact else "not_applicable"
        await session.commit()
        notification_status = await _notify_question(session, question.id)
        return AnswerSubmissionResponse(
            question_id=question.id,
            document_id=document.id,
            document_status=document.status,
            resolved=True,
            notification_status=notification_status,
        )

    await session.commit()
    return AnswerSubmissionResponse(
        question_id=question.id,
        document_id=document.id,
        document_status=document.status,
        resolved=False,
        notification_status=None,
    )


async def _publish_document(
    session: AsyncSession,
    document: Document,
    approving_admin: AdminUser,
    *,
    supersedes_id: UUID | None = None,
) -> int:
    if not document.pending_content:
        raise HTTPException(status_code=422, detail="Document has no content to publish")
    document.status = "active"
    document.approved_by = approving_admin.id
    document.approved_at = datetime.now(UTC)
    if supersedes_id is not None:
        if supersedes_id == document.id:
            raise HTTPException(status_code=422, detail="A document cannot supersede itself")
        old_document = await session.scalar(
            select(Document).where(Document.id == supersedes_id).with_for_update()
        )
        if old_document is None or old_document.status != "active":
            raise HTTPException(status_code=422, detail="Only an active document can be superseded")
        document.supersedes_id = old_document.id
        old_document.status = "superseded"
    chunks = await ingest_approved_document(session, document.id)
    return len(chunks)


async def _validate_supersedes_id(session: AsyncSession, supersedes_id: UUID | None) -> None:
    if supersedes_id is None:
        return
    old_document = await session.scalar(
        select(Document).where(Document.id == supersedes_id).with_for_update()
    )
    if old_document is None or old_document.status != "active":
        raise HTTPException(
            status_code=422,
            detail="supersedes_id must identify an existing active document",
        )


async def _locked_pending_document(session: AsyncSession, document_id: UUID) -> Document:
    document = await session.scalar(
        select(Document).where(Document.id == document_id).with_for_update()
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status != "pending":
        raise HTTPException(status_code=409, detail="Document is not pending review")
    return document


async def _mark_linked_questions_resolved(
    session: AsyncSession,
    document_id: UUID,
    admin: AdminUser,
) -> list[QuestionLog]:
    questions = (
        await session.scalars(
            select(QuestionLog)
            .where(QuestionLog.answer_document_id == document_id)
            .with_for_update()
        )
    ).all()
    for question in questions:
        question.resolved_by = admin.id
        question.resolved_at = datetime.now(UTC)
        question.escalated = False
        question.notification_status = "pending" if question.student_contact else "not_applicable"
    return list(questions)


async def _notify_linked_questions(
    session: AsyncSession,
    questions: list[QuestionLog],
) -> str | None:
    outcomes = [await _notify_question(session, question.id) for question in questions]
    if not outcomes:
        return None
    return outcomes[0] if len(set(outcomes)) == 1 else "mixed"


async def _notify_question(session: AsyncSession, question_id: UUID) -> str:
    question = await session.get(QuestionLog, question_id)
    if question is None:
        return "failed"
    if not question.student_contact:
        question.notification_status = "not_applicable"
        await session.commit()
        return question.notification_status

    document = await session.get(Document, question.answer_document_id)
    if document is None or document.status != "active" or document.pending_content is not None:
        # Published answers clear pending_content during indexing.
        question.notification_status = "failed"
        await session.commit()
        return question.notification_status

    chunks = (
        await session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.chunk_index)
        )
    ).all()
    answer_text = " ".join(chunk.content for chunk in chunks)
    answer_prefix = f"Question: {question.query_text} Approved staff answer:"
    if answer_text.startswith(answer_prefix):
        answer_text = answer_text[len(answer_prefix) :].strip()
    if not answer_text:
        question.notification_status = "failed"
        await session.commit()
        return question.notification_status
    try:
        await send_resolution_email(
            question.student_contact,
            question.query_text,
            answer_text,
        )
    except EmailError:
        question.notification_status = "failed"
    else:
        question.notification_status = "sent"
        question.notification_sent_at = datetime.now(UTC)
    await session.commit()
    return question.notification_status


async def _linked_question_ids(
    session: AsyncSession,
    document_ids: list[UUID],
) -> dict[UUID, UUID]:
    if not document_ids:
        return {}
    rows = (
        await session.execute(
            select(QuestionLog.answer_document_id, QuestionLog.id).where(
                QuestionLog.answer_document_id.in_(document_ids)
            )
        )
    ).all()
    return {document_id: question_id for document_id, question_id in rows}


def _document_response(
    document: Document,
    *,
    chunk_count: int = 0,
    linked_question_id: UUID | None = None,
    notification_status: str | None = None,
) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        authority_office=document.authority_office,
        source_type=document.source_type,
        status=document.status,
        publication_date=document.publication_date,
        effective_date=document.effective_date,
        expiry_date=document.expiry_date,
        authored_by=document.authored_by,
        approved_by=document.approved_by,
        supersedes_id=document.supersedes_id,
        pending_content=document.pending_content,
        chunk_count=chunk_count,
        linked_question_id=linked_question_id,
        notification_status=notification_status,
    )


def _validate_dates(effective_date: date, expiry_date: date | None) -> None:
    if expiry_date is not None and expiry_date < effective_date:
        raise HTTPException(status_code=422, detail="expiry_date must be on or after effective_date")


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
