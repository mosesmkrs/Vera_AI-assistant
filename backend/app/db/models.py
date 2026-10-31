from datetime import date, datetime
from typing import Optional
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

EMBEDDING_DIMENSION = 768  # intfloat/multilingual-e5-base


class AdminUser(Base):
    __tablename__ = "admin_users"
    __table_args__ = (
        CheckConstraint("role IN ('admin', 'assistant')", name="ck_admin_users_role"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="assistant")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    authored_documents: Mapped[list["Document"]] = relationship(
        back_populates="author", foreign_keys="Document.authored_by"
    )
    approved_documents: Mapped[list["Document"]] = relationship(
        back_populates="approver", foreign_keys="Document.approved_by"
    )
    answered_questions: Mapped[list["QuestionLog"]] = relationship(
        back_populates="answering_user", foreign_keys="QuestionLog.answered_by"
    )
    resolved_questions: Mapped[list["QuestionLog"]] = relationship(
        back_populates="resolver", foreign_keys="QuestionLog.resolved_by"
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("source_type IN ('uploaded', 'manual')", name="ck_documents_source_type"),
        CheckConstraint(
            "status IN ('pending', 'active', 'superseded', 'withdrawn', 'rejected')",
            name="ck_documents_status",
        ),
        CheckConstraint(
            "expiry_date IS NULL OR expiry_date >= effective_date",
            name="ck_documents_expiry_after_effective",
        ),
        CheckConstraint(
            "supersedes_id IS NULL OR supersedes_id <> id", name="ck_documents_not_self_superseding"
        ),
        CheckConstraint(
            "status NOT IN ('active', 'superseded', 'withdrawn') OR approved_by IS NOT NULL",
            name="ck_documents_published_have_approver",
        ),
        Index("ix_documents_status_effective_expiry", "status", "effective_date", "expiry_date"),
        Index("ix_documents_supersedes_id", "supersedes_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    authority_office: Mapped[str] = mapped_column(String(200), nullable=False)
    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    pending_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, default=date.today)
    expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    authored_by: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("admin_users.id", ondelete="RESTRICT"), nullable=False
    )
    approved_by: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("admin_users.id", ondelete="RESTRICT"), nullable=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    author: Mapped[AdminUser] = relationship(
        back_populates="authored_documents", foreign_keys=[authored_by]
    )
    approver: Mapped[AdminUser | None] = relationship(
        back_populates="approved_documents", foreign_keys=[approved_by]
    )
    supersedes: Mapped[Optional["Document"]] = relationship(
        remote_side="Document.id", back_populates="superseded_by"
    )
    superseded_by: Mapped[list["Document"]] = relationship(back_populates="supersedes")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    resolved_questions: Mapped[list["QuestionLog"]] = relationship(
        back_populates="answer_document", foreign_keys="QuestionLog.answer_document_id"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
        Index("ix_document_chunks_document_id", "document_id"),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_document_chunks_content_fts_simple",
            sql_text("to_tsvector('simple', content)"),
            postgresql_using="gin",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class QuestionLog(Base):
    __tablename__ = "questions_log"
    __table_args__ = (
        CheckConstraint(
            "evidence_status IN ('sufficient', 'insufficient', 'conflicting')",
            name="ck_questions_log_evidence_status",
        ),
        CheckConstraint(
            "escalation_reason IS NULL OR escalation_reason IN ('no_evidence', 'conflicting_evidence')",
            name="ck_questions_log_escalation_reason",
        ),
        CheckConstraint(
            "notification_status IN ('not_ready', 'pending', 'sent', 'failed', 'not_applicable')",
            name="ck_questions_log_notification_status",
        ),
        CheckConstraint(
            "retrieval_confidence IS NULL OR (retrieval_confidence >= 0 AND retrieval_confidence <= 1)",
            name="ck_questions_log_retrieval_confidence",
        ),
        CheckConstraint(
            "answer_confidence IS NULL OR (answer_confidence >= 0 AND answer_confidence <= 1)",
            name="ck_questions_log_answer_confidence",
        ),
        Index("ix_questions_log_escalated_created", "escalated", "created_at"),
        Index("ix_questions_log_notification_status", "notification_status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    student_contact: Mapped[str | None] = mapped_column(String(320), nullable=True)
    retrieval_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Grounding verifier's answer support score"
    )
    evidence_status: Mapped[str] = mapped_column(String(20), nullable=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    answered_by: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("admin_users.id", ondelete="RESTRICT"), nullable=True
    )
    answer_document_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("documents.id", ondelete="RESTRICT"), nullable=True
    )
    resolved_by: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("admin_users.id", ondelete="RESTRICT"), nullable=True
    )
    notification_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="not_ready", server_default="not_ready"
    )
    notification_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    answering_user: Mapped[AdminUser | None] = relationship(
        back_populates="answered_questions", foreign_keys=[answered_by]
    )
    resolver: Mapped[AdminUser | None] = relationship(
        back_populates="resolved_questions", foreign_keys=[resolved_by]
    )
    answer_document: Mapped[Document | None] = relationship(
        back_populates="resolved_questions", foreign_keys=[answer_document_id]
    )

